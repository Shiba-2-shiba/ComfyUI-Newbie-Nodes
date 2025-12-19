import torch
import numpy as np
import os
import json
import traceback

# コア機能をインポート
from .magcache_newbie_core import (
    get_model_hash, load_mag_ratios, save_mag_ratios,
    interpolate_mag_ratios, MagCacheState, CALIBRATION_DIR
)

# --- ヘルパー関数 ---
def _get_model_name(model, manual_name=""):
    if manual_name and manual_name.strip():
        return manual_name.strip()
    if hasattr(model, "magcache_source_ckpt_name"):
        return model.magcache_source_ckpt_name
    if hasattr(model, "model") and hasattr(model.model, "magcache_source_ckpt_name"):
        return model.model.magcache_source_ckpt_name
    if hasattr(model, "ckpt_path"):
        return os.path.basename(model.ckpt_path)
    print("[MagCache-NewBie] Warning: Could not find model name automatically. Please use 'manual_model_name'.")
    return "UnknownModel"

# --- グローバル変数 (キャリブレーション用) ---
calibration_data = {
    'cond': [], 'uncond': [],
    'last_cond_eps': None, 'last_uncond_eps': None
}

def reset_calibration_data():
    global calibration_data
    calibration_data = { 'cond': [], 'uncond': [], 'last_cond_eps': None, 'last_uncond_eps': None }

# --- Calibration Hook Function ---
def newbie_calibration_hook(self, x, t, *args, **kwargs):
    # 【修正】popではなくgetを使用し、元のkwargsを破壊しないように変更
    transformer_options = kwargs.get("transformer_options", {})
    if not transformer_options and "c" in kwargs:
         transformer_options = kwargs["c"].get("transformer_options", {})
            
    computed_output = self.magcache_original_forward(x, t, *args, **kwargs)
    try:
        output_tensor = computed_output[0] if isinstance(computed_output, tuple) else computed_output
        if not isinstance(output_tensor, torch.Tensor): return computed_output

        current_batch_size = output_tensor.shape[0]
        input_batch_size = x.shape[0] if isinstance(x, torch.Tensor) else current_batch_size
        
        cond_or_uncond = transformer_options.get("cond_or_uncond", None)

        if current_batch_size == input_batch_size: 
            if current_batch_size % 2 == 0:
                # バッチ内で分割されている場合 (Batch >= 2)
                half_len = current_batch_size // 2
                current_cond_eps = output_tensor[:half_len].mean(dim=0, keepdim=True).detach()
                current_uncond_eps = output_tensor[half_len:].mean(dim=0, keepdim=True).detach()
                
                if calibration_data['last_cond_eps'] is not None:
                    ratio = torch.linalg.norm(current_cond_eps) / (torch.linalg.norm(calibration_data['last_cond_eps']) + 1e-9)
                    calibration_data['cond'].append(ratio.item())
                calibration_data['last_cond_eps'] = current_cond_eps
                
                if calibration_data['last_uncond_eps'] is not None:
                    ratio = torch.linalg.norm(current_uncond_eps) / (torch.linalg.norm(calibration_data['last_uncond_eps']) + 1e-9)
                    calibration_data['uncond'].append(ratio.item())
                calibration_data['last_uncond_eps'] = current_uncond_eps
            else:
                # 分割実行されている場合 (Batch = 1など)
                is_uncond = False
                if cond_or_uncond is not None and 1 in cond_or_uncond: 
                    is_uncond = True
                
                current_eps = output_tensor.mean(dim=0, keepdim=True).detach()
                if is_uncond:
                    if calibration_data['last_uncond_eps'] is not None:
                        ratio = torch.linalg.norm(current_eps) / (torch.linalg.norm(calibration_data['last_uncond_eps']) + 1e-9)
                        calibration_data['uncond'].append(ratio.item())
                    calibration_data['last_uncond_eps'] = current_eps
                else:
                    if calibration_data['last_cond_eps'] is not None:
                        ratio = torch.linalg.norm(current_eps) / (torch.linalg.norm(calibration_data['last_cond_eps']) + 1e-9)
                        calibration_data['cond'].append(ratio.item())
                    calibration_data['last_cond_eps'] = current_eps
        else:
            current_fused_eps = output_tensor.mean(dim=0, keepdim=True).detach()
            if calibration_data['last_cond_eps'] is not None:
                ratio = torch.linalg.norm(current_fused_eps) / (torch.linalg.norm(calibration_data['last_cond_eps']) + 1e-9)
                calibration_data['cond'].append(ratio.item())
                calibration_data['uncond'].append(ratio.item())
            calibration_data['last_cond_eps'] = current_fused_eps
            calibration_data['last_uncond_eps'] = current_fused_eps
    except Exception as e:
        print(f"[MagCache-NewBie] Calibration error: {e}")
        traceback.print_exc()
    return computed_output

# --- Inference Hook Function ---
def newbie_inference_hook(self, x, t, *args, **kwargs):
    params = getattr(self, 'magcache_params', None)
    state_manager = getattr(self, 'magcache_state', None)
    
    # 【修正】popではなくgetを使用。元のkwargsを維持する。
    transformer_options = kwargs.get("transformer_options", {})
    if not transformer_options and "c" in kwargs:
         transformer_options = kwargs["c"].get("transformer_options", {})
    
    if not params:
        return self.magcache_original_forward(x, t, *args, **kwargs)

    current_step = getattr(self, 'magcache_current_step', 0)
    
    cond_or_uncond = transformer_options.get("cond_or_uncond", None)

    # バッチサイズから現在の実行モードを判定
    current_batch_size = x.shape[0]
    mode = 'combined'
    
    if current_batch_size % 2 != 0:
        # 奇数バッチの場合、transformer_optionsで判定
        is_uncond = False
        if cond_or_uncond is not None and 1 in cond_or_uncond:
            is_uncond = True
        
        mode = 'uncond_only' if is_uncond else 'cond_only'

    skip_this_step = False
    debug_info = {}
    in_range = params['start_step_abs'] <= current_step < params['end_step_abs']

    if in_range:
        def check_skip(guidance_type, ratio_offset):
            state = state_manager.get_state(guidance_type)
            ratio_index = current_step * 2 + ratio_offset
            
            # Cache check
            if state['residual_cache'] is None:
                debug_info[guidance_type] = {"status": "NoData (Cache is None)"}
                return False
            if ratio_index >= len(params['mag_ratios']):
                debug_info[guidance_type] = {"status": "NoData (Index Error)"}
                return False
                
            ratio = params['mag_ratios'][ratio_index]
            new_acc_ratio = state['accumulated_ratio'] * ratio
            new_acc_steps = state['accumulated_steps'] + 1
            new_acc_err = state['accumulated_err'] + abs(1.0 - new_acc_ratio)
            
            is_err_ok = new_acc_err < params['delta_threshold']
            is_k_ok = new_acc_steps <= params['K_skips']
            
            debug_info[guidance_type] = {
                "err": new_acc_err, "k": new_acc_steps, "ratio": ratio,
                "ok": is_err_ok and is_k_ok,
                "reason": "OK" if (is_err_ok and is_k_ok) else ("ErrHigh" if not is_err_ok else "K_Max")
            }

            if is_err_ok and is_k_ok:
                return True
            else:
                return False

        # モードに応じたスキップ判定
        if mode == 'combined':
            if check_skip("cond", 0) and check_skip("uncond", 1):
                skip_this_step = True
        elif mode == 'cond_only':
            if check_skip("cond", 0):
                skip_this_step = True
        elif mode == 'uncond_only':
            if check_skip("uncond", 1):
                skip_this_step = True

    # --- スキップ実行 ---
    if skip_this_step:
        # ステート更新 (判定時に計算した値を適用)
        def update_state(guidance_type, ratio_offset):
            state = state_manager.get_state(guidance_type)
            ratio_index = current_step * 2 + ratio_offset
            ratio = params['mag_ratios'][ratio_index]
            state['accumulated_ratio'] *= ratio
            state['accumulated_steps'] += 1
            state['accumulated_err'] += abs(1.0 - state['accumulated_ratio'])

        if mode == 'combined':
            update_state("cond", 0)
            update_state("uncond", 1)
            c_inf = debug_info["cond"]
            print(f"[MagCache] Step {current_step:2d} | ✅ SKIP (Comb) | Err: {c_inf['err']:.4f} | K: {c_inf['k']}")
            cached_cond = state_manager.get_state('cond')['residual_cache']
            cached_uncond = state_manager.get_state('uncond')['residual_cache']
            return torch.cat([cached_cond, cached_uncond], dim=0).to(x.device, x.dtype)
            
        elif mode == 'cond_only':
            update_state("cond", 0)
            c_inf = debug_info["cond"]
            print(f"[MagCache] Step {current_step:2d} | ✅ SKIP (Cond) | Err: {c_inf['err']:.4f} | K: {c_inf['k']}")
            return state_manager.get_state('cond')['residual_cache'].to(x.device, x.dtype)
            
        elif mode == 'uncond_only':
            update_state("uncond", 1)
            u_inf = debug_info["uncond"]
            print(f"[MagCache] Step {current_step:2d} | ✅ SKIP (Uncd) | Err: {u_inf['err']:.4f} | K: {u_inf['k']}")
            return state_manager.get_state('uncond')['residual_cache'].to(x.device, x.dtype)
    
    # --- 計算実行 (スキップしない場合) ---
    if in_range:
         target_type = "cond" if mode != 'uncond_only' else "uncond"
         reason = debug_info.get(target_type, {}).get("reason", None)
         status = debug_info.get(target_type, {}).get("status", "Unknown")
         final_reason = reason if reason else status
         
         # ログを少し抑制しつつ出力
         # print(f"[MagCache] Step {current_step:2d} | 🟦 RUN  | Reason: {final_reason} | Mode: {mode}")

    # ステートリセット
    if mode == 'combined':
        state_manager.get_state("cond").update({'accumulated_err': 0.0, 'accumulated_steps': 0, 'accumulated_ratio': 1.0})
        state_manager.get_state("uncond").update({'accumulated_err': 0.0, 'accumulated_steps': 0, 'accumulated_ratio': 1.0})
    elif mode == 'cond_only':
        state_manager.get_state("cond").update({'accumulated_err': 0.0, 'accumulated_steps': 0, 'accumulated_ratio': 1.0})
    elif mode == 'uncond_only':
        state_manager.get_state("uncond").update({'accumulated_err': 0.0, 'accumulated_steps': 0, 'accumulated_ratio': 1.0})
    
    # 計算実行
    computed_output = self.magcache_original_forward(x, t, *args, **kwargs)
    output_tensor = computed_output[0] if isinstance(computed_output, tuple) else computed_output
    
    # --- Cache Storage ---
    if isinstance(output_tensor, torch.Tensor):
        if mode == 'combined':
            if output_tensor.shape[0] % 2 == 0:
                half_len = output_tensor.shape[0] // 2
                state_manager.store_residual(output_tensor[:half_len], "cond")
                state_manager.store_residual(output_tensor[half_len:], "uncond")
            else:
                print(f"[MagCache-Error] Step {current_step}: Mode combined but odd batch size.")
        elif mode == 'cond_only':
            state_manager.store_residual(output_tensor, "cond")
        elif mode == 'uncond_only':
            state_manager.store_residual(output_tensor, "uncond")
    else:
        # print(f"[MagCache-Error] Step {current_step}: Output is not a tensor. Cache not saved.")
        pass
        
    return computed_output


# --- Node Definitions ---

class MagCacheNewBieCalibration:
    @classmethod
    def INPUT_TYPES(s):
        return { 
            "required": { "model": ("MODEL",) },
            "optional": { "manual_model_name": ("STRING", {"default": "", "multiline": False, "placeholder": "newbie"}) }
        }
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply_calibration"
    CATEGORY = "NewBie/MagCache"
    TITLE = "Calibrate MagCache for NewBie"

    def apply_calibration(self, model, manual_model_name=""):
        model_name = _get_model_name(model, manual_model_name)
        if not model_name: 
            return (model,)
            
        comfy_model = model.model
        target_model = comfy_model
        if hasattr(comfy_model, "diffusion_model"): target_model = comfy_model.diffusion_model
        
        target_method = 'forward'
        if not hasattr(target_model, target_method): return (model,)

        if not hasattr(target_model, 'magcache_original_forward'):
            setattr(target_model, 'magcache_original_forward', getattr(target_model, target_method))
        setattr(target_model, target_method, newbie_calibration_hook.__get__(target_model, type(target_model)))
        
        def unet_wrapper_function(model_function, kwargs):
            transformer_options = kwargs.get('c', {}).get('transformer_options', {})
            all_sigmas = transformer_options.get('sample_sigmas') if transformer_options else None
            
            current_sigma = kwargs.get('timestep')
            if isinstance(current_sigma, torch.Tensor): current_sigma = current_sigma[0]
            elif isinstance(current_sigma, (list, tuple)): current_sigma = current_sigma[0]

            if all_sigmas is not None:
                comparison = torch.isclose(all_sigmas, current_sigma)
                indices = torch.where(comparison)[0]
                target_model.magcache_current_step = indices[0].item() if len(indices) > 0 else 0
                target_model.magcache_total_steps = len(all_sigmas) -1 if len(all_sigmas) > 1 else len(all_sigmas)
            else:
                target_model.magcache_current_step = 0
                target_model.magcache_total_steps = 1

            current_step = target_model.magcache_current_step
            if current_step == 0:
                print(f"[MagCache-NewBie] Start Calibration for: {model_name}")
                reset_calibration_data()
                
            output = model_function(kwargs['input'], kwargs['timestep'], **kwargs['c'])
            
            if target_model.magcache_total_steps > 0 and current_step == target_model.magcache_total_steps - 1:
                cond_len = len(calibration_data['cond'])
                uncond_len = len(calibration_data['uncond'])
                print(f"[MagCache-NewBie] Calibration Finished. Data Points - Cond: {cond_len}, Uncond: {uncond_len}")
                
                # 【修正】Condデータがあれば保存するように条件を緩和 (CFG=1対応)
                if cond_len > 0:
                    model_hash = get_model_hash(model_name)
                    
                    # Uncondがない場合はCondデータを複製して補完 (データ形式を合わせるため)
                    c_data = calibration_data['cond']
                    u_data = calibration_data['uncond'] if uncond_len > 0 else c_data
                    
                    min_len = min(len(c_data), len(u_data))
                    interleaved = np.empty((min_len * 2,), dtype=np.float32)
                    interleaved[0::2] = c_data[:min_len]
                    interleaved[1::2] = u_data[:min_len]
                    
                    save_mag_ratios(model_hash, interleaved.tolist(), model_name)
                
                if hasattr(target_model, 'magcache_original_forward'):
                    setattr(target_model, target_method, target_model.magcache_original_forward)
                    delattr(target_model, 'magcache_original_forward')
            return output

        new_model = model.clone()
        new_model.set_model_unet_function_wrapper(unet_wrapper_function)
        return (new_model,)

class MagCacheNewBie:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",), 
                "enabled": ("BOOLEAN", {"default": True}),
                "Magcache_thresh": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01}),
                "Magcache_K": ("INT", {"default": 2, "min": 1, "max": 10, "step": 1}),
                "retention_ratio": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01}),
                "exclude_last_step": ("BOOLEAN", {"default": True}),
            },
            "optional": { "manual_model_name": ("STRING", {"default": "", "placeholder": "newbie"}) }
        }
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply_magcache"
    CATEGORY = "NewBie/MagCache"
    TITLE = "MagCache for NewBie"

    def apply_magcache(self, model, enabled, Magcache_thresh, Magcache_K, retention_ratio, exclude_last_step, manual_model_name=""):
        if not enabled: return (model,)

        model_name = _get_model_name(model, manual_model_name)
        model_hash = get_model_hash(model_name)
        
        print(f"\n[MagCache-Debug] === Loading Check ===")
        print(f"[MagCache-Debug] Target Model Name: '{model_name}'")
        print(f"[MagCache-Debug] Target Hash:       '{model_hash}'")
        
        mag_ratios = load_mag_ratios(model_hash)
        if mag_ratios is None:
            print(f"[MagCache-Error] ❌ Load Failed! No data found for hash: {model_hash}")
            return (model,)
        else:
            print(f"[MagCache-Debug] ✅ Load Success! Found {len(mag_ratios)} data points.")

        comfy_model = model.model
        target_model = comfy_model
        if hasattr(comfy_model, "diffusion_model"): target_model = comfy_model.diffusion_model

        target_method = 'forward'
        if not hasattr(target_model, 'magcache_original_forward'):
            setattr(target_model, 'magcache_original_forward', getattr(target_model, target_method))
        setattr(target_model, target_method, newbie_inference_hook.__get__(target_model, type(target_model)))
        
        if not hasattr(target_model, 'magcache_state'):
            target_model.magcache_state = MagCacheState()

        def unet_wrapper_function(model_function, kwargs):
            transformer_options = kwargs.get('c', {}).get('transformer_options', {})
            all_sigmas = transformer_options.get('sample_sigmas') if transformer_options else None
            current_sigma = kwargs.get('timestep')
            if isinstance(current_sigma, torch.Tensor): current_sigma = current_sigma[0]
            elif isinstance(current_sigma, (list, tuple)): current_sigma = current_sigma[0]

            if all_sigmas is not None:
                comparison = torch.isclose(all_sigmas, current_sigma)
                indices = torch.where(comparison)[0]
                current_step_val = indices[0].item() if len(indices) > 0 else 0
                target_model.magcache_current_step = current_step_val
                target_model.magcache_total_steps = len(all_sigmas) -1 if len(all_sigmas) > 1 else len(all_sigmas)
            else:
                target_model.magcache_current_step = 0
                target_model.magcache_total_steps = 1
            
            if target_model.magcache_current_step == 0:
                state = target_model.magcache_state
                state.reset()
                interpolated_ratios = interpolate_mag_ratios(mag_ratios, target_model.magcache_total_steps)
                
                if interpolated_ratios is not None:
                    target_model.magcache_params = {
                        'mag_ratios': interpolated_ratios, 
                        'delta_threshold': Magcache_thresh, 
                        'K_skips': Magcache_K,
                        'start_step_abs': int(target_model.magcache_total_steps * retention_ratio),
                        'end_step_abs': target_model.magcache_total_steps - 1 if exclude_last_step else target_model.magcache_total_steps
                    }

            return model_function(kwargs['input'], kwargs['timestep'], **kwargs['c'])
        
        new_model = model.clone()
        new_model.set_model_unet_function_wrapper(unet_wrapper_function)
        return (new_model,)