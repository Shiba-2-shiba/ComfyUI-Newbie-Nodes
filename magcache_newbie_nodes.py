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

# --- sigma -> step 解決（iscloseに依存しない堅牢版） ---
def _resolve_step_and_total(sample_sigmas, current_sigma):
    """Return (current_step, total_steps).

    - sample_sigmas: 1D Tensor (len = steps+1 が多い)
    - current_sigma: scalar (Tensor/float)

    iscloseで一致しないケース（dtype差/丸め誤差/補間）に備えて、
    単調列の区間探索 -> 最近傍 の順で解決する。
    """
    if sample_sigmas is None:
        return 0, 1

    sig = sample_sigmas
    if isinstance(sig, (list, tuple)):
        sig = torch.tensor(sig)

    if not isinstance(sig, torch.Tensor) or sig.numel() == 0:
        return 0, 1

    sig = sig.flatten()

    cur = current_sigma
    if isinstance(cur, torch.Tensor):
        cur = cur.flatten()[0]
    elif isinstance(cur, (list, tuple)):
        cur = cur[0]
        if isinstance(cur, torch.Tensor):
            cur = cur.flatten()[0]
    cur_f = float(cur)

    # 1) iscloseで直撃
    try:
        close = torch.isclose(sig, torch.tensor(cur_f, device=sig.device, dtype=sig.dtype), rtol=1e-4, atol=1e-6)
        idxs = torch.where(close)[0]
        if idxs.numel() > 0:
            idx = int(idxs[0].item())
        else:
            raise RuntimeError("no isclose match")
    except Exception:
        # 2) 単調列なら区間探索（より「今いるステップ」に近い）
        s = sig.detach().cpu().numpy()
        idx = None
        if len(s) >= 2:
            descending = s[0] >= s[-1]
            for i in range(len(s) - 1):
                a, b = s[i], s[i + 1]
                if descending:
                    if a >= cur_f >= b:
                        idx = i
                        break
                else:
                    if a <= cur_f <= b:
                        idx = i
                        break
        if idx is None:
            # 3) 最後の手段: 最近傍
            idx = int(torch.argmin(torch.abs(sig - torch.tensor(cur_f, device=sig.device, dtype=sig.dtype))).item())

    total = (len(sig) - 1) if len(sig) > 1 else len(sig)

    # samplerによっては最後のsigmaが「モデル呼び出し無し」なので、上限をclamp
    if total <= 0:
        total = 1
    if idx >= total:
        idx = total - 1
    if idx < 0:
        idx = 0
    return idx, total

# --- グローバル変数 (キャリブレーション用) ---
# 論文 (MagCache, Eq.(4)) の定義に合わせて residual r_t = v_θ(x_t, t) - x_t を扱う
calibration_data = {
    'cond': [], 'uncond': [],                 # γ_t = mean(||r_t|| / ||r_{t-1}||)
    'cond_cos': [], 'uncond_cos': [],         # cos(r_t, r_{t-1}) の平均（方向の安定性確認用）
    'cond_std_ratio': [], 'uncond_std_ratio': [],  # token-wise norm の std 比（安定性確認用）
    'last_cond_res': None, 'last_uncond_res': None,
    'last_cond_std': None, 'last_uncond_std': None,
}

def reset_calibration_data():
    global calibration_data
    calibration_data = {
        'cond': [], 'uncond': [],
        'cond_cos': [], 'uncond_cos': [],
        'cond_std_ratio': [], 'uncond_std_ratio': [],
        'last_cond_res': None, 'last_uncond_res': None,
        'last_cond_std': None, 'last_uncond_std': None,
    }

def _extract_transformer_options(kwargs: dict):
    topts = kwargs.get("transformer_options", {}) or {}
    if (not topts) and ("c" in kwargs) and isinstance(kwargs["c"], dict):
        topts = kwargs["c"].get("transformer_options", {}) or {}
    return topts

def _token_norm_map(residual: torch.Tensor) -> torch.Tensor:
    """token-wise L2 norm map.
    - BCHW: -> BHW (norm over C)
    - BTC:  -> BT  (norm over C)
    fallback: -> B1
    """
    r = residual.float()
    if r.dim() == 4:
        return torch.linalg.norm(r, dim=1)  # B,H,W
    if r.dim() == 3:
        return torch.linalg.norm(r, dim=-1)  # B,T
    flat = r.reshape(r.shape[0], -1)
    return torch.linalg.norm(flat, dim=-1, keepdim=True)  # B,1

def _cosine_sim(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-6) -> float:
    aa = a.float().reshape(a.shape[0], -1)
    bb = b.float().reshape(b.shape[0], -1)
    num = (aa * bb).sum(dim=1)
    den = aa.norm(dim=1) * bb.norm(dim=1) + eps
    return (num / den).mean().item()

def _compute_mag_ratio_and_stats(curr_res: torch.Tensor, prev_res: torch.Tensor, prev_std: float | None):
    """Return (mag_ratio, cos_sim, std_ratio, curr_std)."""
    curr_map = _token_norm_map(curr_res)
    prev_map = _token_norm_map(prev_res)
    mag_ratio = (curr_map / (prev_map + 1e-6)).mean().item()

    cos_sim = _cosine_sim(curr_res, prev_res)

    curr_std = curr_map.reshape(-1).std().item()
    if prev_std is None:
        std_ratio = 1.0
    else:
        std_ratio = float(curr_std / (prev_std + 1e-6))
    return mag_ratio, cos_sim, std_ratio, curr_std

# --- Calibration Hook Function ---
def newbie_calibration_hook(self, x, t, *args, **kwargs):
    transformer_options = _extract_transformer_options(kwargs)

    computed_output = self.magcache_original_forward(x, t, *args, **kwargs)
    try:
        output_tensor = computed_output[0] if isinstance(computed_output, tuple) else computed_output
        if not isinstance(output_tensor, torch.Tensor):
            return computed_output

        # MagCache residual: r_t = vθ(x_t,t) - x_t
        # （NewBie / Lumina2 の forward が vθ を返す前提）
        input_tensor = x if isinstance(x, torch.Tensor) else None

        current_batch_size = output_tensor.shape[0]
        input_batch_size = input_tensor.shape[0] if input_tensor is not None else current_batch_size
        cond_or_uncond = transformer_options.get("cond_or_uncond", None)

        def _update_branch(branch: str, curr_res: torch.Tensor):
            last_key = 'last_cond_res' if branch == 'cond' else 'last_uncond_res'
            last_std_key = 'last_cond_std' if branch == 'cond' else 'last_uncond_std'
            cos_key = 'cond_cos' if branch == 'cond' else 'uncond_cos'
            std_ratio_key = 'cond_std_ratio' if branch == 'cond' else 'uncond_std_ratio'

            prev_res = calibration_data[last_key]
            prev_std = calibration_data[last_std_key]
            if prev_res is not None:
                mag_ratio, cos_sim, std_ratio, curr_std = _compute_mag_ratio_and_stats(curr_res, prev_res, prev_std)
                calibration_data[branch].append(float(mag_ratio))
                calibration_data[cos_key].append(float(cos_sim))
                calibration_data[std_ratio_key].append(float(std_ratio))
                calibration_data[last_std_key] = float(curr_std)
            else:
                # 1st step: only store baseline std
                curr_std = _token_norm_map(curr_res).reshape(-1).std().item()
                calibration_data[last_std_key] = float(curr_std)

            calibration_data[last_key] = curr_res.detach()

        if input_tensor is None or current_batch_size != input_batch_size:
            # fallback: treat as fused
            out_mean = output_tensor.mean(dim=0, keepdim=True)
            x_mean = input_tensor.mean(dim=0, keepdim=True) if input_tensor is not None else 0.0
            fused_res = (out_mean - x_mean).detach()
            _update_branch('cond', fused_res)
            _update_branch('uncond', fused_res)
        else:
            if current_batch_size % 2 == 0 and current_batch_size >= 2:
                half = current_batch_size // 2
                out_c = output_tensor[:half].mean(dim=0, keepdim=True)
                out_u = output_tensor[half:].mean(dim=0, keepdim=True)
                x_c = input_tensor[:half].mean(dim=0, keepdim=True)
                x_u = input_tensor[half:].mean(dim=0, keepdim=True)
                _update_branch('cond', (out_c - x_c).detach())
                _update_branch('uncond', (out_u - x_u).detach())
            else:
                is_uncond = bool(cond_or_uncond is not None and 1 in cond_or_uncond)
                out_m = output_tensor.mean(dim=0, keepdim=True)
                x_m = input_tensor.mean(dim=0, keepdim=True)
                res = (out_m - x_m).detach()
                _update_branch('uncond' if is_uncond else 'cond', res)

    except Exception as e:
        print(f"[MagCache-NewBie] Calibration error: {e}")
        traceback.print_exc()

    return computed_output

# --- Inference Hook Function ---
def newbie_inference_hook(self, x, t, *args, **kwargs):
    params = getattr(self, 'magcache_params', None)
    state_manager = getattr(self, 'magcache_state', None)

    transformer_options = _extract_transformer_options(kwargs)

    if not params or state_manager is None:
        return self.magcache_original_forward(x, t, *args, **kwargs)

    current_step = int(getattr(self, 'magcache_current_step', 0))
    cond_or_uncond = transformer_options.get("cond_or_uncond", None)

    # バッチサイズから現在の実行モードを判定
    x_tensor = x
    if not isinstance(x_tensor, torch.Tensor):
        return self.magcache_original_forward(x, t, *args, **kwargs)

    bsz = x_tensor.shape[0]
    mode = 'combined'
    if bsz % 2 != 0:
        is_uncond = bool(cond_or_uncond is not None and 1 in cond_or_uncond)
        mode = 'uncond_only' if is_uncond else 'cond_only'

    # ratio は「step=1 から」定義される（step=0 は r_{-1} が無い）
    in_range = (current_step > 0) and (params['start_step_abs'] <= current_step < params['end_step_abs'])

    skip_this_step = False
    debug_info = {}

    if in_range:
        def _ratio_index(ratio_offset: int) -> int:
            # calibrationで保存されるγ_tは step t のもので、配列上は (t-1) に入る
            return (current_step - 1) * 2 + ratio_offset

        def check_skip(guidance_type: str, ratio_offset: int) -> bool:
            state = state_manager.get_state(guidance_type)
            idx = _ratio_index(ratio_offset)

            if state['residual_cache'] is None:
                debug_info[guidance_type] = {"status": "NoData (Cache is None)"}
                return False
            if idx < 0 or idx >= len(params['mag_ratios']):
                debug_info[guidance_type] = {"status": "NoData (Index Error)"}
                return False

            ratio = float(params['mag_ratios'][idx])
            new_acc_ratio = float(state['accumulated_ratio'] * ratio)
            new_acc_steps = int(state['accumulated_steps'] + 1)
            new_acc_err = float(state['accumulated_err'] + abs(1.0 - new_acc_ratio))

            is_err_ok = new_acc_err < params['delta_threshold']
            is_k_ok = new_acc_steps <= params['K_skips']

            debug_info[guidance_type] = {
                "err": new_acc_err, "k": new_acc_steps, "ratio": ratio,
                "acc_ratio": new_acc_ratio,
                "ok": bool(is_err_ok and is_k_ok),
                "reason": "OK" if (is_err_ok and is_k_ok) else ("ErrHigh" if not is_err_ok else "K_Max")
            }
            return bool(is_err_ok and is_k_ok)

        # モードに応じたスキップ判定（combinedは両方OKの時だけ）
        if mode == 'combined':
            if check_skip("cond", 0) and check_skip("uncond", 1):
                skip_this_step = True
        elif mode == 'cond_only':
            if check_skip("cond", 0):
                skip_this_step = True
        else:  # uncond_only
            if check_skip("uncond", 1):
                skip_this_step = True

    # --- スキップ実行 ---
    if skip_this_step:
        def update_state(guidance_type: str, ratio_offset: int):
            state = state_manager.get_state(guidance_type)
            idx = (current_step - 1) * 2 + ratio_offset
            ratio = float(params['mag_ratios'][idx])
            state['accumulated_ratio'] = float(state['accumulated_ratio'] * ratio)
            state['accumulated_steps'] = int(state['accumulated_steps'] + 1)
            state['accumulated_err'] = float(state['accumulated_err'] + abs(1.0 - state['accumulated_ratio']))

        def approx_output(x_chunk: torch.Tensor, guidance_type: str) -> torch.Tensor:
            state = state_manager.get_state(guidance_type)
            # r_t ≈ (∏γ) r_{t0}, vθ(x_t,t) ≈ x_t + r_t
            r0 = state['residual_cache']
            r = (r0 * state['accumulated_ratio']).to(device=x_chunk.device, dtype=x_chunk.dtype)
            return x_chunk + r

        if mode == 'combined':
            update_state("cond", 0)
            update_state("uncond", 1)
            c_inf = debug_info.get("cond", {})
            print(f"[MagCache] Step {current_step:2d} | ✅ SKIP (Comb) | Err: {c_inf.get('err',0):.4f} | K: {c_inf.get('k',0)}")
            half = bsz // 2
            out_c = approx_output(x_tensor[:half], "cond")
            out_u = approx_output(x_tensor[half:], "uncond")
            return torch.cat([out_c, out_u], dim=0)

        if mode == 'cond_only':
            update_state("cond", 0)
            c_inf = debug_info.get("cond", {})
            print(f"[MagCache] Step {current_step:2d} | ✅ SKIP (Cond) | Err: {c_inf.get('err',0):.4f} | K: {c_inf.get('k',0)}")
            return approx_output(x_tensor, "cond")

        # uncond_only
        update_state("uncond", 1)
        u_inf = debug_info.get("uncond", {})
        print(f"[MagCache] Step {current_step:2d} | ✅ SKIP (Uncd) | Err: {u_inf.get('err',0):.4f} | K: {u_inf.get('k',0)}")
        return approx_output(x_tensor, "uncond")

    # --- 計算実行 (スキップしない場合) ---
    computed_output = self.magcache_original_forward(x, t, *args, **kwargs)
    output_tensor = computed_output[0] if isinstance(computed_output, tuple) else computed_output

    # --- Cache Storage (residual) ---
    if isinstance(output_tensor, torch.Tensor):
        if mode == 'combined' and output_tensor.shape[0] % 2 == 0:
            half = output_tensor.shape[0] // 2
            # residual r_t = v - x
            state_manager.store_residual((output_tensor[:half] - x_tensor[:half]).detach(), "cond")
            state_manager.store_residual((output_tensor[half:] - x_tensor[half:]).detach(), "uncond")
            state_manager.get_state("cond").update({'accumulated_err': 0.0, 'accumulated_steps': 0, 'accumulated_ratio': 1.0})
            state_manager.get_state("uncond").update({'accumulated_err': 0.0, 'accumulated_steps': 0, 'accumulated_ratio': 1.0})
        elif mode == 'cond_only':
            state_manager.store_residual((output_tensor - x_tensor).detach(), "cond")
            state_manager.get_state("cond").update({'accumulated_err': 0.0, 'accumulated_steps': 0, 'accumulated_ratio': 1.0})
        elif mode == 'uncond_only':
            state_manager.store_residual((output_tensor - x_tensor).detach(), "uncond")
            state_manager.get_state("uncond").update({'accumulated_err': 0.0, 'accumulated_steps': 0, 'accumulated_ratio': 1.0})

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
                step, total = _resolve_step_and_total(all_sigmas, current_sigma)
                target_model.magcache_current_step = step
                target_model.magcache_total_steps = total
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

                    # 追加: 方向安定性（cos）とtoken-wise std比も保存（後方互換）
                    c_cos = calibration_data.get('cond_cos', [])
                    u_cos = calibration_data.get('uncond_cos', []) if uncond_len > 0 else c_cos
                    c_std = calibration_data.get('cond_std_ratio', [])
                    u_std = calibration_data.get('uncond_std_ratio', []) if uncond_len > 0 else c_std
                    min_len_stats = min(len(c_cos), len(u_cos), len(c_std), len(u_std), min_len)

                    stats = None
                    if min_len_stats > 0:
                        cos_inter = np.empty((min_len_stats * 2,), dtype=np.float32)
                        cos_inter[0::2] = np.array(c_cos[:min_len_stats], dtype=np.float32)
                        cos_inter[1::2] = np.array(u_cos[:min_len_stats], dtype=np.float32)

                        std_inter = np.empty((min_len_stats * 2,), dtype=np.float32)
                        std_inter[0::2] = np.array(c_std[:min_len_stats], dtype=np.float32)
                        std_inter[1::2] = np.array(u_std[:min_len_stats], dtype=np.float32)

                        stats = {
                            "cos_sims": cos_inter.tolist(),
                            "std_ratios": std_inter.tolist(),
                        }

                    save_mag_ratios(model_hash, interleaved.tolist(), model_name, stats=stats)
                
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
                step, total = _resolve_step_and_total(all_sigmas, current_sigma)
                target_model.magcache_current_step = step
                target_model.magcache_total_steps = total
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
