import torch
import numpy as np
import os
import traceback

# コア機能をインポート
from .magcache_newbie_core import (
    get_model_hash, load_mag_ratios, load_magcache_stats, save_mag_ratios,
    interpolate_mag_ratios, interpolate_data, MagCacheState, compute_magnitude_ratio
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
    return "UnknownModel"

def _resolve_step_and_total(sample_sigmas, current_sigma):
    """Step数とTotalStep数を堅牢に特定する"""
    if sample_sigmas is None: return 0, 1
    
    sig = sample_sigmas
    if isinstance(sig, (list, tuple)): sig = torch.tensor(sig)
    if not isinstance(sig, torch.Tensor) or sig.numel() == 0: return 0, 1
    sig = sig.flatten().cpu()

    cur = current_sigma
    if isinstance(cur, torch.Tensor): cur = cur.flatten()[0].cpu()
    elif isinstance(cur, (list, tuple)): cur = cur[0]
    
    cur_f = float(cur)

    try:
        close = torch.isclose(sig, torch.tensor(cur_f, dtype=sig.dtype), rtol=1e-3, atol=1e-4)
        idxs = torch.where(close)[0]
        if idxs.numel() > 0:
            idx = int(idxs[0].item())
        else:
            raise ValueError("No match")
    except:
        idx = int(torch.argmin(torch.abs(sig - cur_f)).item())

    total = max(1, len(sig) - 1)
    if idx >= total: idx = total - 1
    if idx < 0: idx = 0
    return idx, total

def _extract_transformer_options(kwargs: dict):
    topts = kwargs.get("transformer_options", {}) or {}
    if (not topts) and ("c" in kwargs) and isinstance(kwargs["c"], dict):
        topts = kwargs["c"].get("transformer_options", {}) or {}
    return topts

# --- 数学ヘルパー ---
def _proj_beta(v: torch.Tensor, x: torch.Tensor, eps=1e-6, clamp=3.0) -> float:
    """
    β = <v, x> / <x, x>  (least squares projection)
    """
    vf = v.detach().float()
    xf = x.detach().float()
    num = torch.sum(vf * xf)
    den = torch.sum(xf * xf) + eps
    beta = (num / den).item()
    if clamp is not None:
        beta = max(-clamp, min(clamp, beta))
    return float(beta)

def _cos_sim(a: torch.Tensor, b: torch.Tensor, eps=1e-6) -> float:
    """
    cos(a,b) をバッチ全体で1スカラーに潰す
    """
    af = a.detach().float()
    bf = b.detach().float()
    num = torch.sum(af * bf)
    den = (torch.linalg.norm(af) * torch.linalg.norm(bf)) + eps
    return float((num / den).item())

# --- Calibration Hook ---
# Residual定義: r_t = v_t - beta(t) * x_t
calibration_data = {
    'cond_ratios': [], 'uncond_ratios': [],
    'cond_betas':  [], 'uncond_betas':  [],
    'cond_cos':    [], 'uncond_cos':    [],
    'last_cond_r': None, 'last_uncond_r': None,
}

def reset_calibration_data():
    global calibration_data
    calibration_data = {
        'cond_ratios': [], 'uncond_ratios': [],
        'cond_betas':  [], 'uncond_betas':  [],
        'cond_cos':    [], 'uncond_cos':    [],
        'last_cond_r': None, 'last_uncond_r': None,
    }

def newbie_calibration_hook(self, x, t, *args, **kwargs):
    # まず本来の出力を計算 (Flow Matchingでは Velocity v が返る)
    computed_output = self.magcache_original_forward(x, t, *args, **kwargs)
    
    try:
        v = computed_output[0] if isinstance(computed_output, tuple) else computed_output
        if not isinstance(v, torch.Tensor): return computed_output

        transformer_options = _extract_transformer_options(kwargs)
        cond_or_uncond = transformer_options.get("cond_or_uncond", None)
        
        bsz = v.shape[0]
        combined = (bsz >= 2 and bsz % 2 == 0)

        def _update(branch: str, v_cur: torch.Tensor, x_cur: torch.Tensor):
            last_key = 'last_cond_r' if branch == 'cond' else 'last_uncond_r'
            ratios_key = 'cond_ratios' if branch == 'cond' else 'uncond_ratios'
            betas_key  = 'cond_betas'  if branch == 'cond' else 'uncond_betas'
            cos_key    = 'cond_cos'    if branch == 'cond' else 'uncond_cos'

            # β(t) と r_t の計算
            beta = _proj_beta(v_cur, x_cur)
            r_cur = v_cur - (beta * x_cur)

            last_r = calibration_data[last_key]
            if last_r is not None:
                # r 同士で比較
                ratio = compute_magnitude_ratio(r_cur, last_r)
                cs = _cos_sim(r_cur, last_r)
                
                calibration_data[ratios_key].append(float(ratio))
                calibration_data[betas_key].append(float(beta))
                calibration_data[cos_key].append(float(cs))

            calibration_data[last_key] = r_cur.detach()

        if combined:
            h = bsz // 2
            _update('cond', v[:h], x[:h])
            _update('uncond', v[h:], x[h:])
        else:
            is_uncond = bool(cond_or_uncond is not None and 1 in cond_or_uncond)
            branch = 'uncond' if is_uncond else 'cond'
            _update(branch, v, x)

    except Exception:
        traceback.print_exc()
    
    return computed_output

# --- Inference Hook ---
def newbie_inference_hook(self, x, t, *args, **kwargs):
    params = getattr(self, 'magcache_params', None)
    state_manager = getattr(self, 'magcache_state', None)
    
    if not params or state_manager is None:
        return self.magcache_original_forward(x, t, *args, **kwargs)

    current_step = getattr(self, 'magcache_current_step', 0)
    transformer_options = _extract_transformer_options(kwargs)
    cond_or_uncond = transformer_options.get("cond_or_uncond", None)

    bsz = x.shape[0]
    mode = 'combined'
    if bsz % 2 != 0:
        is_uncond = bool(cond_or_uncond is not None and 1 in cond_or_uncond)
        mode = 'uncond_only' if is_uncond else 'cond_only'

    in_range = (current_step > 0) and (params['start_step_abs'] <= current_step < params['end_step_abs'])
    skip_this_step = False

    # スキップ判定ロジック
    if in_range:
        def check_skip(guidance_type, offset):
            state = state_manager.get_state(guidance_type)
            idx = (current_step - 1) * 2 + offset 
            
            if state['residual_cache'] is None: return False
            if idx >= len(params['mag_ratios']): return False
            
            # CosGuard: 方向が不安定なステップはSkip禁止
            if params.get('use_cos_guard', False):
                cos_sims = params.get('cos_sims', None)
                if cos_sims is not None and idx < len(cos_sims):
                    if cos_sims[idx] < params.get('cos_guard', 0.985):
                        return False

            ratio = params['mag_ratios'][idx]
            
            next_acc_ratio = state['accumulated_ratio'] * ratio
            next_acc_err = state['accumulated_err'] + abs(1.0 - next_acc_ratio)
            next_acc_steps = state['accumulated_steps'] + 1
            
            return (next_acc_err < params['delta_threshold']) and (next_acc_steps <= params['K_skips'])

        if mode == 'combined':
            if check_skip('cond', 0) and check_skip('uncond', 1):
                skip_this_step = True
        elif mode == 'cond_only':
            if check_skip('cond', 0): skip_this_step = True
        elif mode == 'uncond_only':
            if check_skip('uncond', 1): skip_this_step = True

    # === SKIP実行 (Reconstruction) ===
    if skip_this_step:
        def get_approx(guidance_type, offset):
            state = state_manager.get_state(guidance_type)
            idx = (current_step - 1) * 2 + offset
            
            ratio = params['mag_ratios'][idx]
            
            # 状態更新
            state['accumulated_ratio'] *= ratio
            state['accumulated_steps'] += 1
            state['accumulated_err'] += abs(1.0 - state['accumulated_ratio'])
            
            # 近似計算: 
            # r_t ≈ r_cache * accumulated_ratio
            # v_t ≈ beta(t) * x_t + r_t
            r_cache = state['residual_cache']
            r_approx = r_cache * state['accumulated_ratio']
            
            betas = params.get('betas', None)
            beta = 0.0
            if betas is not None and idx < len(betas):
                beta = float(betas[idx])
            
            # x_slice 取得
            if mode == 'combined':
                h = x.shape[0] // 2
                x_slice = x[:h] if guidance_type == 'cond' else x[h:]
            else:
                x_slice = x

            v_approx = r_approx + (beta * x_slice)
            return v_approx.to(x.device, x.dtype)

        if mode == 'combined':
            return torch.cat([get_approx('cond', 0), get_approx('uncond', 1)], dim=0)
        elif mode == 'cond_only':
            return get_approx('cond', 0)
        elif mode == 'uncond_only':
            return get_approx('uncond', 1)

    # === 実計算 (Compute) ===
    def reset_state(guidance_type):
        s = state_manager.get_state(guidance_type)
        s['accumulated_err'] = 0.0
        s['accumulated_steps'] = 0
        s['accumulated_ratio'] = 1.0
    
    if mode == 'combined': reset_state('cond'); reset_state('uncond')
    elif mode == 'cond_only': reset_state('cond')
    elif mode == 'uncond_only': reset_state('uncond')

    computed_output = self.magcache_original_forward(x, t, *args, **kwargs)
    output_tensor = computed_output[0] if isinstance(computed_output, tuple) else computed_output

    # キャッシュ保存 (r = v - beta*x を保存)
    if isinstance(output_tensor, torch.Tensor):
        def _store(branch: str, v_cur: torch.Tensor, x_cur: torch.Tensor, offset: int):
            idx = (current_step - 1) * 2 + offset
            
            # キャリブレーション済みのbetaを使用 (なければ0)
            betas = params.get('betas', None)
            beta = 0.0
            if betas is not None and idx < len(betas):
                beta = float(betas[idx])
            
            r_cur = v_cur - (beta * x_cur)
            state_manager.store_residual(r_cur, branch)

        if mode == 'combined':
            h = output_tensor.shape[0] // 2
            _store('cond', output_tensor[:h], x[:h], 0)
            _store('uncond', output_tensor[h:], x[h:], 1)
        elif mode == 'cond_only':
            _store('cond', output_tensor, x, 0)
        elif mode == 'uncond_only':
            _store('uncond', output_tensor, x, 1)

    return computed_output

# --- Nodes ---

class MagCacheNewBieCalibration:
    @classmethod
    def INPUT_TYPES(s):
        return { 
            "required": { "model": ("MODEL",) },
            "optional": { "manual_model_name": ("STRING", {"default": "", "placeholder": "newbie"}) }
        }
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "exec"
    CATEGORY = "NewBie/MagCache"
    TITLE = "Calibrate MagCache (NewBie)"

    def exec(self, model, manual_model_name=""):
        model_name = _get_model_name(model, manual_model_name)
        target = model.model.diffusion_model if hasattr(model.model, "diffusion_model") else model.model
        
        if hasattr(target, 'magcache_original_forward'):
            setattr(target, 'forward', target.magcache_original_forward)
            
        setattr(target, 'magcache_original_forward', target.forward)
        setattr(target, 'forward', newbie_calibration_hook.__get__(target, type(target)))
        
        def wrapper(fn, kwargs):
            topts = kwargs.get('c', {}).get('transformer_options', {})
            all_sigmas = topts.get('sample_sigmas')
            curr_sig = kwargs.get('timestep')
            step, total = _resolve_step_and_total(all_sigmas, curr_sig)
            
            target.magcache_current_step = step
            target.magcache_total_steps = total

            if step == 0:
                print(f"[MagCache] Calibrating: {model_name}")
                reset_calibration_data()
            
            out = fn(kwargs['input'], kwargs['timestep'], **kwargs['c'])
            
            if total > 0 and step >= total - 1:
                # データの保存処理
                cR, uR = calibration_data['cond_ratios'], calibration_data['uncond_ratios']
                cB, uB = calibration_data['cond_betas'], calibration_data['uncond_betas']
                cC, uC = calibration_data['cond_cos'], calibration_data['uncond_cos']

                if len(uR) == 0: uR, uB, uC = cR, cB, cC # Uncondがない場合

                if len(cR) > 0:
                    min_len = min(len(cR), len(uR), len(cB), len(uB), len(cC), len(uC))
                    
                    # Interleave helper
                    def interleave(c_list, u_list, length):
                        arr = np.empty((length * 2,), dtype=np.float32)
                        arr[0::2] = c_list[:length]
                        arr[1::2] = u_list[:length]
                        return arr

                    interR = interleave(cR, uR, min_len)
                    interB = interleave(cB, uB, min_len)
                    interC = interleave(cC, uC, min_len)

                    stats = {
                        "betas": interB.tolist(),
                        "cos_sims": interC.tolist(),
                    }
                    save_mag_ratios(get_model_hash(model_name), interR.tolist(), model_name, stats=stats)
                
                if hasattr(target, 'magcache_original_forward'):
                    setattr(target, 'forward', target.magcache_original_forward)
                    delattr(target, 'magcache_original_forward')
            return out

        m = model.clone()
        m.set_model_unet_function_wrapper(wrapper)
        return (m,)

class MagCacheNewBie:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",), 
                "enabled": ("BOOLEAN", {"default": True}),
                "Magcache_thresh": ("FLOAT", {"default": 0.10, "step": 0.005}),
                "Magcache_K": ("INT", {"default": 2, "min": 0, "max": 10}),
                "retention_ratio": ("FLOAT", {"default": 0.2, "step": 0.05}),
                "exclude_last_step": ("BOOLEAN", {"default": True}),
                # 新機能: CosGuard
                "cos_guard": ("FLOAT", {"default": 0.985, "min": -1.0, "max": 1.0, "step": 0.005}),
                "use_cos_guard": ("BOOLEAN", {"default": True}),
            },
            "optional": { "manual_model_name": ("STRING", {"default": ""}) }
        }
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "exec"
    CATEGORY = "NewBie/MagCache"
    TITLE = "Apply MagCache (NewBie)"

    def exec(self, model, enabled, Magcache_thresh, Magcache_K, retention_ratio, exclude_last_step, cos_guard, use_cos_guard, manual_model_name=""):
        if not enabled: return (model,)
        
        model_name = _get_model_name(model, manual_model_name)
        model_hash = get_model_hash(model_name)
        
        ratios = load_mag_ratios(model_hash)
        stats = load_magcache_stats(model_hash) or {}
        
        if ratios is None:
            print(f"[MagCache] ⚠️ No data for {model_name}. Run Calibration first.")
            return (model,)
        
        target = model.model.diffusion_model if hasattr(model.model, "diffusion_model") else model.model
        
        if hasattr(target, 'magcache_original_forward'):
            setattr(target, 'forward', target.magcache_original_forward)

        setattr(target, 'magcache_original_forward', target.forward)
        setattr(target, 'forward', newbie_inference_hook.__get__(target, type(target)))
        
        if not hasattr(target, 'magcache_state'):
            target.magcache_state = MagCacheState()

        def wrapper(fn, kwargs):
            topts = kwargs.get('c', {}).get('transformer_options', {})
            all_sigmas = topts.get('sample_sigmas')
            curr_sig = kwargs.get('timestep')
            step, total = _resolve_step_and_total(all_sigmas, curr_sig)
            
            target.magcache_current_step = step
            target.magcache_total_steps = total

            if step == 0:
                target.magcache_state.reset()
                
                # データの補間
                interp_ratios = interpolate_mag_ratios(ratios, total)
                
                betas = stats.get("betas", None)
                coss = stats.get("cos_sims", None)
                
                interp_betas = interpolate_data(betas, total, clip_min=-5.0, clip_max=5.0) if betas is not None else None
                interp_coss = interpolate_data(coss, total, clip_min=-1.0, clip_max=1.0) if coss is not None else None
                
                if interp_ratios is not None:
                    target.magcache_params = {
                        'mag_ratios': interp_ratios,
                        'betas': interp_betas,
                        'cos_sims': interp_coss,
                        'delta_threshold': Magcache_thresh,
                        'K_skips': Magcache_K,
                        'start_step_abs': int(total * retention_ratio),
                        'end_step_abs': total - 1 if exclude_last_step else total,
                        'cos_guard': cos_guard,
                        'use_cos_guard': use_cos_guard
                    }
            return fn(kwargs['input'], kwargs['timestep'], **kwargs['c'])

        m = model.clone()
        m.set_model_unet_function_wrapper(wrapper)
        return (m,)
