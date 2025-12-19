import torch
import comfy.utils
import comfy.model_sampling
from comfy.patcher_extension import WrappersMP, add_wrapper
import functools
import logging
import traceback

# -----------------------------------------------------------------------------
# ヘルパー関数群 (Wan版と同様のロジック)
# -----------------------------------------------------------------------------

def _scale_backbone_token(x, b, safe_scale=False):
    """
    トークン列の前半チャンネルを、トークン全体の平均応答に基づいてスケーリングします。
    """
    # x shape: (Batch, SequenceLength, Channels)
    B, L, C = x.shape
    
    # チャンネルごとの平均を計算
    hidden_mean = x.mean(dim=1, keepdim=True) # (B, 1, C)
    
    # min-max正規化
    vmin = hidden_mean.min(dim=-1, keepdim=True)[0]
    vmax = hidden_mean.max(dim=-1, keepdim=True)[0]
    
    # ゼロ除算回避
    range_val = vmax - vmin
    # range_valが0に近い場合の安全策
    range_val = torch.where(range_val < 1e-6, torch.ones_like(range_val) * 1e-6, range_val)
    
    mask = (hidden_mean - vmin) / range_val # (B, 1, C)

    # チャンネルの前半にスケールを適用 (NewBie/NextDiTの構造に依存するが、一般的なDiTのヒューリスティックとして有効)
    half_c = C // 2
    
    # safe_scale=Trueの場合、cloneして破壊的変更を避ける
    x_out = x.clone() if safe_scale else x
    
    gain = (b - 1.0) * mask[..., :half_c] + 1.0
    x_out[..., :half_c] *= gain
    
    return x_out

def _choose_params(block_idx, total_blocks, b1, b2):
    """
    ブロックの深さに応じて適用するbパラメータを選択します。
    """
    one_third = total_blocks // 3
    if block_idx < one_third:
        return b1
    elif block_idx < 2 * one_third:
        return b2
    else:
        return 1.0 # 深い層では適用しない

def _hooked_block_forward(*args, original_forward, block_idx, b_val, safe_scale, **kwargs):
    """
    各Transformerブロックのforwardをフックする関数。
    NewBie(NextDiT)のBlock forwardは通常 x (hidden_states) を第一引数に取ります。
    """
    # オリジナルのブロックを実行
    original_output = original_forward(*args, **kwargs)
    
    # bパラメータを適用 (出力がTensorであることを期待)
    if b_val != 1.0 and isinstance(original_output, torch.Tensor):
        original_output = _scale_backbone_token(original_output, b_val, safe_scale)
        
    return original_output

# -----------------------------------------------------------------------------
# ラッパー定義
# -----------------------------------------------------------------------------

# --- ラッパーB (ステップ毎の実行 - Model側) ---
def _make_newbie_freeu_model_wrapper(b1, b2, safe_scale, diffusion_model):
    """
    [ラッパーB] DIFFUSION_MODELラッパー。
    NextDiTWrapperの内側にある実際のモデル(NextDiT)のblocksにフックを仕掛けます。
    """
    
    # NextDiTWrapperから実体のモデルを取り出すロジック
    # newbie_model_support.py の NextDiTWrapper は .model 属性に実体を持っている
    inner_model = diffusion_model
    if hasattr(diffusion_model, "model") and hasattr(diffusion_model.model, "blocks"):
        inner_model = diffusion_model.model
        print(f"[NewBieFreeU] Detected NextDiTWrapper. Targeting inner model: {type(inner_model)}")
    
    if not hasattr(inner_model, "blocks"):
        print(f"[NewBieFreeU] Error: 'blocks' attribute not found in {type(inner_model)}. FreeU cannot be applied.")
        # 何もしないラッパーを返す
        def dummy_wrapper(executor, *args, **kwargs):
            return executor(*args, **kwargs)
        return dummy_wrapper

    total_blocks = len(inner_model.blocks)
    print(f"[NewBieFreeU] Target model has {total_blocks} blocks.")

    def wrapper(executor, *args, **kwargs):
        # 現在のタイムステップなどを取得するための引数解析
        # ComfyUIの仕様変更に備えてtry-exceptで囲む
        try:
            # args[1] は通常 timestep tensor
            current_timestep = float(args[1][0])
        except Exception:
            current_timestep = 0.0
        
        # Wrapper A (Sampler) で計算された範囲を取得
        # 属性は diffusion_model (Wrapper) に付与されているはず
        timestep_range = getattr(diffusion_model, "_newbie_freeu_sigma_range", None)
        
        if timestep_range is None:
            return executor(*args, **kwargs)

        timestep_start_val = timestep_range['sigma_start'] 
        timestep_end_val = timestep_range['sigma_end']
        
        # 範囲外なら何もしない
        if not (timestep_start_val >= current_timestep >= timestep_end_val):
            return executor(*args, **kwargs)
        
        # 初回適用時のみログ
        if 'last_applied_timestep' not in timestep_range:
            print(f"[NewBieFreeU] Applying FreeU at timestep {current_timestep:.4f} (Range: {timestep_start_val:.2f}-{timestep_end_val:.2f})")
            timestep_range['last_applied_timestep'] = current_timestep
        
        original_forwards = {}
        try:
            # ブロックごとにフックを適用
            for i, block in enumerate(inner_model.blocks):
                original_forwards[i] = block.forward
                
                b_val = _choose_params(i, total_blocks, b1, b2)
                
                if b_val != 1.0:
                    block.forward = functools.partial(
                        _hooked_block_forward,
                        original_forward=block.forward,
                        block_idx=i,
                        b_val=b_val,
                        safe_scale=safe_scale
                    )
            
            # モデル実行
            return executor(*args, **kwargs)
            
        finally:
            # フック解除 (必ず実行)
            for i in original_forwards:
                inner_model.blocks[i].forward = original_forwards[i]

    return wrapper

# --- ラッパーA (シグマスケジュール捕捉 - Sampler側) ---
def _make_newbie_freeu_sampler_wrapper(start_percent, end_percent, diffusion_model):
    """
    [ラッパーA] SAMPLER_SAMPLEラッパー。
    サンプリング開始時に一度だけ実行され、start_percent/end_percent を実際のタイムステップ値に変換して
    diffusion_model に保存します。
    """
    
    def wrapper(executor, guider_self, sigmas, *args, **kwargs):
        print(f"[NewBieFreeU] Sampler wrapper initialized. Range: {start_percent*100}% - {end_percent*100}%")
        
        try:
            # ModelSamplingオブジェクトを取得 (timestep変換に必要)
            model_patcher = guider_self.model_patcher
            model_sampling = model_patcher.get_model_object("model_sampling")
            
            if model_sampling is None:
                print("[NewBieFreeU] Warning: 'model_sampling' object not found. Is the model loaded correctly?")
                setattr(diffusion_model, "_newbie_freeu_sigma_range", None)
                return executor(guider_self, sigmas, *args, **kwargs)

            if sigmas is None or len(sigmas) == 0:
                return executor(guider_self, sigmas, *args, **kwargs)

            # シグマの最大・最小を取得
            sigma_max_actual = sigmas[0].item()
            sigma_min_actual = sigmas[-1].item()
            
            ts_max_tensor = torch.tensor(sigma_max_actual, device=sigmas.device)
            ts_min_tensor = torch.tensor(sigma_min_actual, device=sigmas.device)
            
            # シグマ -> タイムステップ 変換
            timestep_max_actual = model_sampling.timestep(ts_max_tensor).item()
            timestep_min_actual = model_sampling.timestep(ts_min_tensor).item()
            
            # 適用範囲の計算
            total_range = timestep_max_actual - timestep_min_actual
            timestep_start_calc = timestep_max_actual - total_range * start_percent
            timestep_end_calc = timestep_max_actual - total_range * end_percent
            
            # 安全策: startとendが逆転しないようにmax/minをとる
            timestep_start_final = max(timestep_start_calc, timestep_end_calc)
            timestep_end_final = min(timestep_start_calc, timestep_end_calc)

            sigma_range = {
                'sigma_start': timestep_start_final,
                'sigma_end': timestep_end_final
            }
            
            # 計算結果をモデルオブジェクトに保存 (ラッパーBで参照する)
            setattr(diffusion_model, "_newbie_freeu_sigma_range", sigma_range)
            
        except Exception as e:
            print(f"[NewBieFreeU] Error calculating sigma range: {e}")
            traceback.print_exc()
            setattr(diffusion_model, "_newbie_freeu_sigma_range", None)
        
        return executor(guider_self, sigmas, *args, **kwargs)
        
    return wrapper

# -----------------------------------------------------------------------------
# ComfyUI ノード定義
# -----------------------------------------------------------------------------

class NewBieFreeULikeNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "b1": ("FLOAT", {"default": 1.10, "min": 0.5, "max": 3.0, "step": 0.01, "tooltip": "Strength for the first 1/3 layers"}),
                "b2": ("FLOAT", {"default": 1.20, "min": 0.5, "max": 3.0, "step": 0.01, "tooltip": "Strength for the second 1/3 layers"}),
                "start_percent": ("FLOAT", {"default": 0.00, "min": 0.0, "max": 1.0, "step": 0.01}),
                "end_percent":   ("FLOAT", {"default": 0.35, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "safe_scale": ("BOOLEAN", {"default": False, "tooltip": "Use clone to avoid in-place modification (slower but safer)"}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "apply_patch"
    CATEGORY = "NewBie/Model"
    TITLE = "NewBie FreeU-Like Patch"

    def apply_patch(self, model, b1, b2, start_percent, end_percent, safe_scale=False):
        
        print(f"\n[NewBieFreeU] Initializing... (b1={b1}, b2={b2})")
        
        m = model.clone()
        
        # モデルの取得
        try:
            diffusion_model = m.model.diffusion_model
            # diffusion_modelは NextDiTWrapper (newbie_model_support.py) または NextDiT そのもの
        except Exception as e:
            logging.error(f"[NewBieFreeU] Failed to retrieve diffusion_model: {e}")
            return (model,)
        
        # Sampler Wrapper (シグマ計算用) を登録
        sampler_wrapper = _make_newbie_freeu_sampler_wrapper(
            start_percent, end_percent, diffusion_model
        )
        
        # Model Wrapper (Blockフック用) を登録
        model_wrapper = _make_newbie_freeu_model_wrapper(
            b1, b2, safe_scale, diffusion_model
        )
        
        add_wrapper(WrappersMP.SAMPLER_SAMPLE, sampler_wrapper, m.model_options, is_model_options=True)
        add_wrapper(WrappersMP.DIFFUSION_MODEL, model_wrapper, m.model_options, is_model_options=True)
        
        return (m,)

# -----------------------------------------------------------------------------
# ComfyUIへの登録
# -----------------------------------------------------------------------------
NODE_CLASS_MAPPINGS = {
    "NewBieFreeULikeNode": NewBieFreeULikeNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NewBieFreeULikeNode": "NewBie FreeU-Like Patch",
}