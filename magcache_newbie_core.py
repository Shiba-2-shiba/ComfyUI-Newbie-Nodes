import os
import json
import hashlib
import numpy as np
import torch
from comfy import model_management

# 保存ディレクトリ
CALIBRATION_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "magcache_newbie_data")
os.makedirs(CALIBRATION_DIR, exist_ok=True)

# 1. モデルハッシュ取得関数
def get_model_hash(model_name: str) -> str:
    return hashlib.sha256(model_name.encode('utf-8')).hexdigest()[:16]

# 2. 厳密なMagnitude Ratio計算関数 (論文 Eq.4, Appendix A.2準拠)
def compute_magnitude_ratio(current_tensor: torch.Tensor, prev_tensor: torch.Tensor) -> float:
    """
    論文の定義に従い、チャネル方向(dim=-1)のL2ノルムを計算し、
    その後トークン方向(dim=1)の平均を取ることで比率を算出する。
    """
    if prev_tensor is None or current_tensor.shape != prev_tensor.shape:
        return 1.0

    channel_dim = -1
    
    curr_norm = torch.norm(current_tensor.float(), p=2, dim=channel_dim)
    prev_norm = torch.norm(prev_tensor.float(), p=2, dim=channel_dim)
    
    ratio_per_token = curr_norm / (prev_norm + 1e-6)
    
    return ratio_per_token.mean().item()

# 3. データロード/セーブ/補間
def load_mag_ratios(model_hash: str):
    filepath = os.path.join(CALIBRATION_DIR, f"{model_hash}.json")
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return np.array(data['mag_ratios'])
        except Exception:
            return None
    return None

def load_magcache_stats(model_hash: str) -> dict | None:
    """追加統計情報(betas, cos_sims等)を読み込む"""
    filepath = os.path.join(CALIBRATION_DIR, f"{model_hash}.json")
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data.get("stats", None)
        except Exception:
            return None
    return None

def save_mag_ratios(model_hash: str, ratios: list, model_name: str, stats: dict | None = None):
    filepath = os.path.join(CALIBRATION_DIR, f"{model_hash}.json")
    data = {
        'schema_version': 2,
        'model_name': model_name,
        'mag_ratios': ratios
    }
    if stats:
        data['stats'] = stats

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    print(f"[MagCache-NewBie] Calibration data saved to {filepath}")

def interpolate_data(data_array, target_steps, clip_min=0.0, clip_max=5.0):
    """
    汎用補間関数: ratios, betas, cos_sims すべてに使用可能
    """
    if data_array is None or len(data_array) == 0:
        return None
    
    # 配列化
    if not isinstance(data_array, np.ndarray):
        data_array = np.array(data_array)

    source_steps = len(data_array) // 2
    if source_steps == target_steps:
        return data_array

    cond_vals = data_array[0::2]
    uncond_vals = data_array[1::2]
    
    source_indices = np.linspace(0, source_steps - 1, source_steps)
    target_indices = np.linspace(0, source_steps - 1, target_steps)
    
    interp_cond = np.interp(target_indices, source_indices, cond_vals)
    interp_uncond = np.interp(target_indices, source_indices, uncond_vals)

    result = np.empty((target_steps * 2,), dtype=data_array.dtype)
    result[0::2] = interp_cond
    result[1::2] = interp_uncond
    
    # クリップ処理
    if clip_min is not None and clip_max is not None:
        result = np.clip(result, clip_min, clip_max)
    
    print(f"[MagCache-NewBie] Data interpolated from {source_steps} to {target_steps} steps.")
    return result

# 後方互換のためのラッパー
def interpolate_mag_ratios(ratios, target_steps):
    return interpolate_data(ratios, target_steps, clip_min=0.0, clip_max=5.0)

# 4. 状態管理クラス
class MagCacheState:
    def __init__(self):
        self.state = {
            "cond": self._create_guidance_state(),
            "uncond": self._create_guidance_state()
        }
        self.cache_device = model_management.get_torch_device()

    def _create_guidance_state(self):
        return {
            'residual_cache': None, 
            'accumulated_err': 0.0, 
            'accumulated_steps': 0, 
            'accumulated_ratio': 1.0
        }
    
    def reset(self):
        self.state = {
            "cond": self._create_guidance_state(),
            "uncond": self._create_guidance_state()
        }
    
    def get_state(self, guidance_type: str):
        return self.state[guidance_type]

    def store_residual(self, residual_tensor: torch.Tensor, guidance_type: str):
        state = self.get_state(guidance_type)
        # メモリ効率のため clone() して保持
        state['residual_cache'] = residual_tensor.detach().clone()
