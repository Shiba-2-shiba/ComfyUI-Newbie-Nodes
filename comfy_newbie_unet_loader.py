import torch
import os
import json
from safetensors.torch import load_file

import folder_paths
import comfy.model_management as model_management
import comfy.model_patcher

from .newbie_model_support import NewBieModelConfig, NewBieBaseModel


class NewBieUNetLoader:
    @classmethod
    def INPUT_TYPES(cls):
        unet_names = folder_paths.get_filename_list("diffusion_models")
        newbie_unets = [name for name in unet_names if "newbie" in name.lower()]
        if not newbie_unets:
            newbie_unets = unet_names
        return {
            "required": {
                "unet_name": (newbie_unets, ),
                "dtype": (["default", "fp16", "bf16", "fp8_e4m3fn", "fp8_e5m2"], {"default": "bf16"}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "load_unet"
    CATEGORY = "loaders"
    TITLE = "NewBie UNet Loader"

    def load_unet(self, unet_name: str, dtype: str = "bf16"):
        unet_path = folder_paths.get_full_path_or_raise("diffusion_models", unet_name)

        if os.path.isdir(unet_path):
            model_dir = unet_path
            safetensors_path = os.path.join(model_dir, "newbie_dit.safetensors")
            if not os.path.exists(safetensors_path):
                safetensors_path = os.path.join(model_dir, "model.safetensors")
            config_path = os.path.join(model_dir, "config.json")
        else:
            model_dir = os.path.dirname(unet_path)
            safetensors_path = unet_path
            config_path = os.path.join(model_dir, "config.json")

        dtype_map = {
            "default": torch.bfloat16,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
            "fp8_e4m3fn": torch.float8_e4m3fn,
            "fp8_e5m2": torch.float8_e5m2,
        }
        torch_dtype = dtype_map.get(dtype, torch.bfloat16)
        load_device = model_management.get_torch_device()
        offload_device = model_management.unet_offload_device()

        print(f"Loading NewBie UNet: {safetensors_path}")
        state_dict = load_file(safetensors_path, device="cpu")

        config = {}
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)

        cap_feat_dim = config.get("cap_feat_dim", None)
        if cap_feat_dim is None and "cap_embedder.0.weight" in state_dict:
            cap_feat_dim = state_dict["cap_embedder.0.weight"].shape[0]
        if cap_feat_dim is None:
            cap_feat_dim = 2560

        from .models.model import NextDiT_3B_GQA_patch2_Adaln_Refiner_WHIT_CLIP

        diffusion_model = NextDiT_3B_GQA_patch2_Adaln_Refiner_WHIT_CLIP(
            in_channels=config.get("in_channels", 16),
            cap_feat_dim=cap_feat_dim,
            qk_norm=config.get("qk_norm", True),
            clip_text_dim=config.get("clip_text_dim", 1024),
            clip_img_dim=config.get("clip_img_dim", 1024),
        )

        missing, unexpected = diffusion_model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"Missing keys: {len(missing)}")
        if unexpected:
            print(f"Unexpected keys: {len(unexpected)}")

        diffusion_model = diffusion_model.to(dtype=torch_dtype, device=offload_device)
        diffusion_model.eval()

        model_config = NewBieModelConfig()
        model = NewBieBaseModel(diffusion_model, model_config, device=offload_device)

        patcher = comfy.model_patcher.ModelPatcher(model, load_device=load_device, offload_device=offload_device)
        return (patcher,)


NODE_CLASS_MAPPINGS = {
    "NewBieUNetLoader": NewBieUNetLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NewBieUNetLoader": "NewBie UNet Loader",
}
