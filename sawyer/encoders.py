"""Frozen encoders: V-JEPA 2.1 ViT-L for camera frames, Qwen3-Embedding-0.6B for instructions."""

import sys
from pathlib import Path

import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).parents[1]
VJEPA_REPO = ROOT / "third_party/vjepa2"
VJEPA_CKPT = ROOT / "checkpoints/vjepa2_1/vjepa2_1_vitl_dist_vitG_384.pt"
QWEN_DIR = ROOT / "checkpoints/qwen3-embedding-0.6b"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class FrozenVJEPA(nn.Module):
    """V-JEPA 2.1 ViT-L/16 in single-image mode. uint8 frames (N, H, W, 3) -> layer-normed patch features (N, P, 1024)."""

    def __init__(self, path: str | Path = VJEPA_CKPT):
        super().__init__()
        sys.path.insert(0, str(VJEPA_REPO))  # the repo's packages are named `app` and `src`
        from app.vjepa_2_1.models import vision_transformer as vit
        from src.hub.backbones import _clean_backbone_key

        # Same construction as the repo's vjepa2_1_vit_large_384 hub entry; RoPE interpolates to other resolutions.
        self.encoder = vit.vit_large(
            patch_size=16, img_size=(384, 384), num_frames=64, tubelet_size=2, use_sdpa=True, use_SiLU=False,
            wide_SiLU=True, uniform_power=False, use_rope=True, img_temporal_dim_size=1, interpolate_rope=True,
        )  # fmt: skip
        state = torch.load(path, map_location="cpu", weights_only=True)["ema_encoder"]
        self.encoder.load_state_dict(_clean_backbone_key(state), strict=True)
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1, 1), persistent=False)
        self.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True):
        return super().train(False)  # frozen: never leave eval mode

    @torch.no_grad()
    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        x = frames.permute(0, 3, 1, 2).unsqueeze(2).float().div_(255)  # (N, 3, T=1, H, W), on the frames' device
        x = (x - self.mean) / self.std
        # The encoder's RoPE upcasts q/k to fp32, so bf16 must come from autocast, not from casting the weights.
        with torch.autocast(x.device.type, dtype=torch.bfloat16):
            return self.encoder(x)


class FrozenQwen(nn.Module):
    """Qwen3-Embedding-0.6B token states (B, L, 1024) for the predictor's cross-attention."""

    def __init__(self, path: str | Path = QWEN_DIR):
        super().__init__()
        self.model = AutoModel.from_pretrained(path, dtype=torch.bfloat16)
        self.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True):
        return super().train(False)

    @torch.no_grad()
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids=input_ids, attention_mask=attention_mask.long()).last_hidden_state


def load_tokenizer(path: str | Path = QWEN_DIR):
    tokenizer = AutoTokenizer.from_pretrained(path)
    tokenizer.padding_side = "right"  # causal model: right padding leaves the real tokens' states untouched
    return tokenizer
