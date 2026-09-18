from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


def _resolve_scripts_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if parent.name == "scripts":
            return parent
    return current.parents[7]


_DINOV3_REPO = (
    Path(__file__).resolve().parents[4]
    / "third_party"
    / "unified-depth-processor"
    / "src"
    / "dap"
    / "depth_anything_v2_metric"
    / "depth_anything_v2"
    / "dinov3"
)

if str(_DINOV3_REPO) not in sys.path:
    sys.path.insert(0, str(_DINOV3_REPO))


def infer_backbone_variant(
    state_dict: dict[str, torch.Tensor],
    checkpoint_args: dict[str, Any] | None = None,
) -> tuple[str, int, int]:
    patch_weight = state_dict["backbone.model.patch_embed.proj.weight"]
    embed_dim = int(patch_weight.shape[0])
    block_indices = {
        int(key.split("backbone.model.blocks.")[1].split(".")[0])
        for key in state_dict
        if key.startswith("backbone.model.blocks.")
    }
    num_blocks = len(block_indices)
    if checkpoint_args:
        backbone_type = str(checkpoint_args.get("backbone_type", "")).lower()
        if backbone_type in {"vits16", "vits"} and embed_dim == 384:
            return "vits16", embed_dim, num_blocks
        if backbone_type in {"vitb16", "vitb"} and embed_dim == 768:
            return "vitb16", embed_dim, num_blocks
        if backbone_type in {"vitl16", "vitl"} and embed_dim == 1024:
            return "vitl16", embed_dim, num_blocks
    if embed_dim == 384 and num_blocks == 12:
        return "vits16", embed_dim, num_blocks
    if embed_dim == 768 and num_blocks == 12:
        return "vitb16", embed_dim, num_blocks
    if embed_dim == 1024 and num_blocks == 24:
        return "vitl16", embed_dim, num_blocks
    raise ValueError(f"Unsupported DINOv3 checkpoint: embed_dim={embed_dim}, num_blocks={num_blocks}")


class DINOv3Backbone(nn.Module):
    def __init__(self, variant: str, freeze: bool = True) -> None:
        super().__init__()
        from dinov3.hub.backbones import dinov3_vitb16, dinov3_vitl16, dinov3_vits16

        builders = {
            "vits16": dinov3_vits16,
            "vitb16": dinov3_vitb16,
            "vitl16": dinov3_vitl16,
        }
        if variant not in builders:
            raise ValueError(f"Unsupported backbone variant: {variant}")
        self.model = builders[variant](pretrained=False)
        self.variant = variant
        self.embed_dim = int(self.model.embed_dim)
        self.patch_size = int(self.model.patch_size)
        self.num_blocks = int(self.model.n_blocks)
        if freeze:
            for parameter in self.model.parameters():
                parameter.requires_grad = False

    def forward_patch_tokens(self, x: torch.Tensor) -> torch.Tensor:
        output = self.model.forward_features(x)
        return output["x_norm_patchtokens"]

    def forward_multiscale_tokens(self, x: torch.Tensor, layer_indices: tuple[int, ...]) -> list[torch.Tensor]:
        outputs = self.model.get_intermediate_layers(
            x,
            n=layer_indices,
            reshape=False,
            return_class_token=False,
            return_extra_tokens=False,
            norm=True,
        )
        return list(outputs)
