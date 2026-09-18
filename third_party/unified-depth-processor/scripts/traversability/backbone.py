"""DINOv3 ViT backbone wrapper for traversability model.

Supports multiple ViT variants (vits16, vitb16, vitl16, etc.) with single-scale
and multi-scale token extraction for FPN-style fusion.
"""

import sys
import os
from typing import List, Optional

import torch
import torch.nn as nn


# Add DINOv3 repo to path for direct import
_DINOV3_REPO = os.path.join(
    os.path.dirname(__file__), '..', '..', 'src',
    'dap', 'depth_anything_v2_metric', 'depth_anything_v2', 'dinov3'
)
if os.path.isdir(_DINOV3_REPO) and _DINOV3_REPO not in sys.path:
    sys.path.insert(0, os.path.abspath(_DINOV3_REPO))

# Supported model names -> hub function names
_SUPPORTED_MODELS = {
    'vits16':     'dinov3_vits16',
    'vitb16':     'dinov3_vitb16',
    'vitl16':     'dinov3_vitl16',
    'vitl16plus': 'dinov3_vitl16plus',
    'vith16plus': 'dinov3_vith16plus',
}


class DINOv3Backbone(nn.Module):
    """DINOv3 ViT backbone that outputs patch tokens.

    Loads the official DINOv3 DinoVisionTransformer and extracts
    normalized patch tokens for downstream use.

    Args:
        weights_path: Path to pretrained .pth weights file.
        model_name: ViT variant to load. One of: vits16, vitb16, vitl16,
            vitl16plus, vith16plus. Default: 'vitb16'.
        freeze: If True, freeze all backbone parameters.
        unfreeze_last_n: Unfreeze the last N transformer blocks
            (only applies when freeze=True).
    """

    def __init__(self,
                 weights_path: Optional[str],
                 model_name: str = 'vitb16',
                 freeze: bool = True,
                 unfreeze_last_n: int = 0):
        super().__init__()

        if model_name not in _SUPPORTED_MODELS:
            raise ValueError(
                f"Unknown backbone model_name '{model_name}'. "
                f"Supported: {list(_SUPPORTED_MODELS)}"
            )

        import importlib
        hub = importlib.import_module('dinov3.hub.backbones')
        factory = getattr(hub, _SUPPORTED_MODELS[model_name])
        # pretrained=False to avoid torch.hub download; load weights manually
        self.model = factory(pretrained=False)
        if weights_path:
            state_dict = torch.load(weights_path, map_location='cpu', weights_only=True)
            self.model.load_state_dict(state_dict, strict=True)
        else:
            print(
                "[WARN] DINOv3 pretrained weights_path is empty; "
                "using factory initialization until the caller loads a full checkpoint.",
                flush=True,
            )

        self.embed_dim = self.model.embed_dim
        self.patch_size = self.model.patch_size
        self.n_blocks = self.model.n_blocks

        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False

            if unfreeze_last_n > 0:
                blocks = self.model.blocks[-unfreeze_last_n:]
                for block in blocks:
                    for param in block.parameters():
                        param.requires_grad = True
                # Also unfreeze final norm
                for param in self.model.norm.parameters():
                    param.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract patch tokens from input images.

        Args:
            x: Input images (B, 3, H, W). H and W should be divisible by patch_size.

        Returns:
            Patch tokens (B, N, embed_dim) where N = (H/patch_size) * (W/patch_size).
        """
        out = self.model.forward_features(x)
        return out['x_norm_patchtokens']

    def forward_multi_scale(self,
                            x: torch.Tensor,
                            block_indices: List[int] = None,
                            ) -> List[torch.Tensor]:
        """Extract patch tokens from multiple intermediate blocks.

        Args:
            x: Input images (B, 3, H, W). H and W should be divisible by patch_size.
            block_indices: 0-based block indices to extract. Defaults to
                evenly-spaced quarter points of the backbone depth.

        Returns:
            List of (B, N, embed_dim) tensors, one per requested block index.
        """
        if block_indices is None:
            n = self.n_blocks
            block_indices = tuple(n // 4 * i - 1 for i in range(1, 5))

        tokens = self.model.get_intermediate_layers(
            x,
            n=block_indices,
            return_class_token=False,
            norm=True,
        )
        return list(tokens)
