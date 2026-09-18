"""Last-block-only B2 model for Revised DINOv2 Panorama FG/FS v3.

The frozen prefix of the official ViT-S/14 is evaluated once and cached as
513-token sequences (CLS plus the native 16x32 patch grid).  This module owns
only the official final transformer block, the frozen final norm, and the v3
direction decoder.  It therefore cannot accidentally unfreeze earlier DINO
layers or fall back to the legacy FS-only heads.
"""
from __future__ import annotations

import torch
from torch import nn

from panorama_fs.dino_fgfs_revision3 import (
    RevisedDINOv2PanoramaFGFSV3,
    RevisedDINOv3Output,
)


class RevisedDINOv2PanoramaFGFSV3LastBlockB2(nn.Module):
    model_name = "revised_dinov2_panorama_fgfs_v3_last_block_b2"

    def __init__(
        self,
        last_block: nn.Module,
        final_norm: nn.Module,
        decoder: RevisedDINOv2PanoramaFGFSV3,
    ) -> None:
        super().__init__()
        self.last_block = last_block
        self.final_norm = final_norm
        self.decoder = decoder
        for parameter in self.final_norm.parameters():
            parameter.requires_grad_(False)

    def encode_cached_prefix(self, tokens):
        if tokens.ndim != 3 or tokens.shape[1:] != (513, 384):
            raise ValueError("cached DINO prefix tokens must have shape [B,513,384]")
        # The installed CUDA/cuBLASLt stack raises SIGFPE for several BF16
        # last-block batch shapes.  Keep the single trainable DINO block in
        # FP32; the direction decoder remains under the caller's BF16
        # autocast context.
        with torch.autocast(device_type="cuda", enabled=False):
            encoded = self.final_norm(self.last_block(tokens.float()))
        return encoded[:, 1:].reshape(tokens.shape[0], 16, 32, 384)

    def forward(self, current_tokens, goal_tokens) -> RevisedDINOv3Output:
        current_grid = self.encode_cached_prefix(current_tokens)
        goal_grid = self.encode_cached_prefix(goal_tokens)
        return self.decoder(current_grid, goal_grid)
