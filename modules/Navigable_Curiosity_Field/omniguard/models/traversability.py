from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import DINOv3Backbone, infer_backbone_variant


@dataclass
class TraversabilityArchitectureConfig:
    backbone_variant: str
    embed_dim: int
    num_blocks: int
    decoder_hidden_dim: int
    decoder_dim: int
    angle_dim: int
    num_angles: int
    multi_scale: bool
    fpn_dim: int
    decoder_dropout: bool
    layer_indices: tuple[int, ...]


def default_multiscale_layers(backbone_variant: str) -> tuple[int, ...]:
    if backbone_variant == "vits16":
        return (2, 5, 8, 11)
    if backbone_variant == "vitb16":
        return (2, 5, 8, 11)
    if backbone_variant == "vitl16":
        return (4, 11, 17, 23)
    raise ValueError(f"No default multiscale layers for backbone {backbone_variant}")


def infer_architecture_from_checkpoint(
    checkpoint_state: dict[str, torch.Tensor],
    checkpoint_args: dict[str, Any] | None = None,
    override_multi_scale: bool | None = None,
    override_layers: tuple[int, ...] | None = None,
) -> TraversabilityArchitectureConfig:
    backbone_variant, embed_dim, num_blocks = infer_backbone_variant(checkpoint_state, checkpoint_args)
    multi_scale = any(key.startswith("ms_fusion.") for key in checkpoint_state)
    if override_multi_scale is not None:
        multi_scale = override_multi_scale
    decoder_conv_indices = sorted(
        int(key.split(".")[2])
        for key, value in checkpoint_state.items()
        if key.startswith("decoder.net.") and key.endswith(".weight") and value.ndim == 4
    )
    if not decoder_conv_indices:
        raise ValueError("Failed to infer decoder output channels from checkpoint.")
    decoder_hidden_dim = int(checkpoint_state[f"decoder.net.{decoder_conv_indices[0]}.weight"].shape[0])
    decoder_dim = int(checkpoint_state[f"decoder.net.{decoder_conv_indices[-1]}.weight"].shape[0])
    angle_dim = int(checkpoint_state["angle_projector.angle_queries"].shape[1])
    num_angles = int(checkpoint_state["angle_projector.angle_queries"].shape[0])
    fpn_dim = 256
    if "ms_fusion.lateral.0.weight" in checkpoint_state:
        fpn_dim = int(checkpoint_state["ms_fusion.lateral.0.weight"].shape[0])
    decoder_dropout = decoder_conv_indices[-1] == 9
    layer_indices = override_layers or default_multiscale_layers(backbone_variant)
    return TraversabilityArchitectureConfig(
        backbone_variant=backbone_variant,
        embed_dim=embed_dim,
        num_blocks=num_blocks,
        decoder_hidden_dim=decoder_hidden_dim,
        decoder_dim=decoder_dim,
        angle_dim=angle_dim,
        num_angles=num_angles,
        multi_scale=multi_scale,
        fpn_dim=fpn_dim,
        decoder_dropout=decoder_dropout,
        layer_indices=tuple(layer_indices),
    )


class MultiScaleFusion(nn.Module):
    def __init__(self, embed_dim: int, channels: int = 256, num_inputs: int = 4) -> None:
        super().__init__()
        self.lateral = nn.ModuleList([nn.Conv2d(embed_dim, channels, kernel_size=1) for _ in range(num_inputs)])
        self.out_proj = nn.Conv2d(channels, embed_dim, kernel_size=1)

    def forward(self, token_features: list[torch.Tensor], hw: tuple[int, int]) -> torch.Tensor:
        h_tokens, w_tokens = hw
        lateral_features = []
        for conv, tokens in zip(self.lateral, token_features):
            feat = tokens.transpose(1, 2).reshape(tokens.shape[0], tokens.shape[2], h_tokens, w_tokens)
            lateral_features.append(conv(feat))
        fused = torch.stack(lateral_features, dim=0).mean(dim=0)
        return self.out_proj(fused)


class LightweightDecoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, use_dropout_slot: bool = False) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_dim, hidden_dim, kernel_size=1),
            nn.BatchNorm2d(hidden_dim, track_running_stats=False),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim, track_running_stats=False),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
        ]
        if use_dropout_slot:
            layers.append(nn.Dropout2d(p=0.0))
        layers.extend(
            [
                nn.Conv2d(hidden_dim, out_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_dim, track_running_stats=False),
            nn.GELU(),
            ]
        )
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AngleProjector(nn.Module):
    def __init__(self, spatial_dim: int, angle_dim: int, num_angles: int) -> None:
        super().__init__()
        self.angle_dim = angle_dim
        self.num_angles = num_angles
        self.angle_queries = nn.Parameter(torch.randn(num_angles, angle_dim) * 0.02)
        self.register_buffer("angle_pos", self._sinusoidal_encoding(num_angles, angle_dim))
        self.azimuth_encoder = nn.Sequential(
            nn.Conv2d(1, spatial_dim // 4, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(spatial_dim // 4, spatial_dim, kernel_size=1),
        )
        self.query_proj = nn.Linear(angle_dim, angle_dim)
        self.key_proj = nn.Linear(spatial_dim, angle_dim)
        self.value_proj = nn.Linear(spatial_dim, angle_dim)
        self.out_proj = nn.Sequential(
            nn.Linear(angle_dim, angle_dim),
            nn.GELU(),
            nn.LayerNorm(angle_dim),
        )

    @staticmethod
    def _sinusoidal_encoding(num_angles: int, dim: int) -> torch.Tensor:
        angles = torch.arange(num_angles, dtype=torch.float32) / float(num_angles) * 2.0 * math.pi
        half_dim = dim // 2
        freqs = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -(math.log(10000.0) / max(half_dim - 1, 1)))
        outer = angles[:, None] * freqs[None, :]
        return torch.cat([torch.sin(outer), torch.cos(outer)], dim=-1)

    def forward(self, features_2d: torch.Tensor, azimuth_map: torch.Tensor | None) -> torch.Tensor:
        batch_size, _, height, width = features_2d.shape
        if azimuth_map is not None:
            azimuth = F.interpolate(azimuth_map, size=(height, width), mode="bilinear", align_corners=False)
            features_2d = features_2d + self.azimuth_encoder(azimuth)
        feat_flat = features_2d.flatten(2).permute(0, 2, 1)
        keys = self.key_proj(feat_flat)
        values = self.value_proj(feat_flat)
        queries = (self.angle_queries + self.angle_pos).unsqueeze(0).expand(batch_size, -1, -1)
        queries = self.query_proj(queries)
        attention = torch.matmul(queries, keys.transpose(-2, -1)) / math.sqrt(self.angle_dim)
        attention = F.softmax(attention, dim=-1)
        fused = torch.matmul(attention, values)
        return self.out_proj(fused)


class TraversabilityModel(nn.Module):
    def __init__(self, architecture: TraversabilityArchitectureConfig, freeze_backbone: bool = True) -> None:
        super().__init__()
        self.architecture = architecture
        self.backbone = DINOv3Backbone(architecture.backbone_variant, freeze=freeze_backbone)
        self.ms_fusion = (
            MultiScaleFusion(architecture.embed_dim, channels=architecture.fpn_dim, num_inputs=len(architecture.layer_indices))
            if architecture.multi_scale
            else None
        )
        self.decoder = LightweightDecoder(
            architecture.embed_dim,
            architecture.decoder_hidden_dim,
            architecture.decoder_dim,
            use_dropout_slot=architecture.decoder_dropout,
        )
        self.angle_projector = AngleProjector(architecture.decoder_dim, architecture.angle_dim, architecture.num_angles)
        self.radar_dist_mlp = nn.Sequential(
            nn.Linear(architecture.angle_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.GELU(),
        )
        self.dist_out = nn.Sequential(nn.Linear(64, 1), nn.Softplus())
        self.exist_logit = nn.Linear(64, 1)

    def forward(
        self,
        x: torch.Tensor,
        azimuth_map: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        height_tokens = x.shape[2] // self.backbone.patch_size
        width_tokens = x.shape[3] // self.backbone.patch_size
        if self.ms_fusion is not None:
            token_features = self.backbone.forward_multiscale_tokens(x, self.architecture.layer_indices)
            fused = self.ms_fusion(token_features, (height_tokens, width_tokens))
        else:
            tokens = self.backbone.forward_patch_tokens(x)
            fused = tokens.transpose(1, 2).reshape(tokens.shape[0], tokens.shape[2], height_tokens, width_tokens)
        decoded = self.decoder(fused)
        angle_features = self.angle_projector(decoded, azimuth_map)
        radar_features = self.radar_dist_mlp(angle_features)
        raw_dist = self.dist_out(radar_features).squeeze(-1)
        exist_logit = self.exist_logit(radar_features).squeeze(-1)
        exist_prob = torch.sigmoid(exist_logit)
        radar_dist = raw_dist * exist_prob + (1.0 - exist_prob) * 100.0
        return radar_dist, exist_logit, raw_dist
