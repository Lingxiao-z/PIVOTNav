from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


def wrap_degrees(angle_degrees: torch.Tensor) -> torch.Tensor:
    return torch.remainder(angle_degrees.float() + 180.0, 360.0) - 180.0


def circular_interpolate(values: torch.Tensor, output_bins: int) -> torch.Tensor:
    if values.ndim != 3:
        raise ValueError("values must be [B,C,L]")
    source_bins = values.shape[-1]
    positions = torch.arange(output_bins, device=values.device, dtype=torch.float32)
    positions = positions * (float(source_bins) / float(output_bins))
    lower = torch.floor(positions).long() % source_bins
    upper = (lower + 1) % source_bins
    fraction = (positions - lower.float()).to(values.dtype).view(1, 1, -1)
    return values.index_select(-1, lower) * (1.0 - fraction) + values.index_select(
        -1, upper
    ) * fraction


class CircularConv1d(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.padding = kernel_size // 2
        self.depthwise = nn.Conv1d(
            channels, channels, kernel_size, groups=channels, bias=False
        )
        self.pointwise = nn.Conv1d(channels, channels, 1, bias=False)
        self.norm = nn.GroupNorm(8, channels)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        residual = value
        value = F.pad(value, (self.padding, self.padding), mode="circular")
        value = self.pointwise(self.depthwise(value))
        return residual + F.gelu(self.norm(value))


class CircularTokenBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.padding = kernel_size // 2
        self.depthwise = nn.Conv2d(
            channels,
            channels,
            kernel_size,
            groups=channels,
            bias=False,
        )
        self.pointwise = nn.Conv2d(channels, channels, 1, bias=False)
        self.norm = nn.GroupNorm(8, channels)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        residual = value
        value = F.pad(value, (self.padding, self.padding, 0, 0), mode="circular")
        value = F.pad(value, (0, 0, self.padding, self.padding), mode="replicate")
        value = self.pointwise(self.depthwise(value))
        return residual + F.gelu(self.norm(value))


class PairCrossAttention(nn.Module):
    def __init__(self, model_dim: int, heads: int, dropout: float, expansion: int) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(model_dim)
        self.context_norm = nn.LayerNorm(model_dim)
        self.attention = nn.MultiheadAttention(
            model_dim, heads, dropout=dropout, batch_first=True
        )
        self.output_norm = nn.LayerNorm(model_dim)
        self.feedforward = nn.Sequential(
            nn.Linear(model_dim, expansion * model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(expansion * model_dim, model_dim),
        )

    def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        normalized_context = self.context_norm(context)
        attended, _ = self.attention(
            self.query_norm(query),
            normalized_context,
            normalized_context,
            need_weights=False,
        )
        value = query + attended
        return value + self.feedforward(self.output_norm(value))


@dataclass(frozen=True)
class R362BearingHeadConfig:
    input_dim: int = 384
    token_height: int = 16
    token_width: int = 32
    vertical_bands: int = 4
    model_dim: int = 192
    cross_attention_layers: int = 2
    attention_heads: int = 6
    ffn_expansion: int = 4
    dropout: float = 0.1
    circular_layers: int = 2
    bearing_bins: int = 72
    center_label_weight: float = 0.70
    adjacent_label_weight: float = 0.15

    @property
    def bin_width_degrees(self) -> float:
        return 360.0 / self.bearing_bins


class R362PatchBearingHead(nn.Module):
    """R36.2 source-frame translation bearing head over frozen ERP spatial tokens."""

    def __init__(self, config: R362BearingHeadConfig | None = None) -> None:
        super().__init__()
        self.config = config or R362BearingHeadConfig()
        cfg = self.config
        if cfg.model_dim % cfg.attention_heads:
            raise ValueError("model_dim must be divisible by attention_heads")
        if not math.isclose(
            cfg.center_label_weight + 2.0 * cfg.adjacent_label_weight, 1.0
        ):
            raise ValueError("circular soft-label weights must sum to one")

        self.token_projection = nn.Sequential(
            nn.Conv2d(cfg.input_dim, cfg.model_dim, 1, bias=False),
            nn.GroupNorm(8, cfg.model_dim),
            nn.GELU(),
        )
        self.token_circular_encoder = nn.Sequential(
            *(CircularTokenBlock(cfg.model_dim) for _ in range(cfg.circular_layers))
        )
        self.position_projection = nn.Linear(4, cfg.model_dim, bias=False)
        self.cross_attention = nn.ModuleList(
            PairCrossAttention(
                cfg.model_dim,
                cfg.attention_heads,
                cfg.dropout,
                cfg.ffn_expansion,
            )
            for _ in range(cfg.cross_attention_layers)
        )
        self.correlation_projection = nn.Sequential(
            nn.LayerNorm(4),
            nn.Linear(4, cfg.model_dim),
            nn.GELU(),
        )
        self.pair_fusion = nn.Sequential(
            nn.LayerNorm(5 * cfg.model_dim),
            nn.Linear(5 * cfg.model_dim, cfg.model_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
        )
        self.vertical_score = nn.Linear(cfg.model_dim, 1)
        self.circular_fusion = nn.Sequential(
            *(CircularConv1d(cfg.model_dim) for _ in range(cfg.circular_layers))
        )
        self.bin_refinement = nn.Sequential(
            CircularConv1d(cfg.model_dim),
            CircularConv1d(cfg.model_dim),
        )
        self.bin_classifier = nn.Conv1d(cfg.model_dim, 1, 1)

        source_position, target_position = self._make_position_encoding()
        self.register_buffer("source_position", source_position, persistent=True)
        self.register_buffer("target_position", target_position, persistent=True)
        bin_longitude = (
            torch.arange(cfg.bearing_bins, dtype=torch.float32) + 0.5
        ) * (2.0 * math.pi / cfg.bearing_bins) - math.pi
        self.register_buffer(
            "bin_position",
            torch.stack(
                (
                    bin_longitude.sin(),
                    bin_longitude.cos(),
                    torch.zeros_like(bin_longitude),
                    torch.ones_like(bin_longitude),
                ),
                dim=-1,
            ).unsqueeze(0),
            persistent=True,
        )

    def _make_position_encoding(self) -> tuple[torch.Tensor, torch.Tensor]:
        cfg = self.config
        longitude = (
            torch.arange(cfg.token_width, dtype=torch.float32) + 0.5
        ) * (2.0 * math.pi / cfg.token_width) - math.pi
        latitude = (
            0.5 - (torch.arange(cfg.vertical_bands, dtype=torch.float32) + 0.5) / cfg.vertical_bands
        ) * math.pi
        lon_grid = longitude.view(1, cfg.token_width).expand(cfg.vertical_bands, -1)
        lat_grid = latitude.view(cfg.vertical_bands, 1).expand(-1, cfg.token_width)
        source = torch.stack(
            (lon_grid.sin(), lon_grid.cos(), lat_grid.sin(), lat_grid.cos()), dim=-1
        )
        target = torch.stack(
            (
                torch.zeros_like(lon_grid),
                torch.zeros_like(lon_grid),
                lat_grid.sin(),
                lat_grid.cos(),
            ),
            dim=-1,
        )
        return source.unsqueeze(0), target.unsqueeze(0)

    @property
    def architecture_record(self) -> dict[str, Any]:
        return {
            "schema_version": "r362_patch_bearing_head_v2",
            "config": asdict(self.config),
            "input": "frozen R36.1 spatial tokens [B,16,32,384] for current and target ERP",
            "source_position_encoding": "sin/cos longitude and latitude; longitude seam is continuous",
            "target_position_encoding": "latitude only; horizontal target roll is a K/V set permutation",
            "pair_fusion": (
                "two lightweight source-to-target cross-attention blocks plus explicit "
                "per-source-sector max, top2-margin, mean and entropy correlation"
            ),
            "circular_aggregation": (
                "circular token encoding, vertical attention, 32-sector circular fusion, "
                "then feature-domain interpolation and native 72-bin circular refinement"
            ),
            "outputs": [
                "bearing_logits",
                "bearing_distribution",
                "bearing_angle_degrees",
                "bearing_confidence",
                "bearing_entropy",
                "topk_bins",
            ],
            "global_descriptor_only": False,
            "vpr_descriptor_modified": False,
            "arrival_or_goal_anchor_modified": False,
        }

    def _project_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        cfg = self.config
        expected = (cfg.token_height, cfg.token_width, cfg.input_dim)
        if tokens.ndim != 4 or tuple(tokens.shape[1:]) != expected:
            raise ValueError(f"expected tokens [B,{expected[0]},{expected[1]},{expected[2]}]")
        value = tokens.permute(0, 3, 1, 2).contiguous()
        value = self.token_circular_encoder(self.token_projection(value.float()))
        value = F.adaptive_avg_pool2d(value, (cfg.vertical_bands, cfg.token_width))
        return value.permute(0, 2, 3, 1).contiguous()

    def _correlation_features(
        self, source: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        source_sector = F.normalize(source.mean(dim=1).float(), dim=-1)
        target_sector = F.normalize(target.mean(dim=1).float(), dim=-1)
        correlation = torch.einsum("bwd,bvd->bwv", source_sector, target_sector)
        top2 = torch.topk(correlation, k=2, dim=-1).values
        probability = torch.softmax(correlation, dim=-1)
        entropy = -(
            probability * probability.clamp_min(1e-8).log()
        ).sum(dim=-1) / math.log(float(self.config.token_width))
        summary = torch.stack(
            (
                top2[..., 0],
                top2[..., 0] - top2[..., 1],
                correlation.mean(dim=-1),
                entropy,
            ),
            dim=-1,
        )
        return self.correlation_projection(summary), correlation

    def forward(self, source_tokens: torch.Tensor, target_tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        if source_tokens.shape != target_tokens.shape:
            raise ValueError("source and target token tensors must have identical shape")
        cfg = self.config
        source = self._project_tokens(source_tokens)
        target = self._project_tokens(target_tokens)
        correlation_feature, local_correlation = self._correlation_features(source, target)
        source = source + self.position_projection(self.source_position.to(source.dtype))
        target = target + self.position_projection(self.target_position.to(target.dtype))

        batch = source.shape[0]
        source_base = source.reshape(batch, cfg.vertical_bands * cfg.token_width, cfg.model_dim)
        target_set = target.reshape(batch, cfg.vertical_bands * cfg.token_width, cfg.model_dim)
        attended = source_base
        for block in self.cross_attention:
            attended = block(attended, target_set)

        pair = torch.cat(
            (
                source_base,
                attended,
                source_base * attended,
                (source_base - attended).abs(),
                correlation_feature.unsqueeze(1)
                .expand(-1, cfg.vertical_bands, -1, -1)
                .reshape(batch, cfg.vertical_bands * cfg.token_width, cfg.model_dim),
            ),
            dim=-1,
        )
        pair = self.pair_fusion(pair).reshape(
            batch, cfg.vertical_bands, cfg.token_width, cfg.model_dim
        )
        vertical_weight = torch.softmax(self.vertical_score(pair).squeeze(-1), dim=1)
        sectors = (pair * vertical_weight.unsqueeze(-1)).sum(dim=1)
        sectors = self.circular_fusion(sectors.transpose(1, 2).contiguous())
        bins = circular_interpolate(sectors, cfg.bearing_bins)
        bin_position = self.position_projection(self.bin_position.to(bins.dtype)).transpose(1, 2)
        bins = self.bin_refinement(bins + bin_position)
        bearing_logits = self.bin_classifier(bins).squeeze(1)
        distribution = torch.softmax(bearing_logits.float(), dim=-1)

        centers = (
            torch.arange(cfg.bearing_bins, device=distribution.device, dtype=torch.float32)
            * cfg.bin_width_degrees
            - 180.0
            + 0.5 * cfg.bin_width_degrees
        )
        radians = torch.deg2rad(centers)
        sine = (distribution * radians.sin()).sum(dim=-1)
        cosine = (distribution * radians.cos()).sum(dim=-1)
        angle = wrap_degrees(torch.rad2deg(torch.atan2(sine, cosine)))
        entropy = -(
            distribution * distribution.clamp_min(1e-8).log()
        ).sum(dim=-1) / math.log(cfg.bearing_bins)
        confidence = distribution.max(dim=-1).values
        top_probability, top_bins = torch.topk(distribution, k=5, dim=-1)
        return {
            "bearing_logits": bearing_logits,
            "bearing_distribution": distribution,
            "bearing_angle_degrees": angle,
            "bearing_confidence": confidence,
            "bearing_entropy": entropy,
            "topk_bins": top_bins,
            "topk_probabilities": top_probability,
            "source_sector_features": sectors.transpose(1, 2),
            "local_correlation_32x32": local_correlation,
        }


def circular_soft_labels(
    target_bins: torch.Tensor, config: R362BearingHeadConfig
) -> torch.Tensor:
    target_bins = target_bins.long()
    if target_bins.ndim != 1:
        raise ValueError("target_bins must be [B]")
    if ((target_bins < 0) | (target_bins >= config.bearing_bins)).any():
        raise ValueError("target bin outside configured circular range")
    labels = torch.zeros(
        target_bins.shape[0],
        config.bearing_bins,
        device=target_bins.device,
        dtype=torch.float32,
    )
    labels.scatter_(1, target_bins[:, None], config.center_label_weight)
    labels.scatter_(
        1,
        ((target_bins - 1) % config.bearing_bins)[:, None],
        config.adjacent_label_weight,
    )
    labels.scatter_(
        1,
        ((target_bins + 1) % config.bearing_bins)[:, None],
        config.adjacent_label_weight,
    )
    return labels


def circular_soft_label_cross_entropy(
    logits: torch.Tensor,
    target_bins: torch.Tensor,
    config: R362BearingHeadConfig,
) -> torch.Tensor:
    if logits.ndim != 2 or logits.shape[1] != config.bearing_bins:
        raise ValueError(f"logits must be [B,{config.bearing_bins}]")
    labels = circular_soft_labels(target_bins, config)
    return -(labels * F.log_softmax(logits.float(), dim=-1)).sum(dim=-1).mean()
