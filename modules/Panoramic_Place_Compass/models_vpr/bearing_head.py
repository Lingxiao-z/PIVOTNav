from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Dict

import torch
from torch import nn
from torch.nn import functional as F

from .yaw_head import CircularConv1d, CircularTokenBlock


@dataclass(frozen=True)
class R35BearingConfig:
    input_dim: int = 384
    model_dim: int = 128
    token_height: int = 16
    token_width: int = 32
    vertical_bands: int = 4
    attention_heads: int = 4
    attention_dropout: float = 0.0
    circular_layers: int = 2
    bearing_bins: int = 72
    pair_embedding_dim: int = 128
    soft_label_sigma_bins: float = 1.0
    confidence_error_degrees: float = 15.0

    @property
    def bin_width_degrees(self) -> float:
        return 360.0 / float(self.bearing_bins)


@dataclass(frozen=True)
class R35BearingLossWeights:
    circular_soft_label: float = 1.0
    sin_cos_regression: float = 0.5
    geodesic: float = 0.5
    rotation_equivariance: float = 0.25
    reciprocal_consistency: float = 0.25
    confidence_calibration: float = 0.10
    valid_classification: float = 0.25


def wrap_degrees_tensor(angle_degrees: torch.Tensor) -> torch.Tensor:
    return torch.remainder(angle_degrees.float() + 180.0, 360.0) - 180.0


def circular_error_degrees_tensor(
    predicted_degrees: torch.Tensor,
    target_degrees: torch.Tensor,
) -> torch.Tensor:
    return wrap_degrees_tensor(predicted_degrees - target_degrees).abs()


def circular_interpolate_1d(values: torch.Tensor, output_size: int) -> torch.Tensor:
    if values.ndim != 3:
        raise ValueError("values must be [B,C,L]")
    if output_size <= 0:
        raise ValueError("output_size must be positive")
    source_size = values.shape[-1]
    positions = torch.arange(output_size, device=values.device, dtype=torch.float32)
    positions = positions * (float(source_size) / float(output_size))
    lower = torch.floor(positions).long() % source_size
    upper = (lower + 1) % source_size
    fraction = (positions - torch.floor(positions)).to(values.dtype).view(1, 1, -1)
    return values.index_select(-1, lower) * (1.0 - fraction) + values.index_select(-1, upper) * fraction


def circular_shift_distribution(
    distribution: torch.Tensor,
    shift_bins: torch.Tensor | float,
) -> torch.Tensor:
    if distribution.ndim != 2:
        raise ValueError("distribution must be [B,L]")
    batch, bins = distribution.shape
    shift = torch.as_tensor(shift_bins, device=distribution.device, dtype=torch.float32)
    if shift.ndim == 0:
        shift = shift.expand(batch)
    if shift.shape != (batch,):
        raise ValueError("shift_bins must be scalar or [B]")
    destination = torch.arange(bins, device=distribution.device, dtype=torch.float32).view(1, -1)
    source = torch.remainder(destination - shift.view(-1, 1), float(bins))
    lower = torch.floor(source).long()
    upper = (lower + 1) % bins
    fraction = (source - lower.float()).to(distribution.dtype)
    return distribution.gather(1, lower) * (1.0 - fraction) + distribution.gather(1, upper) * fraction


def wrapped_soft_labels(
    target_degrees: torch.Tensor,
    cfg: R35BearingConfig,
) -> torch.Tensor:
    target_position = torch.remainder(target_degrees.float() + 180.0, 360.0) / cfg.bin_width_degrees
    indices = torch.arange(
        cfg.bearing_bins,
        device=target_degrees.device,
        dtype=torch.float32,
    ).view(1, -1)
    distance = (indices - target_position.view(-1, 1)).abs()
    distance = torch.minimum(distance, cfg.bearing_bins - distance)
    labels = torch.exp(-0.5 * (distance / cfg.soft_label_sigma_bins) ** 2)
    return labels / labels.sum(dim=-1, keepdim=True).clamp_min(1e-8)


class CircularSpatialTokenEncoder(nn.Module):
    def __init__(self, cfg: R35BearingConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.projection = nn.Conv2d(cfg.input_dim, cfg.model_dim, 1, bias=False)
        self.normalization = nn.GroupNorm(8, cfg.model_dim)
        self.blocks = nn.Sequential(
            *[CircularTokenBlock(cfg.model_dim) for _ in range(cfg.circular_layers)]
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        expected = (self.cfg.token_height, self.cfg.token_width, self.cfg.input_dim)
        if tokens.ndim != 4 or tuple(tokens.shape[1:]) != expected:
            raise ValueError(f"expected spatial tokens [B,{expected[0]},{expected[1]},{expected[2]}]")
        features = tokens.permute(0, 3, 1, 2).contiguous()
        features = F.gelu(self.normalization(self.projection(features.float())))
        features = self.blocks(features)
        return F.adaptive_avg_pool2d(
            features,
            (self.cfg.vertical_bands, self.cfg.token_width),
        )


class CrossAttentionPairBlock(nn.Module):
    def __init__(self, cfg: R35BearingConfig) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(cfg.model_dim)
        self.context_norm = nn.LayerNorm(cfg.model_dim)
        self.attention = nn.MultiheadAttention(
            cfg.model_dim,
            cfg.attention_heads,
            dropout=cfg.attention_dropout,
            batch_first=True,
        )
        self.output_norm = nn.LayerNorm(cfg.model_dim)
        self.feedforward = nn.Sequential(
            nn.Linear(cfg.model_dim, 2 * cfg.model_dim),
            nn.GELU(),
            nn.Linear(2 * cfg.model_dim, cfg.model_dim),
        )

    def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        attended, _ = self.attention(
            self.query_norm(query),
            self.context_norm(context),
            self.context_norm(context),
            need_weights=False,
        )
        output = query + attended
        return output + self.feedforward(self.output_norm(output))


class R35RelativeTranslationBearingHead(nn.Module):
    """Spatial-token relative translation bearing head with circular ERP structure."""

    def __init__(self, cfg: R35BearingConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or R35BearingConfig()
        self._confidence_output_mode = "gated_evidence"
        self.token_encoder = CircularSpatialTokenEncoder(self.cfg)
        self.source_to_target = CrossAttentionPairBlock(self.cfg)
        self.target_to_source = CrossAttentionPairBlock(self.cfg)
        self.pair_projection = nn.Sequential(
            nn.LayerNorm(4 * self.cfg.model_dim),
            nn.Linear(4 * self.cfg.model_dim, self.cfg.model_dim),
            nn.GELU(),
        )
        self.vertical_score = nn.Linear(self.cfg.model_dim, 1)
        circular_layers: list[nn.Module] = []
        for _ in range(self.cfg.circular_layers):
            circular_layers.extend(
                (
                    CircularConv1d(self.cfg.model_dim, self.cfg.model_dim, 3, bias=False),
                    nn.GroupNorm(8, self.cfg.model_dim),
                    nn.GELU(),
                )
            )
        self.circular_fusion = nn.Sequential(*circular_layers)
        self.sector_logit = CircularConv1d(self.cfg.model_dim, 1, 3)
        self.regression_head = nn.Sequential(
            nn.LayerNorm(self.cfg.model_dim),
            nn.Linear(self.cfg.model_dim, self.cfg.model_dim),
            nn.GELU(),
            nn.Linear(self.cfg.model_dim, 2),
        )
        self.valid_head = nn.Sequential(
            nn.LayerNorm(self.cfg.model_dim),
            nn.Linear(self.cfg.model_dim, 1),
        )
        self.confidence_head = nn.Sequential(
            nn.LayerNorm(self.cfg.model_dim + 4),
            nn.Linear(self.cfg.model_dim + 4, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )
        self.pair_embedding = nn.Sequential(
            nn.LayerNorm(self.cfg.model_dim + 4),
            nn.Linear(self.cfg.model_dim + 4, self.cfg.pair_embedding_dim),
        )

    @property
    def architecture_record(self) -> Dict[str, Any]:
        return {
            "schema_version": "r35_relative_translation_bearing_head_v1",
            "config": asdict(self.cfg),
            "input": "ordered DINOv2 spatial patch tokens [B,16,32,384] for source and target ERP",
            "circular_structure": "horizontal circular convolutions before and after cross-attention",
            "pair_fusion": (
                f"bidirectional cross-attention over {self.cfg.vertical_bands} vertical bands "
                f"and {self.cfg.token_width} longitude sectors"
            ),
            "target_roll_invariance": "target tokens are consumed as an unordered key/value set after circular-equivariant encoding",
            "source_roll_equivariance": "source query tokens remain longitude ordered through circular sector logits",
            "outputs": [
                "bearing_distribution_72",
                "bearing_angle_degrees",
                "bearing_confidence",
                "bearing_valid_probability",
                "pair_embedding",
            ],
            "not_relative_yaw": True,
            "global_descriptor_only": False,
            "confidence_output_mode": self._confidence_output_mode,
        }

    @property
    def confidence_output_mode(self) -> str:
        return self._confidence_output_mode

    def set_confidence_output_mode(self, mode: str) -> None:
        if mode not in ("gated_evidence", "raw_learned"):
            raise ValueError(f"unsupported bearing confidence output mode: {mode}")
        self._confidence_output_mode = mode

    def forward(self, source_tokens: torch.Tensor, target_tokens: torch.Tensor) -> Dict[str, torch.Tensor]:
        source_map = self.token_encoder(source_tokens)
        target_map = self.token_encoder(target_tokens)
        batch, channels, bands, width = source_map.shape
        source_sector_base = F.normalize(source_map.mean(dim=2).transpose(1, 2).float(), dim=-1)
        target_sector_base = F.normalize(target_map.mean(dim=2).transpose(1, 2).float(), dim=-1)
        local_correlation = torch.einsum(
            "bwd,bvd->bwv", source_sector_base, target_sector_base
        )
        local_flat = local_correlation.flatten(1)
        local_top2 = torch.topk(local_flat, k=2, dim=-1).values
        local_probability = torch.softmax(local_flat, dim=-1)
        local_entropy = -(
            local_probability * local_probability.clamp_min(1e-8).log()
        ).sum(dim=-1) / math.log(float(width * width))
        local_correlation_summary = torch.stack(
            (
                local_top2[:, 0],
                local_correlation.mean(dim=(1, 2)),
                local_top2[:, 0] - local_top2[:, 1],
                local_entropy,
            ),
            dim=-1,
        )
        source_sequence = source_map.permute(0, 2, 3, 1).reshape(batch, bands * width, channels)
        target_sequence = target_map.permute(0, 2, 3, 1).reshape(batch, bands * width, channels)
        source_context = self.source_to_target(source_sequence, target_sequence)
        target_context = self.target_to_source(target_sequence, source_sequence)

        source_pair = torch.cat(
            (
                source_sequence,
                source_context,
                source_sequence * source_context,
                (source_sequence - source_context).abs(),
            ),
            dim=-1,
        )
        pair_tokens = self.pair_projection(source_pair).reshape(batch, bands, width, channels)
        vertical_weights = torch.softmax(self.vertical_score(pair_tokens).squeeze(-1), dim=1)
        sectors = (pair_tokens * vertical_weights.unsqueeze(-1)).sum(dim=1)
        fused = self.circular_fusion(sectors.transpose(1, 2).contiguous())
        sector_logits = self.sector_logit(fused).squeeze(1)
        bearing_logits = circular_interpolate_1d(sector_logits.unsqueeze(1), self.cfg.bearing_bins).squeeze(1)
        bearing_distribution = torch.softmax(bearing_logits.float(), dim=-1)

        centers = torch.arange(
            self.cfg.bearing_bins,
            device=bearing_logits.device,
            dtype=torch.float32,
        ) * self.cfg.bin_width_degrees - 180.0 + 0.5 * self.cfg.bin_width_degrees
        radians = torch.deg2rad(centers)
        distribution_vector = torch.stack(
            (
                (bearing_distribution * torch.sin(radians)).sum(dim=-1),
                (bearing_distribution * torch.cos(radians)).sum(dim=-1),
            ),
            dim=-1,
        )
        pooled_pair = fused.mean(dim=-1)
        regression_vector = F.normalize(self.regression_head(pooled_pair).float(), dim=-1, eps=1e-6)
        combined_vector = F.normalize(distribution_vector + regression_vector, dim=-1, eps=1e-6)
        bearing_angle = wrap_degrees_tensor(
            torch.rad2deg(torch.atan2(combined_vector[:, 0], combined_vector[:, 1]))
        )

        top2 = torch.topk(bearing_distribution, k=2, dim=-1).values
        entropy = -(
            bearing_distribution * bearing_distribution.clamp_min(1e-8).log()
        ).sum(dim=-1) / math.log(float(self.cfg.bearing_bins))
        peak_margin = top2[:, 0] - top2[:, 1]
        vector_norm = distribution_vector.norm(dim=-1)
        reverse_pooled = target_context.mean(dim=1)
        symmetry = F.cosine_similarity(pooled_pair.float(), reverse_pooled.float(), dim=-1)
        summary = torch.cat(
            (
                pooled_pair,
                (1.0 - entropy).unsqueeze(-1),
                peak_margin.unsqueeze(-1),
                vector_norm.unsqueeze(-1),
                symmetry.unsqueeze(-1),
            ),
            dim=-1,
        )
        confidence_logit = self.confidence_head(summary).squeeze(-1)
        evidence_gate = (1.0 - entropy).clamp(0.0, 1.0) * torch.sigmoid(12.0 * peak_margin)
        raw_learned_confidence = torch.sigmoid(confidence_logit)
        confidence = (
            raw_learned_confidence * evidence_gate
            if self._confidence_output_mode == "gated_evidence"
            else raw_learned_confidence
        )
        valid_logit = self.valid_head(pooled_pair).squeeze(-1)
        return {
            "bearing_logits": bearing_logits,
            "bearing_distribution": bearing_distribution,
            "bearing_angle_degrees": bearing_angle,
            "bearing_confidence": confidence,
            "bearing_confidence_logit": confidence_logit,
            "bearing_raw_learned_confidence": raw_learned_confidence,
            "bearing_evidence_gate": evidence_gate,
            "bearing_valid_probability": torch.sigmoid(valid_logit),
            "bearing_valid_logit": valid_logit,
            "distribution_vector_sin_cos": distribution_vector,
            "regression_vector_sin_cos": regression_vector,
            "sector_logits_32": sector_logits,
            "normalized_entropy": entropy,
            "peak_margin": peak_margin,
            "pair_embedding": self.pair_embedding(summary),
            "source_sector_features": fused.transpose(1, 2),
            "local_correlation_summary": local_correlation_summary,
        }


def r35_bearing_losses(
    output: Dict[str, torch.Tensor],
    target_degrees: torch.Tensor,
    target_valid: torch.Tensor,
    cfg: R35BearingConfig,
    *,
    weights: R35BearingLossWeights | None = None,
    rolled_source_output: Dict[str, torch.Tensor] | None = None,
    source_roll_degrees: torch.Tensor | None = None,
    reverse_output: Dict[str, torch.Tensor] | None = None,
    source_yaw_degrees: torch.Tensor | None = None,
    target_yaw_degrees: torch.Tensor | None = None,
) -> Dict[str, torch.Tensor]:
    loss_weights = weights or R35BearingLossWeights()
    valid = target_valid.bool()
    valid_float = valid.float()
    valid_count = valid_float.sum().clamp_min(1.0)
    labels = wrapped_soft_labels(target_degrees, cfg)
    per_sample_ce = -(
        labels * F.log_softmax(output["bearing_logits"].float(), dim=-1)
    ).sum(dim=-1)
    circular_soft_label = (per_sample_ce * valid_float).sum() / valid_count

    target_radians = torch.deg2rad(target_degrees.float())
    target_vector = torch.stack((torch.sin(target_radians), torch.cos(target_radians)), dim=-1)
    vector_error = 1.0 - (
        output["regression_vector_sin_cos"].float() * target_vector
    ).sum(dim=-1)
    sin_cos_regression = (vector_error * valid_float).sum() / valid_count
    angular_error = circular_error_degrees_tensor(
        output["bearing_angle_degrees"], target_degrees
    )
    geodesic_per_sample = 1.0 - torch.cos(torch.deg2rad(angular_error))
    geodesic = (geodesic_per_sample * valid_float).sum() / valid_count

    rotation_equivariance = output["bearing_logits"].sum() * 0.0
    if rolled_source_output is not None:
        if source_roll_degrees is None:
            raise ValueError("source_roll_degrees is required with rolled_source_output")
        expected = circular_shift_distribution(
            output["bearing_distribution"].detach(),
            source_roll_degrees.float() / cfg.bin_width_degrees,
        )
        rolled_log = rolled_source_output["bearing_distribution"].float().clamp_min(1e-8).log()
        per_sample_equivariance = F.kl_div(rolled_log, expected, reduction="none").sum(dim=-1)
        rotation_equivariance = (per_sample_equivariance * valid_float).sum() / valid_count

    reciprocal_consistency = output["bearing_logits"].sum() * 0.0
    if reverse_output is not None:
        if source_yaw_degrees is None or target_yaw_degrees is None:
            raise ValueError("source and target yaw are required with reverse_output")
        expected_reverse = wrap_degrees_tensor(
            output["bearing_angle_degrees"].detach()
            + 180.0
            + target_yaw_degrees.float()
            - source_yaw_degrees.float()
        )
        reciprocal_error = circular_error_degrees_tensor(
            reverse_output["bearing_angle_degrees"], expected_reverse
        )
        reciprocal_per_sample = 1.0 - torch.cos(torch.deg2rad(reciprocal_error))
        reciprocal_consistency = (reciprocal_per_sample * valid_float).sum() / valid_count

    confidence_target = (valid & (angular_error.detach() <= cfg.confidence_error_degrees)).float()
    confidence_calibration = F.binary_cross_entropy_with_logits(
        output["bearing_confidence_logit"].float(),
        confidence_target,
    )
    valid_classification = F.binary_cross_entropy_with_logits(
        output["bearing_valid_logit"].float(),
        target_valid.float(),
    )
    total = (
        loss_weights.circular_soft_label * circular_soft_label
        + loss_weights.sin_cos_regression * sin_cos_regression
        + loss_weights.geodesic * geodesic
        + loss_weights.rotation_equivariance * rotation_equivariance
        + loss_weights.reciprocal_consistency * reciprocal_consistency
        + loss_weights.confidence_calibration * confidence_calibration
        + loss_weights.valid_classification * valid_classification
    )
    return {
        "loss": total,
        "circular_soft_label": circular_soft_label,
        "sin_cos_regression": sin_cos_regression,
        "circular_geodesic": geodesic,
        "rotation_equivariance": rotation_equivariance,
        "reciprocal_pair_consistency": reciprocal_consistency,
        "confidence_calibration": confidence_calibration,
        "valid_classification": valid_classification,
        "valid_bearing_mae_degrees": (angular_error * valid_float).sum() / valid_count,
        "valid_accuracy_le_15deg": (
            (angular_error <= 15.0).float() * valid_float
        ).sum() / valid_count,
        "valid_catastrophic_gt_45deg": (
            (angular_error > 45.0).float() * valid_float
        ).sum() / valid_count,
    }
