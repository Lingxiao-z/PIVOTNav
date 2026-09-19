"""Candidate-frontier bearing heads and final refinement chain."""
from __future__ import annotations
from .orientation import CircularConv1d, CircularTokenBlock


# ---------------------------------------------------------------------------
# Base translation-bearing head
# ---------------------------------------------------------------------------

import math
from dataclasses import asdict, dataclass
from typing import Any, Dict

import torch
from torch import nn
from torch.nn import functional as F



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


# ---------------------------------------------------------------------------
# Bearing objective and calibrated residual
# ---------------------------------------------------------------------------

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F



@dataclass(frozen=True)
class R36BearingRevision4Config:
    residual_limit_degrees: float = 2.5
    soft_label_sigma_bins: float = 0.5
    confidence_tolerance_degrees: float = 5.0
    confidence_rank_margin: float = 0.10


@dataclass(frozen=True)
class R36BearingRevision4LossWeights:
    circular_soft_label: float = 1.25
    true_bin_residual: float = 1.00
    sin_cos_regression: float = 0.50
    circular_geodesic: float = 0.75
    direct_accuracy_5deg: float = 1.00
    close_range_accuracy_5deg: float = 0.75
    rotation_equivariance: float = 0.25
    reciprocal_consistency: float = 0.25
    confidence_5deg_calibration: float = 0.35
    confidence_risk_ranking: float = 0.20
    high_confidence_catastrophic: float = 0.25
    valid_classification: float = 0.25


def wrapped_bearing_soft_labels(
    target_degrees: torch.Tensor,
    cfg: R35BearingConfig,
    sigma_bins: float,
) -> torch.Tensor:
    target_position = torch.remainder(target_degrees.float() + 180.0, 360.0) / cfg.bin_width_degrees
    indices = torch.arange(cfg.bearing_bins, device=target_degrees.device, dtype=torch.float32).view(1, -1)
    distance = (indices - target_position.view(-1, 1)).abs()
    distance = torch.minimum(distance, cfg.bearing_bins - distance)
    labels = torch.exp(-0.5 * (distance / sigma_bins) ** 2)
    return labels / labels.sum(dim=-1, keepdim=True).clamp_min(1e-8)


class R36BearingRevision4Head(R35RelativeTranslationBearingHead):
    """R35 spatial matcher with decoupled 5-degree bin and bounded local residual."""

    def __init__(
        self,
        cfg: R35BearingConfig | None = None,
        revision_cfg: R36BearingRevision4Config | None = None,
    ) -> None:
        super().__init__(cfg)
        self.revision_cfg = revision_cfg or R36BearingRevision4Config()
        self.local_residual_decoder = nn.Sequential(
            CircularConv1d(self.cfg.model_dim, 64, 3, bias=False),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            CircularConv1d(64, 1, 1),
        )
        confidence_features = self.cfg.model_dim + 7
        self.confidence_5deg_head = nn.Sequential(
            nn.LayerNorm(confidence_features),
            nn.Linear(confidence_features, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    @property
    def architecture_record(self) -> dict[str, Any]:
        record = dict(super().architecture_record)
        record.update(
            {
                "schema_version": "r36_bearing_revision4_local_residual_v2",
                "revision_config": asdict(self.revision_cfg),
                "angle_decoder": "argmax 5-degree bin center plus bounded per-bin local residual",
                "angle_decoder_gradient": "straight-through hard bin preserves formal forward semantics and differentiable bin selection",
                "legacy_global_regression_decoder_used": False,
                "confidence_target": "circular error <= 5 degrees",
                "legacy_15deg_confidence_head_used": False,
            }
        )
        return record

    def freeze_legacy_decoders(self) -> None:
        for module in (self.regression_head, self.confidence_head):
            for parameter in module.parameters():
                parameter.requires_grad_(False)

    def forward(self, source_tokens: torch.Tensor, target_tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        output = super().forward(source_tokens, target_tokens)
        distribution = output["bearing_distribution"].float()
        sector_features = output["source_sector_features"].transpose(1, 2).contiguous()
        bin_features = circular_interpolate_1d(sector_features, self.cfg.bearing_bins)
        residual_by_bin = torch.tanh(self.local_residual_decoder(bin_features).squeeze(1).float())
        residual_by_bin = residual_by_bin * self.revision_cfg.residual_limit_degrees

        selected_bin = distribution.argmax(dim=-1)
        hard_bin = F.one_hot(selected_bin, self.cfg.bearing_bins).float()
        straight_through_bin = hard_bin + distribution - distribution.detach()
        bin_centers = (
            torch.arange(self.cfg.bearing_bins, device=distribution.device, dtype=torch.float32)
            * self.cfg.bin_width_degrees
            - 180.0
            + 0.5 * self.cfg.bin_width_degrees
        )
        candidate_angles = bin_centers.unsqueeze(0) + residual_by_bin
        candidate_radians = torch.deg2rad(candidate_angles)
        selected_vector = torch.stack(
            (
                (straight_through_bin * torch.sin(candidate_radians)).sum(dim=-1),
                (straight_through_bin * torch.cos(candidate_radians)).sum(dim=-1),
            ),
            dim=-1,
        )
        angle = wrap_degrees_tensor(
            torch.rad2deg(torch.atan2(selected_vector[:, 0], selected_vector[:, 1]))
        )
        selected_residual = residual_by_bin.gather(1, selected_bin.unsqueeze(1)).squeeze(1)

        pooled = output["source_sector_features"].mean(dim=1).float()
        distribution_vector_norm = output["distribution_vector_sin_cos"].float().norm(dim=-1)
        confidence_features = torch.cat(
            (
                pooled,
                (1.0 - output["normalized_entropy"].float()).unsqueeze(1),
                output["peak_margin"].float().unsqueeze(1),
                distribution_vector_norm.unsqueeze(1),
                output["local_correlation_summary"].float(),
            ),
            dim=1,
        )
        confidence_logit = self.confidence_5deg_head(confidence_features).squeeze(1)
        confidence = torch.sigmoid(confidence_logit)
        output.update(
            {
                "bearing_angle_degrees": angle,
                "bearing_confidence": confidence,
                "bearing_confidence_logit": confidence_logit,
                "bearing_raw_learned_confidence": confidence,
                "bearing_local_residual_by_bin_degrees": residual_by_bin,
                "bearing_selected_bin": selected_bin,
                "bearing_selected_residual_degrees": selected_residual,
            }
        )
        return output


def initialize_revision4_from_r36_checkpoint(
    head: R36BearingRevision4Head,
    checkpoint_payload: dict[str, Any],
) -> dict[str, Any]:
    if checkpoint_payload.get("schema_version") != "r36_bearing_checkpoint_v1":
        raise RuntimeError("Bearing Revision 4 requires an R36 Bearing checkpoint")
    incompatible = head.load_state_dict(checkpoint_payload["bearing_head"], strict=False)
    expected_prefixes = ("local_residual_decoder.", "confidence_5deg_head.")
    unexpected_missing = [name for name in incompatible.missing_keys if not name.startswith(expected_prefixes)]
    if incompatible.unexpected_keys or unexpected_missing:
        raise RuntimeError(
            f"Revision 4 initialization mismatch: missing={unexpected_missing} "
            f"unexpected={incompatible.unexpected_keys}"
        )
    return {
        "source_schema": checkpoint_payload["schema_version"],
        "source_global_step": int(checkpoint_payload["global_step"]),
        "loaded_legacy_keys": len(checkpoint_payload["bearing_head"]),
        "new_parameter_keys": sorted(incompatible.missing_keys),
        "unexpected_keys": [],
    }


def r36_bearing_revision4_losses(
    output: dict[str, torch.Tensor],
    target_degrees: torch.Tensor,
    target_valid: torch.Tensor,
    distance_bucket: torch.Tensor,
    cfg: R35BearingConfig,
    revision_cfg: R36BearingRevision4Config,
    *,
    rolled_source_output: dict[str, torch.Tensor] | None = None,
    source_roll_degrees: torch.Tensor | None = None,
    reverse_output: dict[str, torch.Tensor] | None = None,
    source_yaw_degrees: torch.Tensor | None = None,
    target_yaw_degrees: torch.Tensor | None = None,
    weights: R36BearingRevision4LossWeights | None = None,
) -> dict[str, torch.Tensor]:
    loss_weights = weights or R36BearingRevision4LossWeights()
    valid = target_valid.bool()
    valid_float = valid.float()
    valid_count = valid_float.sum().clamp_min(1.0)

    labels = wrapped_bearing_soft_labels(target_degrees, cfg, revision_cfg.soft_label_sigma_bins)
    per_sample_ce = -(labels * F.log_softmax(output["bearing_logits"].float(), dim=-1)).sum(dim=-1)
    circular_soft_label = (per_sample_ce * valid_float).sum() / valid_count

    target_position = torch.remainder(target_degrees.float() + 180.0, 360.0) / cfg.bin_width_degrees
    target_bin = torch.floor(target_position).long().remainder(cfg.bearing_bins)
    target_center = target_bin.float() * cfg.bin_width_degrees - 180.0 + 0.5 * cfg.bin_width_degrees
    target_residual = wrap_degrees_tensor(target_degrees.float() - target_center).clamp(
        -revision_cfg.residual_limit_degrees, revision_cfg.residual_limit_degrees
    )
    predicted_true_residual = output["bearing_local_residual_by_bin_degrees"].gather(
        1, target_bin.unsqueeze(1)
    ).squeeze(1)
    residual_per_sample = F.smooth_l1_loss(
        predicted_true_residual / revision_cfg.residual_limit_degrees,
        target_residual / revision_cfg.residual_limit_degrees,
        reduction="none",
    )
    true_bin_residual = (residual_per_sample * valid_float).sum() / valid_count

    angular_error = circular_error_degrees_tensor(output["bearing_angle_degrees"], target_degrees)
    predicted_radians = torch.deg2rad(output["bearing_angle_degrees"].float())
    target_radians = torch.deg2rad(target_degrees.float())
    predicted_vector = torch.stack((torch.sin(predicted_radians), torch.cos(predicted_radians)), dim=-1)
    target_vector = torch.stack((torch.sin(target_radians), torch.cos(target_radians)), dim=-1)
    sin_cos_per_sample = 1.0 - (predicted_vector * target_vector).sum(dim=-1)
    sin_cos_regression = (sin_cos_per_sample * valid_float).sum() / valid_count
    geodesic_per_sample = 1.0 - torch.cos(torch.deg2rad(angular_error))
    circular_geodesic = (geodesic_per_sample * valid_float).sum() / valid_count
    miss = F.smooth_l1_loss(
        (angular_error[valid] / 5.0).clamp_min(1.0),
        torch.ones_like(angular_error[valid]),
    ) if bool(valid.any()) else angular_error.sum() * 0.0
    close = valid & (distance_bucket == 0)
    close_miss = F.smooth_l1_loss(
        (angular_error[close] / 5.0).clamp_min(1.0),
        torch.ones_like(angular_error[close]),
    ) if bool(close.any()) else angular_error.sum() * 0.0

    rotation_equivariance = output["bearing_logits"].sum() * 0.0
    if rolled_source_output is not None:
        if source_roll_degrees is None:
            raise ValueError("source_roll_degrees is required with rolled_source_output")
        expected = circular_shift_distribution(
            output["bearing_distribution"].detach(),
            source_roll_degrees.float() / cfg.bin_width_degrees,
        )
        rolled_log = rolled_source_output["bearing_distribution"].float().clamp_min(1e-8).log()
        per_sample = F.kl_div(rolled_log, expected, reduction="none").sum(dim=-1)
        rotation_equivariance = (per_sample * valid_float).sum() / valid_count

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

    confidence_target = (valid & (angular_error.detach() <= revision_cfg.confidence_tolerance_degrees)).float()
    confidence_calibration = F.binary_cross_entropy_with_logits(
        output["bearing_confidence_logit"].float(), confidence_target
    )
    confidence = output["bearing_confidence"].float()
    paired_confidence = confidence.roll(1)
    paired_target = confidence_target.roll(1)
    ranking_mask = valid & valid.roll(1) & (confidence_target != paired_target)
    correct_confidence = torch.where(confidence_target > paired_target, confidence, paired_confidence)
    wrong_confidence = torch.where(confidence_target > paired_target, paired_confidence, confidence)
    confidence_ranking = F.relu(
        revision_cfg.confidence_rank_margin + wrong_confidence[ranking_mask] - correct_confidence[ranking_mask]
    ).mean() if bool(ranking_mask.any()) else confidence.sum() * 0.0

    catastrophic = F.relu(confidence[valid] - 0.5) * F.relu((angular_error[valid] - 45.0) / 45.0)
    catastrophic_loss = catastrophic.mean() if catastrophic.numel() else angular_error.sum() * 0.0
    valid_classification = F.binary_cross_entropy_with_logits(
        output["bearing_valid_logit"].float(), target_valid.float()
    )
    total = (
        loss_weights.circular_soft_label * circular_soft_label
        + loss_weights.true_bin_residual * true_bin_residual
        + loss_weights.sin_cos_regression * sin_cos_regression
        + loss_weights.circular_geodesic * circular_geodesic
        + loss_weights.direct_accuracy_5deg * miss
        + loss_weights.close_range_accuracy_5deg * close_miss
        + loss_weights.rotation_equivariance * rotation_equivariance
        + loss_weights.reciprocal_consistency * reciprocal_consistency
        + loss_weights.confidence_5deg_calibration * confidence_calibration
        + loss_weights.confidence_risk_ranking * confidence_ranking
        + loss_weights.high_confidence_catastrophic * catastrophic_loss
        + loss_weights.valid_classification * valid_classification
    )
    return {
        "loss": total,
        "circular_soft_label": circular_soft_label,
        "true_bin_residual": true_bin_residual,
        "sin_cos_regression": sin_cos_regression,
        "circular_geodesic": circular_geodesic,
        "direct_accuracy_5deg": miss,
        "close_range_accuracy_5deg": close_miss,
        "rotation_equivariance": rotation_equivariance,
        "reciprocal_pair_consistency": reciprocal_consistency,
        "confidence_5deg_calibration": confidence_calibration,
        "confidence_risk_ranking": confidence_ranking,
        "high_confidence_catastrophic": catastrophic_loss,
        "valid_classification": valid_classification,
        "valid_bearing_mae_degrees": (angular_error * valid_float).sum() / valid_count,
        "valid_accuracy_le_5deg": ((angular_error <= 5.0).float() * valid_float).sum() / valid_count,
    }


# ---------------------------------------------------------------------------
# Circular bearing decoder
# ---------------------------------------------------------------------------

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch.nn import functional as F



@dataclass(frozen=True)
class R36BearingRevision5Config(R36BearingRevision4Config):
    soft_decode_temperature: float = 1.0


class R36BearingRevision5Head(R36BearingRevision4Head):
    """Revision 4 head with a differentiable circular expectation decoder."""

    def __init__(self, cfg=None, revision_cfg: R36BearingRevision5Config | None = None) -> None:
        super().__init__(cfg, revision_cfg or R36BearingRevision5Config())
        self.revision_cfg = revision_cfg or R36BearingRevision5Config()

    @property
    def architecture_record(self) -> dict[str, Any]:
        record = dict(super().architecture_record)
        record.update(
            {
                "schema_version": "r36_bearing_revision5_soft_circular_expectation_v1",
                "angle_decoder": "temperature-scaled circular expectation over per-bin local residual candidates",
                "angle_decoder_gradient": "fully differentiable through all bearing bins and local residuals",
                "soft_decode_temperature": self.revision_cfg.soft_decode_temperature,
                "revision_number": 5,
            }
        )
        return record

    def forward(self, source_tokens: torch.Tensor, target_tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        output = super().forward(source_tokens, target_tokens)
        distribution = output["bearing_distribution"].float()
        residual = output["bearing_local_residual_by_bin_degrees"].float()
        bins = distribution.shape[-1]
        centers = (
            torch.arange(bins, device=distribution.device, dtype=torch.float32)
            * self.cfg.bin_width_degrees
            - 180.0
            + 0.5 * self.cfg.bin_width_degrees
        )
        candidates = centers.unsqueeze(0) + residual
        weights = F.softmax(
            output["bearing_logits"].float() / self.revision_cfg.soft_decode_temperature,
            dim=-1,
        )
        radians = torch.deg2rad(candidates)
        vector = torch.stack(
            (
                (weights * torch.sin(radians)).sum(dim=-1),
                (weights * torch.cos(radians)).sum(dim=-1),
            ),
            dim=-1,
        )
        angle = wrap_degrees_tensor(torch.rad2deg(torch.atan2(vector[:, 0], vector[:, 1])))
        output.update(
            {
                "bearing_angle_degrees": angle,
                "bearing_soft_decode_weights": weights,
                "bearing_soft_decode_candidates_degrees": candidates,
                "bearing_soft_decode_vector_sin_cos": vector,
            }
        )
        return output


def initialize_revision5_from_revision4_checkpoint(
    head: R36BearingRevision5Head,
    checkpoint_payload: dict[str, Any],
) -> dict[str, Any]:
    schema = str(checkpoint_payload.get("schema_version", ""))
    if schema not in {"r36_bearing_revision4_checkpoint_v2", "r36_bearing_revision4_checkpoint_v3"}:
        raise RuntimeError(f"Revision 5 requires a Revision 4 checkpoint, got {schema}")
    incompatible = head.load_state_dict(checkpoint_payload["bearing_head"], strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"Revision 5 initialization mismatch: {incompatible}")
    return {
        "source_schema": schema,
        "source_global_step": int(checkpoint_payload["global_step"]),
        "loaded_parameter_keys": len(checkpoint_payload["bearing_head"]),
        "soft_decoder_parameters_initialized_from_revision4": True,
    }


# ---------------------------------------------------------------------------
# Structural bearing refinement
# ---------------------------------------------------------------------------

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F



@dataclass(frozen=True)
class R36BearingStructuralRevision4Config(R36BearingRevision5Config):
    fine_correction_limit_degrees: float = 5.0
    fine_correction_hidden_dim: int = 64
    fine_correction_loss_weight: float = 1.0
    boundary_focus_weight: float = 1.5


class R36BearingStructuralRevision4Head(R36BearingRevision5Head):
    """R5 decoder plus a pair-conditioned bounded sub-bin correction."""

    def __init__(
        self,
        cfg=None,
        revision_cfg: R36BearingStructuralRevision4Config | None = None,
    ) -> None:
        revision_cfg = revision_cfg or R36BearingStructuralRevision4Config()
        super().__init__(cfg, revision_cfg)
        self.revision_cfg = revision_cfg
        correction_features = self.cfg.model_dim + 7
        self.fine_correction_head = nn.Sequential(
            nn.LayerNorm(correction_features),
            nn.Linear(correction_features, revision_cfg.fine_correction_hidden_dim),
            nn.GELU(),
            nn.Linear(revision_cfg.fine_correction_hidden_dim, 1),
        )
        # The new revision starts as an exact R5 decoder and learns only an
        # evidence-backed correction from formal Train labels.
        nn.init.zeros_(self.fine_correction_head[-1].weight)
        nn.init.zeros_(self.fine_correction_head[-1].bias)

    @property
    def architecture_record(self) -> dict[str, Any]:
        record = dict(super().architecture_record)
        record.update(
            {
                "schema_version": "r36_bearing_structural_revision4_fine_correction_v1",
                "structural_revision_index": 4,
                "revision_config": asdict(self.revision_cfg),
                "angle_decoder": (
                    "R5 differentiable circular expectation plus a pair-conditioned "
                    "bounded continuous correction"
                ),
                "fine_correction_inputs": (
                    "pair-fused spatial sector mean, distribution entropy, peak margin, "
                    "circular vector norm, and local correlation summary"
                ),
                "initialization_preserves_r5_angle_exactly": True,
                "global_descriptor_only": False,
                "not_relative_yaw": True,
            }
        )
        return record

    def forward(
        self,
        source_tokens: torch.Tensor,
        target_tokens: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        output = super().forward(source_tokens, target_tokens)
        base_angle = output["bearing_angle_degrees"].float()
        pooled = output["source_sector_features"].mean(dim=1).float()
        vector_norm = output["distribution_vector_sin_cos"].float().norm(dim=-1)
        correction_features = torch.cat(
            (
                pooled,
                (1.0 - output["normalized_entropy"].float()).unsqueeze(1),
                output["peak_margin"].float().unsqueeze(1),
                vector_norm.unsqueeze(1),
                output["local_correlation_summary"].float(),
            ),
            dim=1,
        )
        correction = torch.tanh(
            self.fine_correction_head(correction_features).squeeze(1).float()
        ) * self.revision_cfg.fine_correction_limit_degrees
        output.update(
            {
                "bearing_base_angle_degrees": base_angle,
                "bearing_fine_correction_degrees": correction,
                "bearing_angle_degrees": wrap_degrees_tensor(base_angle + correction),
                "bearing_fine_correction_features": correction_features,
            }
        )
        return output


def initialize_structural_revision4_from_revision5_checkpoint(
    head: R36BearingStructuralRevision4Head,
    checkpoint_payload: dict[str, Any],
) -> dict[str, Any]:
    schema = str(checkpoint_payload.get("schema_version", ""))
    if schema != "r36_bearing_revision5_checkpoint_v1":
        raise RuntimeError(
            "Structural Revision 4 requires a formal Revision 5 checkpoint, "
            f"got {schema}"
        )
    incompatible = head.load_state_dict(checkpoint_payload["bearing_head"], strict=False)
    unexpected_missing = [
        name
        for name in incompatible.missing_keys
        if not name.startswith("fine_correction_head.")
    ]
    if incompatible.unexpected_keys or unexpected_missing:
        raise RuntimeError(
            "Structural Revision 4 initialization mismatch: "
            f"missing={unexpected_missing} unexpected={incompatible.unexpected_keys}"
        )
    return {
        "source_schema": schema,
        "source_global_step": int(checkpoint_payload["global_step"]),
        "loaded_parameter_keys": len(checkpoint_payload["bearing_head"]),
        "new_parameter_keys": sorted(incompatible.missing_keys),
        "r5_angle_preserved_at_initialization": True,
    }


def r36_bearing_structural_revision4_losses(
    output: dict[str, torch.Tensor],
    target_degrees: torch.Tensor,
    target_valid: torch.Tensor,
    distance_bucket: torch.Tensor,
    cfg,
    revision_cfg: R36BearingStructuralRevision4Config,
    **kwargs,
) -> dict[str, torch.Tensor]:
    base = r36_bearing_revision4_losses(
        output,
        target_degrees,
        target_valid,
        distance_bucket,
        cfg,
        revision_cfg,
        **kwargs,
    )
    valid = target_valid.bool()
    base_angle = output["bearing_base_angle_degrees"].float()
    target_correction = wrap_degrees_tensor(target_degrees.float() - base_angle.detach())
    target_correction = target_correction.clamp(
        -revision_cfg.fine_correction_limit_degrees,
        revision_cfg.fine_correction_limit_degrees,
    )
    predicted_correction = output["bearing_fine_correction_degrees"].float()
    base_error = circular_error_degrees_tensor(base_angle, target_degrees).detach()
    boundary_focus = ((base_error > 5.0) & (base_error <= 12.0)).float()
    sample_weight = 1.0 + revision_cfg.boundary_focus_weight * boundary_focus
    correction_per_sample = F.smooth_l1_loss(
        predicted_correction / revision_cfg.fine_correction_limit_degrees,
        target_correction / revision_cfg.fine_correction_limit_degrees,
        reduction="none",
    )
    valid_weight = sample_weight * valid.float()
    fine_correction = (correction_per_sample * valid_weight).sum() / valid_weight.sum().clamp_min(1.0)
    total = base["loss"] + revision_cfg.fine_correction_loss_weight * fine_correction
    return {
        **base,
        "loss": total,
        "fine_correction_regression": fine_correction,
        "fine_correction_target_abs_mean_degrees": (
            target_correction.abs() * valid.float()
        ).sum()
        / valid.float().sum().clamp_min(1.0),
        "boundary_focus_fraction": (
            boundary_focus * valid.float()
        ).sum()
        / valid.float().sum().clamp_min(1.0),
    }


# ---------------------------------------------------------------------------
# Final bearing refinement
# ---------------------------------------------------------------------------

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F



@dataclass(frozen=True)
class R36BearingStructuralRevision5Config(R36BearingStructuralRevision4Config):
    phase_offsets_degrees: tuple[float, ...] = (
        -4.0,
        -3.0,
        -2.0,
        -1.0,
        0.0,
        1.0,
        2.0,
        3.0,
        4.0,
    )
    local_feature_offsets_degrees: tuple[float, ...] = (
        -22.5,
        -11.25,
        0.0,
        11.25,
        22.5,
    )
    local_distribution_offsets_degrees: tuple[float, ...] = (
        -10.0,
        -5.0,
        0.0,
        5.0,
        10.0,
    )
    phase_hidden_dim: int = 128
    phase_residual_limit_degrees: float = 1.0
    phase_soft_label_sigma_steps: float = 0.75
    phase_classification_loss_weight: float = 1.25
    phase_residual_loss_weight: float = 0.75
    boundary_focus_weight: float = 2.0
    close_range_focus_weight: float = 1.0


def circular_sample_sequence(
    sequence: torch.Tensor,
    angle_degrees: torch.Tensor,
    offsets_degrees: tuple[float, ...],
) -> torch.Tensor:
    """Linearly sample [B,L,C] or [B,L] at circular angular positions."""
    squeeze = sequence.ndim == 2
    if squeeze:
        sequence = sequence.unsqueeze(-1)
    if sequence.ndim != 3:
        raise ValueError("sequence must be [B,L,C] or [B,L]")
    batch, length, channels = sequence.shape
    offsets = torch.as_tensor(
        offsets_degrees,
        device=sequence.device,
        dtype=torch.float32,
    )
    positions = torch.remainder(
        (angle_degrees.float().unsqueeze(1) + offsets.unsqueeze(0) + 180.0)
        * (float(length) / 360.0),
        float(length),
    )
    lower = torch.floor(positions).long()
    upper = (lower + 1).remainder(length)
    fraction = (positions - lower.float()).to(sequence.dtype).unsqueeze(-1)
    lower_values = sequence.gather(
        1, lower.unsqueeze(-1).expand(batch, lower.shape[1], channels)
    )
    upper_values = sequence.gather(
        1, upper.unsqueeze(-1).expand(batch, upper.shape[1], channels)
    )
    sampled = lower_values * (1.0 - fraction) + upper_values * fraction
    return sampled.squeeze(-1) if squeeze else sampled


class R36BearingStructuralRevision5Head(R36BearingStructuralRevision4Head):
    """Revision 4 plus prediction-aligned local phase refinement."""

    def __init__(
        self,
        cfg=None,
        revision_cfg: R36BearingStructuralRevision5Config | None = None,
    ) -> None:
        revision_cfg = revision_cfg or R36BearingStructuralRevision5Config()
        super().__init__(cfg, revision_cfg)
        self.revision_cfg = revision_cfg

        # Preserve the validated Revision 4 model and learn only the new local
        # phase branch. This prevents late-stage drift of the coarse matcher.
        for parameter in self.parameters():
            parameter.requires_grad_(False)

        local_feature_count = len(revision_cfg.local_feature_offsets_degrees)
        local_distribution_count = len(
            revision_cfg.local_distribution_offsets_degrees
        )
        phase_features = (
            local_feature_count * self.cfg.model_dim
            + self.cfg.model_dim
            + 2 * local_distribution_count
            + 8
        )
        self.phase_feature_norm = nn.LayerNorm(phase_features)
        self.phase_trunk = nn.Sequential(
            nn.Linear(phase_features, revision_cfg.phase_hidden_dim),
            nn.GELU(),
            nn.Linear(revision_cfg.phase_hidden_dim, revision_cfg.phase_hidden_dim),
            nn.GELU(),
        )
        self.phase_offset_logits = nn.Linear(
            revision_cfg.phase_hidden_dim,
            len(revision_cfg.phase_offsets_degrees),
        )
        self.phase_continuous_residual = nn.Linear(
            revision_cfg.phase_hidden_dim,
            1,
        )
        nn.init.zeros_(self.phase_offset_logits.weight)
        nn.init.zeros_(self.phase_offset_logits.bias)
        nn.init.zeros_(self.phase_continuous_residual.weight)
        nn.init.zeros_(self.phase_continuous_residual.bias)

    @property
    def architecture_record(self) -> dict[str, Any]:
        record = dict(super().architecture_record)
        record.update(
            {
                "schema_version": (
                    "r36_bearing_structural_revision5_local_phase_refinement_v1"
                ),
                "structural_revision_index": 5,
                "revision_config": asdict(self.revision_cfg),
                "angle_decoder": (
                    "validated Structural Revision 4 angle plus a prediction-aligned "
                    "local phase distribution and bounded continuous residual"
                ),
                "phase_inputs": (
                    "circularly sampled pair-fused sector features around the predicted "
                    "direction, local bearing-distribution profile, local per-bin residual "
                    "profile, global pair context, and uncertainty summaries"
                ),
                "inherited_revision4_frozen": True,
                "initialization_preserves_revision4_angle_exactly": True,
                "global_descriptor_only": False,
                "not_relative_yaw": True,
            }
        )
        return record

    def forward(
        self,
        source_tokens: torch.Tensor,
        target_tokens: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        output = super().forward(source_tokens, target_tokens)
        base_angle = output["bearing_angle_degrees"].float()
        sector_features = output["source_sector_features"].float()
        local_sector = circular_sample_sequence(
            sector_features,
            base_angle,
            self.revision_cfg.local_feature_offsets_degrees,
        ).flatten(1)
        local_distribution = circular_sample_sequence(
            output["bearing_distribution"].float(),
            base_angle,
            self.revision_cfg.local_distribution_offsets_degrees,
        )
        local_residual = circular_sample_sequence(
            output["bearing_local_residual_by_bin_degrees"].float(),
            base_angle,
            self.revision_cfg.local_distribution_offsets_degrees,
        ) / self.revision_cfg.residual_limit_degrees
        pooled = sector_features.mean(dim=1)
        vector_norm = output["distribution_vector_sin_cos"].float().norm(dim=-1)
        scalar_summary = torch.cat(
            (
                (1.0 - output["normalized_entropy"].float()).unsqueeze(1),
                output["peak_margin"].float().unsqueeze(1),
                vector_norm.unsqueeze(1),
                output["local_correlation_summary"].float(),
                (
                    output["bearing_fine_correction_degrees"].float()
                    / self.revision_cfg.fine_correction_limit_degrees
                ).unsqueeze(1),
            ),
            dim=1,
        )
        phase_features = torch.cat(
            (
                local_sector,
                pooled,
                local_distribution,
                local_residual,
                scalar_summary,
            ),
            dim=1,
        )
        hidden = self.phase_trunk(self.phase_feature_norm(phase_features))
        logits = self.phase_offset_logits(hidden).float()
        probability = F.softmax(logits, dim=-1)
        offsets = torch.as_tensor(
            self.revision_cfg.phase_offsets_degrees,
            device=probability.device,
            dtype=torch.float32,
        )
        discrete_phase = (probability * offsets.unsqueeze(0)).sum(dim=-1)
        continuous_phase = torch.tanh(
            self.phase_continuous_residual(hidden).squeeze(1).float()
        ) * self.revision_cfg.phase_residual_limit_degrees
        phase_correction = discrete_phase + continuous_phase
        output.update(
            {
                "bearing_phase_base_angle_degrees": base_angle,
                "bearing_phase_features": phase_features,
                "bearing_phase_offset_logits": logits,
                "bearing_phase_offset_probability": probability,
                "bearing_phase_discrete_degrees": discrete_phase,
                "bearing_phase_continuous_degrees": continuous_phase,
                "bearing_phase_correction_degrees": phase_correction,
                "bearing_angle_degrees": wrap_degrees_tensor(
                    base_angle + phase_correction
                ),
            }
        )
        return output


def initialize_structural_revision5_from_structural_revision4_checkpoint(
    head: R36BearingStructuralRevision5Head,
    checkpoint_payload: dict[str, Any],
) -> dict[str, Any]:
    schema = str(checkpoint_payload.get("schema_version", ""))
    if schema != "r36_bearing_structural_revision4_checkpoint_v1":
        raise RuntimeError(
            "Structural Revision 5 requires a formal Structural Revision 4 "
            f"checkpoint, got {schema}"
        )
    incompatible = head.load_state_dict(checkpoint_payload["bearing_head"], strict=False)
    expected_prefixes = (
        "phase_feature_norm.",
        "phase_trunk.",
        "phase_offset_logits.",
        "phase_continuous_residual.",
    )
    unexpected_missing = [
        name
        for name in incompatible.missing_keys
        if not name.startswith(expected_prefixes)
    ]
    if incompatible.unexpected_keys or unexpected_missing:
        raise RuntimeError(
            "Structural Revision 5 initialization mismatch: "
            f"missing={unexpected_missing} unexpected={incompatible.unexpected_keys}"
        )
    return {
        "source_schema": schema,
        "source_global_step": int(checkpoint_payload["global_step"]),
        "loaded_parameter_keys": len(checkpoint_payload["bearing_head"]),
        "new_parameter_keys": sorted(incompatible.missing_keys),
        "revision4_angle_preserved_at_initialization": True,
    }


def r36_bearing_structural_revision5_losses(
    output: dict[str, torch.Tensor],
    target_degrees: torch.Tensor,
    target_valid: torch.Tensor,
    distance_bucket: torch.Tensor,
    cfg,
    revision_cfg: R36BearingStructuralRevision5Config,
    **kwargs,
) -> dict[str, torch.Tensor]:
    base = r36_bearing_structural_revision4_losses(
        output,
        target_degrees,
        target_valid,
        distance_bucket,
        cfg,
        revision_cfg,
        **kwargs,
    )
    valid = target_valid.bool()
    base_angle = output["bearing_phase_base_angle_degrees"].float()
    raw_target = wrap_degrees_tensor(target_degrees.float() - base_angle.detach())
    phase_limit = max(abs(value) for value in revision_cfg.phase_offsets_degrees)
    target_phase = raw_target.clamp(
        -phase_limit,
        phase_limit,
    )
    offsets = torch.as_tensor(
        revision_cfg.phase_offsets_degrees,
        device=target_phase.device,
        dtype=torch.float32,
    )
    offset_distance = target_phase.unsqueeze(1) - offsets.unsqueeze(0)
    labels = torch.exp(
        -0.5
        * (offset_distance / revision_cfg.phase_soft_label_sigma_steps).square()
    )
    labels = labels / labels.sum(dim=1, keepdim=True).clamp_min(1e-8)
    classification_per_sample = -(
        labels * F.log_softmax(output["bearing_phase_offset_logits"].float(), dim=1)
    ).sum(dim=1)
    nearest = offset_distance.abs().argmin(dim=1)
    nearest_offset = offsets.gather(0, nearest)
    residual_target = (target_phase - nearest_offset).clamp(
        -revision_cfg.phase_residual_limit_degrees,
        revision_cfg.phase_residual_limit_degrees,
    )
    residual_per_sample = F.smooth_l1_loss(
        output["bearing_phase_continuous_degrees"].float()
        / revision_cfg.phase_residual_limit_degrees,
        residual_target / revision_cfg.phase_residual_limit_degrees,
        reduction="none",
    )
    base_error = circular_error_degrees_tensor(base_angle, target_degrees).detach()
    boundary = ((base_error > 3.0) & (base_error <= 12.0)).float()
    close = (distance_bucket == 0).float()
    sample_weight = (
        1.0
        + revision_cfg.boundary_focus_weight * boundary
        + revision_cfg.close_range_focus_weight * close
    ) * valid.float()
    denominator = sample_weight.sum().clamp_min(1.0)
    phase_classification = (
        classification_per_sample * sample_weight
    ).sum() / denominator
    phase_residual = (residual_per_sample * sample_weight).sum() / denominator
    total = (
        base["loss"]
        + revision_cfg.phase_classification_loss_weight * phase_classification
        + revision_cfg.phase_residual_loss_weight * phase_residual
    )
    return {
        **base,
        "loss": total,
        "phase_offset_classification": phase_classification,
        "phase_continuous_residual": phase_residual,
        "phase_target_abs_mean_degrees": (
            target_phase.abs() * valid.float()
        ).sum()
        / valid.float().sum().clamp_min(1.0),
        "phase_boundary_focus_fraction": (
            boundary * valid.float()
        ).sum()
        / valid.float().sum().clamp_min(1.0),
        "phase_close_range_fraction": (
            close * valid.float()
        ).sum()
        / valid.float().sum().clamp_min(1.0),
    }
