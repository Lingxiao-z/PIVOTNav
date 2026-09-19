from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .yaw_head import CircularConv1d
from .bearing_head import (
    R35BearingConfig,
    R35RelativeTranslationBearingHead,
    circular_error_degrees_tensor,
    circular_interpolate_1d,
    circular_shift_distribution,
    wrap_degrees_tensor,
)


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


def wrapped_soft_labels(
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

    labels = wrapped_soft_labels(target_degrees, cfg, revision_cfg.soft_label_sigma_bins)
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
