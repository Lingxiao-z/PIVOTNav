from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .r35_bearing_head import (
    circular_error_degrees_tensor,
    wrap_degrees_tensor,
)
from .r36_bearing_structural_revision4 import (
    R36BearingStructuralRevision4Config,
    R36BearingStructuralRevision4Head,
    r36_bearing_structural_revision4_losses,
)


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
