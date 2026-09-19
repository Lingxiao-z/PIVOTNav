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
from .r36_bearing_revision4 import r36_bearing_revision4_losses
from .r36_bearing_revision5 import (
    R36BearingRevision5Config,
    R36BearingRevision5Head,
)


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
