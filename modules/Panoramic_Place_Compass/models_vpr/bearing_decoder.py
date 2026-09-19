from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch.nn import functional as F

from .bearing_head import wrap_degrees_tensor
from .bearing_objective import (
    R36BearingRevision4Config,
    R36BearingRevision4Head,
)


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
