from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class R36TemporalArrivalConfig:
    input_dim: int = 280
    visual_hidden_dim: int = 256
    temporal_hidden_dim: int = 128
    gru_layers: int = 1
    dropout: float = 0.10


class R36TemporalArrivalHead(nn.Module):
    """Arrival head over ordered frozen visual-pair features only."""

    def __init__(self, config: R36TemporalArrivalConfig | None = None) -> None:
        super().__init__()
        self.config = config or R36TemporalArrivalConfig()
        self.visual_encoder = nn.Sequential(
            nn.LayerNorm(self.config.input_dim),
            nn.Linear(self.config.input_dim, self.config.visual_hidden_dim),
            nn.GELU(), nn.Dropout(self.config.dropout),
            nn.Linear(self.config.visual_hidden_dim, self.config.visual_hidden_dim),
            nn.GELU(),
        )
        self.frame_logit = nn.Linear(self.config.visual_hidden_dim, 1)
        self.temporal_encoder = nn.GRU(
            self.config.visual_hidden_dim + 1,
            self.config.temporal_hidden_dim,
            num_layers=self.config.gru_layers,
            batch_first=True,
        )
        self.temporal_logit = nn.Linear(self.config.temporal_hidden_dim, 1)
        self.confidence_logit = nn.Linear(self.config.temporal_hidden_dim, 1)

    @property
    def architecture_record(self) -> dict[str, Any]:
        return {
            "schema_version": "r36_temporal_arrival_head_v2",
            "config": asdict(self.config),
            "shared_phase_a_b_features_frozen": True,
            "online_inputs": ["280维冻结视觉对特征", "帧顺序"],
            "forbidden_online_inputs": [
                "GT距离", "GT pose/yaw", "NavMesh", "Depth GT", "GPS/Compass",
                "collision", "success", "SPL",
            ],
            "temporal_evidence": "因果GRU累计独立位置观测；正式Stop仍要求至少两份独立视觉证据",
        }

    def initialize_visual_encoder_from_r1(self, state: dict[str, torch.Tensor]) -> None:
        self.visual_encoder.load_state_dict(
            {k.removeprefix("encoder."): v for k, v in state.items() if k.startswith("encoder.")},
            strict=True,
        )
        self.frame_logit.load_state_dict(
            {k.removeprefix("arrival_logit."): v for k, v in state.items() if k.startswith("arrival_logit.")},
            strict=True,
        )

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        if features.ndim != 3 or features.shape[-1] != self.config.input_dim:
            raise ValueError("features must be [B,T,280]")
        visual = self.visual_encoder(features.float())
        frame_logit = self.frame_logit(visual).squeeze(-1)
        frame_probability = torch.sigmoid(frame_logit)
        temporal, _ = self.temporal_encoder(
            torch.cat((visual, frame_probability.unsqueeze(-1)), dim=-1)
        )
        temporal_logit = self.temporal_logit(temporal).squeeze(-1)
        temporal_probability = torch.sigmoid(temporal_logit)
        confidence = torch.sigmoid(self.confidence_logit(temporal).squeeze(-1)) * (
            2.0 * (temporal_probability - 0.5).abs()
        )
        return {
            "frame_logit": frame_logit,
            "frame_probability": frame_probability,
            "temporal_logit": temporal_logit,
            "temporal_probability": temporal_probability,
            "confidence": confidence,
        }


def r36_temporal_arrival_losses(
    output: dict[str, torch.Tensor],
    *,
    target_arrived: torch.Tensor,
    offline_geodesic_distance_m: torch.Tensor,
    frame_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Use offline distance only for loss weighting, never as model input."""
    target = target_arrived.float()
    valid = frame_mask.bool()
    distance = offline_geodesic_distance_m.float()
    boundary = valid & (distance > 1.0) & (distance <= 1.25)
    positive = valid & target_arrived.bool()
    weight = torch.ones_like(target)
    weight = torch.where(boundary, torch.full_like(weight, 4.0), weight)
    weight = torch.where(valid & (distance > 1.25) & (distance <= 1.5), torch.full_like(weight, 2.5), weight)
    weight = torch.where(valid & (distance > 1.5) & (distance <= 2.0), torch.full_like(weight, 1.5), weight)
    frame = (F.binary_cross_entropy_with_logits(output["frame_logit"], target, reduction="none")[valid] * weight[valid]).mean()
    temporal = (F.binary_cross_entropy_with_logits(output["temporal_logit"], target, reduction="none")[valid] * weight[valid]).mean()
    false_stop = F.softplus(output["temporal_logit"][boundary] + 1.5).mean() if boundary.any() else temporal * 0.0
    preserve = F.softplus(1.5 - output["temporal_logit"][positive]).mean() if positive.any() else temporal * 0.0
    adjacent = valid[:, 1:] & valid[:, :-1] & target_arrived[:, 1:] & target_arrived[:, :-1]
    delta = output["temporal_probability"][:, :-1] - output["temporal_probability"][:, 1:]
    monotonic = F.relu(delta[adjacent]).mean() if adjacent.any() else temporal * 0.0
    correctness = 1.0 - (output["temporal_probability"] - target).abs()
    calibration = F.mse_loss(output["confidence"][valid], correctness[valid])
    total = frame + 1.5 * temporal + 2.0 * false_stop + preserve + 0.25 * monotonic + 0.1 * calibration
    return {"loss": total, "frame_classification": frame, "temporal_classification": temporal,
            "boundary_false_stop": false_stop, "positive_preserving": preserve,
            "entry_monotonicity": monotonic, "confidence_calibration": calibration}
