from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class R36ArrivalHeadConfig:
    input_dim: int = 280
    hidden_dim: int = 256
    dropout: float = 0.10
    boundary_logit_margin: float = -1.5
    positive_logit_margin: float = 1.5


@dataclass(frozen=True)
class R36ArrivalLossWeights:
    classification: float = 1.0
    boundary_false_stop: float = 1.5
    positive_preserving: float = 1.0
    pair_ranking: float = 0.75
    confidence_calibration: float = 0.10
    near_risk_auxiliary: float = 0.25


class R36ArrivalHead(nn.Module):
    """Independent 1m arrival head over frozen online visual pair features."""

    def __init__(self, config: R36ArrivalHeadConfig | None = None) -> None:
        super().__init__()
        self.config = config or R36ArrivalHeadConfig()
        self.encoder = nn.Sequential(
            nn.LayerNorm(self.config.input_dim),
            nn.Linear(self.config.input_dim, self.config.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.GELU(),
        )
        self.arrival_logit = nn.Linear(self.config.hidden_dim, 1)
        self.near_risk_logit = nn.Linear(self.config.hidden_dim, 1)
        self.confidence_logit = nn.Linear(self.config.hidden_dim, 1)

    @property
    def architecture_record(self) -> dict[str, Any]:
        return {
            "schema_version": "r36_arrival_head_v1",
            "config": asdict(self.config),
            "independent_from_goal_anchor_presence": True,
            "shared_phase_a_b_features_frozen": True,
            "online_gt_inputs": [],
            "temporal_stop_rule": "至少连续两份视觉证据；状态机在head外实现",
        }

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        token = self.encoder(features.float())
        logit = self.arrival_logit(token).squeeze(-1)
        near_logit = self.near_risk_logit(token).squeeze(-1)
        probability = torch.sigmoid(logit)
        confidence = torch.sigmoid(self.confidence_logit(token).squeeze(-1)) * (
            2.0 * (probability - 0.5).abs()
        )
        return {
            "arrival_logit": logit,
            "arrival_probability": probability,
            "near_risk_logit": near_logit,
            "near_risk_probability": torch.sigmoid(near_logit),
            "confidence": confidence,
        }


def r36_arrival_losses(
    output: dict[str, torch.Tensor],
    *,
    target_arrived: torch.Tensor,
    category_id: torch.Tensor,
    config: R36ArrivalHeadConfig,
    weights: R36ArrivalLossWeights | None = None,
) -> dict[str, torch.Tensor]:
    w = weights or R36ArrivalLossWeights()
    logits = output["arrival_logit"].float()
    target = target_arrived.float()
    category = category_id.long()
    # 0=positive, 1=1-1.25m, 2=1.25-1.5m, 3=1.5-2m, 4=wall, 5=strong negative.
    sample_weight = torch.ones_like(target)
    sample_weight = torch.where(category == 1, torch.full_like(sample_weight, 4.0), sample_weight)
    sample_weight = torch.where(category == 2, torch.full_like(sample_weight, 2.5), sample_weight)
    sample_weight = torch.where(category == 3, torch.full_like(sample_weight, 1.5), sample_weight)
    sample_weight = torch.where(category == 4, torch.full_like(sample_weight, 3.0), sample_weight)
    classification = (
        F.binary_cross_entropy_with_logits(logits, target, reduction="none") * sample_weight
    ).mean()
    boundary = (category == 1) | (category == 4)
    boundary_false_stop = (
        F.relu(logits[boundary] - config.boundary_logit_margin).mean()
        if boundary.any() else logits.sum() * 0.0
    )
    positive = target_arrived.bool()
    positive_preserving = (
        F.relu(config.positive_logit_margin - logits[positive]).mean()
        if positive.any() else logits.sum() * 0.0
    )
    hard_negative = (category == 1) | (category == 2) | (category == 4)
    pair_ranking = (
        F.relu(1.0 - logits[positive].mean() + logits[hard_negative].mean())
        if positive.any() and hard_negative.any() else logits.sum() * 0.0
    )
    confidence_calibration = F.mse_loss(
        output["confidence"].float(),
        1.0 - (output["arrival_probability"].float() - target).abs(),
    )
    near_risk_target = ((category == 1) | (category == 2) | (category == 4)).float()
    near_risk_auxiliary = F.binary_cross_entropy_with_logits(
        output["near_risk_logit"].float(), near_risk_target
    )
    total = (
        w.classification * classification
        + w.boundary_false_stop * boundary_false_stop
        + w.positive_preserving * positive_preserving
        + w.pair_ranking * pair_ranking
        + w.confidence_calibration * confidence_calibration
        + w.near_risk_auxiliary * near_risk_auxiliary
    )
    return {
        "loss": total,
        "classification": classification,
        "boundary_false_stop": boundary_false_stop,
        "positive_preserving": positive_preserving,
        "pair_ranking": pair_ranking,
        "confidence_calibration": confidence_calibration,
        "near_risk_auxiliary": near_risk_auxiliary,
    }
