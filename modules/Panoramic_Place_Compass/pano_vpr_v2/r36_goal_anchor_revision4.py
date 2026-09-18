from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .r36_goal_anchor_revision3 import R36GoalAnchorRevision2Model


@dataclass(frozen=True)
class R36GoalAnchorRevision4Config:
    evidence_dim: int = 17
    hidden_dim: int = 96
    class_count: int = 5
    dropout: float = 0.05
    hard_mining_count: int = 4096
    ranking_margin: float = 1.0


@dataclass(frozen=True)
class R36GoalAnchorRevision4LossWeights:
    set_classification: float = 1.0
    presence_binary: float = 1.0
    positive_preserving: float = 2.0
    hard_positive_boundary_ranking: float = 1.5
    hard_positive_near_ranking: float = 1.0
    hard_positive_wall_ranking: float = 2.0
    confidence_calibration: float = 0.25
    brier_calibration: float = 0.25


class R36GoalAnchorRevision4Calibrator(nn.Module):
    """Five-way set decision over frozen Revision 3 evidence."""

    def __init__(self, cfg: R36GoalAnchorRevision4Config | None = None) -> None:
        super().__init__()
        self.cfg = cfg or R36GoalAnchorRevision4Config()
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.cfg.evidence_dim),
            nn.Linear(self.cfg.evidence_dim, self.cfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.cfg.dropout),
            nn.Linear(self.cfg.hidden_dim, self.cfg.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.cfg.dropout),
            nn.Linear(self.cfg.hidden_dim, self.cfg.class_count),
        )
        self.temperature_raw = nn.Parameter(torch.tensor(0.0))
        self.confidence_head = nn.Sequential(
            nn.LayerNorm(self.cfg.evidence_dim + self.cfg.class_count),
            nn.Linear(self.cfg.evidence_dim + self.cfg.class_count, 48),
            nn.GELU(),
            nn.Linear(48, 1),
        )

    @property
    def architecture_record(self) -> dict[str, Any]:
        return {
            "schema_version": "r36_goal_anchor_revision4_frozen_evidence_calibrator_v1",
            "config": asdict(self.cfg),
            "classes": [
                "present",
                "boundary_1_1_25",
                "near_1_25_2",
                "wall_separated",
                "other_absent",
            ],
            "base_revision3_frozen": True,
            "candidate_match_frozen": True,
            "candidate_verifier_frozen": True,
            "presence_base_frozen": True,
            "fixed_presence_threshold": 0.5,
            "online_gt_inputs": False,
        }

    def forward(self, base: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        evidence = torch.stack(
            (
                base["base_presence_logit"],
                base["presence_logit"],
                base["presence_probability"],
                base["present_expert_logit"],
                base["unknown_expert_logit"],
                base["near_wrong_logit"],
                base["boundary_risk_logit"],
                base["wall_risk_logit"],
                base["risk_conditioned_penalty"],
                base["boundary_risk_penalty"],
                base["wall_risk_penalty"],
                base["candidate_conditioned_residual"],
                base["verifier_top1_probability"],
                base["verifier_probability_margin"],
                base["verifier_probability_entropy"],
                base["selected_candidate_probability"],
                base["candidate_probability_margin"],
            ),
            dim=-1,
        ).float()
        raw_logits = self.classifier(evidence)
        temperature = F.softplus(self.temperature_raw) + 0.5
        class_logits = raw_logits / temperature
        distribution = torch.softmax(class_logits, dim=-1)
        presence_probability = distribution[:, 0]
        presence_logit = class_logits[:, 0] - torch.logsumexp(class_logits[:, 1:], dim=-1)
        confidence_logit = self.confidence_head(
            torch.cat((evidence, distribution.detach()), dim=-1)
        ).squeeze(-1)
        return {
            "presence_logit": presence_logit,
            "presence_probability": presence_probability,
            "unknown_probability": 1.0 - presence_probability,
            "confidence_logit": confidence_logit,
            "confidence": torch.sigmoid(confidence_logit),
            "set_class_logits": class_logits,
            "set_class_distribution": distribution,
            "set_class_prediction": distribution.argmax(dim=-1),
            "calibration_temperature": temperature.expand_as(presence_probability),
            "frozen_evidence_features": evidence,
            "selected_candidate": base["selected_candidate"],
            "selected_candidate_probability": base["selected_candidate_probability"],
            "candidate_probability_margin": base["candidate_probability_margin"],
            "candidate_mask": base["candidate_mask"],
        }


class R36GoalAnchorRevision4Model(nn.Module):
    def __init__(self, cfg: R36GoalAnchorRevision4Config | None = None) -> None:
        super().__init__()
        self.frozen_revision3 = R36GoalAnchorRevision2Model()
        self.calibrator = R36GoalAnchorRevision4Calibrator(cfg)
        for parameter in self.frozen_revision3.parameters():
            parameter.requires_grad_(False)

    @property
    def architecture_record(self) -> dict[str, Any]:
        return {
            "schema_version": "r36_goal_anchor_revision4_model_v1",
            "frozen_revision3": self.frozen_revision3.architecture_record,
            "calibrator": self.calibrator.architecture_record,
            "shared_backbone_features_frozen": True,
            "test_time_gt_inputs": False,
        }

    def train(self, mode: bool = True):
        super().train(mode)
        self.frozen_revision3.eval()
        self.calibrator.train(mode)
        return self

    def forward(
        self,
        spatial_pair_features: torch.Tensor,
        scalar_features: torch.Tensor,
        candidate_mask: torch.Tensor,
    ) -> dict[str, Any]:
        with torch.no_grad():
            base = self.frozen_revision3(
                spatial_pair_features, scalar_features, candidate_mask
            )
        calibrated = self.calibrator(base["set_presence"])
        return {
            "candidate_match": base["candidate_match"],
            "candidate_verifier": base["candidate_verifier"],
            "frozen_set_presence": base["set_presence"],
            "set_presence": calibrated,
        }


def initialize_r36_goal_anchor_revision4(
    model: R36GoalAnchorRevision4Model,
    payload: dict[str, Any],
) -> dict[str, Any]:
    if payload.get("schema_version") != "r36_goal_anchor_revision3_checkpoint_v1":
        raise RuntimeError("Revision 4 requires a Goal Anchor Revision 3 checkpoint")
    frozen = model.frozen_revision3
    frozen.candidate_match_head.load_state_dict(payload["candidate_match_head"], strict=True)
    frozen.raw_pair_candidate_verifier.load_state_dict(
        payload["raw_pair_candidate_verifier"], strict=True
    )
    frozen.set_presence_head.load_state_dict(payload["set_presence_head"], strict=True)
    return {
        "schema_version": "r36_goal_anchor_revision4_initialization_v1",
        "source_global_step": int(payload["global_step"]),
        "source_checkpoint_schema": payload["schema_version"],
        "candidate_match_frozen": True,
        "candidate_verifier_frozen": True,
        "revision3_presence_frozen": True,
        "new_trainable_scope": "five_way_set_calibrator_and_confidence_only",
        "optimizer_state_reused": False,
        "fixed_presence_threshold": 0.5,
        "test_r32_confirmation_accessed": False,
    }


def _hard_rank(
    positive: torch.Tensor,
    negative: torch.Tensor,
    margin: float,
    count: int,
) -> torch.Tensor:
    if not positive.numel() or not negative.numel():
        return (positive.sum() + negative.sum()) * 0.0
    n = min(int(count), positive.numel(), negative.numel())
    hardest_positive = torch.topk(positive, k=n, largest=False).values
    hardest_negative = torch.topk(negative, k=n, largest=True).values
    return F.relu(margin - hardest_positive + hardest_negative).mean()


def r36_goal_anchor_revision4_losses(
    output: dict[str, Any],
    batch: dict[str, torch.Tensor],
    cfg: R36GoalAnchorRevision4Config | None = None,
    weights: R36GoalAnchorRevision4LossWeights | None = None,
) -> dict[str, torch.Tensor]:
    from scripts.r36_goal_anchor_data import LOGICAL_CATEGORIES

    config = cfg or R36GoalAnchorRevision4Config()
    loss_weights = weights or R36GoalAnchorRevision4LossWeights()
    category = batch["logical_category_id"].long()
    present = batch["same_targets"].bool().any(dim=1)
    hard_positive = category == LOGICAL_CATEGORIES.index("hard_positive")
    boundary = category == LOGICAL_CATEGORIES.index("boundary_1_1_25")
    near = (category == LOGICAL_CATEGORIES.index("near_1_25_1_5")) | (
        category == LOGICAL_CATEGORIES.index("early_1_5_2")
    )
    wall = category == LOGICAL_CATEGORIES.index("wall_separated")
    class_target = torch.full_like(category, 4)
    class_target[present] = 0
    class_target[boundary] = 1
    class_target[near] = 2
    class_target[wall] = 3

    class_counts = torch.bincount(class_target, minlength=config.class_count).float()
    class_weights = class_counts.sum() / class_counts.clamp_min(1.0)
    class_weights = class_weights / class_weights.mean()
    set_classification = F.cross_entropy(
        output["set_presence"]["set_class_logits"].float(),
        class_target,
        weight=class_weights,
    )
    presence_logit = output["set_presence"]["presence_logit"].float()
    presence_binary = F.binary_cross_entropy_with_logits(presence_logit, present.float())
    positive_preserving = F.binary_cross_entropy_with_logits(
        presence_logit[present], torch.ones_like(presence_logit[present])
    ) if bool(present.any()) else presence_logit.sum() * 0.0
    hard_boundary = _hard_rank(
        presence_logit[hard_positive], presence_logit[boundary],
        config.ranking_margin, config.hard_mining_count,
    )
    hard_near = _hard_rank(
        presence_logit[hard_positive], presence_logit[near],
        config.ranking_margin, config.hard_mining_count,
    )
    hard_wall = _hard_rank(
        presence_logit[hard_positive], presence_logit[wall],
        config.ranking_margin, config.hard_mining_count,
    )
    predicted_present = output["set_presence"]["presence_probability"].detach() >= 0.5
    confidence_target = (predicted_present == present).float()
    confidence_calibration = F.binary_cross_entropy_with_logits(
        output["set_presence"]["confidence_logit"].float(), confidence_target
    )
    brier = F.mse_loss(
        output["set_presence"]["presence_probability"].float(), present.float()
    )
    total = (
        loss_weights.set_classification * set_classification
        + loss_weights.presence_binary * presence_binary
        + loss_weights.positive_preserving * positive_preserving
        + loss_weights.hard_positive_boundary_ranking * hard_boundary
        + loss_weights.hard_positive_near_ranking * hard_near
        + loss_weights.hard_positive_wall_ranking * hard_wall
        + loss_weights.confidence_calibration * confidence_calibration
        + loss_weights.brier_calibration * brier
    )
    return {
        "loss": total,
        "set_classification": set_classification,
        "presence_binary": presence_binary,
        "positive_preserving": positive_preserving,
        "hard_positive_boundary_ranking": hard_boundary,
        "hard_positive_near_ranking": hard_near,
        "hard_positive_wall_ranking": hard_wall,
        "confidence_calibration": confidence_calibration,
        "brier_calibration": brier,
    }
