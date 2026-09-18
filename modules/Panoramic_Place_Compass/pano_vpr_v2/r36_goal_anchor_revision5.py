from __future__ import annotations

"""R36 Goal Anchor Revision 5: dual-evidence set resolver.

Revision 4 proved that a five-way softmax alone cannot preserve difficult
positive sets while rejecting the high-score boundary and wall tails.  This
module keeps that entire model frozen.  It learns separate positive and risk
evidence paths, then resolves their disagreement with an explicitly bounded
conflict residual.  It never consumes geometric labels at inference time.
"""

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .r36_goal_anchor_revision4 import R36GoalAnchorRevision4Model


@dataclass(frozen=True)
class R36GoalAnchorRevision5Config:
    frozen_evidence_dim: int = 29
    hidden_dim: int = 128
    dropout: float = 0.05
    tail_weight: float = 2.5
    ranking_margin: float = 1.0
    hard_mining_count: int = 4096
    conflict_residual_bound: float = 1.25


@dataclass(frozen=True)
class R36GoalAnchorRevision5LossWeights:
    decision: float = 1.0
    positive_evidence: float = 1.25
    risk_evidence: float = 1.25
    tail_expert: float = 1.5
    hard_positive_preserving: float = 3.0
    boundary_ranking: float = 2.0
    near_ranking: float = 1.5
    wall_ranking: float = 3.0
    conflict_calibration: float = 0.4
    brier: float = 0.2


def _mlp(input_dim: int, hidden_dim: int, dropout: float, output_dim: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.LayerNorm(input_dim),
        nn.Linear(input_dim, hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, output_dim),
    )


class R36GoalAnchorRevision5Resolver(nn.Module):
    """Separate support and risk estimates before a bounded final decision."""

    def __init__(self, cfg: R36GoalAnchorRevision5Config | None = None) -> None:
        super().__init__()
        self.cfg = cfg or R36GoalAnchorRevision5Config()
        self.positive_head = _mlp(self.cfg.frozen_evidence_dim, self.cfg.hidden_dim, self.cfg.dropout)
        self.risk_head = _mlp(self.cfg.frozen_evidence_dim, self.cfg.hidden_dim, self.cfg.dropout)
        self.tail_head = _mlp(self.cfg.frozen_evidence_dim, self.cfg.hidden_dim, self.cfg.dropout)
        self.conflict_head = _mlp(self.cfg.frozen_evidence_dim + 3, self.cfg.hidden_dim, self.cfg.dropout)
        self.confidence_head = _mlp(self.cfg.frozen_evidence_dim + 4, self.cfg.hidden_dim // 2, self.cfg.dropout)

    @property
    def architecture_record(self) -> dict[str, Any]:
        return {
            "schema_version": "r36_goal_anchor_revision5_dual_evidence_resolver_v1",
            "config": asdict(self.cfg),
            "frozen_revision4": True,
            "positive_evidence_head": True,
            "negative_risk_head": True,
            "hard_tail_expert": True,
            "bounded_conflict_resolver": True,
            "fixed_presence_threshold": 0.5,
            "online_gt_inputs": False,
        }

    @staticmethod
    def _frozen_evidence(revision4: dict[str, torch.Tensor]) -> torch.Tensor:
        raw = revision4["frozen_evidence_features"].float()
        logits = revision4["set_class_logits"].float()
        distribution = revision4["set_class_distribution"].float()
        old_logit = revision4["presence_logit"].float().unsqueeze(-1)
        old_confidence = revision4["confidence"].float().unsqueeze(-1)
        evidence = torch.cat((raw, logits, distribution, old_logit, old_confidence), dim=-1)
        if evidence.shape[-1] != 29:
            raise RuntimeError(f"unexpected frozen evidence dimension {evidence.shape[-1]}")
        return evidence

    def forward(self, revision4: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        evidence = self._frozen_evidence(revision4)
        positive_logit = self.positive_head(evidence).squeeze(-1)
        risk_logit = self.risk_head(evidence).squeeze(-1)
        tail_logit = self.tail_head(evidence).squeeze(-1)
        positive_probability = torch.sigmoid(positive_logit)
        risk_probability = torch.sigmoid(risk_logit)
        tail_probability = torch.sigmoid(tail_logit)
        conflict_inputs = torch.cat(
            (evidence, positive_logit[:, None], risk_logit[:, None], tail_logit[:, None]), dim=-1
        )
        conflict_residual = self.cfg.conflict_residual_bound * torch.tanh(
            self.conflict_head(conflict_inputs).squeeze(-1)
        )
        # The explicit support-minus-risk representation avoids using one softmax
        # logit as both a positive score and a negative-class gate.
        presence_logit = positive_logit - risk_logit - self.cfg.tail_weight * tail_probability + conflict_residual
        presence_probability = torch.sigmoid(presence_logit)
        conflict_strength = 4.0 * positive_probability * risk_probability
        confidence_input = torch.cat(
            (evidence, presence_logit[:, None], conflict_strength[:, None], tail_probability[:, None], revision4["confidence"].float()[:, None]),
            dim=-1,
        )
        confidence = torch.sigmoid(self.confidence_head(confidence_input).squeeze(-1)) * (1.0 - 0.5 * conflict_strength)
        return {
            "presence_logit": presence_logit,
            "presence_probability": presence_probability,
            "unknown_probability": 1.0 - presence_probability,
            "positive_evidence_logit": positive_logit,
            "positive_evidence_probability": positive_probability,
            "risk_evidence_logit": risk_logit,
            "risk_evidence_probability": risk_probability,
            "tail_expert_logit": tail_logit,
            "tail_expert_probability": tail_probability,
            "conflict_residual": conflict_residual,
            "conflict_strength": conflict_strength,
            "confidence": confidence.clamp(0.0, 1.0),
            "frozen_evidence_features": evidence,
            "selected_candidate": revision4["selected_candidate"],
            "selected_candidate_probability": revision4["selected_candidate_probability"],
            "candidate_probability_margin": revision4["candidate_probability_margin"],
            "candidate_mask": revision4["candidate_mask"],
        }


class R36GoalAnchorRevision5Model(nn.Module):
    def __init__(self, cfg: R36GoalAnchorRevision5Config | None = None) -> None:
        super().__init__()
        self.frozen_revision4 = R36GoalAnchorRevision4Model()
        self.resolver = R36GoalAnchorRevision5Resolver(cfg)
        for parameter in self.frozen_revision4.parameters():
            parameter.requires_grad_(False)

    @property
    def architecture_record(self) -> dict[str, Any]:
        return {
            "schema_version": "r36_goal_anchor_revision5_model_v1",
            "frozen_revision4": self.frozen_revision4.architecture_record,
            "resolver": self.resolver.architecture_record,
            "shared_backbone_features_frozen": True,
            "test_time_gt_inputs": False,
        }

    def train(self, mode: bool = True):
        super().train(mode)
        self.frozen_revision4.eval()
        self.resolver.train(mode)
        return self

    def forward(self, spatial_pair_features: torch.Tensor, scalar_features: torch.Tensor, candidate_mask: torch.Tensor) -> dict[str, Any]:
        with torch.no_grad():
            revision4_output = self.frozen_revision4(spatial_pair_features, scalar_features, candidate_mask)
        set_presence = self.resolver(revision4_output["set_presence"])
        return {
            "candidate_match": revision4_output["candidate_match"],
            "candidate_verifier": revision4_output["candidate_verifier"],
            "frozen_revision4_set_presence": revision4_output["set_presence"],
            "frozen_set_presence": revision4_output["frozen_set_presence"],
            "set_presence": set_presence,
        }


def initialize_r36_goal_anchor_revision5(model: R36GoalAnchorRevision5Model, payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("schema_version") != "r36_goal_anchor_revision4_checkpoint_v1":
        raise RuntimeError("Revision 5 requires a Goal Anchor Revision 4 checkpoint")
    frozen = model.frozen_revision4
    frozen.frozen_revision3.candidate_match_head.load_state_dict(payload["frozen_candidate_match_head"], strict=True)
    frozen.frozen_revision3.raw_pair_candidate_verifier.load_state_dict(payload["frozen_raw_pair_candidate_verifier"], strict=True)
    frozen.frozen_revision3.set_presence_head.load_state_dict(payload["frozen_revision3_set_presence_head"], strict=True)
    frozen.calibrator.load_state_dict(payload["revision4_calibrator"], strict=True)
    return {
        "schema_version": "r36_goal_anchor_revision5_initialization_v1",
        "source_global_step": int(payload["global_step"]),
        "source_checkpoint_schema": payload["schema_version"],
        "frozen_revision4": True,
        "new_trainable_scope": "dual_evidence_resolver_only",
        "fixed_presence_threshold": 0.5,
        "test_r32_confirmation_accessed": False,
    }


def _hard_rank(positive: torch.Tensor, negative: torch.Tensor, margin: float, count: int) -> torch.Tensor:
    if not positive.numel() or not negative.numel():
        return (positive.sum() + negative.sum()) * 0.0
    n = min(int(count), positive.numel(), negative.numel())
    return F.relu(margin - torch.topk(positive, n, largest=False).values + torch.topk(negative, n, largest=True).values).mean()


def r36_goal_anchor_revision5_losses(
    output: dict[str, Any], batch: dict[str, torch.Tensor], cfg: R36GoalAnchorRevision5Config | None = None,
    weights: R36GoalAnchorRevision5LossWeights | None = None,
) -> dict[str, torch.Tensor]:
    from scripts.r36_goal_anchor_data import LOGICAL_CATEGORIES

    config = cfg or R36GoalAnchorRevision5Config()
    loss_weights = weights or R36GoalAnchorRevision5LossWeights()
    category = batch["logical_category_id"].long()
    present = batch["same_targets"].bool().any(dim=1)
    hard_positive = category == LOGICAL_CATEGORIES.index("hard_positive")
    boundary = category == LOGICAL_CATEGORIES.index("boundary_1_1_25")
    near = (category == LOGICAL_CATEGORIES.index("near_1_25_1_5")) | (category == LOGICAL_CATEGORIES.index("early_1_5_2"))
    wall = category == LOGICAL_CATEGORIES.index("wall_separated")
    difficult_negative = boundary | near | wall
    decision = output["set_presence"]["presence_logit"].float()
    positive_logit = output["set_presence"]["positive_evidence_logit"].float()
    risk_logit = output["set_presence"]["risk_evidence_logit"].float()
    tail_logit = output["set_presence"]["tail_expert_logit"].float()
    decision_loss = F.binary_cross_entropy_with_logits(decision, present.float())
    positive_loss = F.binary_cross_entropy_with_logits(positive_logit, present.float())
    risk_loss = F.binary_cross_entropy_with_logits(risk_logit, (~present).float())
    tail_loss = F.binary_cross_entropy_with_logits(tail_logit, difficult_negative.float())
    positive_preserving = F.binary_cross_entropy_with_logits(
        decision[hard_positive], torch.ones_like(decision[hard_positive])
    ) if bool(hard_positive.any()) else decision.sum() * 0.0
    boundary_ranking = _hard_rank(decision[hard_positive], decision[boundary], config.ranking_margin, config.hard_mining_count)
    near_ranking = _hard_rank(decision[hard_positive], decision[near], config.ranking_margin, config.hard_mining_count)
    wall_ranking = _hard_rank(decision[hard_positive], decision[wall], config.ranking_margin, config.hard_mining_count)
    conflict = output["set_presence"]["conflict_strength"].float()
    # Confident support/risk disagreement must not produce confident acceptance.
    conflict_calibration = (conflict * output["set_presence"]["confidence"].float()).mean()
    brier = F.mse_loss(output["set_presence"]["presence_probability"].float(), present.float())
    total = (
        loss_weights.decision * decision_loss + loss_weights.positive_evidence * positive_loss
        + loss_weights.risk_evidence * risk_loss + loss_weights.tail_expert * tail_loss
        + loss_weights.hard_positive_preserving * positive_preserving
        + loss_weights.boundary_ranking * boundary_ranking + loss_weights.near_ranking * near_ranking
        + loss_weights.wall_ranking * wall_ranking + loss_weights.conflict_calibration * conflict_calibration
        + loss_weights.brier * brier
    )
    return {
        "loss": total, "decision": decision_loss, "positive_evidence": positive_loss,
        "risk_evidence": risk_loss, "tail_expert": tail_loss, "positive_preserving": positive_preserving,
        "boundary_ranking": boundary_ranking, "near_ranking": near_ranking,
        "wall_ranking": wall_ranking, "conflict_calibration": conflict_calibration, "brier": brier,
    }
