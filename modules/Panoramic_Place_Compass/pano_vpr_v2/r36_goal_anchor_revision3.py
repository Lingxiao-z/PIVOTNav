from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .r35_stage2_revision4 import (
    R35RawPairCandidateVerifier,
    R35RawPairVerifierLossWeights,
    R35RawPairVerifierPresenceConfig,
    R35SetPresenceConfig,
    _set_feature,
    r35_raw_pair_verifier_presence_losses,
)
from .r35_open_set_heads import R35CandidateMatchConfig, R35CandidateMatchHead


@dataclass(frozen=True)
class R36GoalAnchorRevision2Config:
    initial_boundary_penalty_scale: float = 1.5
    initial_wall_penalty_scale: float = 3.0
    ranking_margin: float = 1.25
    hard_mining_count: int = 4096


@dataclass(frozen=True)
class R36GoalAnchorRevision2LossWeights:
    boundary_risk_classification: float = 1.5
    wall_risk_classification: float = 2.0
    hard_positive_boundary_ranking: float = 1.5
    hard_positive_wall_ranking: float = 2.0
    risk_penalty_regularization: float = 0.02


def _head(feature_dim: int, hidden: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.LayerNorm(feature_dim),
        nn.Linear(feature_dim, hidden),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden, 1),
    )


class R36GoalAnchorRevision2PresenceHead(nn.Module):
    """Presence head with independent boundary/wall risks used at inference."""

    def __init__(
        self,
        set_cfg: R35SetPresenceConfig | None = None,
        robust_cfg: R35RawPairVerifierPresenceConfig | None = None,
        revision_cfg: R36GoalAnchorRevision2Config | None = None,
    ) -> None:
        super().__init__()
        self.set_cfg = set_cfg or R35SetPresenceConfig()
        self.robust_cfg = robust_cfg or R35RawPairVerifierPresenceConfig()
        self.revision_cfg = revision_cfg or R36GoalAnchorRevision2Config()

        # Reuse the frozen set encoder and expert topology from revision 1.
        from .r35_stage2_revision4 import R35SetPresenceHead

        self.evidence_encoder = R35SetPresenceHead(self.set_cfg)
        for obsolete in (
            self.evidence_encoder.presence_head,
            self.evidence_encoder.near_wrong_head,
            self.evidence_encoder.confidence_head,
        ):
            for parameter in obsolete.parameters():
                parameter.requires_grad_(False)

        feature_dim = 3 * self.set_cfg.model_dim + self.set_cfg.scalar_summary_dim
        self.present_expert = _head(feature_dim, self.set_cfg.model_dim, self.set_cfg.dropout)
        self.unknown_expert = _head(feature_dim, self.set_cfg.model_dim // 2, self.set_cfg.dropout)
        self.boundary_risk_head = _head(feature_dim, self.set_cfg.model_dim // 2, self.set_cfg.dropout)
        self.wall_risk_head = _head(feature_dim, self.set_cfg.model_dim // 2, self.set_cfg.dropout)
        self.near_risk_aux_head = _head(feature_dim, self.set_cfg.model_dim // 2, self.set_cfg.dropout)

        residual_input_dim = feature_dim + 3
        self.candidate_residual_head = _head(
            residual_input_dim, self.set_cfg.model_dim // 2, self.set_cfg.dropout
        )
        # Revision 3 uses an explicitly monotonic risk gate. Revision 2 learned
        # separable risks but its free conditioner suppressed the mean penalty
        # to 0.011, so risk evidence barely affected the presence decision.
        self.boundary_penalty_scale_raw = nn.Parameter(
            torch.tensor(self.revision_cfg.initial_boundary_penalty_scale).expm1().log()
        )
        self.wall_penalty_scale_raw = nn.Parameter(
            torch.tensor(self.revision_cfg.initial_wall_penalty_scale).expm1().log()
        )
        self.confidence_head = _head(feature_dim, self.set_cfg.model_dim // 2, self.set_cfg.dropout)

    @property
    def architecture_record(self) -> dict[str, Any]:
        return {
            "schema_version": "r36_goal_anchor_revision3_presence_v1",
            "set_config": asdict(self.set_cfg),
            "robust_config": asdict(self.robust_cfg),
            "revision_config": asdict(self.revision_cfg),
            "boundary_and_wall_risk_heads_independent": True,
            "risk_conditioned_presence_logit": True,
            "monotonic_risk_gate": True,
            "free_risk_conditioner_removed": True,
            "candidate_match_frozen": True,
            "shared_backbone_features_frozen": True,
            "presence_threshold": 0.5,
            "online_gt_inputs": False,
        }

    def forward(
        self,
        candidate_output: dict[str, torch.Tensor],
        global_similarity: torch.Tensor,
        yaw_confidence: torch.Tensor,
        bearing_confidence: torch.Tensor,
        normalized_rank: torch.Tensor,
        verifier_output: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        feature, base = _set_feature(
            self.evidence_encoder,
            candidate_output,
            global_similarity,
            yaw_confidence,
            bearing_confidence,
            normalized_rank,
        )
        present_logit = self.present_expert(feature).squeeze(-1)
        unknown_logit = self.unknown_expert(feature).squeeze(-1)
        boundary_logit = self.boundary_risk_head(feature).squeeze(-1)
        wall_logit = self.wall_risk_head(feature).squeeze(-1)
        near_logit = self.near_risk_aux_head(feature).squeeze(-1)

        verifier_probability = verifier_output["probability"].masked_fill(
            ~base["candidate_mask"].bool(), -1.0
        )
        top = torch.topk(verifier_probability, k=min(2, verifier_probability.shape[1]), dim=-1).values
        verifier_top1 = top[:, 0]
        verifier_margin = top[:, 0] - top[:, 1] if top.shape[1] > 1 else top[:, 0]
        valid_probability = verifier_output["probability"].float().clamp(1e-6, 1.0 - 1e-6)
        entropy_terms = -(
            valid_probability * valid_probability.log()
            + (1.0 - valid_probability) * (1.0 - valid_probability).log()
        )
        valid = base["candidate_mask"].float()
        verifier_entropy = (entropy_terms * valid).sum(-1) / valid.sum(-1).clamp_min(1.0)
        competition = torch.stack((verifier_top1, verifier_margin, verifier_entropy), dim=-1)

        residual = self.robust_cfg.candidate_residual_bound * torch.tanh(
            self.candidate_residual_head(torch.cat((feature, competition), dim=-1)).squeeze(-1)
        )
        boundary_probability = torch.sigmoid(boundary_logit)
        wall_probability = torch.sigmoid(wall_logit)
        boundary_scale = F.softplus(self.boundary_penalty_scale_raw)
        wall_scale = F.softplus(self.wall_penalty_scale_raw)
        boundary_penalty = boundary_scale * boundary_probability
        wall_penalty = wall_scale * wall_probability
        risk_penalty = boundary_penalty + wall_penalty
        base_presence_logit = present_logit - unknown_logit + residual
        presence_logit = base_presence_logit - risk_penalty
        presence_probability = torch.sigmoid(presence_logit)
        confidence = torch.sigmoid(self.confidence_head(feature).squeeze(-1)) * (
            2.0 * (presence_probability - 0.5).abs()
        )
        return {
            "presence_logit": presence_logit,
            "base_presence_logit": base_presence_logit,
            "presence_probability": presence_probability,
            "unknown_probability": 1.0 - presence_probability,
            "present_expert_logit": present_logit,
            "unknown_expert_logit": unknown_logit,
            "near_wrong_logit": near_logit,
            "near_wrong_probability": torch.sigmoid(near_logit),
            "boundary_risk_logit": boundary_logit,
            "boundary_risk_probability": boundary_probability,
            "wall_risk_logit": wall_logit,
            "wall_risk_probability": wall_probability,
            "risk_conditioned_penalty": risk_penalty,
            "boundary_risk_penalty": boundary_penalty,
            "wall_risk_penalty": wall_penalty,
            "boundary_penalty_scale": boundary_scale.expand_as(risk_penalty),
            "wall_penalty_scale": wall_scale.expand_as(risk_penalty),
            "candidate_conditioned_residual": residual,
            "verifier_top1_probability": verifier_top1,
            "verifier_probability_margin": verifier_margin,
            "verifier_probability_entropy": verifier_entropy,
            "confidence": confidence,
            "selected_candidate": base["selected_candidate"],
            "selected_candidate_probability": base["selected_candidate_probability"],
            "candidate_probability_margin": base["candidate_probability_margin"],
            "candidate_mask": base["candidate_mask"],
        }


class R36GoalAnchorRevision2Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.candidate_match_head = R35CandidateMatchHead(R35CandidateMatchConfig())
        self.raw_pair_candidate_verifier = R35RawPairCandidateVerifier()
        self.set_presence_head = R36GoalAnchorRevision2PresenceHead()
        for parameter in self.candidate_match_head.parameters():
            parameter.requires_grad_(False)

    @property
    def architecture_record(self) -> dict[str, Any]:
        return {
            "schema_version": "r36_goal_anchor_revision3_model_v1",
            "candidate_match_frozen": True,
            "candidate_verifier_trainable": True,
            "presence": self.set_presence_head.architecture_record,
            "shared_backbone_features_frozen": True,
            "test_time_gt_inputs": False,
        }

    def forward(
        self,
        spatial_pair_features: torch.Tensor,
        scalar_features: torch.Tensor,
        candidate_mask: torch.Tensor,
    ) -> dict[str, Any]:
        with torch.no_grad():
            candidate = self.candidate_match_head(spatial_pair_features, scalar_features, candidate_mask)
        verifier = self.raw_pair_candidate_verifier(spatial_pair_features, scalar_features, candidate_mask)
        presence = self.set_presence_head(
            candidate,
            global_similarity=scalar_features[:, :, 0],
            yaw_confidence=scalar_features[:, :, 6],
            bearing_confidence=scalar_features[:, :, 8],
            normalized_rank=scalar_features[:, :, 15],
            verifier_output=verifier,
        )
        return {"candidate_match": candidate, "candidate_verifier": verifier, "set_presence": presence}


def initialize_r36_goal_anchor_revision2(
    model: R36GoalAnchorRevision2Model, payload: dict[str, Any]
) -> dict[str, Any]:
    if payload.get("schema_version") != "r36_goal_anchor_revision2_checkpoint_v1":
        raise RuntimeError("revision 3 requires an R36 Goal Anchor revision 2 checkpoint")
    model.candidate_match_head.load_state_dict(payload["candidate_match_head"], strict=True)
    model.raw_pair_candidate_verifier.load_state_dict(payload["raw_pair_candidate_verifier"], strict=True)
    source = payload["set_presence_head"]
    target = model.set_presence_head.state_dict()
    copied = {
        name: value for name, value in source.items()
        if name in target and target[name].shape == value.shape
    }
    target.update(copied)
    model.set_presence_head.load_state_dict(target, strict=True)
    return {
        "schema_version": "r36_goal_anchor_revision3_initialization_v1",
        "source_global_step": int(payload["global_step"]),
        "candidate_match_loaded_strict_and_frozen": True,
        "candidate_verifier_loaded_strict_and_trainable": True,
        "compatible_presence_keys_loaded": sorted(copied),
        "boundary_and_wall_heads_loaded_from_revision2": True,
        "monotonic_boundary_penalty_scale_initial": model.set_presence_head.revision_cfg.initial_boundary_penalty_scale,
        "monotonic_wall_penalty_scale_initial": model.set_presence_head.revision_cfg.initial_wall_penalty_scale,
        "free_risk_conditioner_removed": True,
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


def r36_goal_anchor_revision2_losses(
    output: dict[str, Any],
    batch: dict[str, torch.Tensor],
    config: R36GoalAnchorRevision2Config | None = None,
    weights: R36GoalAnchorRevision2LossWeights | None = None,
) -> dict[str, torch.Tensor]:
    from scripts.r36_goal_anchor_data import LOGICAL_CATEGORIES

    cfg = config or R36GoalAnchorRevision2Config()
    loss_weights = weights or R36GoalAnchorRevision2LossWeights()
    category = batch["logical_category_id"].long()
    target_present = batch["same_targets"].bool().any(dim=1)
    ordinary = category == LOGICAL_CATEGORIES.index("ordinary_positive")
    hard_positive = category == LOGICAL_CATEGORIES.index("hard_positive")
    boundary = category == LOGICAL_CATEGORIES.index("boundary_1_1_25")
    near = category == LOGICAL_CATEGORIES.index("near_1_25_1_5")
    early = category == LOGICAL_CATEGORIES.index("early_1_5_2")
    wall = category == LOGICAL_CATEGORIES.index("wall_separated")
    unified_near = boundary | near | early | wall
    same_targets = batch["same_targets"].bool()
    base = r35_raw_pair_verifier_presence_losses(
        output["set_presence"],
        target_present=target_present,
        hard_positive_set=hard_positive,
        ordinary_positive_set=ordinary,
        near_wrong_set=unified_near,
        scene_group_ids=batch["scene_group_id"],
        verifier_output=output["candidate_verifier"],
        candidate_same_targets=same_targets,
        hard_positive_candidate_mask=same_targets & hard_positive.unsqueeze(1),
        hard_negative_candidate_mask=batch["hard_negative_targets"].bool(),
        config=R35RawPairVerifierPresenceConfig(),
        weights=R35RawPairVerifierLossWeights(
            positive_preserving=1.75,
            hard_positive_near_ranking=1.0,
            worst_group_present_false_negative=1.5,
            worst_group_near_false_accept=1.5,
        ),
    )
    presence = output["set_presence"]
    boundary_weight = ((~boundary).sum().float() / boundary.sum().clamp_min(1)).clamp(1.0, 20.0)
    wall_weight = ((~wall).sum().float() / wall.sum().clamp_min(1)).clamp(1.0, 20.0)
    boundary_classification = F.binary_cross_entropy_with_logits(
        presence["boundary_risk_logit"].float(), boundary.float(), pos_weight=boundary_weight
    )
    wall_classification = F.binary_cross_entropy_with_logits(
        presence["wall_risk_logit"].float(), wall.float(), pos_weight=wall_weight
    )
    logits = presence["presence_logit"].float()
    hard_positive_boundary = _hard_rank(
        logits[hard_positive], logits[boundary], cfg.ranking_margin, cfg.hard_mining_count
    )
    hard_positive_wall = _hard_rank(
        logits[hard_positive], logits[wall], cfg.ranking_margin, cfg.hard_mining_count
    )
    risk_regularization = presence["risk_conditioned_penalty"].square().mean()
    total = (
        base["loss"]
        + loss_weights.boundary_risk_classification * boundary_classification
        + loss_weights.wall_risk_classification * wall_classification
        + loss_weights.hard_positive_boundary_ranking * hard_positive_boundary
        + loss_weights.hard_positive_wall_ranking * hard_positive_wall
        + loss_weights.risk_penalty_regularization * risk_regularization
    )
    return {
        "loss": total,
        **{f"base_{name}": value for name, value in base.items() if name != "loss"},
        "boundary_risk_classification": boundary_classification,
        "wall_risk_classification": wall_classification,
        "hard_positive_boundary_ranking": hard_positive_boundary,
        "hard_positive_wall_ranking": hard_positive_wall,
        "risk_penalty_regularization": risk_regularization,
    }
