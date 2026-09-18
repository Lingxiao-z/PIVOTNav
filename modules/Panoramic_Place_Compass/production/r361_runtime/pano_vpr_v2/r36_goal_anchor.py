from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .r35_stage2_revision4 import (
    R35RawPairVerifierLossWeights,
    R35RawPairVerifierPresenceConfig,
    R35Stage2Revision4Model,
    r35_raw_pair_verifier_presence_losses,
)


@dataclass(frozen=True)
class R36GoalAnchorLossWeights:
    boundary_suppression: float = 2.0
    near_suppression: float = 1.25
    early_suppression: float = 0.75
    wall_suppression: float = 2.0
    repeated_suppression: float = 0.75
    hard_positive_boundary_ranking: float = 1.25
    positive_preserving_extra: float = 0.75


class R36GoalAnchorModel(R35Stage2Revision4Model):
    """R35 best-initialized heads over frozen pair/set features under the 1m protocol."""

    @property
    def architecture_record(self) -> dict[str, Any]:
        return {
            "schema_version": "r36_goal_anchor_head_v1",
            "base": super().architecture_record,
            "initialization": "R35 Goal Anchor best checkpoint",
            "candidate_match_frozen": True,
            "candidate_verifier_trainable": True,
            "presence_near_risk_confidence_trainable": True,
            "shared_backbone_features_frozen": True,
            "online_gt_inputs": False,
            "fixed_presence_threshold": 0.5,
        }


def initialize_r36_goal_anchor(model: R36GoalAnchorModel, payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("schema_version") != "r35_stage2_checkpoint_v5":
        raise RuntimeError("R36 Goal Anchor requires R35 revision4 v5 checkpoint")
    model.candidate_match_head.load_state_dict(payload["candidate_match_head"], strict=True)
    model.raw_pair_candidate_verifier.load_state_dict(payload["raw_pair_candidate_verifier"], strict=True)
    model.set_presence_head.load_state_dict(payload["set_presence_head"], strict=True)
    return {
        "candidate_match_loaded_strict_and_frozen": True,
        "raw_pair_candidate_verifier_loaded_strict_and_trainable": True,
        "set_presence_head_loaded_strict_and_trainable": True,
        "optimizer_state_reused": False,
        "test_r32_confirmation_accessed": False,
    }


def _masked_softplus(logits: torch.Tensor, mask: torch.Tensor, margin: float) -> torch.Tensor:
    return F.softplus(logits[mask] + margin).mean() if bool(mask.any()) else logits.sum() * 0.0


def r36_goal_anchor_losses(
    output: dict[str, Any],
    batch: dict[str, torch.Tensor],
    weights: R36GoalAnchorLossWeights | None = None,
) -> dict[str, torch.Tensor]:
    from scripts.r36_goal_anchor_data import LOGICAL_CATEGORIES

    extra = weights or R36GoalAnchorLossWeights()
    category = batch["logical_category_id"].long()
    target_present = batch["same_targets"].bool().any(dim=1)
    ordinary = category == LOGICAL_CATEGORIES.index("ordinary_positive")
    hard_positive = category == LOGICAL_CATEGORIES.index("hard_positive")
    boundary = category == LOGICAL_CATEGORIES.index("boundary_1_1_25")
    near = category == LOGICAL_CATEGORIES.index("near_1_25_1_5")
    early = category == LOGICAL_CATEGORIES.index("early_1_5_2")
    wall = category == LOGICAL_CATEGORIES.index("wall_separated")
    repeated = category == LOGICAL_CATEGORIES.index("repeated_texture")
    unified_near = boundary | near | early | wall
    same_targets = batch["same_targets"].bool()
    hard_positive_candidate = same_targets & hard_positive.unsqueeze(1)
    base_weights = R35RawPairVerifierLossWeights(
        positive_preserving=1.5,
        hard_positive_near_ranking=1.0,
        worst_group_present_false_negative=1.25,
        worst_group_near_false_accept=1.25,
    )
    base = r35_raw_pair_verifier_presence_losses(
        output["set_presence"],
        target_present=target_present,
        hard_positive_set=hard_positive,
        ordinary_positive_set=ordinary,
        near_wrong_set=unified_near,
        scene_group_ids=batch["scene_group_id"],
        verifier_output=output["candidate_verifier"],
        candidate_same_targets=same_targets,
        hard_positive_candidate_mask=hard_positive_candidate,
        hard_negative_candidate_mask=batch["hard_negative_targets"].bool(),
        config=R35RawPairVerifierPresenceConfig(),
        weights=base_weights,
    )
    logits = output["set_presence"]["presence_logit"].float()
    losses = {
        "boundary_suppression": _masked_softplus(logits, boundary, 1.25),
        "near_suppression": _masked_softplus(logits, near, 0.75),
        "early_suppression": _masked_softplus(logits, early, 0.25),
        "wall_suppression": _masked_softplus(logits, wall, 1.25),
        "repeated_suppression": _masked_softplus(logits, repeated, 0.5),
    }
    hard_logits = logits[hard_positive]
    boundary_logits = logits[boundary]
    losses["hard_positive_boundary_ranking"] = (
        F.relu(1.25 - hard_logits[:, None] + boundary_logits[None, :]).mean()
        if hard_logits.numel() and boundary_logits.numel()
        else logits.sum() * 0.0
    )
    positive = ordinary | hard_positive
    losses["positive_preserving_extra"] = (
        F.relu(torch.logit(torch.tensor(0.85, device=logits.device)) - logits[positive]).mean()
        if bool(positive.any()) else logits.sum() * 0.0
    )
    total = base["loss"] + sum(
        getattr(extra, name) * value for name, value in losses.items()
    )
    return {"loss": total, **{f"base_{name}": value for name, value in base.items() if name != "loss"}, **losses}
