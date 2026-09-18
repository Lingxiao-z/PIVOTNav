from __future__ import annotations

from dataclasses import asdict
from typing import Any

import torch
from torch import nn

from .r35_open_set_heads import (
    R35CandidateLossWeights,
    R35CandidateMatchConfig,
    R35CandidateMatchHead,
    R35SetPresenceConfig,
    R35SetPresenceHead,
    R35SetPresenceLossWeights,
    r35_candidate_match_losses,
    r35_set_presence_losses,
)


class R35TrackCStage2Model(nn.Module):
    """Frozen-feature Stage 2 workload with separate candidate and set heads."""

    def __init__(
        self,
        candidate_cfg: R35CandidateMatchConfig | None = None,
        presence_cfg: R35SetPresenceConfig | None = None,
    ) -> None:
        super().__init__()
        self.candidate_cfg = candidate_cfg or R35CandidateMatchConfig()
        self.presence_cfg = presence_cfg or R35SetPresenceConfig()
        self.candidate_match_head = R35CandidateMatchHead(self.candidate_cfg)
        self.set_presence_head = R35SetPresenceHead(self.presence_cfg)

    @property
    def architecture_record(self) -> dict[str, Any]:
        return {
            "schema_version": "r35_track_c_stage2_model_v2",
            "candidate_match_config": asdict(self.candidate_cfg),
            "set_presence_config": asdict(self.presence_cfg),
            "candidate_and_presence_heads_are_separate": True,
            "shared_backbone_features_are_frozen": True,
            "explicit_candidate_competition": True,
            "dedicated_near_wrong_auxiliary": True,
            "test_time_gt_inputs": False,
        }

    def forward(
        self,
        spatial_pair_features: torch.Tensor,
        scalar_features: torch.Tensor,
        candidate_mask: torch.Tensor,
    ) -> dict[str, Any]:
        candidate = self.candidate_match_head(
            spatial_pair_features,
            scalar_features,
            candidate_mask,
        )
        presence = self.set_presence_head(
            candidate,
            global_similarity=scalar_features[:, :, 0],
            yaw_confidence=scalar_features[:, :, 6],
            bearing_confidence=scalar_features[:, :, 8],
            normalized_rank=scalar_features[:, :, 15],
        )
        return {
            "candidate_match": candidate,
            "set_presence": presence,
        }


def r35_stage2_losses(
    output: dict[str, Any],
    *,
    candidate_same_place: torch.Tensor,
    hard_positive_candidate_mask: torch.Tensor,
    hard_negative_candidate_mask: torch.Tensor,
    target_present: torch.Tensor,
    target_candidate_index: torch.Tensor,
    hard_positive_set: torch.Tensor,
    ordinary_positive_set: torch.Tensor,
    near_wrong_set: torch.Tensor,
    candidate_cfg: R35CandidateMatchConfig,
    presence_cfg: R35SetPresenceConfig,
    candidate_weights: R35CandidateLossWeights | None = None,
    presence_weights: R35SetPresenceLossWeights | None = None,
) -> dict[str, torch.Tensor]:
    candidate = r35_candidate_match_losses(
        output["candidate_match"],
        candidate_same_place,
        hard_positive_candidate_mask,
        hard_negative_candidate_mask,
        cfg=candidate_cfg,
        weights=candidate_weights,
    )
    presence = r35_set_presence_losses(
        output["set_presence"],
        target_present,
        target_candidate_index,
        output["candidate_match"]["same_place_probability"],
        hard_positive_set,
        ordinary_positive_set,
        near_wrong_set,
        cfg=presence_cfg,
        weights=presence_weights,
    )
    total = candidate["loss"] + presence["loss"]
    return {
        "loss": total,
        **{
            f"candidate_{name}": value
            for name, value in candidate.items()
            if name != "loss"
        },
        **{
            f"presence_{name}": value
            for name, value in presence.items()
            if name != "loss"
        },
        "candidate_loss": candidate["loss"],
        "presence_loss": presence["loss"],
    }
