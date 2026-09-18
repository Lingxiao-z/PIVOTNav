from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .r35_open_set_heads import (
    R35CandidateMatchConfig,
    R35CandidateMatchHead,
    R35SetPresenceConfig,
    R35SetPresenceHead,
)


@dataclass(frozen=True)
class R35SceneRobustPresenceConfig:
    minimum_near_suppression: float = 0.25
    maximum_near_suppression: float = 1.75
    ordinary_positive_target: float = 0.85
    hard_positive_target: float = 0.90
    near_wrong_target: float = 0.20
    hard_positive_to_near_margin: float = 1.0
    group_softmax_temperature: float = 0.25


@dataclass(frozen=True)
class R35SceneRobustLossWeights:
    present_unknown_classification: float = 1.0
    positive_preserving: float = 1.25
    near_wrong_classification: float = 0.75
    hard_positive_near_ranking: float = 0.75
    worst_group_present_false_negative: float = 1.0
    worst_group_near_false_accept: float = 1.0
    gate_regularization: float = 0.05
    confidence_calibration: float = 0.10


def _set_feature(
    encoder: R35SetPresenceHead,
    candidate_output: dict[str, torch.Tensor],
    global_similarity: torch.Tensor,
    yaw_confidence: torch.Tensor,
    bearing_confidence: torch.Tensor,
    normalized_rank: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    base = encoder(
        candidate_output,
        global_similarity=global_similarity,
        yaw_confidence=yaw_confidence,
        bearing_confidence=bearing_confidence,
        normalized_rank=normalized_rank,
    )
    encoded = base["encoded_candidate_tokens"]
    mask = base["candidate_mask"].bool()
    pooled = (
        (encoded * mask.unsqueeze(-1).float()).sum(dim=1)
        / mask.sum(dim=1, keepdim=True).clamp_min(1)
    )
    same = candidate_output["same_place_probability"].float().masked_fill(~mask, -1.0)
    top_indices = torch.topk(same, k=min(2, same.shape[1]), dim=-1).indices
    gather = top_indices.unsqueeze(-1).expand(-1, -1, encoded.shape[-1])
    top = encoded.gather(1, gather)
    top1 = top[:, 0]
    top2 = top[:, 1] if top.shape[1] > 1 else torch.zeros_like(top1)
    competition = encoder.competition_projection(
        torch.cat((top1, top2, top1 - top2, top1 * top2), dim=-1)
    )
    feature = torch.cat(
        (
            base["encoded_unknown_token"],
            pooled,
            competition,
            base["set_summary"],
        ),
        dim=-1,
    )
    return feature, base


class R35SceneRobustPresenceHead(nn.Module):
    """Separate positive, near-risk and uncertainty evidence without scene input."""

    def __init__(
        self,
        set_cfg: R35SetPresenceConfig | None = None,
        robust_cfg: R35SceneRobustPresenceConfig | None = None,
    ) -> None:
        super().__init__()
        self.set_cfg = set_cfg or R35SetPresenceConfig()
        self.robust_cfg = robust_cfg or R35SceneRobustPresenceConfig()
        self.evidence_encoder = R35SetPresenceHead(self.set_cfg)
        for obsolete_head in (
            self.evidence_encoder.presence_head,
            self.evidence_encoder.near_wrong_head,
            self.evidence_encoder.confidence_head,
        ):
            for parameter in obsolete_head.parameters():
                parameter.requires_grad_(False)
        feature_dim = 3 * self.set_cfg.model_dim + self.set_cfg.scalar_summary_dim

        def head(hidden: int) -> nn.Sequential:
            return nn.Sequential(
                nn.LayerNorm(feature_dim),
                nn.Linear(feature_dim, hidden),
                nn.GELU(),
                nn.Dropout(self.set_cfg.dropout),
                nn.Linear(hidden, 1),
            )

        self.positive_evidence_head = head(self.set_cfg.model_dim)
        self.near_risk_head = head(self.set_cfg.model_dim // 2)
        self.uncertainty_gate_head = head(self.set_cfg.model_dim // 2)
        self.confidence_head = head(self.set_cfg.model_dim // 2)

    @property
    def architecture_record(self) -> dict[str, Any]:
        return {
            "schema_version": "r35_set_presence_scene_robust_v1",
            "set_config": asdict(self.set_cfg),
            "robust_config": asdict(self.robust_cfg),
            "positive_near_uncertainty_evidence_separate": True,
            "scene_id_is_training_loss_only": True,
            "scene_id_is_inference_input": False,
            "fixed_global_near_suppression_removed": True,
            "presence_threshold": 0.5,
        }

    def forward(
        self,
        candidate_output: dict[str, torch.Tensor],
        global_similarity: torch.Tensor,
        yaw_confidence: torch.Tensor,
        bearing_confidence: torch.Tensor,
        normalized_rank: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        feature, base = _set_feature(
            self.evidence_encoder,
            candidate_output,
            global_similarity,
            yaw_confidence,
            bearing_confidence,
            normalized_rank,
        )
        positive_logit = self.positive_evidence_head(feature).squeeze(-1)
        near_logit = self.near_risk_head(feature).squeeze(-1)
        gate_unit = torch.sigmoid(self.uncertainty_gate_head(feature).squeeze(-1))
        cfg = self.robust_cfg
        suppression = cfg.minimum_near_suppression + (
            cfg.maximum_near_suppression - cfg.minimum_near_suppression
        ) * gate_unit
        near_evidence = F.softplus(near_logit)
        presence_logit = positive_logit - suppression * near_evidence
        presence_probability = torch.sigmoid(presence_logit)
        confidence = torch.sigmoid(self.confidence_head(feature).squeeze(-1)) * (
            2.0 * (presence_probability - 0.5).abs()
        )
        return {
            "presence_logit": presence_logit,
            "presence_probability": presence_probability,
            "unknown_probability": 1.0 - presence_probability,
            "positive_evidence_logit": positive_logit,
            "near_wrong_logit": near_logit,
            "near_wrong_probability": torch.sigmoid(near_logit),
            "uncertainty_gate": gate_unit,
            "near_suppression": suppression,
            "confidence": confidence,
            "selected_candidate": base["selected_candidate"],
            "selected_candidate_probability": base["selected_candidate_probability"],
            "candidate_probability_margin": base["candidate_probability_margin"],
            "candidate_mask": base["candidate_mask"],
        }


def _smooth_worst_group(
    values: torch.Tensor,
    group_ids: torch.Tensor,
    mask: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    groups = torch.unique(group_ids[mask])
    if groups.numel() == 0:
        return values.sum() * 0.0
    means = torch.stack(
        [values[mask & (group_ids == group)].mean() for group in groups]
    )
    return temperature * (
        torch.logsumexp(means / temperature, dim=0) - math.log(means.numel())
    )


def r35_scene_robust_presence_losses(
    output: dict[str, torch.Tensor],
    *,
    target_present: torch.Tensor,
    hard_positive_set: torch.Tensor,
    ordinary_positive_set: torch.Tensor,
    near_wrong_set: torch.Tensor,
    scene_group_ids: torch.Tensor,
    config: R35SceneRobustPresenceConfig,
    weights: R35SceneRobustLossWeights | None = None,
) -> dict[str, torch.Tensor]:
    loss_weights = weights or R35SceneRobustLossWeights()
    present = target_present.bool()
    hard_positive = hard_positive_set.bool() & present
    ordinary_positive = ordinary_positive_set.bool() & present
    near_wrong = near_wrong_set.bool() & ~present
    if scene_group_ids.shape != present.shape:
        raise ValueError("scene_group_ids必须与set batch一致")
    logits = output["presence_logit"].float()
    positive_weight = ((~present).sum().float() / present.sum().clamp_min(1)).clamp(1.0, 10.0)
    classification = F.binary_cross_entropy_with_logits(
        logits, present.float(), pos_weight=positive_weight
    )
    positive_terms = []
    for mask, target_probability in (
        (ordinary_positive, config.ordinary_positive_target),
        (hard_positive, config.hard_positive_target),
    ):
        if mask.any():
            target = torch.logit(torch.tensor(target_probability, device=logits.device))
            positive_terms.append(F.relu(target - logits[mask]).mean())
    positive_preserving = (
        torch.stack(positive_terms).mean() if positive_terms else logits.sum() * 0.0
    )
    near_logits = output["near_wrong_logit"].float()
    near_weight = ((~near_wrong).sum().float() / near_wrong.sum().clamp_min(1)).clamp(1.0, 10.0)
    near_classification = F.binary_cross_entropy_with_logits(
        near_logits, near_wrong.float(), pos_weight=near_weight
    )
    ranking = (
        F.relu(
            config.hard_positive_to_near_margin
            - logits[hard_positive][:, None]
            + logits[near_wrong][None, :]
        ).mean()
        if hard_positive.any() and near_wrong.any()
        else logits.sum() * 0.0
    )
    positive_target = torch.logit(
        torch.tensor(config.hard_positive_target, device=logits.device)
    )
    near_target = torch.logit(
        torch.tensor(config.near_wrong_target, device=logits.device)
    )
    per_set_fn = F.relu(positive_target - logits)
    per_set_far = F.relu(logits - near_target)
    worst_group_fn = _smooth_worst_group(
        per_set_fn, scene_group_ids, present, config.group_softmax_temperature
    )
    worst_group_far = _smooth_worst_group(
        per_set_far, scene_group_ids, near_wrong, config.group_softmax_temperature
    )
    gate_regularization = (
        output["uncertainty_gate"][present | near_wrong]
        * (1.0 - output["uncertainty_gate"][present | near_wrong])
    ).mean()
    probability = output["presence_probability"].float()
    with torch.no_grad():
        correct = ((probability >= 0.5) == present).float()
        confidence_target = 2.0 * (probability - 0.5).abs() * correct
    confidence_calibration = ((output["confidence"] - confidence_target) ** 2).mean()
    total = (
        loss_weights.present_unknown_classification * classification
        + loss_weights.positive_preserving * positive_preserving
        + loss_weights.near_wrong_classification * near_classification
        + loss_weights.hard_positive_near_ranking * ranking
        + loss_weights.worst_group_present_false_negative * worst_group_fn
        + loss_weights.worst_group_near_false_accept * worst_group_far
        + loss_weights.gate_regularization * gate_regularization
        + loss_weights.confidence_calibration * confidence_calibration
    )
    return {
        "loss": total,
        "present_unknown_classification": classification,
        "positive_preserving": positive_preserving,
        "near_wrong_classification": near_classification,
        "hard_positive_near_ranking": ranking,
        "worst_group_present_false_negative": worst_group_fn,
        "worst_group_near_false_accept": worst_group_far,
        "gate_regularization": gate_regularization,
        "confidence_calibration": confidence_calibration,
    }


class R35Stage2Revision2Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.candidate_match_head = R35CandidateMatchHead(R35CandidateMatchConfig())
        self.set_presence_head = R35SceneRobustPresenceHead()
        for parameter in self.candidate_match_head.parameters():
            parameter.requires_grad_(False)

    @property
    def architecture_record(self) -> dict[str, Any]:
        return {
            "schema_version": "r35_track_c_stage2_revision2_model_v1",
            "candidate_match_frozen": True,
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
            candidate = self.candidate_match_head(
                spatial_pair_features, scalar_features, candidate_mask
            )
        presence = self.set_presence_head(
            candidate,
            global_similarity=scalar_features[:, :, 0],
            yaw_confidence=scalar_features[:, :, 6],
            bearing_confidence=scalar_features[:, :, 8],
            normalized_rank=scalar_features[:, :, 15],
        )
        return {"candidate_match": candidate, "set_presence": presence}


def initialize_revision2_from_checkpoint(
    model: R35Stage2Revision2Model,
    payload: dict[str, Any],
) -> dict[str, Any]:
    if payload.get("schema_version") != "r35_stage2_checkpoint_v2":
        raise RuntimeError("revision 2只接受完整Stage 2 r2 checkpoint")
    model.candidate_match_head.load_state_dict(payload["candidate_match_head"], strict=True)
    source = payload["set_presence_head"]
    current = model.set_presence_head.evidence_encoder.state_dict()
    transferred = {
        name: value
        for name, value in source.items()
        if name in current
        and current[name].shape == value.shape
        and not name.startswith(("presence_head.", "near_wrong_head.", "confidence_head."))
    }
    current.update(transferred)
    model.set_presence_head.evidence_encoder.load_state_dict(current, strict=True)
    required = ("candidate_projection.", "set_encoder.", "competition_projection.", "unknown_token")
    if not all(any(name.startswith(prefix) for name in transferred) for prefix in required):
        raise RuntimeError("revision 2 evidence encoder迁移不完整")
    return {
        "schema_version": "r35_stage2_revision2_initialization_v1",
        "candidate_match_loaded_strict_and_frozen": True,
        "evidence_encoder_transferred_keys": sorted(transferred),
        "new_heads_randomly_initialized": [
            "positive_evidence_head",
            "near_risk_head",
            "uncertainty_gate_head",
            "confidence_head",
        ],
        "optimizer_state_reused": False,
        "presence_threshold": 0.5,
        "test_r32_confirmation_accessed": False,
    }
