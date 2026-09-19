from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .open_set import (
    R35CandidateMatchConfig,
    R35CandidateMatchHead,
    R35SetPresenceConfig,
    R35SetPresenceHead,
)


@dataclass(frozen=True)
class R35RawPairVerifierPresenceConfig:
    ordinary_positive_target: float = 0.85
    hard_positive_target: float = 0.90
    near_wrong_target: float = 0.20
    hard_positive_to_near_margin: float = 1.0
    candidate_residual_bound: float = 2.0
    worst_group_cvar_fraction: float = 0.25


@dataclass(frozen=True)
class R35RawPairVerifierLossWeights:
    candidate_verifier_classification: float = 1.0
    candidate_verifier_hard_positive: float = 1.5
    candidate_verifier_hard_negative: float = 0.75
    candidate_verifier_ranking: float = 0.75
    present_unknown_classification: float = 1.0
    positive_preserving: float = 1.25
    near_wrong_classification: float = 0.75
    hard_positive_near_ranking: float = 0.75
    worst_group_present_false_negative: float = 1.0
    worst_group_near_false_accept: float = 1.0
    expert_specialization: float = 0.25
    candidate_residual_regularization: float = 0.02
    confidence_calibration: float = 0.10


class R35RawPairCandidateVerifier(nn.Module):
    """Candidate-level same-place verifier over frozen 256+17 pair features."""

    def __init__(self, spatial_dim: int = 256, scalar_dim: int = 17) -> None:
        super().__init__()
        self.spatial_projection = nn.Sequential(
            nn.LayerNorm(spatial_dim),
            nn.Linear(spatial_dim, 192),
            nn.GELU(),
        )
        self.scalar_projection = nn.Sequential(
            nn.LayerNorm(scalar_dim),
            nn.Linear(scalar_dim, 64),
            nn.GELU(),
        )
        self.fusion = nn.Sequential(
            nn.LayerNorm(256),
            nn.Linear(256, 192),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(192, 192),
            nn.GELU(),
        )
        self.logit_head = nn.Linear(192, 1)

    def forward(
        self,
        spatial_pair_features: torch.Tensor,
        scalar_features: torch.Tensor,
        candidate_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        token = self.fusion(
            torch.cat(
                (
                    self.spatial_projection(spatial_pair_features.float()),
                    self.scalar_projection(scalar_features.float()),
                ),
                dim=-1,
            )
        )
        logit = self.logit_head(token).squeeze(-1)
        logit = logit.masked_fill(~candidate_mask.bool(), -20.0)
        return {
            "logit": logit,
            "probability": torch.sigmoid(logit),
            "token": token,
            "candidate_mask": candidate_mask.bool(),
        }


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


class R35RawPairVerifierPresenceHead(nn.Module):
    """Independent present/UNKNOWN experts with a bounded candidate residual."""

    def __init__(
        self,
        set_cfg: R35SetPresenceConfig | None = None,
        robust_cfg: R35RawPairVerifierPresenceConfig | None = None,
    ) -> None:
        super().__init__()
        self.set_cfg = set_cfg or R35SetPresenceConfig()
        self.robust_cfg = robust_cfg or R35RawPairVerifierPresenceConfig()
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

        self.present_expert = head(self.set_cfg.model_dim)
        self.unknown_expert = head(self.set_cfg.model_dim // 2)
        self.near_risk_aux_head = head(self.set_cfg.model_dim // 2)
        residual_input_dim = feature_dim + 3
        self.candidate_residual_head = nn.Sequential(
            nn.LayerNorm(residual_input_dim),
            nn.Linear(residual_input_dim, self.set_cfg.model_dim // 2),
            nn.GELU(),
            nn.Dropout(self.set_cfg.dropout),
            nn.Linear(self.set_cfg.model_dim // 2, 1),
        )
        self.confidence_head = head(self.set_cfg.model_dim // 2)

    @property
    def architecture_record(self) -> dict[str, Any]:
        return {
            "schema_version": "r35_set_presence_raw_pair_verifier_v1",
            "set_config": asdict(self.set_cfg),
            "robust_config": asdict(self.robust_cfg),
            "present_unknown_experts_independent": True,
            "candidate_conditioned_residual": True,
            "candidate_residual_bound": self.robust_cfg.candidate_residual_bound,
            "scene_id_is_training_loss_only": True,
            "scene_id_is_inference_input": False,
            "monotonic_near_suppression_removed": True,
            "presence_threshold": 0.5,
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
        near_logit = self.near_risk_aux_head(feature).squeeze(-1)
        verifier_probability = verifier_output["probability"].masked_fill(
            ~base["candidate_mask"].bool(), -1.0
        )
        top = torch.topk(
            verifier_probability,
            k=min(2, verifier_probability.shape[1]),
            dim=-1,
        ).values
        verifier_top1 = top[:, 0]
        verifier_margin = (
            top[:, 0] - top[:, 1]
            if top.shape[1] > 1
            else top[:, 0]
        )
        # BF16 rounds 1 - 1e-6 back to exactly 1. Compute entropy in FP32 so
        # saturated verifier probabilities cannot produce 0 * log(0) NaNs.
        valid_probability = verifier_output["probability"].float().clamp(
            1e-6, 1.0 - 1e-6
        )
        entropy_terms = -(
            valid_probability * valid_probability.log()
            + (1.0 - valid_probability) * (1.0 - valid_probability).log()
        )
        valid = base["candidate_mask"].float()
        verifier_entropy = (entropy_terms * valid).sum(-1) / valid.sum(-1).clamp_min(1.0)
        residual_input = torch.cat(
            (
                feature,
                verifier_top1.unsqueeze(-1),
                verifier_margin.unsqueeze(-1),
                verifier_entropy.unsqueeze(-1),
            ),
            dim=-1,
        )
        residual = self.robust_cfg.candidate_residual_bound * torch.tanh(
            self.candidate_residual_head(residual_input).squeeze(-1)
        )
        presence_logit = present_logit - unknown_logit + residual
        presence_probability = torch.sigmoid(presence_logit)
        confidence = torch.sigmoid(self.confidence_head(feature).squeeze(-1)) * (
            2.0 * (presence_probability - 0.5).abs()
        )
        return {
            "presence_logit": presence_logit,
            "presence_probability": presence_probability,
            "unknown_probability": 1.0 - presence_probability,
            "present_expert_logit": present_logit,
            "unknown_expert_logit": unknown_logit,
            "near_wrong_logit": near_logit,
            "near_wrong_probability": torch.sigmoid(near_logit),
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


def _worst_group_cvar(
    values: torch.Tensor,
    group_ids: torch.Tensor,
    mask: torch.Tensor,
    fraction: float,
) -> torch.Tensor:
    groups = torch.unique(group_ids[mask])
    if groups.numel() == 0:
        return values.sum() * 0.0
    means = torch.stack(
        [values[mask & (group_ids == group)].mean() for group in groups]
    )
    count = max(1, int(math.ceil(means.numel() * fraction)))
    return torch.topk(means, k=count, largest=True).values.mean()


def r35_raw_pair_verifier_presence_losses(
    output: dict[str, torch.Tensor],
    *,
    target_present: torch.Tensor,
    hard_positive_set: torch.Tensor,
    ordinary_positive_set: torch.Tensor,
    near_wrong_set: torch.Tensor,
    scene_group_ids: torch.Tensor,
    verifier_output: dict[str, torch.Tensor],
    candidate_same_targets: torch.Tensor,
    hard_positive_candidate_mask: torch.Tensor,
    hard_negative_candidate_mask: torch.Tensor,
    config: R35RawPairVerifierPresenceConfig,
    weights: R35RawPairVerifierLossWeights | None = None,
) -> dict[str, torch.Tensor]:
    loss_weights = weights or R35RawPairVerifierLossWeights()
    present = target_present.bool()
    hard_positive = hard_positive_set.bool() & present
    ordinary_positive = ordinary_positive_set.bool() & present
    near_wrong = near_wrong_set.bool() & ~present
    if scene_group_ids.shape != present.shape:
        raise ValueError("scene_group_ids必须与set batch一致")
    logits = output["presence_logit"].float()
    verifier_logits = verifier_output["logit"].float()
    candidate_mask = verifier_output["candidate_mask"].bool()
    same_targets = candidate_same_targets.bool() & candidate_mask
    negatives = (~candidate_same_targets.bool()) & candidate_mask
    positive_count = same_targets.sum().clamp_min(1)
    negative_count = negatives.sum().clamp_min(1)
    verifier_positive_weight = (negative_count.float() / positive_count.float()).clamp(1.0, 20.0)
    verifier_classification = F.binary_cross_entropy_with_logits(
        verifier_logits[candidate_mask],
        candidate_same_targets.float()[candidate_mask],
        pos_weight=verifier_positive_weight,
    )
    hard_positive_mask = hard_positive_candidate_mask.bool() & same_targets
    verifier_hard_positive = (
        F.softplus(1.5 - verifier_logits[hard_positive_mask]).mean()
        if hard_positive_mask.any()
        else verifier_logits.sum() * 0.0
    )
    hard_negative_mask = hard_negative_candidate_mask.bool() & negatives
    verifier_hard_negative = (
        F.softplus(verifier_logits[hard_negative_mask] + 1.0).mean()
        if hard_negative_mask.any()
        else verifier_logits.sum() * 0.0
    )
    positive_by_set = verifier_logits.masked_fill(~same_targets, -20.0).max(dim=-1).values
    negative_by_set = verifier_logits.masked_fill(~negatives, -20.0).max(dim=-1).values
    sets_with_target = same_targets.any(dim=-1)
    verifier_ranking = (
        F.relu(1.0 - positive_by_set[sets_with_target] + negative_by_set[sets_with_target]).mean()
        if sets_with_target.any()
        else verifier_logits.sum() * 0.0
    )
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
    worst_group_fn = _worst_group_cvar(
        per_set_fn, scene_group_ids, present, config.worst_group_cvar_fraction
    )
    worst_group_far = _worst_group_cvar(
        per_set_far, scene_group_ids, near_wrong, config.worst_group_cvar_fraction
    )
    present_expert = output["present_expert_logit"].float()
    unknown_expert = output["unknown_expert_logit"].float()
    expert_specialization = logits.sum() * 0.0
    if present.any():
        expert_specialization = expert_specialization + F.softplus(
            -present_expert[present]
        ).mean()
    if (~present).any():
        expert_specialization = expert_specialization + F.softplus(
            -unknown_expert[~present]
        ).mean()
    residual_regularization = output["candidate_conditioned_residual"].square().mean()
    probability = output["presence_probability"].float()
    with torch.no_grad():
        correct = ((probability >= 0.5) == present).float()
        confidence_target = 2.0 * (probability - 0.5).abs() * correct
    confidence_calibration = ((output["confidence"] - confidence_target) ** 2).mean()
    total = (
        loss_weights.candidate_verifier_classification * verifier_classification
        + loss_weights.candidate_verifier_hard_positive * verifier_hard_positive
        + loss_weights.candidate_verifier_hard_negative * verifier_hard_negative
        + loss_weights.candidate_verifier_ranking * verifier_ranking
        + loss_weights.present_unknown_classification * classification
        + loss_weights.positive_preserving * positive_preserving
        + loss_weights.near_wrong_classification * near_classification
        + loss_weights.hard_positive_near_ranking * ranking
        + loss_weights.worst_group_present_false_negative * worst_group_fn
        + loss_weights.worst_group_near_false_accept * worst_group_far
        + loss_weights.expert_specialization * expert_specialization
        + loss_weights.candidate_residual_regularization * residual_regularization
        + loss_weights.confidence_calibration * confidence_calibration
    )
    return {
        "loss": total,
        "candidate_verifier_classification": verifier_classification,
        "candidate_verifier_hard_positive": verifier_hard_positive,
        "candidate_verifier_hard_negative": verifier_hard_negative,
        "candidate_verifier_ranking": verifier_ranking,
        "present_unknown_classification": classification,
        "positive_preserving": positive_preserving,
        "near_wrong_classification": near_classification,
        "hard_positive_near_ranking": ranking,
        "worst_group_present_false_negative": worst_group_fn,
        "worst_group_near_false_accept": worst_group_far,
        "expert_specialization": expert_specialization,
        "candidate_residual_regularization": residual_regularization,
        "confidence_calibration": confidence_calibration,
    }


class R35Stage2Revision4Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.candidate_match_head = R35CandidateMatchHead(R35CandidateMatchConfig())
        self.raw_pair_candidate_verifier = R35RawPairCandidateVerifier()
        self.set_presence_head = R35RawPairVerifierPresenceHead()
        for parameter in self.candidate_match_head.parameters():
            parameter.requires_grad_(False)

    @property
    def architecture_record(self) -> dict[str, Any]:
        return {
            "schema_version": "r35_track_c_stage2_revision4_model_v1",
            "candidate_match_frozen": True,
            "raw_pair_candidate_verifier_trainable": True,
            "raw_pair_feature_dimensions": [256, 17],
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
        verifier = self.raw_pair_candidate_verifier(
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
            verifier_output=verifier,
        )
        return {
            "candidate_match": candidate,
            "candidate_verifier": verifier,
            "set_presence": presence,
        }


def initialize_revision4_from_checkpoint(
    model: R35Stage2Revision4Model,
    payload: dict[str, Any],
) -> dict[str, Any]:
    if payload.get("schema_version") != "r35_stage2_checkpoint_v4":
        raise RuntimeError("revision 4只接受完整Stage 2 revision3 checkpoint")
    model.candidate_match_head.load_state_dict(payload["candidate_match_head"], strict=True)
    source = payload["set_presence_head"]
    current = model.set_presence_head.evidence_encoder.state_dict()
    transferred = {
        name[len("evidence_encoder."):]: value
        for name, value in source.items()
        if name.startswith("evidence_encoder.")
        and name[len("evidence_encoder."):] in current
        and current[name[len("evidence_encoder."):]].shape == value.shape
    }
    current.update(transferred)
    model.set_presence_head.evidence_encoder.load_state_dict(current, strict=True)
    required = (
        "candidate_projection.",
        "set_encoder.",
        "competition_projection.",
        "unknown_token",
    )
    if not all(any(name.startswith(prefix) for name in transferred) for prefix in required):
        raise RuntimeError("revision 4 evidence encoder迁移不完整")
    copied_heads = {}
    for source_prefix, target_name, target_head in (
        (
            "present_expert.",
            "present_expert",
            model.set_presence_head.present_expert,
        ),
        ("unknown_expert.", "unknown_expert", model.set_presence_head.unknown_expert),
        (
            "near_risk_aux_head.",
            "near_risk_aux_head",
            model.set_presence_head.near_risk_aux_head,
        ),
        ("confidence_head.", "confidence_head", model.set_presence_head.confidence_head),
    ):
        target_state = target_head.state_dict()
        compatible = {
            name[len(source_prefix):]: value
            for name, value in source.items()
            if name.startswith(source_prefix)
            and name[len(source_prefix):] in target_state
            and target_state[name[len(source_prefix):]].shape == value.shape
        }
        target_state.update(compatible)
        target_head.load_state_dict(target_state, strict=True)
        copied_heads[target_name] = {
            "source_prefix": source_prefix,
            "transferred_keys": sorted(compatible),
        }
    return {
        "schema_version": "r35_stage2_revision4_initialization_v1",
        "candidate_match_loaded_strict_and_frozen": True,
        "evidence_encoder_transferred_keys": sorted(transferred),
        "compatible_expert_head_transfers": copied_heads,
        "new_heads_randomly_initialized": [
            "raw_pair_candidate_verifier",
        ],
        "optimizer_state_reused": False,
        "presence_threshold": 0.5,
        "test_r32_confirmation_accessed": False,
    }
