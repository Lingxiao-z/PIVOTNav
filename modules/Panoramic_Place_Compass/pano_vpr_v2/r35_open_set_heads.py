from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class R35CandidateMatchConfig:
    spatial_pair_dim: int = 256
    scalar_feature_dim: int = 17
    model_dim: int = 192
    dropout: float = 0.1
    ranking_margin: float = 0.20
    hard_negative_margin: float = 0.50
    hard_positive_target_probability: float = 0.90


@dataclass(frozen=True)
class R35CandidateLossWeights:
    same_place_classification: float = 1.0
    candidate_ranking: float = 0.50
    hard_positive_false_negative: float = 1.25
    hard_negative_margin: float = 0.75
    calibration: float = 0.10


@dataclass(frozen=True)
class R35SetPresenceConfig:
    candidate_token_dim: int = 192
    model_dim: int = 192
    attention_heads: int = 4
    attention_layers: int = 2
    feedforward_dim: int = 384
    dropout: float = 0.1
    scalar_summary_dim: int = 22
    ordinary_positive_target: float = 0.85
    hard_positive_target: float = 0.90
    near_wrong_target: float = 0.20
    ordinary_to_near_margin: float = 0.50
    hard_positive_to_near_margin: float = 1.00
    near_risk_suppression_scale: float = 1.50


@dataclass(frozen=True)
class R35SetPresenceLossWeights:
    present_unknown_classification: float = 1.0
    set_level_ranking: float = 0.50
    confidence_calibration: float = 0.10
    positive_preserving_false_negative: float = 1.25
    near_wrong_classification: float = 0.75
    near_wrong_false_accept: float = 1.00
    hard_positive_near_ranking: float = 0.75
    selected_candidate: float = 0.25


class R35CandidateMatchHead(nn.Module):
    """Per-candidate same-place verifier, independent of set-level UNKNOWN."""

    def __init__(self, cfg: R35CandidateMatchConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or R35CandidateMatchConfig()
        self.spatial_projection = nn.Sequential(
            nn.LayerNorm(self.cfg.spatial_pair_dim),
            nn.Linear(self.cfg.spatial_pair_dim, self.cfg.model_dim),
            nn.GELU(),
        )
        self.scalar_projection = nn.Sequential(
            nn.LayerNorm(self.cfg.scalar_feature_dim),
            nn.Linear(self.cfg.scalar_feature_dim, self.cfg.model_dim),
            nn.GELU(),
        )
        self.fusion = nn.Sequential(
            nn.LayerNorm(2 * self.cfg.model_dim),
            nn.Linear(2 * self.cfg.model_dim, self.cfg.model_dim),
            nn.GELU(),
            nn.Dropout(self.cfg.dropout),
            nn.Linear(self.cfg.model_dim, self.cfg.model_dim),
            nn.GELU(),
        )
        self.same_place_head = nn.Linear(self.cfg.model_dim, 1)
        self.quality_head = nn.Linear(self.cfg.model_dim, 1)
        self.uncertainty_head = nn.Linear(self.cfg.model_dim, 1)
        self.ranking_head = nn.Linear(self.cfg.model_dim, 1)

    @property
    def architecture_record(self) -> Dict[str, Any]:
        return {
            "schema_version": "r35_candidate_match_head_v1",
            "config": asdict(self.cfg),
            "scope": "one decision per query-candidate pair; no UNKNOWN decision",
            "required_spatial_input": "concatenated yaw and bearing spatial pair embeddings",
            "scalar_feature_order": [
                "global_descriptor_similarity",
                "sector_match_max",
                "sector_match_mean",
                "sector_match_margin",
                "sector_match_entropy",
                "yaw_probability",
                "yaw_confidence",
                "yaw_entropy",
                "bearing_confidence",
                "bearing_valid_probability",
                "bearing_entropy",
                "local_correlation_max",
                "local_correlation_mean",
                "local_correlation_margin",
                "local_correlation_entropy",
                "normalized_rank",
                "reciprocal_rank_score",
            ],
            "outputs": [
                "same_place_probability",
                "candidate_quality",
                "pair_uncertainty",
                "candidate_ranking_score",
                "candidate_token",
            ],
        }

    def forward(
        self,
        spatial_pair_features: torch.Tensor,
        scalar_features: torch.Tensor,
        candidate_mask: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        if spatial_pair_features.ndim != 3:
            raise ValueError("spatial_pair_features must be [B,K,D]")
        batch, candidate_count, spatial_dim = spatial_pair_features.shape
        if spatial_dim != self.cfg.spatial_pair_dim:
            raise ValueError(f"expected spatial pair dim {self.cfg.spatial_pair_dim}")
        if scalar_features.shape != (batch, candidate_count, self.cfg.scalar_feature_dim):
            raise ValueError(
                f"scalar_features must be [B,K,{self.cfg.scalar_feature_dim}]"
            )
        if candidate_mask is None:
            candidate_mask = torch.ones(
                batch,
                candidate_count,
                dtype=torch.bool,
                device=spatial_pair_features.device,
            )
        if candidate_mask.shape != (batch, candidate_count):
            raise ValueError("candidate_mask must be [B,K]")
        spatial = self.spatial_projection(spatial_pair_features.float())
        scalar = self.scalar_projection(scalar_features.float())
        token = self.fusion(torch.cat((spatial, scalar), dim=-1))
        same_place_logit = self.same_place_head(token).squeeze(-1)
        quality_logit = self.quality_head(token).squeeze(-1)
        uncertainty_logit = self.uncertainty_head(token).squeeze(-1)
        ranking_score = self.ranking_head(token).squeeze(-1)
        invalid_fill = torch.full_like(same_place_logit, -1.0e4)
        return {
            "same_place_logit": torch.where(candidate_mask, same_place_logit, invalid_fill),
            "same_place_probability": torch.sigmoid(same_place_logit) * candidate_mask.float(),
            "candidate_quality": torch.sigmoid(quality_logit) * candidate_mask.float(),
            "pair_uncertainty": torch.sigmoid(uncertainty_logit) * candidate_mask.float(),
            "candidate_ranking_score": torch.where(candidate_mask, ranking_score, invalid_fill),
            "candidate_token": token,
            "candidate_mask": candidate_mask,
        }


def r35_candidate_match_losses(
    output: Dict[str, torch.Tensor],
    candidate_same_place: torch.Tensor,
    hard_positive_mask: torch.Tensor,
    hard_negative_mask: torch.Tensor,
    *,
    cfg: R35CandidateMatchConfig,
    weights: R35CandidateLossWeights | None = None,
) -> Dict[str, torch.Tensor]:
    loss_weights = weights or R35CandidateLossWeights()
    mask = output["candidate_mask"].bool()
    if candidate_same_place.shape != mask.shape:
        raise ValueError("candidate_same_place must match candidate mask")
    if hard_positive_mask.shape != mask.shape or hard_negative_mask.shape != mask.shape:
        raise ValueError("hard positive/negative masks must match candidate mask")
    target = candidate_same_place.float()
    logits = output["same_place_logit"].float()
    positive_count = (mask & candidate_same_place.bool()).sum().clamp_min(1)
    negative_count = (mask & ~candidate_same_place.bool()).sum().clamp_min(1)
    positive_weight = (negative_count.float() / positive_count.float()).clamp(1.0, 10.0)
    classification_raw = F.binary_cross_entropy_with_logits(
        logits,
        target,
        pos_weight=positive_weight,
        reduction="none",
    )
    same_place_classification = (classification_raw * mask.float()).sum() / mask.sum().clamp_min(1)

    ranking_terms: list[torch.Tensor] = []
    ranking_score = output["candidate_ranking_score"].float()
    for row in range(mask.shape[0]):
        positives = ranking_score[row][mask[row] & candidate_same_place[row].bool()]
        negatives = ranking_score[row][mask[row] & ~candidate_same_place[row].bool()]
        if positives.numel() and negatives.numel():
            ranking_terms.append(
                F.relu(cfg.ranking_margin - positives[:, None] + negatives[None, :]).mean()
            )
    candidate_ranking = (
        torch.stack(ranking_terms).mean()
        if ranking_terms
        else ranking_score.sum() * 0.0
    )

    hard_positive = mask & hard_positive_mask.bool() & candidate_same_place.bool()
    hard_positive_target_logit = torch.logit(
        torch.tensor(
            cfg.hard_positive_target_probability,
            device=logits.device,
            dtype=logits.dtype,
        )
    )
    hard_positive_false_negative = (
        F.relu(hard_positive_target_logit - logits[hard_positive]).mean()
        if hard_positive.any()
        else logits.sum() * 0.0
    )
    hard_negative = mask & hard_negative_mask.bool() & ~candidate_same_place.bool()
    hard_negative_margin = (
        F.relu(logits[hard_negative] + cfg.hard_negative_margin).mean()
        if hard_negative.any()
        else logits.sum() * 0.0
    )
    probability = output["same_place_probability"].float()
    calibration = (((probability - target) ** 2) * mask.float()).sum() / mask.sum().clamp_min(1)
    total = (
        loss_weights.same_place_classification * same_place_classification
        + loss_weights.candidate_ranking * candidate_ranking
        + loss_weights.hard_positive_false_negative * hard_positive_false_negative
        + loss_weights.hard_negative_margin * hard_negative_margin
        + loss_weights.calibration * calibration
    )
    return {
        "loss": total,
        "same_place_classification": same_place_classification,
        "candidate_ranking": candidate_ranking,
        "hard_positive_false_negative": hard_positive_false_negative,
        "hard_negative_margin": hard_negative_margin,
        "calibration": calibration,
    }


class R35SetPresenceHead(nn.Module):
    """Top-K set-level presence/UNKNOWN decision, separate from candidate matching."""

    def __init__(self, cfg: R35SetPresenceConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or R35SetPresenceConfig()
        self.candidate_projection = nn.Sequential(
            nn.LayerNorm(self.cfg.candidate_token_dim + 7),
            nn.Linear(self.cfg.candidate_token_dim + 7, self.cfg.model_dim),
            nn.GELU(),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=self.cfg.model_dim,
            nhead=self.cfg.attention_heads,
            dim_feedforward=self.cfg.feedforward_dim,
            dropout=self.cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.set_encoder = nn.TransformerEncoder(layer, self.cfg.attention_layers)
        self.unknown_token = nn.Parameter(torch.zeros(1, 1, self.cfg.model_dim))
        self.competition_projection = nn.Sequential(
            nn.LayerNorm(4 * self.cfg.model_dim),
            nn.Linear(4 * self.cfg.model_dim, self.cfg.model_dim),
            nn.GELU(),
            nn.Dropout(self.cfg.dropout),
        )
        set_feature_dim = 3 * self.cfg.model_dim + self.cfg.scalar_summary_dim
        self.presence_head = nn.Sequential(
            nn.LayerNorm(set_feature_dim),
            nn.Linear(set_feature_dim, self.cfg.model_dim),
            nn.GELU(),
            nn.Dropout(self.cfg.dropout),
            nn.Linear(self.cfg.model_dim, 1),
        )
        self.near_wrong_head = nn.Sequential(
            nn.LayerNorm(set_feature_dim),
            nn.Linear(set_feature_dim, self.cfg.model_dim // 2),
            nn.GELU(),
            nn.Dropout(self.cfg.dropout),
            nn.Linear(self.cfg.model_dim // 2, 1),
        )
        self.confidence_head = nn.Sequential(
            nn.LayerNorm(set_feature_dim),
            nn.Linear(set_feature_dim, 1),
        )
        nn.init.normal_(self.unknown_token, std=0.02)

    @property
    def architecture_record(self) -> Dict[str, Any]:
        return {
            "schema_version": "r35_set_presence_unknown_head_v2",
            "config": asdict(self.cfg),
            "scope": "binary presence/UNKNOWN over the entire Top-K set",
            "candidate_selection": "argmax of independent Candidate Match probability, not a joint candidate-plus-UNKNOWN softmax",
            "set_inputs": [
                "Top-1 same-place probability",
                "Top-1 minus Top-2 margin",
                "Top-K score distribution and entropy",
                "candidate quality and uncertainty",
                "yaw confidence and set consistency",
                "bearing confidence and set consistency",
                "candidate rank",
                "global descriptor query-set relative statistics",
                "explicit Top-1/Top-2 competition token",
                "dedicated near-but-wrong risk",
                "learned UNKNOWN token",
            ],
            "outputs": [
                "presence_probability",
                "unknown_probability",
                "selected_candidate",
                "confidence",
                "near_wrong_probability",
            ],
        }

    @staticmethod
    def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return (values * mask.float()).sum(dim=1) / mask.sum(dim=1).clamp_min(1)

    @staticmethod
    def _masked_std(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mean = R35SetPresenceHead._masked_mean(values, mask)
        variance = R35SetPresenceHead._masked_mean((values - mean[:, None]) ** 2, mask)
        # sqrt has an unbounded derivative at exactly zero. Cached Stage-2
        # features hide that singularity, but it produces NaN backbone
        # gradients once Stage 3 propagates through set-level statistics.
        return variance.clamp_min(1.0e-12).sqrt()

    def forward(
        self,
        candidate_output: Dict[str, torch.Tensor],
        global_similarity: torch.Tensor,
        yaw_confidence: torch.Tensor,
        bearing_confidence: torch.Tensor,
        normalized_rank: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        candidate_token = candidate_output["candidate_token"]
        mask = candidate_output["candidate_mask"].bool()
        batch, candidate_count, _ = candidate_token.shape
        if candidate_count < 1 or not mask.any(dim=1).all():
            raise ValueError("every set must contain at least one valid candidate")
        expected = (batch, candidate_count)
        for name, value in (
            ("global_similarity", global_similarity),
            ("yaw_confidence", yaw_confidence),
            ("bearing_confidence", bearing_confidence),
            ("normalized_rank", normalized_rank),
        ):
            if value.shape != expected:
                raise ValueError(f"{name} must be [B,K]")
        same = candidate_output["same_place_probability"].float()
        quality = candidate_output["candidate_quality"].float()
        uncertainty = candidate_output["pair_uncertainty"].float()
        per_candidate_scalar = torch.stack(
            (
                same,
                quality,
                uncertainty,
                global_similarity.float(),
                yaw_confidence.float(),
                bearing_confidence.float(),
                normalized_rank.float(),
            ),
            dim=-1,
        )
        tokens = self.candidate_projection(
            torch.cat((candidate_token.float(), per_candidate_scalar), dim=-1)
        )
        unknown = self.unknown_token.expand(batch, -1, -1)
        encoded = self.set_encoder(
            torch.cat((tokens, unknown), dim=1),
            src_key_padding_mask=torch.cat(
                (~mask, torch.zeros(batch, 1, dtype=torch.bool, device=mask.device)),
                dim=1,
            ),
        )
        candidate_encoded = encoded[:, :candidate_count]
        unknown_encoded = encoded[:, candidate_count]
        pooled = (
            (candidate_encoded * mask.unsqueeze(-1).float()).sum(dim=1)
            / mask.sum(dim=1, keepdim=True).clamp_min(1)
        )

        masked_same = same.masked_fill(~mask, -1.0)
        top_values, top_indices = torch.topk(masked_same, k=min(2, candidate_count), dim=-1)
        top1 = top_values[:, 0]
        top2 = top_values[:, 1] if candidate_count > 1 else torch.zeros_like(top1)
        margin = top1 - top2
        gather_index = top_indices[:, : min(2, candidate_count)].unsqueeze(-1).expand(
            -1, -1, candidate_encoded.shape[-1]
        )
        top_encoded = candidate_encoded.gather(1, gather_index)
        top1_encoded = top_encoded[:, 0]
        top2_encoded = (
            top_encoded[:, 1]
            if candidate_count > 1
            else torch.zeros_like(top1_encoded)
        )
        competition = self.competition_projection(
            torch.cat(
                (
                    top1_encoded,
                    top2_encoded,
                    top1_encoded - top2_encoded,
                    top1_encoded * top2_encoded,
                ),
                dim=-1,
            )
        )
        normalized_scores = torch.softmax(masked_same.masked_fill(~mask, -1.0e4), dim=-1)
        score_entropy = -(
            normalized_scores * normalized_scores.clamp_min(1e-8).log()
        ).sum(dim=-1) / mask.sum(dim=-1).float().clamp_min(2.0).log()
        summary = torch.stack(
            (
                top1,
                margin,
                score_entropy,
                self._masked_mean(same, mask),
                self._masked_std(same, mask),
                self._masked_mean(quality, mask),
                self._masked_mean(uncertainty, mask),
                self._masked_mean(global_similarity.float(), mask),
                self._masked_std(global_similarity.float(), mask),
                global_similarity.float().masked_fill(~mask, -1.0e4).max(dim=-1).values,
                self._masked_mean(yaw_confidence.float(), mask),
                self._masked_std(yaw_confidence.float(), mask),
                self._masked_mean(bearing_confidence.float(), mask),
                self._masked_std(bearing_confidence.float(), mask),
                self._masked_mean(normalized_rank.float(), mask),
                mask.sum(dim=-1).float() / float(candidate_count),
                self._masked_mean(quality - uncertainty, mask),
                self._masked_std(quality - uncertainty, mask),
                quality.masked_fill(~mask, -1.0e4).max(dim=-1).values,
                uncertainty.masked_fill(~mask, -1.0e4).max(dim=-1).values,
                self._masked_mean(yaw_confidence.float() * bearing_confidence.float(), mask),
                self._masked_std(yaw_confidence.float() * bearing_confidence.float(), mask),
            ),
            dim=-1,
        )
        if summary.shape[-1] != self.cfg.scalar_summary_dim:
            raise RuntimeError("set summary dimension does not match config")
        set_feature = torch.cat((unknown_encoded, pooled, competition, summary), dim=-1)
        presence_evidence_logit = self.presence_head(set_feature).squeeze(-1)
        near_wrong_logit = self.near_wrong_head(set_feature).squeeze(-1)
        near_wrong_probability = torch.sigmoid(near_wrong_logit)
        presence_logit = (
            presence_evidence_logit
            - self.cfg.near_risk_suppression_scale * near_wrong_probability
        )
        presence_probability = torch.sigmoid(presence_logit)
        confidence_logit = self.confidence_head(set_feature).squeeze(-1)
        confidence = torch.sigmoid(confidence_logit) * (2.0 * (presence_probability - 0.5).abs())
        selected_candidate = top_indices[:, 0]
        return {
            "presence_logit": presence_logit,
            "presence_evidence_logit": presence_evidence_logit,
            "presence_probability": presence_probability,
            "unknown_probability": 1.0 - presence_probability,
            "selected_candidate": selected_candidate,
            "selected_candidate_probability": top1,
            "candidate_probability_margin": margin,
            "confidence_logit": confidence_logit,
            "confidence": confidence,
            "near_wrong_logit": near_wrong_logit,
            "near_wrong_probability": near_wrong_probability,
            "set_summary": summary,
            "encoded_candidate_tokens": candidate_encoded,
            "encoded_unknown_token": unknown_encoded,
            "candidate_mask": mask,
        }


def r35_set_presence_losses(
    output: Dict[str, torch.Tensor],
    target_present: torch.Tensor,
    target_candidate_index: torch.Tensor,
    candidate_same_place: torch.Tensor,
    hard_positive_set: torch.Tensor,
    ordinary_positive_set: torch.Tensor,
    near_wrong_set: torch.Tensor,
    *,
    cfg: R35SetPresenceConfig,
    weights: R35SetPresenceLossWeights | None = None,
) -> Dict[str, torch.Tensor]:
    loss_weights = weights or R35SetPresenceLossWeights()
    present = target_present.bool()
    hard_positive = hard_positive_set.bool() & present
    ordinary_positive = ordinary_positive_set.bool() & present
    near_wrong = near_wrong_set.bool() & ~present
    expected_shape = present.shape
    if any(
        value.shape != expected_shape
        for value in (hard_positive, ordinary_positive, near_wrong)
    ):
        raise ValueError("set-level category masks must match target_present")
    logits = output["presence_logit"].float()
    positive_count = present.sum().clamp_min(1)
    negative_count = (~present).sum().clamp_min(1)
    positive_weight = (negative_count.float() / positive_count.float()).clamp(1.0, 10.0)
    present_unknown_classification = F.binary_cross_entropy_with_logits(
        logits,
        present.float(),
        pos_weight=positive_weight,
    )

    ranking_terms: list[torch.Tensor] = []
    near_logits = logits[near_wrong]
    if near_logits.numel() and logits[ordinary_positive].numel():
        ranking_terms.append(
            F.relu(
                cfg.ordinary_to_near_margin
                - logits[ordinary_positive][:, None]
                + near_logits[None, :]
            ).mean()
        )
    if near_logits.numel() and logits[hard_positive].numel():
        ranking_terms.append(
            F.relu(
                cfg.hard_positive_to_near_margin
                - logits[hard_positive][:, None]
                + near_logits[None, :]
            ).mean()
        )
    set_level_ranking = (
        torch.stack(ranking_terms).mean()
        if ranking_terms
        else logits.sum() * 0.0
    )
    probability = output["presence_probability"].float()
    presence_probability_calibration = (
        (probability - present.float()) ** 2
    ).mean()
    # Confidence is confidence in the current binary decision, not another
    # presence probability. Correct decisions should approach their available
    # probability margin; incorrect decisions should report zero confidence.
    with torch.no_grad():
        decision_correct = (probability >= 0.5) == present
        confidence_target = (
            2.0 * (probability - 0.5).abs() * decision_correct.float()
        )
    decision_confidence_calibration = (
        (output["confidence"].float() - confidence_target) ** 2
    ).mean()
    confidence_calibration = (
        presence_probability_calibration + decision_confidence_calibration
    )
    positive_terms: list[torch.Tensor] = []
    for category_mask, target_probability in (
        (ordinary_positive, cfg.ordinary_positive_target),
        (hard_positive, cfg.hard_positive_target),
    ):
        if category_mask.any():
            target_logit = torch.logit(
                torch.tensor(
                    target_probability,
                    device=logits.device,
                    dtype=logits.dtype,
                )
            )
            positive_terms.append(F.relu(target_logit - logits[category_mask]).mean())
    positive_preserving_false_negative = (
        torch.stack(positive_terms).mean()
        if positive_terms
        else logits.sum() * 0.0
    )

    near_logits_aux = output["near_wrong_logit"].float()
    near_count = near_wrong.sum().clamp_min(1)
    not_near_count = (~near_wrong).sum().clamp_min(1)
    near_positive_weight = (not_near_count.float() / near_count.float()).clamp(1.0, 10.0)
    near_wrong_classification = F.binary_cross_entropy_with_logits(
        near_logits_aux,
        near_wrong.float(),
        pos_weight=near_positive_weight,
    )
    near_target_logit = torch.logit(
        torch.tensor(
            cfg.near_wrong_target,
            device=logits.device,
            dtype=logits.dtype,
        )
    )
    near_wrong_false_accept = (
        F.relu(logits[near_wrong] - near_target_logit).mean()
        if near_wrong.any()
        else logits.sum() * 0.0
    )
    hard_positive_near_ranking = (
        F.relu(
            cfg.hard_positive_to_near_margin
            - logits[hard_positive][:, None]
            + logits[near_wrong][None, :]
        ).mean()
        if hard_positive.any() and near_wrong.any()
        else logits.sum() * 0.0
    )

    candidate_probabilities = candidate_same_place.float().clamp_min(1e-8)
    candidate_log_probabilities = (
        candidate_probabilities / candidate_probabilities.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    ).log()
    valid_target = present & (target_candidate_index >= 0)
    selected_candidate = (
        F.nll_loss(
            candidate_log_probabilities[valid_target],
            target_candidate_index[valid_target].long(),
        )
        if valid_target.any()
        else logits.sum() * 0.0
    )
    total = (
        loss_weights.present_unknown_classification * present_unknown_classification
        + loss_weights.set_level_ranking * set_level_ranking
        + loss_weights.confidence_calibration * confidence_calibration
        + loss_weights.positive_preserving_false_negative * positive_preserving_false_negative
        + loss_weights.near_wrong_classification * near_wrong_classification
        + loss_weights.near_wrong_false_accept * near_wrong_false_accept
        + loss_weights.hard_positive_near_ranking * hard_positive_near_ranking
        + loss_weights.selected_candidate * selected_candidate
    )
    return {
        "loss": total,
        "present_unknown_classification": present_unknown_classification,
        "set_level_ranking": set_level_ranking,
        "confidence_calibration": confidence_calibration,
        "presence_probability_calibration": presence_probability_calibration,
        "decision_confidence_calibration": decision_confidence_calibration,
        "positive_preserving_false_negative": positive_preserving_false_negative,
        "near_wrong_classification": near_wrong_classification,
        "near_wrong_false_accept": near_wrong_false_accept,
        "hard_positive_near_ranking": hard_positive_near_ranking,
        "selected_candidate": selected_candidate,
        "present_false_negative_rate_at_0_5": (
            (probability[present] < 0.5).float().mean()
            if present.any()
            else logits.sum() * 0.0
        ),
    }
