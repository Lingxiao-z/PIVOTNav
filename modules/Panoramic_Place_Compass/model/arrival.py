"""Arrival and open-set heads preserved by the released checkpoint."""
from __future__ import annotations
from .bearing import R35RelativeTranslationBearingHead
from .core import PanoramicVPRV2System


# ---------------------------------------------------------------------------
# Temporal arrival head
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Open-set candidate heads
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Candidate verifier
# ---------------------------------------------------------------------------

from dataclasses import asdict, dataclass
import math
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F



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


# ---------------------------------------------------------------------------
# Shared multitask checkpoint system
# ---------------------------------------------------------------------------

import math
from enum import Enum
from typing import Any, Dict

import torch
from torch import nn
from torch.nn import functional as F



class R35TrainingStage(str, Enum):
    BEARING_HEAD = "stage_1_bearing_head"
    OPEN_SET_HEADS = "stage_2_candidate_and_presence_heads"
    LIMITED_JOINT = "stage_3_limited_joint_finetuning"


class R35MultitaskSystem(nn.Module):
    """Shared VPR encoder with frozen Track Y and independent R35 task heads."""

    def __init__(
        self,
        encoder_system: PanoramicVPRV2System,
        frozen_track_y: nn.Module,
        bearing_head: R35RelativeTranslationBearingHead | None = None,
        candidate_match_head: R35CandidateMatchHead | None = None,
        set_presence_head: R35SetPresenceHead | None = None,
        raw_pair_candidate_verifier: R35RawPairCandidateVerifier | None = None,
    ) -> None:
        super().__init__()
        self.encoder_system = encoder_system
        self.track_y = frozen_track_y
        self.bearing_head = bearing_head or R35RelativeTranslationBearingHead()
        self.candidate_match_head = candidate_match_head or R35CandidateMatchHead()
        self.set_presence_head = set_presence_head or R35SetPresenceHead()
        self.raw_pair_candidate_verifier = raw_pair_candidate_verifier
        self._training_stage: R35TrainingStage | None = None
        self._stage3_unfreeze_blocks = 0
        for parameter in self.track_y.parameters():
            parameter.requires_grad_(False)

    @property
    def architecture_record(self) -> Dict[str, Any]:
        return {
            "schema_version": "r35_shared_backbone_multitask_system_v1",
            "shared_encoder": "Track-Y-r3-adapted DINOv2-S/14 plus SALAD descriptor head",
            "heads": {
                "vpr_global_descriptor": "existing frozen-compatible 2112-dimensional SALAD output",
                "relative_yaw": "frozen Track Y-r3",
                "relative_translation_bearing": self.bearing_head.architecture_record,
                "candidate_match": self.candidate_match_head.architecture_record,
                "set_presence_unknown": self.set_presence_head.architecture_record,
                "raw_pair_candidate_verifier": (
                    self.raw_pair_candidate_verifier.__class__.__name__
                    if self.raw_pair_candidate_verifier is not None
                    else None
                ),
            },
            "training_stages": {
                R35TrainingStage.BEARING_HEAD.value: (
                    "shared encoder and Track Y frozen; bearing prediction and calibration layers only; "
                    "the Track-C-only pair embedding adapter remains frozen until limited joint tuning"
                ),
                R35TrainingStage.OPEN_SET_HEADS.value: "shared encoder, Track Y and bearing head frozen; Candidate Match and Set Presence only",
                R35TrainingStage.LIMITED_JOINT.value: "bearing/open-set heads plus only the final one or two DINO blocks; VPR distillation required externally",
            },
            "single_shared_backbone": True,
            "test_time_gt_inputs": False,
        }

    @staticmethod
    def _set_trainable(module: nn.Module, trainable: bool) -> None:
        for parameter in module.parameters():
            parameter.requires_grad_(trainable)

    def _freeze_obsolete_presence_heads(self) -> None:
        evidence_encoder = getattr(self.set_presence_head, "evidence_encoder", None)
        if evidence_encoder is None:
            return
        for name in ("presence_head", "near_wrong_head", "confidence_head"):
            obsolete = getattr(evidence_encoder, name, None)
            if obsolete is not None:
                self._set_trainable(obsolete, False)

    def configure_training_stage(
        self,
        stage: R35TrainingStage | str,
        *,
        stage3_unfreeze_blocks: int = 2,
    ) -> Dict[str, Any]:
        selected = R35TrainingStage(stage)
        if selected is R35TrainingStage.LIMITED_JOINT and stage3_unfreeze_blocks not in (1, 2):
            raise ValueError("Stage 3 may unfreeze only the final one or two DINO blocks")
        self._training_stage = selected
        self._stage3_unfreeze_blocks = 0
        self._set_trainable(self.encoder_system, False)
        self._set_trainable(self.track_y, False)
        self._set_trainable(self.bearing_head, False)
        self._set_trainable(self.candidate_match_head, False)
        self._set_trainable(self.set_presence_head, False)
        if self.raw_pair_candidate_verifier is not None:
            self._set_trainable(self.raw_pair_candidate_verifier, False)
        if selected is R35TrainingStage.BEARING_HEAD:
            self._set_trainable(self.bearing_head, True)
            # pair_embedding is not consumed by any Stage-1 bearing loss. Keeping
            # it trainable would silently apply only AdamW decay and violates the
            # declared DDP trainable scope.
            self._set_trainable(self.bearing_head.pair_embedding, False)
        elif selected is R35TrainingStage.OPEN_SET_HEADS:
            self._set_trainable(self.candidate_match_head, True)
            self._set_trainable(self.set_presence_head, True)
            if self.raw_pair_candidate_verifier is not None:
                self._set_trainable(self.raw_pair_candidate_verifier, True)
        else:
            self.encoder_system.backbone.unfreeze_last_blocks(stage3_unfreeze_blocks)
            self._stage3_unfreeze_blocks = stage3_unfreeze_blocks
            self._set_trainable(self.bearing_head, True)
            self._set_trainable(self.candidate_match_head, True)
            self._set_trainable(self.set_presence_head, True)
            if self.raw_pair_candidate_verifier is not None:
                self._set_trainable(self.raw_pair_candidate_verifier, True)
        self._freeze_obsolete_presence_heads()
        audit = self.trainable_scope_audit()
        if audit["track_y_trainable_parameters"] != 0:
            raise RuntimeError("Track Y must remain frozen in every R35 stage")
        if selected is not R35TrainingStage.LIMITED_JOINT and audit["encoder_trainable_parameters"] != 0:
            raise RuntimeError("shared encoder must be frozen in R35 Stage 1/2")
        return audit

    def trainable_scope_audit(self) -> Dict[str, Any]:
        count = lambda module: sum(
            parameter.numel() for parameter in module.parameters() if parameter.requires_grad
        )
        return {
            "training_stage": self._training_stage.value if self._training_stage else None,
            "stage3_unfreeze_blocks": self._stage3_unfreeze_blocks,
            "encoder_trainable_parameters": count(self.encoder_system),
            "track_y_trainable_parameters": count(self.track_y),
            "bearing_head_trainable_parameters": count(self.bearing_head),
            "candidate_match_trainable_parameters": count(self.candidate_match_head),
            "set_presence_trainable_parameters": count(self.set_presence_head),
            "raw_pair_candidate_verifier_trainable_parameters": (
                count(self.raw_pair_candidate_verifier)
                if self.raw_pair_candidate_verifier is not None
                else 0
            ),
            "total_trainable_parameters": count(self),
        }

    def train(self, mode: bool = True) -> "R35MultitaskSystem":
        super().train(mode)
        self.track_y.eval()
        self.encoder_system.descriptor_head.eval()
        self.encoder_system.matcher.eval()
        if self._training_stage is R35TrainingStage.LIMITED_JOINT:
            self.encoder_system.backbone.train(mode)
        else:
            self.encoder_system.backbone.eval()
        if self._training_stage is R35TrainingStage.BEARING_HEAD:
            self.bearing_head.train(mode)
            self.candidate_match_head.eval()
            self.set_presence_head.eval()
            if self.raw_pair_candidate_verifier is not None:
                self.raw_pair_candidate_verifier.eval()
        elif self._training_stage is R35TrainingStage.OPEN_SET_HEADS:
            self.bearing_head.eval()
            self.candidate_match_head.train(mode)
            self.set_presence_head.train(mode)
            if self.raw_pair_candidate_verifier is not None:
                self.raw_pair_candidate_verifier.train(mode)
        elif self._training_stage is R35TrainingStage.LIMITED_JOINT:
            self.bearing_head.train(mode)
            self.candidate_match_head.train(mode)
            self.set_presence_head.train(mode)
            if self.raw_pair_candidate_verifier is not None:
                self.raw_pair_candidate_verifier.train(mode)
        return self

    def _encode_track_y(
        self,
        tokens: torch.Tensor,
        descriptor_ring: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        base_head = getattr(self.track_y, "base_head", None)
        ring_encoder = getattr(base_head, "ring_encoder", None)
        if callable(ring_encoder):
            return {"dense_ring": ring_encoder(tokens)}
        legacy_encode = getattr(self.track_y, "encode", None)
        if callable(legacy_encode):
            return legacy_encode(tokens, descriptor_ring)
        raise TypeError("Track Y must expose base_head.ring_encoder or the legacy encode interface")

    def _forward_track_y_encoded(
        self,
        source_dense_ring: torch.Tensor,
        target_dense_ring: torch.Tensor,
        source_descriptor_ring: torch.Tensor,
        target_descriptor_ring: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        forward_rings = getattr(self.track_y, "forward_rings", None)
        if callable(forward_rings):
            return forward_rings(
                source_dense_ring,
                target_dense_ring,
                source_descriptor_ring,
                target_descriptor_ring,
            )
        legacy_forward = getattr(self.track_y, "forward_encoded", None)
        if callable(legacy_forward):
            return legacy_forward(
                source_dense_ring,
                target_dense_ring,
                source_descriptor_ring,
                target_descriptor_ring,
            )
        raise TypeError("Track Y must expose forward_rings or the legacy forward_encoded interface")

    def encode(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        tokens = self.encoder_system.backbone.forward_tokens(images)
        descriptor = self.encoder_system.descriptor_head.forward_tokens(tokens.float())
        yaw_encoding = self._encode_track_y(tokens, descriptor["ring"])
        return {
            "tokens": tokens,
            "global": descriptor["global"],
            "descriptor_ring": descriptor["ring"],
            "dense_ring": yaw_encoding["dense_ring"],
        }

    @staticmethod
    def _correlation_summary(scores: torch.Tensor) -> torch.Tensor:
        if scores.ndim != 2 or scores.shape[1] < 2:
            raise ValueError("correlation scores must be [N,L] with L >= 2")
        probability = torch.softmax(scores.float(), dim=-1)
        top2 = torch.topk(probability, k=2, dim=-1).values
        entropy = -(
            probability * probability.clamp_min(1e-8).log()
        ).sum(dim=-1) / math.log(float(scores.shape[-1]))
        return torch.stack(
            (
                scores.float().max(dim=-1).values,
                scores.float().mean(dim=-1),
                top2[:, 0] - top2[:, 1],
                entropy,
            ),
            dim=-1,
        )

    def forward_pairs(
        self,
        source_images: torch.Tensor,
        target_images: torch.Tensor,
    ) -> Dict[str, Any]:
        if source_images.shape != target_images.shape:
            raise ValueError("source and target image batches must have equal shape")
        encoded = self.encode(torch.cat((source_images, target_images), dim=0))
        source = {key: value.chunk(2, dim=0)[0] for key, value in encoded.items()}
        target = {key: value.chunk(2, dim=0)[1] for key, value in encoded.items()}
        yaw = self._forward_track_y_encoded(
            source["dense_ring"],
            target["dense_ring"],
            source["descriptor_ring"],
            target["descriptor_ring"],
        )
        bearing = self.bearing_head(source["tokens"], target["tokens"])
        return {"source": source, "target": target, "yaw": yaw, "bearing": bearing}

    def match_candidate_set(
        self,
        query_images: torch.Tensor,
        candidate_images: torch.Tensor,
        *,
        candidate_mask: torch.Tensor | None = None,
        reciprocal_rank_score: torch.Tensor | None = None,
    ) -> Dict[str, Any]:
        if query_images.ndim != 4 or candidate_images.ndim != 5:
            raise ValueError("query images must be [B,3,H,W] and candidates [B,K,3,H,W]")
        batch, candidate_count = candidate_images.shape[:2]
        if query_images.shape[0] != batch or query_images.shape[1:] != candidate_images.shape[2:]:
            raise ValueError("query and candidate image shapes are incompatible")
        if candidate_mask is None:
            candidate_mask = torch.ones(
                batch, candidate_count, dtype=torch.bool, device=query_images.device
            )
        if candidate_mask.shape != (batch, candidate_count):
            raise ValueError("candidate_mask must be [B,K]")
        if reciprocal_rank_score is None:
            reciprocal_rank_score = torch.zeros(
                batch, candidate_count, device=query_images.device
            )
        if reciprocal_rank_score.shape != (batch, candidate_count):
            raise ValueError("reciprocal_rank_score must be [B,K]")

        flat_candidates = candidate_images.flatten(0, 1)
        encoded = self.encode(torch.cat((query_images, flat_candidates), dim=0))
        query = {key: value[:batch] for key, value in encoded.items()}
        candidate = {
            key: value[batch:].reshape(batch, candidate_count, *value.shape[1:])
            for key, value in encoded.items()
        }
        expand = lambda value: value[:, None].expand(
            batch, candidate_count, *value.shape[1:]
        ).flatten(0, 1)
        candidate_flat = {key: value.flatten(0, 1) for key, value in candidate.items()}
        yaw = self._forward_track_y_encoded(
            expand(query["dense_ring"]),
            candidate_flat["dense_ring"],
            expand(query["descriptor_ring"]),
            candidate_flat["descriptor_ring"],
        )
        bearing = self.bearing_head(
            expand(query["tokens"]),
            candidate_flat["tokens"],
        )
        global_similarity = F.cosine_similarity(
            expand(query["global"]).float(),
            candidate_flat["global"].float(),
            dim=-1,
        )
        sector_scores = yaw.get("descriptor_correlation_32")
        if sector_scores is None:
            scale_scores = yaw["scale_scores"].float()
            sector_scores = scale_scores.mean(dim=1)
        sector_summary = self._correlation_summary(sector_scores)
        yaw_probability = yaw["top1_probability"].float()
        yaw_confidence = yaw["yaw_confidence"].float()
        yaw_entropy = yaw["normalized_entropy"].float()
        bearing_summary = torch.stack(
            (
                bearing["bearing_confidence"].float(),
                bearing["bearing_valid_probability"].float(),
                bearing["normalized_entropy"].float(),
            ),
            dim=-1,
        )
        normalized_rank = (
            torch.arange(candidate_count, device=query_images.device, dtype=torch.float32)
            .view(1, -1)
            .expand(batch, -1)
            / float(max(candidate_count - 1, 1))
        ).reshape(-1)
        scalar_features = torch.cat(
            (
                global_similarity.unsqueeze(-1),
                sector_summary,
                torch.stack((yaw_probability, yaw_confidence, yaw_entropy), dim=-1),
                bearing_summary,
                bearing["local_correlation_summary"].float(),
                normalized_rank.unsqueeze(-1),
                reciprocal_rank_score.reshape(-1, 1).float(),
            ),
            dim=-1,
        ).reshape(batch, candidate_count, -1)
        spatial_pair = torch.cat(
            (yaw["pair_embedding"].float(), bearing["pair_embedding"].float()),
            dim=-1,
        ).reshape(batch, candidate_count, -1)
        candidate_output = self.candidate_match_head(
            spatial_pair,
            scalar_features,
            candidate_mask,
        )
        verifier_output = None
        if self.raw_pair_candidate_verifier is not None:
            verifier_output = self.raw_pair_candidate_verifier(
                spatial_pair, scalar_features, candidate_mask
            )
            presence_output = self.set_presence_head(
                candidate_output,
                global_similarity.reshape(batch, candidate_count),
                yaw_confidence.reshape(batch, candidate_count),
                bearing["bearing_confidence"].reshape(batch, candidate_count),
                normalized_rank.reshape(batch, candidate_count),
                verifier_output,
            )
        else:
            presence_output = self.set_presence_head(
                candidate_output,
                global_similarity.reshape(batch, candidate_count),
                yaw_confidence.reshape(batch, candidate_count),
                bearing["bearing_confidence"].reshape(batch, candidate_count),
                normalized_rank.reshape(batch, candidate_count),
            )
        return {
            "query_encoding": query,
            "candidate_encoding": candidate,
            "yaw": {
                key: value.reshape(batch, candidate_count, *value.shape[1:])
                for key, value in yaw.items()
                if torch.is_tensor(value) and value.shape[0] == batch * candidate_count
            },
            "bearing": {
                key: value.reshape(batch, candidate_count, *value.shape[1:])
                for key, value in bearing.items()
                if torch.is_tensor(value) and value.shape[0] == batch * candidate_count
            },
            "scalar_features": scalar_features,
            "candidate_match": candidate_output,
            "candidate_verifier": verifier_output,
            "set_presence": presence_output,
        }
