from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class R34OpenSetConfig:
    pair_embedding_dim: int = 128
    model_dim: int = 192
    attention_heads: int = 4
    attention_layers: int = 2
    feedforward_dim: int = 384
    dropout: float = 0.1
    minimum_temperature: float = 0.05
    initial_temperature: float = 1.0
    hard_negative_margin: float = 0.20


@dataclass(frozen=True)
class R34OpenSetLossWeights:
    set_cross_entropy: float = 1.0
    pair_same_place: float = 0.25
    hard_negative_margin: float = 0.25
    calibration: float = 0.10
    yaw_consistency: float = 0.10
    descriptor_distillation: float = 0.25


@dataclass(frozen=True)
class R34OpenSetThresholds:
    unknown_threshold: float = 0.50
    anchor_threshold: float = 0.50
    anchor_margin: float = 0.10
    threshold_offset: float = 0.0


class R34SetOpenSetHead(nn.Module):
    """Permutation-equivariant Top-K verifier with an explicit UNKNOWN token."""

    def __init__(self, cfg: R34OpenSetConfig | None = None):
        super().__init__()
        self.cfg = cfg or R34OpenSetConfig()
        self.input_projection = nn.Sequential(
            nn.LayerNorm(self.cfg.pair_embedding_dim),
            nn.Linear(self.cfg.pair_embedding_dim, self.cfg.model_dim),
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
        self.set_encoder = nn.TransformerEncoder(layer, num_layers=self.cfg.attention_layers)
        self.unknown_token = nn.Parameter(torch.zeros(1, 1, self.cfg.model_dim))
        self.candidate_head = nn.Linear(self.cfg.model_dim, 1)
        self.same_place_head = nn.Linear(self.cfg.model_dim, 1)
        self.unknown_head = nn.Linear(self.cfg.model_dim, 1)
        self.unknown_bias = nn.Parameter(torch.zeros(()))
        initial_raw = torch.log(
            torch.expm1(torch.tensor(self.cfg.initial_temperature - self.cfg.minimum_temperature))
        )
        self.raw_temperature = nn.Parameter(initial_raw)
        nn.init.normal_(self.unknown_token, std=0.02)

    @property
    def temperature(self) -> torch.Tensor:
        return F.softplus(self.raw_temperature) + self.cfg.minimum_temperature

    @property
    def architecture_record(self) -> Dict[str, Any]:
        return {
            "schema_version": "r34_set_open_set_head_v1",
            "config": asdict(self.cfg),
            "set_reasoning": "two-layer self-attention over K pair tokens and one learnable UNKNOWN token",
            "candidate_order": "no positional embedding; candidate permutation equivariant",
            "outputs": "K node logits plus one UNKNOWN logit with learned temperature and UNKNOWN bias",
            "external_thresholds": [
                "unknown_threshold",
                "anchor_threshold",
                "anchor_margin",
                "threshold_offset",
            ],
        }

    def forward(
        self,
        pair_embeddings: torch.Tensor,
        candidate_mask: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        if pair_embeddings.ndim != 3 or pair_embeddings.shape[-1] != self.cfg.pair_embedding_dim:
            raise ValueError(
                f"pair_embeddings must be [B,K,{self.cfg.pair_embedding_dim}], got {tuple(pair_embeddings.shape)}"
            )
        batch, candidate_count, _ = pair_embeddings.shape
        if candidate_count < 1:
            raise ValueError("at least one candidate is required")
        if candidate_mask is None:
            candidate_mask = torch.ones(
                batch, candidate_count, dtype=torch.bool, device=pair_embeddings.device
            )
        if candidate_mask.shape != (batch, candidate_count) or not candidate_mask.any(dim=1).all():
            raise ValueError("candidate_mask must be [B,K] with at least one valid candidate per set")

        candidate_tokens = self.input_projection(pair_embeddings.float())
        unknown = self.unknown_token.expand(batch, -1, -1)
        tokens = torch.cat((candidate_tokens, unknown), dim=1)
        padding_mask = torch.cat(
            (~candidate_mask, torch.zeros(batch, 1, dtype=torch.bool, device=candidate_mask.device)),
            dim=1,
        )
        encoded = self.set_encoder(tokens, src_key_padding_mask=padding_mask)
        candidate_encoded = encoded[:, :candidate_count]
        unknown_encoded = encoded[:, candidate_count]

        candidate_logits = self.candidate_head(candidate_encoded).squeeze(-1)
        candidate_logits = candidate_logits.masked_fill(~candidate_mask, -1.0e4)
        unknown_logit = self.unknown_head(unknown_encoded).squeeze(-1) + self.unknown_bias
        unscaled_logits = torch.cat((candidate_logits, unknown_logit.unsqueeze(-1)), dim=-1)
        calibrated_logits = unscaled_logits / self.temperature
        probabilities = torch.softmax(calibrated_logits.float(), dim=-1)
        node_probabilities = probabilities[:, :candidate_count] * candidate_mask.to(probabilities.dtype)
        unknown_probability = probabilities[:, candidate_count]
        top_count = min(2, candidate_count)
        top = torch.topk(node_probabilities, k=top_count, dim=-1)
        best_probability = top.values[:, 0]
        second_probability = (
            top.values[:, 1] if candidate_count > 1 else torch.zeros_like(best_probability)
        )

        return {
            "candidate_logits": candidate_logits,
            "unknown_logit": unknown_logit,
            "unscaled_logits_with_unknown": unscaled_logits,
            "logits_with_unknown": calibrated_logits,
            "node_probabilities": node_probabilities,
            "unknown_probability": unknown_probability,
            "present_probability": 1.0 - unknown_probability,
            "best_candidate_index": node_probabilities.argmax(dim=-1),
            "best_candidate_probability": best_probability,
            "candidate_probability_margin": best_probability - second_probability,
            "pair_same_place_logits": self.same_place_head(candidate_encoded).squeeze(-1),
            "encoded_candidate_tokens": candidate_encoded,
            "encoded_unknown_token": unknown_encoded,
            "candidate_mask": candidate_mask,
            "temperature": self.temperature,
        }


def r34_open_set_decision(
    output: Dict[str, torch.Tensor],
    thresholds: R34OpenSetThresholds,
) -> Dict[str, torch.Tensor]:
    effective_unknown_threshold = min(
        1.0, max(0.0, thresholds.unknown_threshold + thresholds.threshold_offset)
    )
    unknown = (
        (output["unknown_probability"] >= effective_unknown_threshold)
        | (output["best_candidate_probability"] < thresholds.anchor_threshold)
        | (output["candidate_probability_margin"] < thresholds.anchor_margin)
    )
    selected_index = torch.where(
        unknown,
        torch.full_like(output["best_candidate_index"], -1),
        output["best_candidate_index"],
    )
    return {
        "is_unknown": unknown,
        "selected_candidate_index": selected_index,
        "effective_unknown_threshold": torch.full_like(
            output["unknown_probability"], effective_unknown_threshold
        ),
    }


def r34_open_set_losses(
    output: Dict[str, torch.Tensor],
    target_index: torch.Tensor,
    candidate_same_place: torch.Tensor,
    *,
    weights: R34OpenSetLossWeights | None = None,
    hard_negative_margin: float = 0.20,
    candidate_hard_negative: torch.Tensor | None = None,
    yaw_consistency_penalty: torch.Tensor | None = None,
    descriptor_distillation_penalty: torch.Tensor | None = None,
) -> Dict[str, torch.Tensor]:
    loss_weights = weights or R34OpenSetLossWeights()
    logits = output["logits_with_unknown"].float()
    candidate_mask = output["candidate_mask"]
    batch, candidate_count = candidate_mask.shape
    if target_index.shape != (batch,) or ((target_index < 0) | (target_index > candidate_count)).any():
        raise ValueError("target_index must be [B] with UNKNOWN encoded as K")
    if candidate_same_place.shape != (batch, candidate_count):
        raise ValueError("candidate_same_place must be [B,K]")
    if candidate_hard_negative is not None and candidate_hard_negative.shape != (batch, candidate_count):
        raise ValueError("candidate_hard_negative must be [B,K]")

    set_ce = F.cross_entropy(logits, target_index.long())
    pair_loss_raw = F.binary_cross_entropy_with_logits(
        output["pair_same_place_logits"].float(),
        candidate_same_place.float(),
        reduction="none",
    )
    pair_same_place = (pair_loss_raw * candidate_mask).sum() / candidate_mask.sum().clamp_min(1)

    candidate_logits = output["candidate_logits"].float()
    unknown_logit = output["unknown_logit"].float()
    hard_mask = (
        candidate_mask & candidate_hard_negative.bool()
        if candidate_hard_negative is not None
        else candidate_mask
    )
    # A row without an explicitly tagged hard negative falls back to every
    # valid candidate so the margin remains defined for small candidate sets.
    hard_mask = torch.where(hard_mask.any(dim=1, keepdim=True), hard_mask, candidate_mask)
    valid_candidates = candidate_logits.masked_fill(~hard_mask, -torch.inf)
    hardest_candidate = valid_candidates.max(dim=-1).values
    absent = target_index == candidate_count
    present = ~absent
    hard_margin_terms = []
    if absent.any():
        hard_margin_terms.append(
            F.relu(hard_negative_margin - unknown_logit[absent] + hardest_candidate[absent]).mean()
        )
    if present.any():
        positive = candidate_logits[present].gather(1, target_index[present, None]).squeeze(1)
        negative_mask = hard_mask[present].clone()
        negative_mask.scatter_(1, target_index[present, None], False)
        hardest_negative = candidate_logits[present].masked_fill(~negative_mask, -torch.inf).max(dim=-1).values
        has_negative = negative_mask.any(dim=-1)
        if has_negative.any():
            hard_margin_terms.append(
                F.relu(
                    hard_negative_margin
                    - positive[has_negative]
                    + hardest_negative[has_negative]
                ).mean()
            )
    hard_negative = (
        torch.stack(hard_margin_terms).mean()
        if hard_margin_terms
        else output["unknown_logit"].sum() * 0.0
    )

    target_one_hot = F.one_hot(target_index.long(), candidate_count + 1).float()
    calibration = ((torch.softmax(logits, dim=-1) - target_one_hot) ** 2).sum(dim=-1).mean()
    yaw_consistency = (
        yaw_consistency_penalty.float().mean()
        if yaw_consistency_penalty is not None
        else output["unknown_logit"].sum() * 0.0
    )
    descriptor_distillation = (
        descriptor_distillation_penalty.float().mean()
        if descriptor_distillation_penalty is not None
        else output["unknown_logit"].sum() * 0.0
    )
    total = (
        loss_weights.set_cross_entropy * set_ce
        + loss_weights.pair_same_place * pair_same_place
        + loss_weights.hard_negative_margin * hard_negative
        + loss_weights.calibration * calibration
        + loss_weights.yaw_consistency * yaw_consistency
        + loss_weights.descriptor_distillation * descriptor_distillation
    )
    return {
        "loss": total,
        "set_cross_entropy": set_ce,
        "pair_same_place": pair_same_place,
        "hard_negative_margin": hard_negative,
        "calibration": calibration,
        "yaw_consistency": yaw_consistency,
        "descriptor_distillation": descriptor_distillation,
    }
