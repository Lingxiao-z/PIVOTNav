from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class MatcherConfig:
    descriptor_dim: int = 2112
    ring_bins: int = 32
    ring_dim: int = 128
    cluster_count: int = 64
    pair_projection_dim: int = 64
    hidden_dim: int = 192
    top_k: int = 8
    dropout: float = 0.1


def _masked_mean(values: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    weights = mask.to(values.dtype)
    while weights.ndim < values.ndim:
        weights = weights.unsqueeze(-1)
    return (values * weights).sum(dim=dim) / weights.sum(dim=dim).clamp_min(1.0)


def ring_correlation_matrix(query_ring: torch.Tensor, candidate_ring: torch.Tensor) -> torch.Tensor:
    """Compute all circular candidate-to-query alignments.

    Args:
        query_ring: [B,L,R]
        candidate_ring: [B,K,L,R]
    Returns:
        Correlation scores [B,K,L]. A positive shift rolls the candidate ring
        right to match the query frame, matching the project yaw convention.
    """
    if query_ring.ndim != 3 or candidate_ring.ndim != 4:
        raise ValueError("query_ring must be [B,L,R] and candidate_ring [B,K,L,R]")
    if query_ring.shape[0] != candidate_ring.shape[0] or query_ring.shape[1:] != candidate_ring.shape[2:]:
        raise ValueError(f"ring shape mismatch: {tuple(query_ring.shape)} vs {tuple(candidate_ring.shape)}")
    scores = []
    query = F.normalize(query_ring.float(), dim=-1).unsqueeze(1)
    candidates = F.normalize(candidate_ring.float(), dim=-1)
    for shift in range(query_ring.shape[1]):
        aligned = torch.roll(candidates, shifts=shift, dims=2)
        scores.append((query * aligned).sum(dim=-1).mean(dim=-1))
    return torch.stack(scores, dim=-1)


class TopKOpenSetMatcher(nn.Module):
    """Small learned verifier applied only to retrieved Top-K candidates.

    Global descriptors perform scalable retrieval. This module then combines
    descriptor interactions, ring alignment and SALAD assignment statistics to
    classify candidate nodes jointly with one explicit UNKNOWN state.
    """

    scalar_feature_dim = 9

    def __init__(self, cfg: MatcherConfig | None = None):
        super().__init__()
        self.cfg = cfg or MatcherConfig()
        p = self.cfg.pair_projection_dim
        h = self.cfg.hidden_dim
        self.abs_projection = nn.Linear(self.cfg.descriptor_dim, p, bias=False)
        self.product_projection = nn.Linear(self.cfg.descriptor_dim, p, bias=False)
        pair_dim = p * 2 + self.scalar_feature_dim
        self.pair_encoder = nn.Sequential(
            nn.LayerNorm(pair_dim),
            nn.Linear(pair_dim, h),
            nn.GELU(),
            nn.Dropout(self.cfg.dropout),
            nn.Linear(h, h),
            nn.GELU(),
        )
        self.same_place_head = nn.Linear(h, 1)
        self.unknown_head = nn.Sequential(
            nn.LayerNorm(h * 2 + 4),
            nn.Linear(h * 2 + 4, h),
            nn.GELU(),
            nn.Dropout(self.cfg.dropout),
            nn.Linear(h, 1),
        )

    def _validate(self, query: Dict[str, torch.Tensor], candidates: Dict[str, torch.Tensor]) -> None:
        required = {"global", "ring", "cluster_mass", "dustbin_fraction"}
        missing_query = required.difference(query)
        missing_candidates = required.difference(candidates)
        if missing_query or missing_candidates:
            raise KeyError(f"missing matcher tensors: query={missing_query}, candidates={missing_candidates}")
        b, k, d = candidates["global"].shape
        if query["global"].shape != (b, d) or d != self.cfg.descriptor_dim:
            raise ValueError("global descriptor shape mismatch")
        if query["ring"].shape != (b, self.cfg.ring_bins, self.cfg.ring_dim):
            raise ValueError("query ring shape mismatch")
        if candidates["ring"].shape != (b, k, self.cfg.ring_bins, self.cfg.ring_dim):
            raise ValueError("candidate ring shape mismatch")
        if query["cluster_mass"].shape != (b, self.cfg.cluster_count):
            raise ValueError("query cluster_mass shape mismatch")
        if candidates["cluster_mass"].shape != (b, k, self.cfg.cluster_count):
            raise ValueError("candidate cluster_mass shape mismatch")

    def forward(
        self,
        query: Dict[str, torch.Tensor],
        candidates: Dict[str, torch.Tensor],
        candidate_mask: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        self._validate(query, candidates)
        b, k, _ = candidates["global"].shape
        if candidate_mask is None:
            candidate_mask = torch.ones(b, k, dtype=torch.bool, device=candidates["global"].device)
        if candidate_mask.shape != (b, k) or not candidate_mask.any(dim=1).all():
            raise ValueError("candidate_mask must be [B,K] with at least one valid candidate per query")

        q_global = F.normalize(query["global"].float(), dim=-1)
        c_global = F.normalize(candidates["global"].float(), dim=-1)
        q_expanded = q_global.unsqueeze(1).expand_as(c_global)
        global_similarity = (q_expanded * c_global).sum(dim=-1)
        descriptor_abs = self.abs_projection(torch.abs(q_expanded - c_global))
        descriptor_product = self.product_projection(q_expanded * c_global)

        ring_scores = ring_correlation_matrix(query["ring"], candidates["ring"])
        ring_top2 = torch.topk(ring_scores, k=2, dim=-1)
        ring_peak = ring_top2.values[..., 0]
        ring_margin = ring_top2.values[..., 0] - ring_top2.values[..., 1]
        ring_prob = torch.softmax(ring_scores, dim=-1)
        ring_entropy = -(ring_prob * ring_prob.clamp_min(1e-8).log()).sum(dim=-1)
        ring_entropy = ring_entropy / torch.log(torch.tensor(float(self.cfg.ring_bins), device=ring_entropy.device))
        yaw_shift_bins = ring_top2.indices[..., 0]

        q_cluster = F.normalize(query["cluster_mass"].float(), dim=-1).unsqueeze(1)
        c_cluster = F.normalize(candidates["cluster_mass"].float(), dim=-1)
        cluster_similarity = (q_cluster * c_cluster).sum(dim=-1)
        cluster_l1 = torch.abs(query["cluster_mass"].float().unsqueeze(1) - candidates["cluster_mass"].float()).mean(dim=-1)
        q_dust = query["dustbin_fraction"].float().reshape(b, 1).expand(b, k)
        c_dust = candidates["dustbin_fraction"].float().reshape(b, k)
        rank = torch.arange(k, device=q_global.device, dtype=q_global.dtype).view(1, k).expand(b, k)
        normalized_rank = rank / max(1, k - 1)

        scalar_features = torch.stack(
            [
                global_similarity,
                ring_peak,
                ring_margin,
                ring_entropy,
                cluster_similarity,
                cluster_l1,
                q_dust,
                c_dust,
                normalized_rank,
            ],
            dim=-1,
        )
        pair_features = torch.cat([descriptor_abs, descriptor_product, scalar_features], dim=-1)
        pair_embedding = self.pair_encoder(pair_features)
        candidate_logits = self.same_place_head(pair_embedding).squeeze(-1)
        candidate_logits = candidate_logits.masked_fill(~candidate_mask, torch.finfo(candidate_logits.dtype).min)

        set_mean = _masked_mean(pair_embedding, candidate_mask, dim=1)
        set_max = pair_embedding.masked_fill(~candidate_mask.unsqueeze(-1), torch.finfo(pair_embedding.dtype).min).max(dim=1).values
        sorted_similarity = global_similarity.masked_fill(~candidate_mask, -1.0).sort(dim=1, descending=True).values
        top1 = sorted_similarity[:, 0]
        top2 = sorted_similarity[:, 1] if k > 1 else torch.full_like(top1, -1.0)
        valid_count = candidate_mask.sum(dim=1).to(global_similarity.dtype)
        score_mean = _masked_mean(global_similarity, candidate_mask, dim=1)
        set_scalars = torch.stack([top1, top1 - top2, score_mean, valid_count / float(k)], dim=-1)
        unknown_logit = self.unknown_head(torch.cat([set_mean, set_max, set_scalars], dim=-1)).squeeze(-1)
        logits_with_unknown = torch.cat([candidate_logits, unknown_logit.unsqueeze(-1)], dim=-1)
        probabilities = torch.softmax(logits_with_unknown, dim=-1)
        node_probabilities = probabilities[:, :k] * candidate_mask.to(probabilities.dtype)
        unknown_probability = probabilities[:, -1]

        return {
            "candidate_logits": candidate_logits,
            "unknown_logit": unknown_logit,
            "logits_with_unknown": logits_with_unknown,
            "node_probabilities": node_probabilities,
            "unknown_probability": unknown_probability,
            "present_probability": 1.0 - unknown_probability,
            "best_candidate_index": node_probabilities.argmax(dim=-1),
            "global_similarity": global_similarity,
            "ring_scores": ring_scores,
            "ring_peak": ring_peak,
            "ring_peak_second_margin": ring_margin,
            "ring_entropy": ring_entropy,
            "yaw_shift_bins": yaw_shift_bins,
            "pair_embedding": pair_embedding,
        }
