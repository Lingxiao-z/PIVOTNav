from __future__ import annotations

from typing import Iterable, Mapping, Sequence

import torch
from torch.nn import functional as F

from .spatial_protocol import positive_place_ids


def place_positive_mask(place_ids: Sequence[str]) -> torch.Tensor:
    ids = list(place_ids)
    n = len(ids)
    mask = torch.zeros(n, n, dtype=torch.bool)
    for i, a in enumerate(ids):
        for j, b in enumerate(ids):
            if i != j and a == b:
                mask[i, j] = True
    return mask


def spatial_multi_positive_mask(
    metadata: Sequence[Mapping[str, object]],
    candidate_place_ids: Sequence[str],
) -> torch.Tensor:
    """Asymmetric query-to-gallery mask from R3 spatial positive sets."""
    if len(metadata) != len(candidate_place_ids):
        raise ValueError("metadata and candidate IDs must have the same length")
    mask = torch.zeros((len(metadata), len(candidate_place_ids)), dtype=torch.bool)
    columns_by_place: dict[str, list[int]] = {}
    for col, place_id in enumerate(candidate_place_ids):
        columns_by_place.setdefault(str(place_id), []).append(col)
    for row, meta in enumerate(metadata):
        for place_id in positive_place_ids(meta):
            columns = columns_by_place.get(str(place_id))
            if columns:
                mask[row, columns] = True
    mask.fill_diagonal_(False)
    return mask


def supervised_contrastive_multi_positive_loss(embeddings: torch.Tensor, place_ids: Sequence[str], temperature: float = 0.07) -> torch.Tensor:
    if embeddings.ndim != 2:
        raise ValueError('embeddings must be [B,D]')
    pos_mask = place_positive_mask(place_ids).to(embeddings.device)
    if not pos_mask.any():
        return embeddings.sum() * 0.0
    z = F.normalize(embeddings, dim=-1)
    logits = z @ z.T / max(float(temperature), 1e-6)
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    self_mask = torch.eye(logits.shape[0], dtype=torch.bool, device=logits.device)
    exp_logits = torch.exp(logits).masked_fill(self_mask, 0.0)
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12))
    per_anchor = -(log_prob * pos_mask).sum(dim=1) / pos_mask.sum(dim=1).clamp_min(1)
    valid = pos_mask.any(dim=1)
    return per_anchor[valid].mean()


def roll_invariance_loss(global_a: torch.Tensor, global_b: torch.Tensor) -> torch.Tensor:
    return (1.0 - (F.normalize(global_a, dim=-1) * F.normalize(global_b, dim=-1)).sum(dim=-1)).mean()


def ring_roll_equivariance_loss(ring_a: torch.Tensor, ring_b_from_rolled_input: torch.Tensor, shifts: torch.Tensor) -> torch.Tensor:
    if ring_a.shape != ring_b_from_rolled_input.shape:
        raise ValueError("ring inputs must have matching shapes")
    if ring_a.ndim != 3 or shifts.shape != (ring_a.shape[0],):
        raise ValueError("expected ring tensors [B,L,D] and shifts [B]")
    if ring_a.shape[0] == 0:
        return ring_a.sum() * 0.0
    bins = ring_a.shape[1]
    source_index = (
        torch.arange(bins, device=ring_a.device).unsqueeze(0)
        - shifts.to(device=ring_a.device, dtype=torch.long).unsqueeze(1)
    ).remainder(bins)
    expected = ring_a.gather(1, source_index.unsqueeze(-1).expand(-1, -1, ring_a.shape[-1]))
    return F.mse_loss(ring_b_from_rolled_input, expected)


def yaw_circular_classification_loss(logits: torch.Tensor, target_bins: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    if valid_mask.dtype is not torch.bool:
        valid_mask = valid_mask.bool()
    if logits.shape[0] != target_bins.shape[0] or logits.shape[0] != valid_mask.shape[0]:
        raise ValueError('logits, target_bins, valid_mask batch mismatch')
    if not valid_mask.any():
        return logits.sum() * 0.0
    return F.cross_entropy(logits[valid_mask], target_bins[valid_mask].long())


def hard_negative_ranking_loss(pos_scores: torch.Tensor, hard_neg_scores: torch.Tensor, margin: float = 0.1) -> torch.Tensor:
    return F.relu(float(margin) + hard_neg_scores - pos_scores).mean()


def pairwise_same_place_loss(
    candidate_logits: torch.Tensor,
    same_place_targets: torch.Tensor,
    candidate_mask: torch.Tensor | None = None,
    positive_weight: float | None = None,
) -> torch.Tensor:
    """Binary candidate supervision for Phase C.

    Targets are 1 only for candidates that satisfy the frozen spatial same-place
    protocol. Invalid padded candidates never contribute to the loss.
    """
    if candidate_logits.shape != same_place_targets.shape:
        raise ValueError("candidate_logits and same_place_targets shape mismatch")
    if candidate_mask is None:
        candidate_mask = torch.ones_like(candidate_logits, dtype=torch.bool)
    if candidate_mask.shape != candidate_logits.shape or not candidate_mask.any():
        raise ValueError("candidate_mask must select at least one candidate")
    weight = None
    if positive_weight is not None:
        weight = torch.as_tensor(float(positive_weight), device=candidate_logits.device, dtype=candidate_logits.dtype)
    return F.binary_cross_entropy_with_logits(
        candidate_logits[candidate_mask],
        same_place_targets.to(candidate_logits.dtype)[candidate_mask],
        pos_weight=weight,
    )


def multi_positive_open_set_loss(
    logits_with_unknown: torch.Tensor,
    positive_mask: torch.Tensor,
    absent_target: torch.Tensor,
) -> torch.Tensor:
    """Set loss: any spatially valid candidate is positive; absent targets UNKNOWN."""
    if logits_with_unknown.ndim != 2 or positive_mask.shape != logits_with_unknown[:, :-1].shape:
        raise ValueError("invalid multi-positive open-set shapes")
    if absent_target.shape != (logits_with_unknown.shape[0],):
        raise ValueError("absent_target shape mismatch")
    logits = logits_with_unknown.float()
    positives = positive_mask.bool()
    present = ~absent_target.bool()
    losses = []
    if present.any():
        rows = logits[present, :-1]
        pos = positives[present]
        valid = pos.any(dim=1)
        if valid.any():
            selected = rows[valid]
            selected_pos = pos[valid]
            numerator = torch.logsumexp(selected.masked_fill(~selected_pos, -1e9), dim=1)
            denominator = torch.logsumexp(logits[present][valid], dim=1)
            losses.append(-(numerator - denominator).mean())
    if (~present).any():
        unknown_index = logits.shape[1] - 1
        losses.append(torch.nn.functional.cross_entropy(logits[~present], torch.full((int((~present).sum()),), unknown_index, dtype=torch.long, device=logits.device)))
    return torch.stack(losses).mean() if losses else logits.sum() * 0.0


def open_set_node_or_unknown_loss(
    logits_with_unknown: torch.Tensor,
    target_candidate_index: torch.Tensor,
) -> torch.Tensor:
    """Set-level cross entropy over K candidates plus explicit UNKNOWN.

    `target_candidate_index == -1` denotes an absent query and is mapped to the
    final UNKNOWN logit. Present queries use their index inside the retrieved
    candidate set. Candidate-recall misses must be excluded or handled by the
    caller as UNKNOWN according to the frozen training protocol.
    """
    if logits_with_unknown.ndim != 2 or target_candidate_index.ndim != 1:
        raise ValueError("expected logits [B,K+1] and targets [B]")
    if logits_with_unknown.shape[0] != target_candidate_index.shape[0]:
        raise ValueError("batch mismatch")
    unknown_index = logits_with_unknown.shape[1] - 1
    target = target_candidate_index.long().clone()
    target[target < 0] = unknown_index
    if (target > unknown_index).any():
        raise ValueError("target candidate index is outside candidate/UNKNOWN logits")
    return F.cross_entropy(logits_with_unknown, target)


def open_set_brier_loss(unknown_probability: torch.Tensor, absent_target: torch.Tensor) -> torch.Tensor:
    if unknown_probability.shape != absent_target.shape:
        raise ValueError("unknown_probability and absent_target shape mismatch")
    return F.mse_loss(unknown_probability.float(), absent_target.float())
