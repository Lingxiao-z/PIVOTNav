from __future__ import annotations

import numpy as np


NUM_DIRECTIONS = 12
DMAX_METERS = 20.0


def fs_scores(
    endpoint_goal_geodesic: np.ndarray,
    valid_direction: np.ndarray,
    *,
    dmax: float = DMAX_METERS,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute paper-defined inter-node scores without hiding unreachable values."""
    distances = np.asarray(endpoint_goal_geodesic, dtype=np.float64)
    valid = np.asarray(valid_direction, dtype=bool)
    if distances.shape[-1] != NUM_DIRECTIONS or valid.shape != distances.shape:
        raise ValueError("distances and mask must have matching [..., 12] shapes")
    if not np.isfinite(dmax) or dmax <= 0:
        raise ValueError("dmax must be finite and positive")
    unreachable = valid & ~np.isfinite(distances)
    finite_valid = valid & np.isfinite(distances)
    scores = np.zeros_like(distances, dtype=np.float32)
    scores[finite_valid] = np.maximum(1.0 - distances[finite_valid] / dmax, 0.0)
    return scores, unreachable


def masked_mse(prediction, target, valid_mask):
    import torch

    mask = valid_mask.to(dtype=prediction.dtype)
    denominator = mask.sum().clamp_min(1.0)
    return ((prediction - target).square() * mask).sum() / denominator

