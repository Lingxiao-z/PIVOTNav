"""Input validation and score utilities for final curiosity inference."""
from __future__ import annotations

from typing import Mapping

import numpy as np
import torch


FORBIDDEN_ONLINE_FIELDS = frozenset({
    "pose", "gt_pose", "yaw", "gt_yaw", "goal_position", "gt_goal_position",
    "navmesh", "depth", "gps", "compass", "geodesic_distance", "collision",
    "shortest_path", "success", "spl", "oracle_direction", "oracle_action",
})
ALLOWED_ONLINE_FIELDS = frozenset({
    "current_erp_rgb", "goal_erp_rgb", "current_features", "goal_features",
})


def validate_online_inputs(inputs: Mapping[str, object]) -> None:
    keys = set(inputs)
    forbidden = keys & FORBIDDEN_ONLINE_FIELDS
    unknown = keys - ALLOWED_ONLINE_FIELDS
    if forbidden:
        raise ValueError(f"forbidden online curiosity inputs: {sorted(forbidden)}")
    if unknown:
        raise ValueError(f"unrecognized online curiosity inputs: {sorted(unknown)}")
    image_pair = {"current_erp_rgb", "goal_erp_rgb"} <= keys
    feature_pair = {"current_features", "goal_features"} <= keys
    if image_pair == feature_pair:
        raise ValueError("provide exactly one complete Current/Goal RGB or feature pair")


def normalize_erp(image: object, *, device: torch.device) -> torch.Tensor:
    tensor = torch.as_tensor(np.asarray(image) if isinstance(image, np.ndarray) else image)
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 4:
        raise ValueError("ERP RGB must be an unbatched or batched image")
    if tensor.shape[-3:] == (3, 224, 448):
        pass
    elif tensor.shape[-3:] == (224, 448, 3):
        tensor = tensor.permute(0, 3, 1, 2)
    else:
        raise ValueError("ERP RGB must have shape [224,448,3] or [B,3,224,448]")
    if tensor.dtype == torch.uint8:
        tensor = tensor.to(device=device, dtype=torch.float32, non_blocking=True).div_(255.0)
    elif tensor.is_floating_point():
        tensor = tensor.to(device=device, dtype=torch.float32, non_blocking=True)
        if not bool(torch.isfinite(tensor).all()) or float(tensor.min()) < 0.0 or float(tensor.max()) > 1.0:
            raise ValueError("floating-point ERP RGB must be finite and normalized to [0,1]")
    else:
        raise TypeError("ERP RGB must use uint8 or floating-point values")
    return tensor.contiguous()


def fs_scores(
    endpoint_goal_geodesic: np.ndarray,
    valid_direction: np.ndarray,
    *,
    dmax: float = 20.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the training target for the twelve candidate directions."""
    distances = np.asarray(endpoint_goal_geodesic, dtype=np.float64)
    valid = np.asarray(valid_direction, dtype=bool)
    if distances.shape[-1] != 12 or valid.shape != distances.shape:
        raise ValueError("distances and mask must have matching [..., 12] shapes")
    if not np.isfinite(dmax) or dmax <= 0:
        raise ValueError("dmax must be finite and positive")
    unreachable = valid & ~np.isfinite(distances)
    finite_valid = valid & np.isfinite(distances)
    scores = np.zeros_like(distances, dtype=np.float32)
    scores[finite_valid] = np.maximum(1.0 - distances[finite_valid] / dmax, 0.0)
    return scores, unreachable


def masked_mse(prediction: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor):
    mask = valid_mask.to(dtype=prediction.dtype)
    return ((prediction - target).square() * mask).sum() / mask.sum().clamp_min(1.0)
