from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import torch

from ...training.fgfs_models import normalized_nts_views
from .geometry import twelve_sector_views


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
        raise ValueError(f"forbidden online FG/FS inputs: {sorted(forbidden)}")
    if unknown:
        raise ValueError(f"unrecognized online FG/FS inputs: {sorted(unknown)}")
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


def model_device(model: torch.nn.Module) -> torch.device:
    parameter = next(model.parameters(), None)
    return parameter.device if parameter is not None else torch.device("cpu")


def _validated_output(outputs: Mapping[str, torch.Tensor], fg_threshold: float) -> dict[str, torch.Tensor]:
    if set(outputs) < {"fg_logits", "fs_scores"}:
        raise RuntimeError("FG/FS model must return fg_logits and fs_scores")
    fg_logits = outputs["fg_logits"]
    fs_scores = outputs["fs_scores"]
    if fg_logits.shape != fs_scores.shape or fg_logits.ndim != 2 or fg_logits.shape[1] != 12:
        raise RuntimeError("FG/FS outputs must have matching [B,12] shapes")
    finite = torch.isfinite(fg_logits) & torch.isfinite(fs_scores)
    fg_probabilities = torch.sigmoid(fg_logits)
    predicted_fg_mask = finite & (fg_probabilities >= fg_threshold)
    masked_fs_scores = torch.where(predicted_fg_mask, fs_scores, torch.full_like(fs_scores, -torch.inf))
    return {
        "fg_logits": fg_logits,
        "fg_probabilities": fg_probabilities,
        "fs_scores": fs_scores,
        "fs_valid_mask": predicted_fg_mask,
        "masked_fs_scores": masked_fs_scores,
    }


@dataclass
class FGFSInference:
    """Pure-RGB joint inference with explicit predicted FG masking."""

    model: torch.nn.Module
    model_type: str
    fg_threshold: float = 0.5

    def __post_init__(self) -> None:
        if self.model_type not in {"resnet", "dino"}:
            raise ValueError(f"unknown FG/FS model type: {self.model_type}")
        if not 0.0 <= self.fg_threshold <= 1.0:
            raise ValueError("fg_threshold must be in [0,1]")

    @torch.inference_mode()
    def __call__(self, inputs: Mapping[str, object]) -> dict[str, torch.Tensor]:
        validate_online_inputs(inputs)
        self.model.eval()
        device = model_device(self.model)
        if "current_features" in inputs:
            if self.model_type != "dino":
                raise ValueError("cached features are only valid for DINO FG/FS inference")
            source = torch.as_tensor(inputs["current_features"], device=device, dtype=torch.float32)
            goal = torch.as_tensor(inputs["goal_features"], device=device, dtype=torch.float32)
            if source.ndim == 2:
                source = source.unsqueeze(0)
            if goal.ndim == 2:
                goal = goal.unsqueeze(0)
            if source.shape != goal.shape or source.ndim != 3 or source.shape[1:] != (12, 384):
                raise ValueError("DINO features must have matching [B,12,384] shape")
            return _validated_output(self.model.forward_features(source, goal), self.fg_threshold)

        source = normalize_erp(inputs["current_erp_rgb"], device=device)
        goal = normalize_erp(inputs["goal_erp_rgb"], device=device)
        if source.shape != goal.shape:
            raise ValueError("Current and Goal ERP RGB batch shapes must match")
        if self.model_type == "resnet":
            outputs = self.model(normalized_nts_views(source), normalized_nts_views(goal))
        else:
            source_views = twelve_sector_views(source)
            goal_views = twelve_sector_views(goal)
            mean = source_views.new_tensor([0.485, 0.456, 0.406])[None, None, :, None, None]
            std = source_views.new_tensor([0.229, 0.224, 0.225])[None, None, :, None, None]
            outputs = self.model((source_views - mean) / std, (goal_views - mean) / std)
        return _validated_output(outputs, self.fg_threshold)


# Compatibility name for downstream imports. It now enforces the joint contract.
FSInference = FGFSInference


def fs_scores(
    endpoint_goal_geodesic: np.ndarray,
    valid_direction: np.ndarray,
    *,
    dmax: float = 20.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute paper-defined inter-node scores without hiding unreachable values."""
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


def masked_mse(prediction, target, valid_mask):
    mask = valid_mask.to(dtype=prediction.dtype)
    return ((prediction - target).square() * mask).sum() / mask.sum().clamp_min(1.0)
