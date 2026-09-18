from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..geometry import bin_to_model_angle_deg, model_angle_rad_to_bin, signed_model_degrees


@dataclass
class RadarState:
    effective_distance_m: np.ndarray
    raw_distance_m: np.ndarray
    exist_probability: np.ndarray
    danger_mask: np.ndarray
    traversable_mask: np.ndarray
    nearest_index: int
    nearest_angle_deg: float
    nearest_distance_m: float
    front_min_distance_m: float
    left_min_distance_m: float
    right_min_distance_m: float
    goal_index: int | None
    goal_blocked: bool


def _sector_mask(angles_deg: np.ndarray, min_deg: float, max_deg: float) -> np.ndarray:
    return (angles_deg >= min_deg) & (angles_deg <= max_deg)


def analyze_radar(
    effective_distance_m: np.ndarray,
    raw_distance_m: np.ndarray,
    exist_probability: np.ndarray,
    config: dict[str, Any],
    goal_angle_rad: float | None = None,
) -> RadarState:
    num_angles = effective_distance_m.shape[0]
    angles_deg = signed_model_degrees(num_angles)
    radar_cfg = config["controllers"]["radar"]
    danger_threshold = float(radar_cfg["danger_distance_m"])
    traversable_threshold = float(radar_cfg["traversable_distance_m"])
    front_half_width = float(radar_cfg["front_sector_half_width_deg"])
    side_min = float(radar_cfg["side_sector_min_deg"])
    side_max = float(radar_cfg["side_sector_max_deg"])

    danger_mask = effective_distance_m < danger_threshold
    traversable_mask = effective_distance_m >= traversable_threshold
    nearest_index = int(np.argmin(effective_distance_m))
    nearest_angle_deg = bin_to_model_angle_deg(nearest_index, num_angles)
    nearest_distance_m = float(effective_distance_m[nearest_index])

    front_mask = _sector_mask(angles_deg, -front_half_width, front_half_width)
    left_mask = _sector_mask(angles_deg, -side_max, -side_min)
    right_mask = _sector_mask(angles_deg, side_min, side_max)

    front_min = float(np.min(effective_distance_m[front_mask])) if np.any(front_mask) else nearest_distance_m
    left_min = float(np.min(effective_distance_m[left_mask])) if np.any(left_mask) else nearest_distance_m
    right_min = float(np.min(effective_distance_m[right_mask])) if np.any(right_mask) else nearest_distance_m

    goal_index = None
    goal_blocked = False
    if goal_angle_rad is not None:
        goal_index = model_angle_rad_to_bin(goal_angle_rad, num_angles)
        goal_blocked = float(effective_distance_m[goal_index]) < float(radar_cfg["goal_blocked_distance_m"])

    return RadarState(
        effective_distance_m=effective_distance_m.astype(np.float32),
        raw_distance_m=raw_distance_m.astype(np.float32),
        exist_probability=exist_probability.astype(np.float32),
        danger_mask=danger_mask,
        traversable_mask=traversable_mask,
        nearest_index=nearest_index,
        nearest_angle_deg=nearest_angle_deg,
        nearest_distance_m=nearest_distance_m,
        front_min_distance_m=front_min,
        left_min_distance_m=left_min,
        right_min_distance_m=right_min,
        goal_index=goal_index,
        goal_blocked=goal_blocked,
    )
