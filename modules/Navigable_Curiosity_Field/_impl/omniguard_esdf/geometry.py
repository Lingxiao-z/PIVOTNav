from __future__ import annotations

import math
from typing import Iterable, Mapping

import numpy as np


def wrap_angle_rad(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def world_delta_to_body(dx_world: float, dy_world: float, yaw_rad: float) -> tuple[float, float]:
    cos_yaw = math.cos(yaw_rad)
    sin_yaw = math.sin(yaw_rad)
    x_body = cos_yaw * dx_world + sin_yaw * dy_world
    y_body = -sin_yaw * dx_world + cos_yaw * dy_world
    return x_body, y_body


def body_goal_to_model_angle_rad(x_body: float, y_body: float) -> float:
    return wrap_angle_rad(-math.atan2(y_body, x_body))


def model_heading_rad_to_body_xy(angle_rad: float, radius: float = 1.0) -> tuple[float, float]:
    distance = float(radius)
    theta = float(angle_rad)
    return distance * math.cos(theta), -distance * math.sin(theta)


def signed_model_degrees(num_angles: int = 360) -> np.ndarray:
    if num_angles <= 0:
        raise ValueError("num_angles must be positive")
    indices = np.arange(num_angles, dtype=np.float32)
    return indices - float(num_angles // 2)


def model_angle_deg_to_bin(angle_deg: float, num_angles: int = 360) -> int:
    if num_angles <= 0:
        raise ValueError("num_angles must be positive")
    wrapped_deg = (float(angle_deg) + 180.0) % 360.0 - 180.0
    return (int(round(wrapped_deg)) + num_angles // 2) % num_angles


def model_angle_rad_to_bin(angle_rad: float, num_angles: int = 360) -> int:
    return model_angle_deg_to_bin(math.degrees(wrap_angle_rad(angle_rad)), num_angles)


def bin_to_model_angle_deg(index: int, num_angles: int = 360) -> float:
    if num_angles <= 0:
        raise ValueError("num_angles must be positive")
    index %= num_angles
    return float(index - num_angles // 2)


def get_world_coordinate(data: Mapping[str, float], axis: str, default: float | None = None) -> float:
    meter_key = f"{axis}_m"
    if meter_key in data and data[meter_key] is not None:
        return float(data[meter_key])
    if default is not None:
        return float(default)
    raise KeyError(f"Missing world coordinate field: {meter_key}")


def cumulative_path_length(points: Iterable[tuple[float, float]]) -> float:
    path = list(points)
    if len(path) < 2:
        return 0.0
    total = 0.0
    for index in range(1, len(path)):
        total += math.hypot(path[index][0] - path[index - 1][0], path[index][1] - path[index - 1][1])
    return total
