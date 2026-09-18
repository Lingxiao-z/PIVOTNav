from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np


def wrap_degrees(value: float) -> float:
    return float(((float(value) + 180.0) % 360.0) - 180.0)


def global_bearing_degrees(source: Sequence[float], target: Sequence[float]) -> float:
    """Habitat XZ bearing: zero is -Z and positive angles point to +X/right."""
    source_array = np.asarray(source, dtype=np.float64)
    target_array = np.asarray(target, dtype=np.float64)
    delta_x = float(target_array[0] - source_array[0])
    delta_z = float(target_array[2] - source_array[2])
    return float(math.degrees(math.atan2(delta_x, -delta_z)))


def gt_goal_heading(
    position: Sequence[float],
    heading_degrees: float,
    target: Sequence[float],
) -> tuple[float, float]:
    """Return one right-positive relative bearing for both OmniGuard and audit."""
    relative_degrees = wrap_degrees(
        global_bearing_degrees(position, target) - float(heading_degrees)
    )
    return math.radians(relative_degrees), relative_degrees


def velocity_action(
    linear_velocity_mps: float,
    angular_velocity_rps: float,
    *,
    allow_sliding: bool,
    min_linear_velocity_mps: float = 0.0,
    max_linear_velocity_mps: float = 0.4,
    max_angular_velocity_rps: float = 0.3,
    time_step_s: float = 0.25,
) -> dict[str, Any]:
    """Map OmniGuard commands to the configured Habitat velocity action.

    The Integration V6 Habitat config freezes ``lin_vel_range`` to
    ``[0, max_linear_velocity_mps]``.  A negative default here makes a zero
    command decode to forward motion, corrupting every in-place rotation.
    OmniGuard's controller already applies angular=-k*theta, so its angular
    sign is passed through unchanged to the normalized Habitat action.
    """
    minimum = float(min_linear_velocity_mps)
    maximum = float(max_linear_velocity_mps)
    if not minimum < maximum:
        raise ValueError("linear velocity range must be increasing")
    max_angular = float(max_angular_velocity_rps)
    if max_angular <= 0.0:
        raise ValueError("max angular velocity must be positive")
    normalized_linear = (
        2.0 * (float(linear_velocity_mps) - minimum) / (maximum - minimum) - 1.0
    )
    normalized_angular = float(angular_velocity_rps) / max_angular
    return {
        "action": "velocity_control",
        "action_args": {
            "linear_velocity": float(np.clip(normalized_linear, -1.0, 1.0)),
            "angular_velocity": float(np.clip(normalized_angular, -1.0, 1.0)),
            "time_step": float(time_step_s),
            "allow_sliding": bool(allow_sliding),
        },
    }
