from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class R35BearingConvention:
    angle_range: str = "[-180, 180)"
    zero_direction: str = "source camera forward"
    positive_direction: str = "right in OmniGuard body frame"
    habitat_zero_forward: str = "-Z"
    habitat_zero_right: str = "+X"
    bins: int = 72


def wrap_degrees(angle_degrees: float) -> float:
    return (float(angle_degrees) + 180.0) % 360.0 - 180.0


def wrap_radians(angle_radians: float) -> float:
    return (float(angle_radians) + math.pi) % (2.0 * math.pi) - math.pi


def habitat_delta_to_omniguard_body(
    delta_x: float,
    delta_z: float,
    source_render_yaw_degrees: float,
) -> tuple[float, float]:
    """Return OmniGuard (forward, left) coordinates for a Habitat X/Z delta.

    Habitat R3 yaw=0 looks toward -Z and +X is camera-right. Positive
    render/action yaw turns left, matching the dataset builder convention.
    """

    yaw = math.radians(float(source_render_yaw_degrees))
    sin_yaw = math.sin(yaw)
    cos_yaw = math.cos(yaw)
    forward = -sin_yaw * float(delta_x) - cos_yaw * float(delta_z)
    left = -cos_yaw * float(delta_x) + sin_yaw * float(delta_z)
    return forward, left


def relative_translation_bearing_degrees(
    source_position: Sequence[float],
    target_position: Sequence[float],
    source_render_yaw_degrees: float,
    *,
    minimum_translation_m: float = 0.05,
) -> tuple[float, bool, float]:
    if len(source_position) != 3 or len(target_position) != 3:
        raise ValueError("Habitat positions must be XYZ triples")
    delta_x = float(target_position[0]) - float(source_position[0])
    delta_z = float(target_position[2]) - float(source_position[2])
    translation_m = math.hypot(delta_x, delta_z)
    if translation_m < float(minimum_translation_m):
        return 0.0, False, translation_m
    forward, left = habitat_delta_to_omniguard_body(
        delta_x,
        delta_z,
        source_render_yaw_degrees,
    )
    bearing = wrap_degrees(math.degrees(-math.atan2(left, forward)))
    return bearing, True, translation_m


def bearing_after_source_erp_roll(
    bearing_degrees: float,
    source_roll_pixels: int,
    panorama_width: int = 448,
) -> float:
    if panorama_width <= 0:
        raise ValueError("panorama_width must be positive")
    roll_degrees = float(source_roll_pixels) * 360.0 / float(panorama_width)
    return wrap_degrees(float(bearing_degrees) + roll_degrees)


def bearing_to_bin(angle_degrees: float, bins: int = 72) -> int:
    if bins <= 0:
        raise ValueError("bins must be positive")
    width = 360.0 / float(bins)
    return int(math.floor((wrap_degrees(angle_degrees) + 180.0) / width)) % bins


def bin_center_degrees(index: int, bins: int = 72) -> float:
    if bins <= 0:
        raise ValueError("bins must be positive")
    width = 360.0 / float(bins)
    return wrap_degrees(-180.0 + (int(index) % bins + 0.5) * width)


def circular_error_degrees(predicted: float, target: float) -> float:
    return abs(wrap_degrees(float(predicted) - float(target)))
