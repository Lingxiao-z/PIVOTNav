from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ControlDecision:
    linear_velocity_mps: float
    angular_velocity_rps: float
    mode: str
    goal_heading_deg: float | None
    chosen_heading_deg: float | None
    min_obstacle_distance_m: float
    goal_distance_m: float | None
    note: str = ""
    goal_reached: bool = False
    debug_data: dict[str, Any] = field(default_factory=dict)
