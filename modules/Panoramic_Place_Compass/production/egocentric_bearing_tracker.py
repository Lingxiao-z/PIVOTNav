"""Track parent-ERP directions in the robot frame from executed angular motion."""
from __future__ import annotations

import math
from dataclasses import dataclass


def wrap_degrees(value: float) -> float:
    return float(((float(value) + 180.0) % 360.0) - 180.0)


@dataclass
class EgocentricBearingTracker:
    """Command-odometry bridge; no pose, yaw, depth or NavMesh input is used.

    Canonical bearings are right-positive. The project's Habitat action bridge
    makes a negative controller angular command increase canonical heading, so
    the executed heading delta is ``-angular * dt``.
    """

    heading_from_parent_degrees: float = 0.0
    hop_generation: int = 0

    def reset_for_new_hop(self) -> None:
        self.heading_from_parent_degrees = 0.0
        self.hop_generation += 1

    def apply_executed_angular(self, angular_velocity_rps: float, time_step_s: float) -> None:
        heading_delta = -math.degrees(float(angular_velocity_rps) * float(time_step_s))
        self.heading_from_parent_degrees = wrap_degrees(
            self.heading_from_parent_degrees + heading_delta
        )

    def local_bearing(self, parent_bearing_degrees: float) -> float:
        return wrap_degrees(float(parent_bearing_degrees) - self.heading_from_parent_degrees)

    @staticmethod
    def omnitrav_index_for_local_bearing(local_bearing_degrees: float) -> int:
        """OmniTrav bins are -180..179 degrees, so body-forward is index 180."""
        return int(round(wrap_degrees(local_bearing_degrees) + 180.0)) % 360

    def parent_sector_for_local(self, local_sector: int, sector_count: int = 12,
                                forward_sector: int | None = None) -> int:
        if sector_count <= 0:
            raise ValueError("sector_count must be positive")
        if forward_sector is None:
            forward_sector = int(sector_count) // 2
        width = 360.0 / int(sector_count)
        local_angle = wrap_degrees((int(local_sector) - int(forward_sector)) * width)
        parent_angle = wrap_degrees(local_angle + self.heading_from_parent_degrees)
        return (int(round(parent_angle / width)) + int(forward_sector)) % int(sector_count)

    def local_sector_for_parent(self, parent_sector: int, sector_count: int = 12,
                                forward_sector: int | None = None) -> int:
        if sector_count <= 0:
            raise ValueError("sector_count must be positive")
        if forward_sector is None:
            forward_sector = int(sector_count) // 2
        width = 360.0 / int(sector_count)
        parent_angle = wrap_degrees((int(parent_sector) - int(forward_sector)) * width)
        local_angle = self.local_bearing(parent_angle)
        return (int(round(local_angle / width)) + int(forward_sector)) % int(sector_count)
