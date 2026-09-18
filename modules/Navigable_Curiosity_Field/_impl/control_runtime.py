from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OMNI_RUNTIME = ROOT / "runtime" / "omniguard_runtime"
sys.path.insert(0, str(OMNI_RUNTIME))
from omniguard_esdf.controllers.polar_esdf_controller import PolarEsdfController
from omniguard_esdf.controllers.radar import analyze_radar


class OmniGuardDistanceController:
    """Pure distance-array adapter around the existing PolarEsdfController."""

    def __init__(self, config: dict) -> None:
        self.config = config
        self.controller = PolarEsdfController(config["control"])

    def step(self, distances_m: np.ndarray, goal_heading_rad: float) -> tuple[float, float, dict]:
        distances = np.asarray(distances_m, dtype=np.float32).reshape(-1)
        if distances.size != 360 or not np.isfinite(distances).all():
            raise ValueError("OmniTrav distance message must contain 360 finite values")
        radar_cfg = {"controllers": {"radar": {
            "danger_distance_m": 0.5,
            "traversable_distance_m": 0.7,
            "front_sector_half_width_deg": 25.0,
            "side_sector_min_deg": 35.0,
            "side_sector_max_deg": 120.0,
            "goal_blocked_distance_m": 0.5,
        }}}
        radar = analyze_radar(distances, distances, np.ones(360, np.float32), radar_cfg, goal_heading_rad)
        geometry = self.controller.build_local_geometry(radar, observed_angle_mask=np.ones(360, dtype=bool))
        scoring = self.controller.score_candidate_directions(
            esdf_grid_m=geometry.esdf_grid_m,
            grid_spec=geometry.occupancy.grid_spec,
            theta_goal_rad=float(goal_heading_rad),
            observed_angle_mask=geometry.processed.observed_angle_mask,
            processed=geometry.processed,
            goal_distance_m=999.0,
            point_goal_active=False,
        )
        linear, angular = self.controller.compute_control_from_scoring(scoring=scoring, goal_distance_m=999.0)
        return float(linear), float(angular), {
            "selected_reason": scoring.selected_reason,
            "selected_mode": scoring.selected_mode,
            "selected_rank": scoring.selected_rank,
            "nearest_distance_m": radar.nearest_distance_m,
        }
