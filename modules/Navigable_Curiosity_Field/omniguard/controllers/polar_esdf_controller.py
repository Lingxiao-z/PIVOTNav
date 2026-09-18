from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

try:
    from scipy.ndimage import distance_transform_edt as _scipy_distance_transform_edt
except Exception:
    _scipy_distance_transform_edt = None

from ..geometry import (
    body_goal_to_model_angle_rad,
    get_world_coordinate,
    model_angle_rad_to_bin,
    model_heading_rad_to_body_xy,
    signed_model_degrees,
    world_delta_to_body,
    wrap_angle_rad,
)
from .radar import RadarState


_OCCUPANCY_UNKNOWN = np.int8(-1)
_OCCUPANCY_FREE = np.int8(0)
_OCCUPANCY_OCCUPIED = np.int8(1)
_POLAR_ESDF_BASE_RULE = "polar_esdf"
POLAR_ESDF_RULE_OVERRIDES: dict[str, dict[str, float | bool]] = {
    "polar_esdf": {},
    "polar_esdf_wo_esdf_min": {
        "use_esdf_min": False,
        "w_esdf_min": 0.0,
    },
    "polar_esdf_wo_goal_alignment": {
        "use_goal_alignment": False,
        "w_goal_alignment": 0.0,
    },
}


@dataclass
class GridSpec:
    x_min_m: float
    x_max_m: float
    y_min_m: float
    y_max_m: float
    resolution_m: float

    @property
    def width(self) -> int:
        return max(1, int(math.ceil((self.x_max_m - self.x_min_m) / self.resolution_m)))

    @property
    def height(self) -> int:
        return max(1, int(math.ceil((self.y_max_m - self.y_min_m) / self.resolution_m)))

    def local_to_index(self, x_m: float, y_m: float) -> tuple[int, int] | None:
        if not (self.x_min_m <= x_m < self.x_max_m and self.y_min_m <= y_m < self.y_max_m):
            return None
        col = int((x_m - self.x_min_m) / self.resolution_m)
        row = int((y_m - self.y_min_m) / self.resolution_m)
        if row < 0 or row >= self.height or col < 0 or col >= self.width:
            return None
        return row, col

    def to_metadata(self) -> dict[str, float]:
        return {
            "x_min_m": float(self.x_min_m),
            "x_max_m": float(self.x_max_m),
            "y_min_m": float(self.y_min_m),
            "y_max_m": float(self.y_max_m),
            "resolution_m": float(self.resolution_m),
        }


@dataclass
class ProcessedDistances:
    clipped_distance_m: np.ndarray
    ema_distance_m: np.ndarray
    smoothed_distance_m: np.ndarray
    observed_angle_mask: np.ndarray
    obstacle_hit_mask: np.ndarray


@dataclass
class OccupancyResult:
    occupancy_grid: np.ndarray
    occupied_mask: np.ndarray
    observed_grid_mask: np.ndarray
    free_grid_mask: np.ndarray
    boundary_points_local_m: np.ndarray
    grid_spec: GridSpec


@dataclass
class CandidateScoring:
    angles_rad: np.ndarray
    scores: np.ndarray
    mean_esdf_m: np.ndarray
    min_esdf_m: np.ndarray
    best_index: int | None
    front_index: int | None
    rollout_points_local_m: list[np.ndarray]
    linear_velocity_mps: np.ndarray | None = None
    angular_velocity_rps: np.ndarray | None = None
    candidate_kinds: np.ndarray | None = None
    observed_fraction: np.ndarray | None = None
    goal_visible: bool = True
    observed_angle_width_deg: float = 360.0
    visible_min_deg: float | None = -180.0
    visible_max_deg: float | None = 180.0
    visible_width_deg: float = 360.0
    observed_bins: int = 0
    point_goal_active: bool = False
    goal_trackable: bool = True
    goal_in_reachable_sector: bool = True
    align_goal_active: bool = False
    pano_goal_align_active: bool = False
    known_goal_align_active: bool = False
    is_panoramic_observation: bool = False
    reachable_min_deg: float = -60.0
    reachable_max_deg: float = 60.0
    align_goal_heading_rad: float | None = None
    side_unknown: bool = False
    left_clearance_m: float | None = None
    right_clearance_m: float | None = None
    selected_mode: str = "polar_esdf"
    selected_reason: str = ""
    selected_rank: int | None = None
    fallback_active: bool = False


@dataclass
class LocalGeometryResult:
    processed: ProcessedDistances
    occupancy: OccupancyResult
    esdf_grid_m: np.ndarray
    geometry_time_ms: float


def _ring_mean(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values.astype(np.float32, copy=True)
    pad = window // 2
    kernel = np.ones((window,), dtype=np.float32) / float(window)
    padded = np.pad(values.astype(np.float32, copy=False), (pad, pad), mode="wrap")
    return np.convolve(padded, kernel, mode="valid").astype(np.float32)


def _ring_median(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values.astype(np.float32, copy=True)
    radius = window // 2
    output = np.empty_like(values, dtype=np.float32)
    for index in range(values.shape[0]):
        offsets = np.arange(index - radius, index + radius + 1)
        output[index] = float(np.median(values[offsets % values.shape[0]]))
    return output


def _build_inflation_offsets(radius_cells: int) -> list[tuple[int, int]]:
    offsets: list[tuple[int, int]] = []
    for d_row in range(-radius_cells, radius_cells + 1):
        for d_col in range(-radius_cells, radius_cells + 1):
            if d_row * d_row + d_col * d_col <= radius_cells * radius_cells:
                offsets.append((d_row, d_col))
    return offsets


def _dilate_binary_mask(mask: np.ndarray, radius_cells: int) -> np.ndarray:
    if radius_cells <= 0 or not np.any(mask):
        return mask.astype(bool, copy=True)
    dilated = np.zeros_like(mask, dtype=bool)
    offsets = _build_inflation_offsets(radius_cells)
    for row, col in np.argwhere(mask):
        for d_row, d_col in offsets:
            n_row = int(row + d_row)
            n_col = int(col + d_col)
            if 0 <= n_row < mask.shape[0] and 0 <= n_col < mask.shape[1]:
                dilated[n_row, n_col] = True
    return dilated


def _resolve_observed_angle_mask(
    *,
    num_angles: int,
    configured_fov_deg: float,
    observed_angle_mask: np.ndarray | None,
) -> np.ndarray:
    if observed_angle_mask is not None:
        mask = np.asarray(observed_angle_mask, dtype=bool).reshape(-1)
        if mask.shape[0] != num_angles:
            raise ValueError(
                f"observed_angle_mask has {mask.shape[0]} bins, expected {num_angles} bins"
            )
        return mask.astype(bool, copy=True)

    angles_deg = signed_model_degrees(num_angles)
    fov_half_deg = max(0.0, float(configured_fov_deg) * 0.5)
    return (np.abs(angles_deg) <= fov_half_deg + 1e-3).astype(bool)


def _fallback_distance_transform_edt(free_mask: np.ndarray) -> np.ndarray:
    blocked_indices = np.argwhere(~free_mask)
    if blocked_indices.size == 0:
        return np.full(free_mask.shape, np.inf, dtype=np.float32)
    free_indices = np.argwhere(free_mask)
    result = np.zeros(free_mask.shape, dtype=np.float32)
    blocked = blocked_indices.astype(np.float32, copy=False)
    batch_size = 256
    for start in range(0, free_indices.shape[0], batch_size):
        batch = free_indices[start : start + batch_size].astype(np.float32, copy=False)
        diff = batch[:, None, :] - blocked[None, :, :]
        distances = np.sqrt(np.min(np.sum(diff * diff, axis=2), axis=1))
        result[free_indices[start : start + batch_size, 0], free_indices[start : start + batch_size, 1]] = distances
    return result


def _distance_transform_edt(free_mask: np.ndarray) -> np.ndarray:
    if _scipy_distance_transform_edt is not None:
        return _scipy_distance_transform_edt(free_mask).astype(np.float32)
    return _fallback_distance_transform_edt(free_mask)


def _normalize_rule_config(config: dict[str, Any]) -> dict[str, Any]:
    resolved = dict(config)
    requested_rule = str(config.get("rule", _POLAR_ESDF_BASE_RULE)).strip().lower() or _POLAR_ESDF_BASE_RULE
    selected_rule = requested_rule if requested_rule in POLAR_ESDF_RULE_OVERRIDES else _POLAR_ESDF_BASE_RULE

    base_rule_cfg = config.get(_POLAR_ESDF_BASE_RULE)
    if isinstance(base_rule_cfg, dict):
        resolved.update(base_rule_cfg)

    if selected_rule != _POLAR_ESDF_BASE_RULE:
        selected_rule_cfg = config.get(selected_rule)
        if isinstance(selected_rule_cfg, dict):
            resolved.update(selected_rule_cfg)

    resolved.update(POLAR_ESDF_RULE_OVERRIDES[selected_rule])
    resolved["rule"] = selected_rule
    return resolved


def _local_goal_heading_rad(
    pose: dict[str, float] | None,
    goal_pose: dict[str, float] | None,
    goal_angle_rad: float | None,
) -> float | None:
    if pose is not None and goal_pose is not None and "yaw_rad" in pose:
        dx_world = get_world_coordinate(goal_pose, "x") - get_world_coordinate(pose, "x")
        dy_world = get_world_coordinate(goal_pose, "y") - get_world_coordinate(pose, "y")
        x_body, y_body = world_delta_to_body(dx_world, dy_world, float(pose["yaw_rad"]))
        return body_goal_to_model_angle_rad(x_body, y_body)
    if goal_angle_rad is None:
        return None
    return wrap_angle_rad(float(goal_angle_rad))


def _nearest_esdf_samples(
    esdf_grid_m: np.ndarray,
    grid_spec: GridSpec,
    x_samples_m: np.ndarray,
    y_samples_m: np.ndarray,
) -> np.ndarray:
    values = np.zeros_like(x_samples_m, dtype=np.float32)
    col_coords = np.floor((x_samples_m - grid_spec.x_min_m) / grid_spec.resolution_m).astype(np.int32)
    row_coords = np.floor((y_samples_m - grid_spec.y_min_m) / grid_spec.resolution_m).astype(np.int32)
    valid_mask = (
        (row_coords >= 0)
        & (row_coords < esdf_grid_m.shape[0])
        & (col_coords >= 0)
        & (col_coords < esdf_grid_m.shape[1])
    )
    values[valid_mask] = esdf_grid_m[row_coords[valid_mask], col_coords[valid_mask]]
    return values


class PolarEsdfController:
    def __init__(self, config: dict[str, Any]) -> None:
        self.navigation_config = config
        self.rule_config = _normalize_rule_config(config)
        self._prev_ema_distances_m: np.ndarray | None = None
        self._prev_best_theta_rad: float = 0.0
        self._state = "polar_esdf"
        self._recovery_active = False
        self._recovery_turn_direction = 1.0
        self._recovery_steps_since_switch = 0

    def preprocess_distances(
        self,
        distances_360_m: np.ndarray,
        exist_probability: np.ndarray | None = None,
        observed_angle_mask: np.ndarray | None = None,
    ) -> ProcessedDistances:
        cfg = self.rule_config
        min_range = float(cfg["min_range"])
        max_range = float(cfg["max_range"])
        clipped = np.nan_to_num(
            np.asarray(distances_360_m, dtype=np.float32),
            nan=max_range,
            posinf=max_range,
            neginf=min_range,
        )
        clipped = np.clip(clipped, min_range, max_range)

        if bool(cfg.get("use_existence_filter", False)) and exist_probability is not None:
            exist_threshold = float(cfg.get("existence_threshold", 0.5))
            clipped = np.where(
                np.asarray(exist_probability, dtype=np.float32) >= exist_threshold,
                clipped,
                max_range,
            ).astype(np.float32)

        alpha = float(cfg["distance_ema_alpha"])
        if self._prev_ema_distances_m is None or self._prev_ema_distances_m.shape != clipped.shape:
            ema = clipped.astype(np.float32, copy=True)
        else:
            ema = (alpha * clipped + (1.0 - alpha) * self._prev_ema_distances_m).astype(np.float32)
        self._prev_ema_distances_m = ema

        smooth_window = max(1, int(cfg.get("angular_smooth_window", 1)))
        smooth_method = str(cfg.get("angular_smooth_method", "mean")).strip().lower()
        smoothed = _ring_median(ema, smooth_window) if smooth_method == "median" else _ring_mean(ema, smooth_window)
        smoothed = np.clip(smoothed, min_range, max_range).astype(np.float32)

        num_angles = smoothed.shape[0]
        resolved_observed_angle_mask = _resolve_observed_angle_mask(
            num_angles=num_angles,
            configured_fov_deg=float(cfg["fov_deg"]),
            observed_angle_mask=observed_angle_mask,
        )
        obstacle_hit_mask = resolved_observed_angle_mask & (smoothed < max_range - 1e-3)
        return ProcessedDistances(
            clipped_distance_m=clipped,
            ema_distance_m=ema,
            smoothed_distance_m=smoothed,
            observed_angle_mask=resolved_observed_angle_mask.astype(bool),
            obstacle_hit_mask=obstacle_hit_mask.astype(bool),
        )

    def build_local_occupancy(self, processed: ProcessedDistances) -> OccupancyResult:
        cfg = self.rule_config
        grid_spec = GridSpec(
            x_min_m=float(cfg["grid_x_min"]),
            x_max_m=float(cfg["grid_x_max"]),
            y_min_m=float(cfg["grid_y_min"]),
            y_max_m=float(cfg["grid_y_max"]),
            resolution_m=float(cfg["grid_resolution"]),
        )
        occupancy = np.full((grid_spec.height, grid_spec.width), _OCCUPANCY_UNKNOWN, dtype=np.int8)
        free_mask = np.zeros_like(occupancy, dtype=bool)
        observed_grid_mask = np.zeros_like(occupancy, dtype=bool)
        occupied_seed_mask = np.zeros_like(occupancy, dtype=bool)
        boundary_points: list[tuple[float, float]] = []
        ray_endpoints_local: list[tuple[float, float] | None] = []
        ray_step_m = max(grid_spec.resolution_m * 0.5, 0.02)
        angles_rad = np.deg2rad(signed_model_degrees(processed.smoothed_distance_m.shape[0]))

        for angle_rad, distance_m, is_observed, is_hit in zip(
            angles_rad,
            processed.smoothed_distance_m,
            processed.observed_angle_mask,
            processed.obstacle_hit_mask,
        ):
            if not bool(is_observed):
                continue
            ray_length_m = float(min(distance_m, self.rule_config["max_range"]))
            if ray_length_m <= 0.0:
                continue
            sample_steps = max(2, int(math.ceil(ray_length_m / ray_step_m)) + 1)
            sample_range = np.linspace(0.0, ray_length_m, sample_steps, dtype=np.float32)
            if bool(is_hit) and sample_range.size > 1:
                sample_range = sample_range[:-1]
            x_samples = sample_range * math.cos(float(angle_rad))
            y_samples = sample_range * (-math.sin(float(angle_rad)))
            last_valid_point: tuple[float, float] | None = None
            for x_m, y_m in zip(x_samples, y_samples):
                index = grid_spec.local_to_index(float(x_m), float(y_m))
                if index is None:
                    continue
                row, col = index
                free_mask[row, col] = True
                observed_grid_mask[row, col] = True
                last_valid_point = (float(x_m), float(y_m))
            if bool(is_hit):
                x_hit_m, y_hit_m = model_heading_rad_to_body_xy(float(angle_rad), float(ray_length_m))
                index = grid_spec.local_to_index(x_hit_m, y_hit_m)
                if index is not None:
                    row, col = index
                    occupied_seed_mask[row, col] = True
                    observed_grid_mask[row, col] = True
                    boundary_points.append((float(x_hit_m), float(y_hit_m)))
                    last_valid_point = (float(x_hit_m), float(y_hit_m))
            ray_endpoints_local.append(last_valid_point)

        origin_index = grid_spec.local_to_index(0.0, 0.0)
        if origin_index is not None:
            free_mask[origin_index] = True
            observed_grid_mask[origin_index] = True
            sector_fill_mask = np.zeros_like(occupancy, dtype=np.uint8)
            origin_row, origin_col = origin_index
            previous_endpoint = None
            for endpoint in ray_endpoints_local:
                if endpoint is None:
                    previous_endpoint = None
                    continue
                if previous_endpoint is not None:
                    prev_index = grid_spec.local_to_index(previous_endpoint[0], previous_endpoint[1])
                    curr_index = grid_spec.local_to_index(endpoint[0], endpoint[1])
                    if prev_index is not None and curr_index is not None:
                        prev_row, prev_col = prev_index
                        curr_row, curr_col = curr_index
                        polygon = np.asarray(
                            [
                                [origin_col, origin_row],
                                [prev_col, prev_row],
                                [curr_col, curr_row],
                            ],
                            dtype=np.int32,
                        )
                        cv2.fillConvexPoly(sector_fill_mask, polygon, color=1, lineType=cv2.LINE_AA)
                previous_endpoint = endpoint
            free_mask |= sector_fill_mask.astype(bool)
            observed_grid_mask |= sector_fill_mask.astype(bool)

        free_space_fill_radius_cells = max(0, int(cfg.get("free_space_fill_radius_cells", 1)))
        if np.any(free_mask):
            free_mask = _dilate_binary_mask(free_mask, free_space_fill_radius_cells)

        inflation_radius_m = max(0.0, float(cfg["robot_radius"]) + float(cfg["safety_margin"]))
        inflation_radius_cells = max(0, int(math.ceil(inflation_radius_m / grid_spec.resolution_m)))
        occupied_mask = _dilate_binary_mask(occupied_seed_mask, inflation_radius_cells)

        occupancy[free_mask] = _OCCUPANCY_FREE
        occupancy[occupied_mask] = _OCCUPANCY_OCCUPIED
        return OccupancyResult(
            occupancy_grid=occupancy,
            occupied_mask=occupied_mask,
            observed_grid_mask=observed_grid_mask,
            free_grid_mask=free_mask,
            boundary_points_local_m=np.asarray(boundary_points, dtype=np.float32).reshape(-1, 2),
            grid_spec=grid_spec,
        )

    def compute_esdf(self, occupancy: OccupancyResult) -> np.ndarray:
        cfg = self.rule_config
        unknown_mask = occupancy.occupancy_grid == _OCCUPANCY_UNKNOWN
        occupied_mask = occupancy.occupancy_grid == _OCCUPANCY_OCCUPIED
        unknown_as_occupied = bool(cfg.get("unknown_as_occupied", True))
        blocked_mask = occupied_mask | unknown_mask if unknown_as_occupied else occupied_mask
        free_mask = ~blocked_mask

        if not np.any(free_mask):
            esdf_grid = np.zeros(occupancy.occupancy_grid.shape, dtype=np.float32)
        elif not np.any(blocked_mask):
            esdf_grid = np.full(occupancy.occupancy_grid.shape, float(cfg["esdf_max_dist"]), dtype=np.float32)
        else:
            esdf_cells = _distance_transform_edt(free_mask)
            esdf_grid = esdf_cells.astype(np.float32) * float(occupancy.grid_spec.resolution_m)
            esdf_grid[blocked_mask] = 0.0

        if not unknown_as_occupied and np.any(unknown_mask):
            unknown_penalty_dist = float(cfg.get("unknown_penalty_dist_m", cfg["phi_safe"]))
            esdf_grid[unknown_mask] = np.minimum(esdf_grid[unknown_mask], unknown_penalty_dist)

        return np.minimum(esdf_grid, float(cfg["esdf_max_dist"])).astype(np.float32)

    def build_local_geometry(
        self,
        radar: RadarState,
        observed_angle_mask: np.ndarray | None = None,
    ) -> LocalGeometryResult:
        geometry_start_time = time.perf_counter()
        processed = self.preprocess_distances(
            distances_360_m=radar.effective_distance_m,
            exist_probability=radar.exist_probability,
            observed_angle_mask=observed_angle_mask,
        )
        occupancy = self.build_local_occupancy(processed)
        esdf_grid_m = self.compute_esdf(occupancy)
        geometry_time_ms = (time.perf_counter() - geometry_start_time) * 1000.0
        return LocalGeometryResult(
            processed=processed,
            occupancy=occupancy,
            esdf_grid_m=esdf_grid_m,
            geometry_time_ms=float(geometry_time_ms),
        )

    def _resolved_observed_mask(self, observed_angle_mask: np.ndarray | None, num_angles: int) -> np.ndarray:
        return _resolve_observed_angle_mask(
            num_angles=num_angles,
            configured_fov_deg=float(self.rule_config["fov_deg"]),
            observed_angle_mask=observed_angle_mask,
        )

    @staticmethod
    def _angle_is_observed(angle_rad: float, observed_angle_mask: np.ndarray) -> bool:
        if observed_angle_mask.size == 0:
            return True
        index = model_angle_rad_to_bin(float(angle_rad), int(observed_angle_mask.shape[0]))
        return bool(observed_angle_mask[index])

    @staticmethod
    def _observed_width_deg(observed_angle_mask: np.ndarray) -> float:
        if observed_angle_mask.size == 0:
            return 360.0
        return float(np.count_nonzero(observed_angle_mask)) * 360.0 / float(observed_angle_mask.shape[0])

    @staticmethod
    def _to_signed_degrees(angle_deg: float) -> float:
        return float(((float(angle_deg) + 180.0) % 360.0) - 180.0)

    @staticmethod
    def _observed_angle_stats(
        observed_angle_mask: np.ndarray,
    ) -> tuple[float | None, float | None, float, int]:
        mask = np.asarray(observed_angle_mask, dtype=bool).reshape(-1)
        if mask.size == 0:
            return -180.0, 180.0, 360.0, 0
        observed_bins = int(np.count_nonzero(mask))
        if observed_bins <= 0:
            return None, None, 0.0, 0
        if observed_bins == mask.shape[0]:
            return -180.0, 180.0, 360.0, observed_bins

        angles_deg = signed_model_degrees(int(mask.shape[0])).astype(np.float64)
        observed_angles = np.sort((angles_deg[mask] + 360.0) % 360.0)
        extended = np.concatenate([observed_angles, observed_angles[:1] + 360.0])
        gaps = np.diff(extended)
        largest_gap_index = int(np.argmax(gaps))
        visible_min_360 = float(extended[largest_gap_index + 1] % 360.0)
        visible_max_360 = float(extended[largest_gap_index] % 360.0)
        visible_width_deg = float((visible_max_360 - visible_min_360) % 360.0)
        return (
            PolarEsdfController._to_signed_degrees(visible_min_360),
            PolarEsdfController._to_signed_degrees(visible_max_360),
            visible_width_deg,
            observed_bins,
        )

    @staticmethod
    def _candidate_observed_mask(
        candidate_angles_rad: np.ndarray,
        observed_angle_mask: np.ndarray,
        margin_deg: float,
    ) -> np.ndarray:
        mask = np.asarray(observed_angle_mask, dtype=bool).reshape(-1)
        if mask.size == 0 or np.all(mask):
            return np.ones(candidate_angles_rad.shape, dtype=bool)
        num_angles = int(mask.shape[0])
        margin_bins = max(0, int(math.ceil(float(margin_deg) * float(num_angles) / 360.0)))
        candidate_bins = np.asarray(
            [model_angle_rad_to_bin(float(angle), num_angles) for angle in candidate_angles_rad],
            dtype=np.int32,
        )
        valid = np.ones(candidate_bins.shape, dtype=bool)
        for offset in range(-margin_bins, margin_bins + 1):
            valid &= mask[(candidate_bins + offset) % num_angles]
        return valid

    @staticmethod
    def _observed_fraction_for_points(points: np.ndarray, observed_angle_mask: np.ndarray, ignore_near_m: float) -> float:
        if observed_angle_mask.size == 0 or np.all(observed_angle_mask):
            return 1.0
        rollout = np.asarray(points, dtype=np.float32).reshape(-1, 2)
        if rollout.size == 0:
            return 1.0
        distances = np.linalg.norm(rollout, axis=1)
        valid = distances >= float(ignore_near_m)
        if not np.any(valid):
            return 1.0
        headings = -np.arctan2(rollout[valid, 1], rollout[valid, 0])
        num_angles = int(observed_angle_mask.shape[0])
        bins = np.asarray(
            [model_angle_rad_to_bin(float(angle), num_angles) for angle in headings],
            dtype=np.int32,
        )
        return float(np.mean(observed_angle_mask[bins]))

    def score_candidate_directions(
        self,
        *,
        esdf_grid_m: np.ndarray,
        grid_spec: GridSpec,
        theta_goal_rad: float,
        observed_angle_mask: np.ndarray | None = None,
        processed: ProcessedDistances | None = None,
        goal_distance_m: float | None = None,
        point_goal_active: bool = False,
    ) -> CandidateScoring:
        cfg = self.rule_config
        if observed_angle_mask is not None:
            num_angles = int(np.asarray(observed_angle_mask).reshape(-1).shape[0])
        elif processed is not None:
            num_angles = int(processed.smoothed_distance_m.shape[0])
        else:
            num_angles = 360
        observed_mask = self._resolved_observed_mask(observed_angle_mask, num_angles)
        visible_min_deg, visible_max_deg, visible_width_deg, observed_bins = self._observed_angle_stats(observed_mask)
        observed_width_deg = self._observed_width_deg(observed_mask)
        goal_visible = self._angle_is_observed(theta_goal_rad, observed_mask)
        footprint_clearance_m = float(cfg["robot_radius"]) + float(cfg["safety_margin"])
        ignore_near_distance_m = max(
            footprint_clearance_m,
            float(cfg.get("rollout_ignore_near_distance_m", footprint_clearance_m)),
        )
        rollout_length = float(cfg.get("rollout_length", 1.5))
        rollout_num_points = max(2, int(cfg.get("rollout_num_points", 10)))
        rollout_distance = np.linspace(0.0, rollout_length, rollout_num_points, dtype=np.float32)
        candidate_min_deg = float(cfg.get("candidate_angle_min_deg", -60.0))
        candidate_max_deg = float(cfg.get("candidate_angle_max_deg", 60.0))
        candidate_step_deg = max(1e-3, float(cfg.get("candidate_angle_step_deg", 5.0)))
        observed_margin_deg = float(cfg.get("observed_sector_margin_deg", 5.0))
        goal_trackable_margin_deg = max(0.0, float(cfg.get("goal_trackable_margin_deg", 0.0)))
        candidate_low_deg = min(candidate_min_deg, candidate_max_deg)
        candidate_high_deg = max(candidate_min_deg, candidate_max_deg)
        reachable_min_deg = candidate_low_deg + goal_trackable_margin_deg
        reachable_max_deg = candidate_high_deg - goal_trackable_margin_deg
        if reachable_min_deg > reachable_max_deg:
            reachable_mid_deg = 0.5 * (candidate_low_deg + candidate_high_deg)
            reachable_min_deg = reachable_mid_deg
            reachable_max_deg = reachable_mid_deg
        goal_heading_deg = math.degrees(wrap_angle_rad(float(theta_goal_rad)))
        goal_in_reachable_sector = bool(
            reachable_min_deg - 1e-3 <= goal_heading_deg <= reachable_max_deg + 1e-3
        )
        pano_align_min_visible_width_deg = float(cfg.get("pano_align_min_visible_width_deg", 300.0))
        pano_min_observed_bins = int(
            math.ceil(max(0.0, pano_align_min_visible_width_deg) * float(num_angles) / 360.0)
        )
        is_panoramic_observation = bool(
            observed_width_deg >= pano_align_min_visible_width_deg
            or visible_width_deg >= pano_align_min_visible_width_deg
            or observed_bins >= pano_min_observed_bins
        )
        goal_reached = bool(
            goal_distance_m is not None and float(goal_distance_m) <= float(cfg["goal_tolerance_m"])
        )
        pano_goal_align_active = bool(
            cfg.get("enable_pano_goal_align", True)
            and is_panoramic_observation
            and goal_visible
            and not goal_in_reachable_sector
            and not goal_reached
        )
        known_goal_align_active = bool(
            cfg.get("enable_known_goal_align", True)
            and bool(point_goal_active)
            and not goal_visible
            and not goal_reached
        )
        align_goal_active = bool(pano_goal_align_active or known_goal_align_active)
        goal_trackable = bool(goal_visible and goal_in_reachable_sector)
        candidate_angles_deg = np.arange(
            candidate_min_deg,
            candidate_max_deg + 0.5 * candidate_step_deg,
            candidate_step_deg,
            dtype=np.float32,
        )
        candidate_angles_rad = np.deg2rad(candidate_angles_deg).astype(np.float32)
        candidate_valid_mask = self._candidate_observed_mask(
            candidate_angles_rad=candidate_angles_rad,
            observed_angle_mask=observed_mask,
            margin_deg=observed_margin_deg,
        )

        front_index = int(np.argmin(np.abs(candidate_angles_rad))) if candidate_angles_rad.size > 0 else None

        scores_list: list[float] = []
        mean_esdf_list: list[float] = []
        min_esdf_list: list[float] = []
        observed_fraction_list: list[float] = []
        rollout_points_local_m: list[np.ndarray] = []
        kind_list: list[str] = []

        use_goal_alignment = bool(cfg.get("use_goal_alignment", True))
        use_esdf_min = bool(cfg.get("use_esdf_min", True))
        w_goal = float(cfg["w_goal"]) * (float(cfg.get("w_goal_alignment", 1.0)) if use_goal_alignment else 0.0)
        w_collision = float(cfg["w_collision"]) * (float(cfg.get("w_esdf_min", 1.0)) if use_esdf_min else 0.0)
        w_turn_change = float(cfg["w_turn_change"])
        phi_safe = float(cfg["phi_safe"])
        prev_theta_rad = float(self._prev_best_theta_rad)

        for angle_rad, is_valid in zip(candidate_angles_rad, candidate_valid_mask):
            points = np.stack(
                [
                    rollout_distance * math.cos(float(angle_rad)),
                    rollout_distance * (-math.sin(float(angle_rad))),
                ],
                axis=1,
            ).astype(np.float32)
            phi_samples = _nearest_esdf_samples(
                esdf_grid_m=esdf_grid_m,
                grid_spec=grid_spec,
                x_samples_m=points[:, 0],
                y_samples_m=points[:, 1],
            )
            distances = np.linalg.norm(points, axis=1)
            aggregate_mask = distances >= ignore_near_distance_m
            phi_eval = phi_samples[aggregate_mask] if np.any(aggregate_mask) else phi_samples
            mean_phi = float(np.mean(phi_eval))
            min_phi = float(np.min(phi_eval))
            observed_fraction = self._observed_fraction_for_points(points, observed_mask, ignore_near_distance_m)
            goal_alignment = math.cos(wrap_angle_rad(float(angle_rad) - float(theta_goal_rad)))
            collision_penalty = max(0.0, phi_safe - min_phi)
            turn_change_penalty = abs(wrap_angle_rad(float(angle_rad) - prev_theta_rad))
            score = float(
                w_goal * goal_alignment
                - w_collision * collision_penalty
                - w_turn_change * turn_change_penalty
            )
            if not bool(is_valid):
                score = -np.inf

            scores_list.append(score)
            mean_esdf_list.append(mean_phi)
            min_esdf_list.append(min_phi)
            observed_fraction_list.append(observed_fraction)
            rollout_points_local_m.append(points)
            kind_list.append("heading")

        scores = np.asarray(scores_list, dtype=np.float32)
        mean_esdf = np.asarray(mean_esdf_list, dtype=np.float32)
        min_esdf = np.asarray(min_esdf_list, dtype=np.float32)
        observed_fractions = np.asarray(observed_fraction_list, dtype=np.float32)
        candidate_kinds = np.asarray(kind_list, dtype=object)

        best_index = int(np.argmax(scores)) if np.any(np.isfinite(scores)) else None
        selected_mode = "polar_esdf"
        selected_reason = "best_heading_candidate" if best_index is not None else "no_valid_candidate"
        if pano_goal_align_active:
            selected_mode = "align_goal"
            selected_reason = "pano_goal_visible_outside_reachable_sector"
        if known_goal_align_active:
            selected_mode = "align_known_goal"
            selected_reason = "known_point_goal_not_visible"
        if goal_reached:
            selected_reason = "goal_reached"

        self._state = selected_mode
        return CandidateScoring(
            angles_rad=candidate_angles_rad,
            scores=scores,
            mean_esdf_m=mean_esdf,
            min_esdf_m=min_esdf,
            best_index=best_index,
            front_index=front_index,
            rollout_points_local_m=rollout_points_local_m,
            linear_velocity_mps=None,
            angular_velocity_rps=None,
            candidate_kinds=candidate_kinds,
            observed_fraction=observed_fractions,
            goal_visible=goal_visible,
            observed_angle_width_deg=observed_width_deg,
            visible_min_deg=visible_min_deg,
            visible_max_deg=visible_max_deg,
            visible_width_deg=visible_width_deg,
            observed_bins=observed_bins,
            point_goal_active=bool(point_goal_active),
            goal_trackable=goal_trackable,
            goal_in_reachable_sector=goal_in_reachable_sector,
            align_goal_active=align_goal_active,
            pano_goal_align_active=pano_goal_align_active,
            known_goal_align_active=known_goal_align_active,
            is_panoramic_observation=is_panoramic_observation,
            reachable_min_deg=reachable_min_deg,
            reachable_max_deg=reachable_max_deg,
            align_goal_heading_rad=float(theta_goal_rad),
            side_unknown=False,
            left_clearance_m=None,
            right_clearance_m=None,
            selected_mode=selected_mode,
            selected_reason=selected_reason,
            selected_rank=None,
            fallback_active=False,
        )

    def compute_control_from_scoring(
        self,
        *,
        scoring: CandidateScoring,
        goal_distance_m: float,
    ) -> tuple[float, float]:
        cfg = self.rule_config
        if scoring.align_goal_active:
            return self._compute_align_goal_control(scoring=scoring)
        if scoring.best_index is None:
            return 0.0, 0.0
        if goal_distance_m <= float(cfg["goal_tolerance_m"]):
            scoring.selected_reason = "goal_reached"
            scoring.selected_mode = "polar_esdf"
            scoring.fallback_active = False
            self._recovery_active = False
            self._recovery_steps_since_switch = 0
            self._state = "polar_esdf"
            return 0.0, 0.0

        finite_indices = np.flatnonzero(np.isfinite(scoring.scores))
        if finite_indices.size == 0:
            scoring.best_index = None
            scoring.selected_reason = "no_valid_candidate"
            scoring.selected_mode = "polar_esdf"
            scoring.fallback_active = False
            return 0.0, 0.0

        ranked_indices = finite_indices[np.argsort(scoring.scores[finite_indices])[::-1]]
        phi_stop = float(cfg["phi_stop"])
        was_recovering = bool(self._recovery_active)

        for rank, candidate_index in enumerate(ranked_indices):
            path_min_esdf_m = float(scoring.min_esdf_m[candidate_index])
            path_mean_esdf_m = float(scoring.mean_esdf_m[candidate_index])
            candidate_theta_rad = float(scoring.angles_rad[candidate_index])
            linear_velocity, angular_velocity = self.compute_control_command(
                best_theta_rad=candidate_theta_rad,
                path_min_esdf_m=path_min_esdf_m,
                path_mean_esdf_m=path_mean_esdf_m,
                goal_distance_m=goal_distance_m,
            )
            if float(path_min_esdf_m) > phi_stop + 1e-6 and linear_velocity > 0.0:
                scoring.best_index = int(candidate_index)
                scoring.selected_rank = int(rank)
                if was_recovering:
                    scoring.selected_reason = "recovery_exit_non_stop_candidate"
                else:
                    scoring.selected_reason = "best_non_stop_candidate" if rank > 0 else "best_heading_candidate"
                scoring.selected_mode = "polar_esdf"
                scoring.fallback_active = rank > 0
                best_theta_rad = candidate_theta_rad
                self._recovery_active = False
                self._recovery_steps_since_switch = 0
                self._state = "polar_esdf"
                self._prev_best_theta_rad = best_theta_rad
                return linear_velocity, angular_velocity

        recovery_index = int(ranked_indices[0])
        best_theta_rad = float(scoring.angles_rad[recovery_index])
        if not self._recovery_active:
            turn_command = -best_theta_rad
            if abs(turn_command) > 1e-3:
                self._recovery_turn_direction = math.copysign(1.0, turn_command)
            else:
                self._recovery_turn_direction *= -1.0
            self._recovery_steps_since_switch = 0
        else:
            self._recovery_steps_since_switch += 1
            switch_period = max(1, int(ranked_indices.size))
            if self._recovery_steps_since_switch >= switch_period:
                self._recovery_turn_direction *= -1.0
                self._recovery_steps_since_switch = 0

        self._recovery_active = True
        self._state = "recovery_turn"
        scoring.best_index = recovery_index
        scoring.selected_rank = 0
        scoring.selected_mode = "recovery_turn"
        scoring.selected_reason = "recovery_turn_scan"
        scoring.fallback_active = True

        angular_velocity = self._recovery_turn_direction * float(cfg["max_angular_speed"])
        linear_velocity = 0.0
        self._prev_best_theta_rad = best_theta_rad
        return linear_velocity, angular_velocity

    def _compute_align_goal_control(self, *, scoring: CandidateScoring) -> tuple[float, float]:
        cfg = self.rule_config
        if scoring.known_goal_align_active:
            selected_mode = "align_known_goal"
            selected_reason = "known_point_goal_not_visible"
        else:
            selected_mode = "align_goal"
            selected_reason = "pano_goal_visible_outside_reachable_sector"

        theta_goal_rad = wrap_angle_rad(float(scoring.align_goal_heading_rad or 0.0))
        angular_velocity = -float(cfg["k_omega"]) * theta_goal_rad
        angular_velocity = max(
            -float(cfg["max_angular_speed"]),
            min(float(cfg["max_angular_speed"]), angular_velocity),
        )

        finite_indices = np.flatnonzero(np.isfinite(scoring.scores))
        if finite_indices.size > 0:
            goal_heading_deg = math.degrees(theta_goal_rad)
            boundary_deg = float(np.clip(goal_heading_deg, scoring.reachable_min_deg, scoring.reachable_max_deg))
            boundary_rad = math.radians(boundary_deg)
            nearest_offset = np.asarray(
                [
                    abs(wrap_angle_rad(float(scoring.angles_rad[index]) - boundary_rad))
                    for index in finite_indices
                ],
                dtype=np.float32,
            )
            align_index = int(finite_indices[int(np.argmin(nearest_offset))])
            scoring.best_index = align_index
            if scoring.candidate_kinds is not None and 0 <= align_index < scoring.candidate_kinds.shape[0]:
                scoring.candidate_kinds[align_index] = selected_mode
            self._prev_best_theta_rad = float(scoring.angles_rad[align_index])

        self._recovery_active = False
        self._recovery_steps_since_switch = 0
        self._state = selected_mode
        scoring.selected_mode = selected_mode
        scoring.selected_reason = selected_reason
        scoring.selected_rank = None
        scoring.fallback_active = False
        return 0.0, float(angular_velocity)

    def compute_control_command(
        self,
        *,
        best_theta_rad: float,
        path_min_esdf_m: float,
        path_mean_esdf_m: float,
        goal_distance_m: float,
    ) -> tuple[float, float]:
        cfg = self.rule_config
        turn_alignment = max(0.0, math.cos(float(best_theta_rad)))
        min_turn_speed_scale = float(np.clip(cfg.get("min_turn_speed_scale", 0.4), 0.0, 1.0))
        turn_scale = min_turn_speed_scale + (1.0 - min_turn_speed_scale) * turn_alignment
        phi_stop = float(cfg["phi_stop"])
        phi_slow = max(float(cfg["phi_slow"]), phi_stop + 1e-6)
        speed_clearance_mean_weight = float(np.clip(cfg.get("speed_clearance_mean_weight", 0.5), 0.0, 1.0))
        speed_clearance_m = float(path_min_esdf_m) + speed_clearance_mean_weight * max(
            0.0,
            float(path_mean_esdf_m) - float(path_min_esdf_m),
        )
        safety_scale = float(np.clip((speed_clearance_m - phi_stop) / (phi_slow - phi_stop), 0.0, 1.0))
        linear_velocity = float(cfg["max_linear_speed"]) * safety_scale * turn_scale
        linear_velocity = min(linear_velocity, float(cfg["max_linear_speed"]))
        linear_velocity = max(float(cfg["min_linear_speed"]), linear_velocity)
        if float(path_min_esdf_m) <= phi_stop + 1e-6:
            linear_velocity = 0.0
        if goal_distance_m <= float(cfg["goal_tolerance_m"]):
            linear_velocity = 0.0

        angular_velocity = -float(cfg["k_omega"]) * float(best_theta_rad)
        angular_velocity = max(
            -float(cfg["max_angular_speed"]),
            min(float(cfg["max_angular_speed"]), angular_velocity),
        )
        return float(linear_velocity), float(angular_velocity)

    def _build_debug_data(
        self,
        *,
        processed: ProcessedDistances,
        occupancy: OccupancyResult,
        esdf_grid_m: np.ndarray,
        scoring: CandidateScoring,
        theta_goal_rad: float,
        best_theta_rad: float | None,
        best_score: float | None,
        front_min_esdf_m: float,
        path_min_esdf_m: float | None,
        path_mean_esdf_m: float | None,
        recovery_active: bool,
        recovery_reason: str,
        geometry_time_ms: float | None = None,
    ) -> dict[str, Any]:
        return {
            "polar_esdf": {
                "state": str(self._state),
                "theta_goal_rad": float(theta_goal_rad),
                "theta_goal_deg": float(math.degrees(theta_goal_rad)),
                "theta_best_rad": None if best_theta_rad is None else float(best_theta_rad),
                "theta_best_deg": None if best_theta_rad is None else float(math.degrees(best_theta_rad)),
                "front_min_esdf_m": float(front_min_esdf_m),
                "path_min_esdf_m": None if path_min_esdf_m is None else float(path_min_esdf_m),
                "path_mean_esdf_m": None if path_mean_esdf_m is None else float(path_mean_esdf_m),
                "geometry_time_ms": None if geometry_time_ms is None else float(geometry_time_ms),
                "recovery_active": bool(recovery_active),
                "recovery_reason": str(recovery_reason),
                "best_candidate_score": None if best_score is None else float(best_score),
                "candidate_angles_deg": np.degrees(scoring.angles_rad).astype(np.float32),
                "candidate_scores": scoring.scores.astype(np.float32),
                "candidate_mean_esdf_m": scoring.mean_esdf_m.astype(np.float32),
                "candidate_min_esdf_m": scoring.min_esdf_m.astype(np.float32),
                "candidate_linear_velocity_mps": np.asarray(
                    [] if scoring.linear_velocity_mps is None else scoring.linear_velocity_mps,
                    dtype=np.float32,
                ),
                "candidate_angular_velocity_rps": np.asarray(
                    [] if scoring.angular_velocity_rps is None else scoring.angular_velocity_rps,
                    dtype=np.float32,
                ),
                "candidate_kinds": []
                if scoring.candidate_kinds is None
                else [str(value) for value in scoring.candidate_kinds.tolist()],
                "candidate_observed_fraction": np.asarray(
                    [] if scoring.observed_fraction is None else scoring.observed_fraction,
                    dtype=np.float32,
                ),
                "goal_visible": bool(scoring.goal_visible),
                "observed_angle_width_deg": float(scoring.observed_angle_width_deg),
                "visible_min_deg": scoring.visible_min_deg,
                "visible_max_deg": scoring.visible_max_deg,
                "visible_width_deg": float(scoring.visible_width_deg),
                "observed_bins": int(scoring.observed_bins),
                "point_goal_active": bool(scoring.point_goal_active),
                "goal_trackable": bool(scoring.goal_trackable),
                "goal_in_reachable_sector": bool(scoring.goal_in_reachable_sector),
                "align_goal_active": bool(scoring.align_goal_active),
                "pano_goal_align_active": bool(scoring.pano_goal_align_active),
                "known_goal_align_active": bool(scoring.known_goal_align_active),
                "is_panoramic_observation": bool(scoring.is_panoramic_observation),
                "reachable_min_deg": float(scoring.reachable_min_deg),
                "reachable_max_deg": float(scoring.reachable_max_deg),
                "side_unknown": bool(scoring.side_unknown),
                "left_clearance_m": scoring.left_clearance_m,
                "right_clearance_m": scoring.right_clearance_m,
                "selected_mode": str(scoring.selected_mode),
                "selected_reason": str(scoring.selected_reason),
                "rollout_points_local_m": [points.astype(np.float32) for points in scoring.rollout_points_local_m],
                "raw_distance_m": processed.clipped_distance_m.astype(np.float32),
                "ema_distance_m": processed.ema_distance_m.astype(np.float32),
                "smoothed_distance_m": processed.smoothed_distance_m.astype(np.float32),
                "observed_angle_mask": processed.observed_angle_mask.astype(np.uint8),
                "boundary_points_local_m": occupancy.boundary_points_local_m.astype(np.float32),
                "occupancy_grid": occupancy.occupancy_grid.astype(np.int8),
                "esdf_grid_m": esdf_grid_m.astype(np.float32),
                "grid_meta": occupancy.grid_spec.to_metadata(),
            }
        }

    def resolve_debug_payload(
        self,
        *,
        radar: RadarState,
        goal_heading_rad: float,
        goal_distance_m: float,
        observed_angle_mask: np.ndarray | None = None,
        point_goal_active: bool = False,
    ) -> tuple[LocalGeometryResult, CandidateScoring, dict[str, Any]]:
        self._state = "OFFLINE_CHECK"
        geometry = self.build_local_geometry(radar, observed_angle_mask=observed_angle_mask)
        scoring = self.score_candidate_directions(
            esdf_grid_m=geometry.esdf_grid_m,
            grid_spec=geometry.occupancy.grid_spec,
            theta_goal_rad=goal_heading_rad,
            observed_angle_mask=geometry.processed.observed_angle_mask,
            processed=geometry.processed,
            goal_distance_m=goal_distance_m,
            point_goal_active=point_goal_active,
        )
        linear_velocity, angular_velocity = self.compute_control_from_scoring(
            scoring=scoring,
            goal_distance_m=goal_distance_m,
        )
        best_theta_rad: float | None = None
        best_score: float | None = None
        path_min_esdf_m: float | None = None
        path_mean_esdf_m: float | None = None
        if scoring.best_index is not None and np.isfinite(scoring.scores[scoring.best_index]):
            best_theta_rad = float(scoring.angles_rad[scoring.best_index])
            best_score = float(scoring.scores[scoring.best_index])
            path_min_esdf_m = float(scoring.min_esdf_m[scoring.best_index])
            path_mean_esdf_m = float(scoring.mean_esdf_m[scoring.best_index])
        front_min_esdf_m = (
            float(scoring.min_esdf_m[scoring.front_index])
            if scoring.front_index is not None and scoring.front_index < scoring.min_esdf_m.shape[0]
            else 0.0
        )
        debug_payload = self._build_debug_data(
            processed=geometry.processed,
            occupancy=geometry.occupancy,
            esdf_grid_m=geometry.esdf_grid_m,
            scoring=scoring,
            theta_goal_rad=goal_heading_rad,
            best_theta_rad=best_theta_rad,
            best_score=best_score,
            front_min_esdf_m=front_min_esdf_m,
            path_min_esdf_m=path_min_esdf_m,
            path_mean_esdf_m=path_mean_esdf_m,
            recovery_active=bool(scoring.fallback_active),
            recovery_reason=str(scoring.selected_reason),
            geometry_time_ms=geometry.geometry_time_ms,
        )["polar_esdf"]
        debug_payload["state"] = "OFFLINE_CHECK"
        debug_payload["offline_goal_distance_m"] = float(goal_distance_m)
        debug_payload["suggested_linear_velocity_mps"] = float(linear_velocity)
        debug_payload["suggested_angular_velocity_rps"] = float(angular_velocity)
        return geometry, scoring, debug_payload
