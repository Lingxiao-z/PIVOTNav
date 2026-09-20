from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Callable

import numpy as np


NUM_DIRECTIONS = 12
SECTOR_WIDTH_DEGREES = 30.0
LOCAL_RADIUS_METERS = 3.0
FG_PATH_RATIO = 1.05
DMAX_METERS = 20.0


def wrap_degrees(degrees: float) -> float:
    return float(degrees % 360.0)


def signed_angular_offset(degrees: float, center_degrees: float) -> float:
    return float((degrees - center_degrees + 180.0) % 360.0 - 180.0)


def direction_center_degrees(direction_index: int) -> float:
    if not 0 <= direction_index < NUM_DIRECTIONS:
        raise ValueError("direction_index must be in [0, 11]")
    return direction_index * SECTOR_WIDTH_DEGREES


def direction_index_from_offset(delta_x: float, delta_z: float) -> int:
    """Map a world offset to the canonical Source-relative direction index.

    Identity Habitat camera forward is world -Z. Positive indices rotate toward
    world +X, matching the empirically frozen ERP convention. Boundaries are
    half-open: direction i owns [-15, +15) degrees around its center.
    """
    if not np.isfinite([delta_x, delta_z]).all() or abs(delta_x) + abs(delta_z) == 0:
        raise ValueError("direction offset must be finite and nonzero")
    degrees = wrap_degrees(math.degrees(math.atan2(float(delta_x), float(-delta_z))))
    return int(math.floor((degrees + 15.0) / SECTOR_WIDTH_DEGREES)) % NUM_DIRECTIONS


def target_offset(direction_index: int, radius_m: float = LOCAL_RADIUS_METERS) -> np.ndarray:
    if not math.isfinite(radius_m) or radius_m <= 0:
        raise ValueError("radius_m must be finite and positive")
    radians = math.radians(direction_center_degrees(direction_index))
    return np.asarray([math.sin(radians) * radius_m, 0.0, -math.cos(radians) * radius_m], dtype=np.float32)


@dataclass(frozen=True)
class FGDecision:
    direction_index: int
    direction_center_degrees: float
    requested_target: list[float]
    snapped_target: list[float]
    snap_offset_m: float
    shortest_path_distance_m: float
    path_limit_m: float
    max_snap_offset_m: float
    navigable: bool
    path_found: bool
    fg_label: int
    rejection_reason: str | None

    def to_dict(self) -> dict:
        return asdict(self)


def decide_fg(
    *,
    direction_index: int,
    requested_target: np.ndarray,
    snapped_target: np.ndarray,
    navigable: bool,
    path_found: bool,
    shortest_path_distance_m: float,
    radius_m: float = LOCAL_RADIUS_METERS,
    path_ratio: float = FG_PATH_RATIO,
    max_snap_offset_m: float = 0.25,
) -> FGDecision:
    requested = np.asarray(requested_target, dtype=np.float64)
    snapped = np.asarray(snapped_target, dtype=np.float64)
    snap_offset = float(np.linalg.norm(snapped - requested)) if np.isfinite(snapped).all() else float("inf")
    path_limit = float(path_ratio * radius_m)
    reason = None
    if not np.isfinite(snapped).all():
        reason = "snap_nonfinite"
    elif not navigable:
        reason = "snapped_target_not_navigable"
    elif snap_offset > max_snap_offset_m:
        reason = "snap_offset_exceeds_limit"
    elif not path_found or not math.isfinite(shortest_path_distance_m):
        reason = "shortest_path_unreachable"
    elif shortest_path_distance_m > path_limit:
        reason = "shortest_path_exceeds_1p05_times_3m"
    return FGDecision(
        direction_index=direction_index,
        direction_center_degrees=direction_center_degrees(direction_index),
        requested_target=[float(x) for x in requested],
        snapped_target=[float(x) for x in snapped],
        snap_offset_m=snap_offset,
        shortest_path_distance_m=float(shortest_path_distance_m),
        path_limit_m=path_limit,
        max_snap_offset_m=float(max_snap_offset_m),
        navigable=bool(navigable),
        path_found=bool(path_found),
        fg_label=int(reason is None),
        rejection_reason=reason,
    )


def local_polar_offsets(
    *,
    radial_resolution_m: float = 0.15,
    angular_resolution_degrees: float = 5.0,
    radius_m: float = LOCAL_RADIUS_METERS,
) -> np.ndarray:
    """Return deterministic local candidates; no global random points are used."""
    if radial_resolution_m <= 0 or angular_resolution_degrees <= 0 or radius_m <= 0:
        raise ValueError("local-grid resolutions and radius must be positive")
    radii = np.arange(radial_resolution_m, radius_m + radial_resolution_m * 0.5, radial_resolution_m)
    angles = np.arange(0.0, 360.0, angular_resolution_degrees)
    offsets = []
    for radius in radii:
        for degrees in angles:
            radians = math.radians(float(degrees))
            offsets.append((math.sin(radians) * radius, 0.0, -math.cos(radians) * radius))
    return np.asarray(offsets, dtype=np.float32)


@dataclass(frozen=True)
class LocalEndpoint:
    direction_index: int
    valid: bool
    endpoint: list[float] | None
    source_geodesic_m: float | None
    source_euclidean_m: float | None
    angular_offset_degrees: float | None
    snap_offset_m: float | None
    accepted_candidate_count: int

    def to_dict(self) -> dict:
        return asdict(self)


def select_local_endpoints(
    source: np.ndarray,
    offsets: np.ndarray,
    *,
    snap: Callable[[np.ndarray], np.ndarray],
    is_navigable: Callable[[np.ndarray], bool],
    geodesic: Callable[[np.ndarray, np.ndarray], tuple[bool, float]],
    radius_m: float = LOCAL_RADIUS_METERS,
    max_snap_offset_m: float = 0.20,
) -> tuple[list[LocalEndpoint], dict]:
    """Select the farthest point per sector from Source-connected local free space.

    A candidate is accepted only if its snapped NavMesh point has a finite path
    from Source whose geodesic is within the 3 m local region. This connectivity
    check prevents points across walls or on disconnected NavMesh islands from
    entering the endpoint set.
    """
    source = np.asarray(source, dtype=np.float32)
    offsets = np.asarray(offsets, dtype=np.float32)
    best: list[tuple[tuple[float, float, float], np.ndarray, float, float, float] | None] = [None] * NUM_DIRECTIONS
    counts = np.zeros(NUM_DIRECTIONS, dtype=np.int32)
    rejection = {
        "snap_nonfinite": 0,
        "snap_offset": 0,
        "not_navigable": 0,
        "not_source_connected": 0,
        "outside_local_geodesic": 0,
        "degenerate": 0,
    }
    for offset in offsets:
        requested = source + offset
        snapped = np.asarray(snap(requested), dtype=np.float32)
        if not np.isfinite(snapped).all():
            rejection["snap_nonfinite"] += 1
            continue
        snap_offset = float(np.linalg.norm(snapped - requested))
        if snap_offset > max_snap_offset_m:
            rejection["snap_offset"] += 1
            continue
        if not is_navigable(snapped):
            rejection["not_navigable"] += 1
            continue
        found, path_distance = geodesic(source, snapped)
        if not found or not math.isfinite(path_distance):
            rejection["not_source_connected"] += 1
            continue
        if path_distance <= 0.05:
            rejection["degenerate"] += 1
            continue
        if path_distance > radius_m + 1e-5:
            rejection["outside_local_geodesic"] += 1
            continue
        delta = snapped - source
        sector = direction_index_from_offset(float(delta[0]), float(delta[2]))
        actual_degrees = wrap_degrees(math.degrees(math.atan2(float(delta[0]), float(-delta[2]))))
        angular_offset = abs(signed_angular_offset(actual_degrees, direction_center_degrees(sector)))
        euclidean = float(np.linalg.norm(delta))
        counts[sector] += 1
        # Maximize geodesic reach, then favor sector center and smaller snap error.
        key = (float(path_distance), -angular_offset, -snap_offset)
        if best[sector] is None or key > best[sector][0]:
            best[sector] = (key, snapped.copy(), float(path_distance), euclidean, snap_offset)
    endpoints = []
    for sector, value in enumerate(best):
        if value is None:
            endpoints.append(LocalEndpoint(sector, False, None, None, None, None, None, int(counts[sector])))
            continue
        _, point, path_distance, euclidean, snap_offset = value
        delta = point - source
        degrees = wrap_degrees(math.degrees(math.atan2(float(delta[0]), float(-delta[2]))))
        endpoints.append(LocalEndpoint(
            sector, True, [float(x) for x in point], path_distance, euclidean,
            signed_angular_offset(degrees, direction_center_degrees(sector)), snap_offset, int(counts[sector]),
        ))
    audit = {
        "candidate_count": int(len(offsets)),
        "accepted_candidate_count": int(counts.sum()),
        "accepted_per_direction": counts.tolist(),
        "rejection_counts": rejection,
        "radius_m": float(radius_m),
        "max_snap_offset_m": float(max_snap_offset_m),
        "connectivity_rule": "finite NavMesh shortest path from source with geodesic <= local radius",
    }
    return endpoints, audit


def fs_targets(
    endpoint_goal_geodesic: np.ndarray,
    fg_labels: np.ndarray,
    endpoint_valid: np.ndarray,
    *,
    dmax_m: float = DMAX_METERS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    distances = np.asarray(endpoint_goal_geodesic, dtype=np.float64)
    fg = np.asarray(fg_labels, dtype=bool)
    endpoint = np.asarray(endpoint_valid, dtype=bool)
    if distances.shape != fg.shape or fg.shape != endpoint.shape or distances.shape[-1] != NUM_DIRECTIONS:
        raise ValueError("FS arrays must have matching [..., 12] shapes")
    candidate = fg & endpoint
    unreachable = candidate & ~np.isfinite(distances)
    valid = candidate & np.isfinite(distances)
    scores = np.zeros(distances.shape, dtype=np.float32)
    scores[valid] = np.maximum(1.0 - distances[valid] / dmax_m, 0.0)
    return scores, valid, unreachable
