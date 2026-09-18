from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Iterable, Mapping, Sequence


PROTOCOL_VERSION = "r3_spatial_multi_positive_v1"
DEFAULT_NODE_SEPARATION_M = 2.0
DEFAULT_POSITIVE_RADIUS_M = 1.0
DEFAULT_STRICT_OWNER_RADIUS_M = 0.25
DEFAULT_ABSENT_CLEARANCE_M = 3.0
DEFAULT_NEIGHBOR_AUDIT_RADIUS_M = 3.0


@dataclass(frozen=True)
class SpatialProtocol:
    node_separation_m: float = DEFAULT_NODE_SEPARATION_M
    positive_radius_m: float = DEFAULT_POSITIVE_RADIUS_M
    strict_owner_radius_m: float = DEFAULT_STRICT_OWNER_RADIUS_M
    absent_clearance_m: float = DEFAULT_ABSENT_CLEARANCE_M
    neighbor_audit_radius_m: float = DEFAULT_NEIGHBOR_AUDIT_RADIUS_M

    def validate(self) -> None:
        values = (
            self.node_separation_m,
            self.positive_radius_m,
            self.strict_owner_radius_m,
            self.absent_clearance_m,
            self.neighbor_audit_radius_m,
        )
        if not all(math.isfinite(v) and v > 0.0 for v in values):
            raise ValueError("all spatial protocol distances must be finite and positive")
        if self.strict_owner_radius_m > self.positive_radius_m:
            raise ValueError("strict owner radius cannot exceed the spatial-positive radius")
        if self.node_separation_m < 2.0 * self.positive_radius_m:
            raise ValueError("node separation is too small for the spatial-positive radius")
        if self.neighbor_audit_radius_m < self.positive_radius_m:
            raise ValueError("neighbor audit radius must cover the positive radius")


def euclidean_distance(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        raise ValueError("position dimensions differ")
    return math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(a, b)))


def nearby_gallery_candidates(
    query_position: Sequence[float],
    gallery_positions: Mapping[str, Sequence[float]],
    radius_m: float,
) -> list[str]:
    """Return candidates that may be within radius using a safe Euclidean prefilter."""
    return [
        place_id
        for place_id, position in gallery_positions.items()
        if euclidean_distance(query_position, position) <= float(radius_m) + 1e-6
    ]


def build_spatial_label(
    query_position: Sequence[float],
    owner_place_id: str | None,
    gallery_positions: Mapping[str, Sequence[float]],
    distance_fn: Callable[[Sequence[float], Sequence[float]], float],
    protocol: SpatialProtocol,
) -> dict[str, object]:
    """Build a multi-positive label without forcing an arbitrary unique node ID.

    Euclidean distance is a lower bound on path distance, so only anchors inside
    the audit radius need the potentially expensive geodesic call.
    """
    protocol.validate()
    candidate_ids = nearby_gallery_candidates(
        query_position, gallery_positions, protocol.neighbor_audit_radius_m
    )
    if owner_place_id and owner_place_id in gallery_positions and owner_place_id not in candidate_ids:
        candidate_ids.append(owner_place_id)
    distances = {
        place_id: float(distance_fn(query_position, gallery_positions[place_id]))
        for place_id in candidate_ids
    }
    distances = {
        place_id: value
        for place_id, value in distances.items()
        if math.isfinite(value) and value >= 0.0
    }
    ordered = sorted(distances, key=lambda place_id: (distances[place_id], place_id))
    positive_ids = [
        place_id
        for place_id in ordered
        if distances[place_id] <= protocol.positive_radius_m + 1e-6
    ]
    nearest_id = ordered[0] if ordered else None
    nearest_distance = distances.get(nearest_id) if nearest_id is not None else None
    owner_distance = distances.get(owner_place_id) if owner_place_id else None
    strict_owner = bool(
        owner_place_id
        and owner_distance is not None
        and owner_distance <= protocol.strict_owner_radius_m + 1e-6
        and positive_ids == [owner_place_id]
    )
    return {
        "spatial_protocol_version": PROTOCOL_VERSION,
        "spatial_positive_radius_m": protocol.positive_radius_m,
        "strict_owner_radius_m": protocol.strict_owner_radius_m,
        "owner_place_id": owner_place_id,
        "owner_geodesic_distance_m": owner_distance,
        "spatial_positive_place_ids": positive_ids,
        "spatial_positive_distances_m": {
            place_id: distances[place_id] for place_id in positive_ids
        },
        "nearby_gallery_distances_m": {
            place_id: distances[place_id] for place_id in ordered
        },
        "nearest_gallery_place_id": nearest_id,
        "nearest_gallery_distance_m": nearest_distance,
        "strict_owner_unique": strict_owner,
        "ambiguous_positive_count": len(positive_ids),
    }


def positive_place_ids(meta: Mapping[str, object]) -> tuple[str, ...]:
    # An explicitly empty list is meaningful for UNKNOWN/absent samples. Only
    # legacy rows without the R3 field may fall back to their owner ID.
    if "spatial_positive_place_ids" in meta:
        values = meta.get("spatial_positive_place_ids")
        if not isinstance(values, Iterable) or isinstance(values, (str, bytes)):
            raise ValueError("spatial_positive_place_ids must be an iterable of IDs")
        return tuple(str(value) for value in values if value)
    values = None
    owner = meta.get("owner_place_id") or meta.get("place_id")
    return (str(owner),) if owner else ()


def spatial_positive_mask(
    metadata: Sequence[Mapping[str, object]],
    candidate_owner_place_ids: Sequence[str],
) -> list[list[bool]]:
    if len(metadata) != len(candidate_owner_place_ids):
        raise ValueError("metadata and candidate owner IDs must have the same length")
    result: list[list[bool]] = []
    for row_index, meta in enumerate(metadata):
        accepted = set(positive_place_ids(meta))
        result.append(
            [
                column_index != row_index and owner_id in accepted
                for column_index, owner_id in enumerate(candidate_owner_place_ids)
            ]
        )
    return result
