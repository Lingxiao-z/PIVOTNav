"""V5 event-driven FS/candidate_frontier protocol and coordinate contract.

This module is intentionally independent from the V4 runner.  FS is called
only at explicit graph events; ordinary control frames never invoke it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

NUM_SECTORS = 12
SECTOR_WIDTH_DEG = 30.0
FS_EVENT_REASONS = frozenset({
    "ROOT_REGULAR_CREATED",
    "CANDIDATE_FRONTIER_FINAL_CONFIRMED",
    "HISTORICAL_NODE_FINAL_CONFIRMED",
    "CANDIDATE_FRONTIER_BLOCKED",
    "CANDIDATE_FRONTIER_EXHAUSTED",
    "CANDIDATE_FRONTIER_EXECUTION_FAILED",
    "FORMAL_LOOP_CONFIRMED",
    "CANDIDATE_INVALIDATED",
})


def wrap_degrees(value: float) -> float:
    return float(((float(value) + 180.0) % 360.0) - 180.0)


def fs_sector_to_robot_relative_bearing(sector_index: int) -> float:
    """Map the frozen FS sector contract to right-positive body bearing.

    Sector 0 is robot-forward.  Increasing sectors rotate right, so the
    canonical mapping is 0, +30, ..., +180, -150, ..., -30 degrees.
    """
    sector = int(sector_index) % NUM_SECTORS
    bearing = wrap_degrees(sector * SECTOR_WIDTH_DEG)
    return 180.0 if sector == 6 else bearing


def robot_relative_bearing_to_sector(bearing_degrees: float) -> int:
    return int(round(wrap_degrees(float(bearing_degrees)) / SECTOR_WIDTH_DEG)) % NUM_SECTORS


@dataclass(frozen=True)
class FSSchedule:
    reason: str
    graph_version: int
    current_node: int
    selected_candidate_frontier_id: str | None
    selected_parent_node_id: int | None
    selected_sector: int | None
    selected_fg_score: float | None
    selected_fs_score: float | None
    candidates: tuple[dict, ...]


def assert_fs_event(reason: str) -> None:
    if reason not in FS_EVENT_REASONS:
        raise AssertionError(f"FS_CALL_ON_NON_EVENT:{reason}")
