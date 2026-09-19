"""Hard event semantics for known-node routing and final Goal arrival."""
from __future__ import annotations

from typing import Any, Mapping


GOAL_FINAL_CONFIRMED = "GOAL_FINAL_CONFIRMED"
CANDIDATE_FRONTIER_FINAL_CONFIRMED = "CANDIDATE_FRONTIER_FINAL_CONFIRMED"
HISTORICAL_NODE_FINAL_CONFIRMED = "HISTORICAL_NODE_FINAL_CONFIRMED"
KNOWN_NODE_LOCALIZED = "KNOWN_NODE_LOCALIZED"

KNOWN_NODE_SWITCH_EVENTS = frozenset({
    KNOWN_NODE_LOCALIZED,
    HISTORICAL_NODE_FINAL_CONFIRMED,
})


def assert_stop_semantics(event: Mapping[str, Any]) -> None:
    """Only final Goal-image confirmation may authorize Stop."""
    if bool(event.get("stop_authorized", False)):
        assert event.get("event_type") == GOAL_FINAL_CONFIRMED
        assert bool(event.get("final_confirmed", False))


def assert_known_node_event(event: Mapping[str, Any]) -> None:
    """Known-node localization is geometry-free and cannot emit Goal/Stop."""
    assert int(event.get("geometry_call_count", 0)) == 0
    assert int(event.get(GOAL_FINAL_CONFIRMED, 0)) == 0
    assert event.get("event_type") != GOAL_FINAL_CONFIRMED
    assert not bool(event.get("stop_authorized", False))
    assert not bool(event.get("ordinary_rgb_frame_bpl_mutation", False))
    assert not bool(event.get("ordinary_rgb_frame_topology_mutation", False))
    if bool(event.get("known_next_hop_switched", False)):
        assert event.get("event_type") in KNOWN_NODE_SWITCH_EVENTS

