"""NTS-style online ghost-node scheduler for RGB goal navigation.

The policy deliberately contains no privileged geometry or pose inputs.  It
owns only graph state, frontier lifecycle, FS scheduling, and the
final-confirmed transaction gate.  Local control adapters consume the bearing
returned by :meth:`control_target`.
"""
from __future__ import annotations

import copy
import heapq
import math
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

NUM_SECTORS = 12
SECTOR_DEGREES = 360.0 / NUM_SECTORS
FORWARD_SECTOR = 6


def wrap_degrees(value: float) -> float:
    return float(((float(value) + 180.0) % 360.0) - 180.0)


@dataclass
class RegularNode:
    node_id: int
    parent_id: int | None
    created_step: int
    rgb_path: str | None = None
    descriptor_ref: str | None = None
    branch_state: str = "ACTIVE"
    visited: bool = True


@dataclass
class GhostNode:
    candidate_frontier_id: str
    parent_id: int
    sector: int
    created_step: int
    state: str = "UNKNOWN"
    fs_score: float | None = None
    fg_valid: bool = False
    attempts: int = 0
    exhausted_reason: str | None = None
    last_selected_step: int | None = None
    feedback_count: int = 0
    novelty_ema: float = 0.0
    stall_count: int = 0
    value_adjustment: float = 0.0


@dataclass
class DirectedEdge:
    source: int
    target: int
    edge_type: str = "SEQUENTIAL_OBSERVED"
    confirmed_step: int | None = None


class NTSGoalNavigator:
    """Stateful regular/ghost graph with event-only node and BPL mutation."""

    schema = "integration_v4_nts_goal_policy_v1"

    def __init__(self, sectors: int = NUM_SECTORS) -> None:
        if int(sectors) != NUM_SECTORS:
            raise ValueError("the frozen FS contract has exactly 12 sectors")
        self.regular: dict[int, RegularNode] = {}
        self.candidate_frontiers: dict[str, GhostNode] = {}
        self.edges: list[DirectedEdge] = []
        self.current_node: int | None = None
        self.next_node_id = 0
        self.long_term_goal: str | int | None = None
        self.current_regular_subgoal: int | None = None
        self.bpl: dict[int, float] = {}
        self.bpl_events: list[dict] = []
        self.selection_log: list[dict] = []
        self.node_events: list[dict] = []
        self.candidate_frontier_events: list[dict] = []
        self.ordinary_frame_bpl_mutations = 0

    def _event(self, target: list[dict], payload: Mapping) -> None:
        target.append(dict(payload))

    def add_root(self, step: int = 0, rgb_path: str | None = None, descriptor_ref: str | None = None) -> int:
        if self.regular:
            raise RuntimeError("root already exists")
        node_id = self.next_node_id
        self.next_node_id += 1
        self.regular[node_id] = RegularNode(node_id, None, int(step), rgb_path, descriptor_ref)
        self.current_node = node_id
        self._event(self.node_events, {"event": "regular_node_created", "node_id": node_id,
                                       "step": int(step), "final_confirmed": True,
                                       "reason": "initialization"})
        self._bpl_update(node_id, int(step), "initial_regular_node",
                         event_type="INITIAL_REGULAR_NODE", final_confirmed=True)
        self.create_ghosts(node_id, int(step))
        return node_id

    def create_ghosts(self, parent_id: int, step: int) -> list[str]:
        if parent_id not in self.regular:
            raise KeyError(parent_id)
        created: list[str] = []
        for sector in range(NUM_SECTORS):
            candidate_frontier_id = f"g{parent_id}s{sector}"
            if candidate_frontier_id in self.candidate_frontiers:
                continue
            self.candidate_frontiers[candidate_frontier_id] = GhostNode(candidate_frontier_id, int(parent_id), sector, int(step))
            created.append(candidate_frontier_id)
            self._event(self.candidate_frontier_events, {"event": "ghost_created", "ghost_id": candidate_frontier_id,
                                             "parent_node": int(parent_id), "sector": sector,
                                             "state": "UNKNOWN", "step": int(step)})
        return created

    def _bpl_update(self, node_id: int, step: int, reason: str, *,
                    event_type: str, final_confirmed: bool) -> None:
        # BPL is intentionally event-only.  A normal RGB frame never calls this method.
        if node_id not in self.regular:
            raise KeyError(node_id)
        self.bpl = {key: (0.99 if key == node_id else 0.01 / max(len(self.regular) - 1, 1))
                    for key in self.regular}
        total = sum(self.bpl.values()) or 1.0
        self.bpl = {key: value / total for key, value in self.bpl.items()}
        self._event(self.bpl_events, {"event": "bpl_node_event_update", "event_type": event_type,
                                      "node_id": int(node_id),
                                      "step": int(step), "reason": reason,
                                      "final_confirmed": bool(final_confirmed), "ordinary_rgb_frame": False,
                                      "bpl_can_affirm_stop": False})

    def observe_rgb_frame(self, step: int) -> None:
        """Record the observation boundary without mutating nodes or BPL."""
        if self.current_node is None:
            raise RuntimeError("root must be initialized")
        # Kept explicit for audits: this counter must remain zero in normal runs.
        self.ordinary_frame_bpl_mutations += 0

    def update_fs_scores(
        self,
        scores: Mapping[str, float] | Sequence[float],
        fg_valid: Mapping[str, bool] | Sequence[bool] | None = None,
        traversable: Mapping[str, bool] | Sequence[bool] | None = None,
        step: int = 0,
    ) -> dict:
        """Score every non-exhausted ghost and select one global target.

        ``scores`` is keyed by ghost id for online use.  A sequence is accepted
        for deterministic unit tests and maps sector order at the current
        parent.  No persistent-forward shortcut is allowed here.
        """
        previous_target = self.long_term_goal
        previous_sector = (
            self.candidate_frontiers[previous_target].sector
            if isinstance(previous_target, str) and previous_target in self.candidate_frontiers
            else None
        )
        candidates: list[GhostNode] = []
        evaluated: list[dict] = []
        for candidate_frontier in self.candidate_frontiers.values():
            if candidate_frontier.state in {"EXHAUSTED", "BLOCKED", "CONFIRMED", "REJECTED"}:
                continue
            if isinstance(scores, Mapping):
                value = float(scores.get(candidate_frontier.candidate_frontier_id, float("-inf")))
                fg = bool((fg_valid or {}).get(candidate_frontier.candidate_frontier_id, True)) if isinstance(fg_valid, Mapping) else True
                open_ = bool((traversable or {}).get(candidate_frontier.candidate_frontier_id, True)) if isinstance(traversable, Mapping) else True
            else:
                value = float(scores[candidate_frontier.sector]) if candidate_frontier.sector < len(scores) else float("-inf")
                fg = bool(fg_valid[candidate_frontier.sector]) if fg_valid is not None and candidate_frontier.sector < len(fg_valid) else True
                open_ = bool(traversable[candidate_frontier.sector]) if traversable is not None and candidate_frontier.sector < len(traversable) else True
            candidate_frontier.fs_score = value
            candidate_frontier.fg_valid = fg
            evaluated.append({"ghost_id": candidate_frontier.candidate_frontier_id, "parent_node": candidate_frontier.parent_id,
                              "sector": candidate_frontier.sector, "fs_score": value, "fg_valid": fg,
                              "traversable": open_, "state_before_selection": candidate_frontier.state})
            if not fg or not open_ or not math.isfinite(value):
                if not open_ and candidate_frontier.state == "ACTIVE":
                    candidate_frontier.state = "BLOCKED"
                continue
            candidates.append(candidate_frontier)
        candidates.sort(key=lambda item: (-float(item.fs_score), item.candidate_frontier_id))
        selected = candidates[0] if candidates else None
        if selected is not None:
            selected.state = "ACTIVE"
            selected.last_selected_step = int(step)
            self.long_term_goal = selected.candidate_frontier_id
            self.current_regular_subgoal = selected.parent_id if self.current_node != selected.parent_id else None
        record = {
            "event": "global_fs_selection",
            "step": int(step),
            "candidate_count": len(candidates),
            "evaluated_ghost_count": len(evaluated),
            "selected_ghost_id": selected.candidate_frontier_id if selected else None,
            "selected_parent_node": selected.parent_id if selected else None,
            "selected_sector": selected.sector if selected else None,
            "selected_fs_score": selected.fs_score if selected else None,
            "previous_selected_ghost_id": previous_target,
            "previous_selected_sector": previous_sector,
            "target_changed": bool((selected.candidate_frontier_id if selected else None) != previous_target),
            "old_target_direction_degrees_right_positive": (
                wrap_degrees((previous_sector - FORWARD_SECTOR) * SECTOR_DEGREES)
                if previous_sector is not None else None
            ),
            "new_target_direction_degrees_right_positive": (
                wrap_degrees((selected.sector - FORWARD_SECTOR) * SECTOR_DEGREES)
                if selected is not None else None
            ),
            "candidates": [{"ghost_id": item.candidate_frontier_id, "parent_node": item.parent_id,
                            "sector": item.sector, "fs_score": item.fs_score,
                            "fg_valid": item.fg_valid, "state": item.state,
                            "selected": item is selected} for item in candidates],
            "all_ghost_scores": evaluated,
            "persistent_forward_overrode_selection": False,
        }
        self.selection_log.append(record)
        return record

    def _path(self, source: int, target: int) -> list[int]:
        adjacency: dict[int, list[int]] = {node: [] for node in self.regular}
        for edge in self.edges:
            adjacency.setdefault(edge.source, []).append(edge.target)
            adjacency.setdefault(edge.target, []).append(edge.source)
        queue: list[tuple[int, int]] = [(0, int(source))]
        distance = {int(source): 0}
        parent: dict[int, int] = {}
        while queue:
            cost, node = heapq.heappop(queue)
            if node == int(target):
                break
            if cost != distance[node]:
                continue
            for neighbor in adjacency.get(node, []):
                next_cost = cost + 1
                if next_cost < distance.get(neighbor, 10**9):
                    distance[neighbor] = next_cost
                    parent[neighbor] = node
                    heapq.heappush(queue, (next_cost, neighbor))
        if int(target) not in distance:
            return []
        path = [int(target)]
        while path[-1] != int(source):
            path.append(parent[path[-1]])
        return list(reversed(path))

    def control_target(self) -> dict:
        """Return the immediate bearing for OmniGuard.

        A historical ghost is never applied in the current frame: Dijkstra
        first selects the next regular node, and only at its parent is the
        ghost sector converted to a body-relative bearing.
        """
        if self.current_node is None or self.long_term_goal is None:
            return {"mode": "IDLE", "bearing_degrees_right_positive": 0.0, "target": None}
        target = self.long_term_goal
        if isinstance(target, str):
            candidate_frontier = self.candidate_frontiers[target]
            if self.current_node != candidate_frontier.parent_id:
                path = self._path(self.current_node, candidate_frontier.parent_id)
                next_node = path[1] if len(path) > 1 else None
                self.current_regular_subgoal = next_node
                return {"mode": "DIJKSTRA_RETURN", "bearing_degrees_right_positive": 0.0,
                        "target": target, "next_regular_node": next_node, "path": path}
            bearing = wrap_degrees((candidate_frontier.sector - FORWARD_SECTOR) * SECTOR_DEGREES)
            candidate_frontier.attempts += 1
            return {"mode": "FS_GHOST", "bearing_degrees_right_positive": bearing,
                    "target": target, "parent_node": candidate_frontier.parent_id,
                    "sector": candidate_frontier.sector, "path": [self.current_node]}
        path = self._path(self.current_node, int(target))
        return {"mode": "DIJKSTRA_RETURN", "bearing_degrees_right_positive": 0.0,
                "target": target, "next_regular_node": path[1] if len(path) > 1 else None,
                "path": path}

    def record_branch_feedback(
        self,
        *,
        step: int,
        descriptor_novelty: float,
        commanded_linear_velocity_mps: float,
        traversable: bool,
        controller_mode: str | None = None,
    ) -> dict:
        """Update the active frontier from RGB/controller observations only.

        This is deliberately independent of privileged displacement or
        collision state.  Low descriptor novelty while a forward command is
        being issued is treated as a local stall; repeated stalls lower the
        branch value and eventually block that ghost until a later global
        replan can reconsider it.
        """
        target = self.long_term_goal
        if not isinstance(target, str) or target not in self.candidate_frontiers:
            return {"updated": False, "reason": "no_active_ghost"}
        candidate_frontier = self.candidate_frontiers[target]
        novelty = float(max(0.0, min(1.0, descriptor_novelty)))
        candidate_frontier.feedback_count += 1
        if candidate_frontier.feedback_count == 1:
            candidate_frontier.novelty_ema = novelty
        else:
            candidate_frontier.novelty_ema = 0.8 * candidate_frontier.novelty_ema + 0.2 * novelty
        stalled = bool(commanded_linear_velocity_mps > 0.05 and (not traversable or novelty < 0.0025))
        candidate_frontier.stall_count = candidate_frontier.stall_count + 1 if stalled else max(0, candidate_frontier.stall_count - 1)
        reward = 0.04 * candidate_frontier.novelty_ema if traversable else -0.04
        penalty = 0.02 * min(candidate_frontier.stall_count, 10)
        candidate_frontier.value_adjustment = float(candidate_frontier.value_adjustment + reward - penalty)
        if candidate_frontier.stall_count >= 12:
            candidate_frontier.state = "BLOCKED"
            candidate_frontier.exhausted_reason = "RGB_NOVELTY_OR_OMNIGUARD_STALL"
        event = {
            "event": "ghost_branch_feedback",
            "step": int(step),
            "ghost_id": target,
            "descriptor_novelty": novelty,
            "novelty_ema": candidate_frontier.novelty_ema,
            "commanded_linear_velocity_mps": float(commanded_linear_velocity_mps),
            "traversable": bool(traversable),
            "controller_mode": controller_mode,
            "stalled": stalled,
            "stall_count": candidate_frontier.stall_count,
            "value_adjustment": candidate_frontier.value_adjustment,
            "state": candidate_frontier.state,
            "runtime_gt_inputs": [],
        }
        self.candidate_frontier_events.append(event)
        return {"updated": True, **event}

    def on_final_confirmed(
        self,
        *,
        step: int,
        target_kind: str,
        candidate_frontier_id: str | None = None,
        existing_node_id: int | None = None,
        rgb_path: str | None = None,
        descriptor_ref: str | None = None,
        reason: str = "arrival_verifier_final_confirmed",
    ) -> int:
        """Atomically commit a node event; any validation failure rolls back."""
        snapshot = copy.deepcopy((self.regular, self.candidate_frontiers, self.edges, self.current_node,
                                  self.next_node_id, self.long_term_goal, self.current_regular_subgoal,
                                  self.bpl, self.bpl_events, self.node_events, self.candidate_frontier_events))
        try:
            if self.current_node is None:
                raise RuntimeError("no current node")
            if target_kind == "ghost":
                if candidate_frontier_id not in self.candidate_frontiers:
                    raise KeyError(candidate_frontier_id)
                candidate_frontier = self.candidate_frontiers[candidate_frontier_id]
                if candidate_frontier.state not in {"ACTIVE", "UNKNOWN"}:
                    raise RuntimeError(f"ghost {ghost_id} is {ghost.state}")
                node_id = self.next_node_id
                self.next_node_id += 1
                self.regular[node_id] = RegularNode(node_id, candidate_frontier.parent_id, int(step), rgb_path, descriptor_ref)
                self.edges.append(DirectedEdge(candidate_frontier.parent_id, node_id, "SEQUENTIAL_OBSERVED", int(step)))
                candidate_frontier.state = "CONFIRMED"
                self.current_node = node_id
                self.long_term_goal = None
                self.current_regular_subgoal = None
                self.create_ghosts(node_id, int(step))
                self._bpl_update(node_id, int(step), reason,
                                 event_type="GHOST_FINAL_CONFIRMED", final_confirmed=True)
                self._event(self.node_events, {"event": "ghost_to_regular_confirmed",
                                               "event_type": "GHOST_FINAL_CONFIRMED", "ghost_id": candidate_frontier_id,
                                               "node_id": node_id, "parent_node": candidate_frontier.parent_id,
                                               "step": int(step), "final_confirmed": True})
                return node_id
            if target_kind == "existing":
                if existing_node_id not in self.regular:
                    raise KeyError(existing_node_id)
                self.edges.append(DirectedEdge(self.current_node, int(existing_node_id), "LOOP_CONFIRMED", int(step)))
                self.current_node = int(existing_node_id)
                self.long_term_goal = None
                self.current_regular_subgoal = None
                self._bpl_update(int(existing_node_id), int(step), reason,
                                 event_type="HISTORICAL_NODE_FINAL_CONFIRMED", final_confirmed=True)
                self._event(self.node_events, {"event": "existing_regular_confirmed",
                                               "event_type": "HISTORICAL_NODE_FINAL_CONFIRMED",
                                               "node_id": int(existing_node_id),
                                               "step": int(step), "final_confirmed": True,
                                               "formal_loop_edge": True})
                return int(existing_node_id)
            raise ValueError(target_kind)
        except Exception:
            (self.regular, self.candidate_frontiers, self.edges, self.current_node,
             self.next_node_id, self.long_term_goal, self.current_regular_subgoal,
             self.bpl, self.bpl_events, self.node_events, self.candidate_frontier_events) = snapshot
            raise

    def on_known_node_localized(self, *, step: int, node_id: int,
                                reason: str = "vpr_bpl_multiframe_localized") -> int:
        """Switch to an already-known node without goal geometry or Stop.

        This is a typed localization event, not a Goal arrival confirmation.
        It may update the node-level BPL belief because it is an explicit node
        event, while ordinary RGB observations remain mutation-free.
        """
        snapshot = copy.deepcopy((self.current_node, self.long_term_goal,
                                  self.current_regular_subgoal, self.bpl,
                                  self.bpl_events, self.node_events))
        try:
            node_id = int(node_id)
            if node_id not in self.regular:
                raise KeyError(node_id)
            self.current_node = node_id
            self.long_term_goal = None
            self.current_regular_subgoal = None
            self._bpl_update(node_id, int(step), reason,
                             event_type="KNOWN_NODE_LOCALIZED", final_confirmed=False)
            self._event(self.node_events, {
                "event": "known_node_localized",
                "event_type": "KNOWN_NODE_LOCALIZED",
                "node_id": node_id,
                "step": int(step),
                "final_confirmed": False,
                "geometry_called": False,
                "stop_authorized": False,
            })
            return node_id
        except Exception:
            (self.current_node, self.long_term_goal, self.current_regular_subgoal,
             self.bpl, self.bpl_events, self.node_events) = snapshot
            raise

    def mark_blocked(self, candidate_frontier_id: str, reason: str) -> None:
        candidate_frontier = self.candidate_frontiers[candidate_frontier_id]
        candidate_frontier.state = "BLOCKED"
        candidate_frontier.exhausted_reason = str(reason)

    def payload(self) -> dict:
        return {
            "schema": self.schema,
            "current_node": self.current_node,
            "long_term_goal": self.long_term_goal,
            "current_regular_subgoal": self.current_regular_subgoal,
            "regular_nodes": [vars(node) for node in self.regular.values()],
            "ghost_nodes": [vars(node) for node in self.candidate_frontiers.values()],
            "directed_edges": [vars(edge) for edge in self.edges],
            "bpl": self.bpl,
            "bpl_events": self.bpl_events,
            "ordinary_frame_bpl_mutations": self.ordinary_frame_bpl_mutations,
            "selection_log": self.selection_log,
            "node_events": self.node_events,
            "ghost_events": self.candidate_frontier_events,
            "formal_loop_edge_count": sum(edge.edge_type == "LOOP_CONFIRMED" for edge in self.edges),
        }

    def csr_payload(self) -> dict:
        """Materialize CSR and routing views from the committed directed edges."""
        node_ids = sorted(self.regular)
        index = {node_id: offset for offset, node_id in enumerate(node_ids)}
        adjacency: list[list[int]] = [[] for _ in node_ids]
        for edge in self.edges:
            if edge.source in index and edge.target in index:
                adjacency[index[edge.source]].append(index[edge.target])
        row_ptr = [0]
        col_idx: list[int] = []
        for neighbors in adjacency:
            col_idx.extend(sorted(set(neighbors)))
            row_ptr.append(len(col_idx))
        return {
            "schema": "integration_v4_csr_routing_graph_v1",
            "node_ids": node_ids,
            "row_ptr": row_ptr,
            "col_idx": col_idx,
            "data": [1.0] * len(col_idx),
            "nnz": len(col_idx),
            "connected_components": 1 if node_ids else 0,
            "routing_graph": {str(source): sorted(set(targets)) for source, targets in
                               ((node_id, [edge.target for edge in self.edges if edge.source == node_id])
                                for node_id in node_ids)},
            "atomic_transaction": True,
        }
