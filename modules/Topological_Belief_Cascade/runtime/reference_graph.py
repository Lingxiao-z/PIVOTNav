from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class RegularNode:
    node_id: int
    rgb: np.ndarray
    created_step: int
    visits: int = 0
    scores: np.ndarray = field(default_factory=lambda: np.zeros(12, dtype=np.float32))
    valid: np.ndarray = field(default_factory=lambda: np.ones(12, dtype=bool))
    heading_deg: float = 0.0


@dataclass
class CandidateFrontier:
    frontier_id: str
    parent_node: int
    sector: int
    state: str = "available"
    attempts: int = 0


class OriginalTopology:
    """Reference global graph update used by default."""

    def __init__(self) -> None:
        self.nodes: dict[int, RegularNode] = {}
        self.frontiers: dict[str, CandidateFrontier] = {}
        self.edges: list[tuple[int, int, str]] = []
        self.current_node: int | None = None
        self.selected_frontier: str | None = None
        self.step = 0

    @property
    def node_count(self) -> int:
        return len(self.nodes)

    def add_regular_node(self, rgb: np.ndarray, scores: np.ndarray | None = None,
                         valid: np.ndarray | None = None, heading_deg: float = 0.0) -> int:
        node_id = len(self.nodes)
        node = RegularNode(node_id, np.asarray(rgb)[..., :3].copy(), self.step,
                           heading_deg=float(heading_deg))
        if scores is not None:
            node.scores = np.asarray(scores, dtype=np.float32).reshape(12)
        if valid is not None:
            node.valid = np.asarray(valid, dtype=bool).reshape(12)
        self.nodes[node_id] = node
        for sector in range(12):
            self.frontiers[f"n{node_id}s{sector}"] = CandidateFrontier(f"n{node_id}s{sector}", node_id, sector)
        self.current_node = node_id
        return node_id

    def add_edge(self, source: int, target: int, kind: str = "sequential") -> None:
        self.edges.append((int(source), int(target), str(kind)))

    def select(self, node_id: int) -> str | None:
        node = self.nodes[node_id]
        choices = [f for f in self.frontiers.values() if f.parent_node == node_id and f.state == "available" and node.valid[f.sector]]
        if not choices:
            self.selected_frontier = None
            return None
        choices.sort(key=lambda f: (-float(node.scores[f.sector]), f.frontier_id))
        selected = choices[0]
        selected.attempts += 1
        self.selected_frontier = selected.frontier_id
        return selected.frontier_id

    def select_global(self, scores_by_node: dict[int, dict[str, np.ndarray]]) -> str | None:
        """Select the best still-available frontier over the full graph."""
        candidates = []
        for node_id, scores in scores_by_node.items():
            node = self.nodes[node_id]
            fs = np.asarray(scores["fs_scores"], dtype=np.float32).reshape(12)
            valid = np.asarray(scores.get("valid_mask", np.ones(12, dtype=bool)), dtype=bool).reshape(12)
            node.scores = fs
            node.valid = valid
            for frontier in self.frontiers.values():
                if frontier.parent_node != node_id or frontier.state != "available":
                    continue
                if not bool(valid[frontier.sector]):
                    continue
                candidates.append((-float(fs[frontier.sector]), frontier.frontier_id))
        if not candidates:
            self.selected_frontier = None
            return None
        candidates.sort()
        selected = self.frontiers[candidates[0][1]]
        selected.attempts += 1
        self.selected_frontier = selected.frontier_id
        return selected.frontier_id

    def promote_selected(self, rgb: np.ndarray, scores: np.ndarray, valid: np.ndarray,
                         heading_deg: float = 0.0) -> int | None:
        selected = self.frontiers.get(self.selected_frontier or "")
        if selected is None:
            return None
        new_id = self.add_regular_node(rgb, scores, valid, heading_deg=heading_deg)
        self.add_edge(selected.parent_node, new_id)
        selected.state = "promoted"
        self.current_node = new_id
        self.selected_frontier = None
        return new_id


class AcceleratedTopology(OriginalTopology):
    """Optional hot-node/delta-edge backend with the reference graph API."""

    def __init__(self, hot_size: int = 16) -> None:
        super().__init__()
        self.hot_size = int(hot_size)
        self.delta_edges: list[tuple[int, int, str]] = []

    def add_edge(self, source: int, target: int, kind: str = "sequential") -> None:
        edge = (int(source), int(target), str(kind))
        self.delta_edges.append(edge)
        self.edges.append(edge)
