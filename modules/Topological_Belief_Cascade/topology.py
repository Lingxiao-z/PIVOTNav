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
                         valid: np.ndarray | None = None) -> int:
        node_id = len(self.nodes)
        node = RegularNode(node_id, np.asarray(rgb)[..., :3].copy(), self.step)
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

    def promote_selected(self, rgb: np.ndarray, scores: np.ndarray, valid: np.ndarray) -> int | None:
        selected = self.frontiers.get(self.selected_frontier or "")
        if selected is None:
            return None
        new_id = self.add_regular_node(rgb, scores, valid)
        self.add_edge(selected.parent_node, new_id)
        selected.state = "promoted"
        self.current_node = new_id
        self.selected_frontier = None
        return new_id
