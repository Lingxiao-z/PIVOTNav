from __future__ import annotations

from typing import Any

import numpy as np

from .topology import OriginalTopology


class TopologicalBeliefCascade:
    def __init__(self, config: dict[str, Any], compass: Any):
        self.config = config
        self.compass = compass
        self.backend_name = str(config.get("topology_backend", "original"))
        if self.backend_name == "accelerated":
            from .accelerated import AcceleratedTopology
            self.graph = AcceleratedTopology()
        else:
            self.graph = OriginalTopology()
        self.last_observation_node: int | None = None

    @property
    def node_count(self) -> int:
        return self.graph.node_count

    @property
    def selected_frontier(self) -> str | None:
        return self.graph.selected_frontier

    def observe(self, rgb: np.ndarray, goal_evidence: dict[str, Any]) -> int:
        self.graph.step += 1
        ranked = self.compass.retrieve(rgb, top_k=1)
        if not self.graph.nodes or not ranked or ranked[0][1] < float(self.config.get("node_match_threshold", 0.92)):
            node_id = self.graph.add_regular_node(rgb)
            self.compass.add_node(node_id, rgb)
        else:
            node_id = int(ranked[0][0])
            self.graph.current_node = node_id
            self.graph.nodes[node_id].visits += 1
        self.last_observation_node = node_id
        return node_id

    def needs_exploration(self, node_id: int) -> bool:
        selected = self.graph.selected_frontier
        return selected is None or self.graph.frontiers.get(selected, None) is None

    def select_candidate_frontier(self, node_id: int, scores: dict[str, np.ndarray]) -> str | None:
        node = self.graph.nodes[node_id]
        node.scores = np.asarray(scores["fs_scores"], dtype=np.float32).reshape(12)
        node.valid = np.asarray(scores["valid_mask"], dtype=bool).reshape(12)
        return self.graph.select(node_id)

    def promote_candidate_frontier(self, rgb: np.ndarray, distances: np.ndarray) -> int | None:
        scores = self.compass.retrieve(rgb, top_k=1)
        value = np.zeros(12, dtype=np.float32)
        if scores:
            value[:] = float(scores[0][1])
        new_id = self.graph.promote_selected(rgb, value, np.ones(12, dtype=bool))
        if new_id is not None:
            self.compass.add_node(new_id, rgb)
        return new_id

    def command(self, rgb: np.ndarray, distances: np.ndarray, goal: dict[str, Any], config: dict[str, Any]) -> tuple[float, float]:
        selected = self.graph.frontiers.get(self.graph.selected_frontier or "")
        if selected is None:
            return 0.0, 0.0
        bearing = float(selected.sector * 30.0)
        if self.graph.current_node is not None and selected.parent_node != self.graph.current_node:
            bearing = self.compass.command_bearing(rgb, selected.parent_node)
        from modules.Navigable_Curiosity_Field.inference import NavigableCuriosityField
        if hasattr(self, "curiosity") and isinstance(self.curiosity, NavigableCuriosityField):
            return self.curiosity.command(distances, np.deg2rad(((bearing + 180.0) % 360.0) - 180.0))[:2]
        target = np.deg2rad(((bearing + 180.0) % 360.0) - 180.0)
        max_linear = float(config.get("control", {}).get("max_linear_mps", 0.20))
        max_angular = float(config.get("control", {}).get("max_angular_rps", 0.30))
        angular = float(np.clip(target, -max_angular, max_angular))
        linear = max_linear if abs(angular) < max_angular else 0.0
        return linear, angular


def smoke_topology(image: np.ndarray) -> dict[str, bool]:
    graph = OriginalTopology()
    graph.add_regular_node(image)
    return {"ok": graph.node_count == 1 and len(graph.frontiers) == 12}
