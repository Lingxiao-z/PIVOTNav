from __future__ import annotations

from typing import Any

import numpy as np

from .runtime.reference_graph import AcceleratedTopology, OriginalTopology


class TopologicalBeliefCascade:
    def __init__(self, config: dict[str, Any], compass: Any):
        self.config = config
        self.compass = compass
        self.backend_name = str(config.get("topology_backend", "original"))
        if self.backend_name == "accelerated":
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

    def observe(self, rgb: np.ndarray, goal_evidence: dict[str, Any], heading_deg: float = 0.0) -> int:
        self.graph.step += 1
        # A weak or missing VPR match is an uncertainty event, not a map
        # insertion event. Regular nodes are created only after the selected
        # candidate frontier reaches its 3 m completion gate.
        if not self.graph.nodes:
            node_id = self.graph.add_regular_node(rgb, heading_deg=heading_deg)
            self.compass.add_node(node_id, rgb)
        else:
            node_id = self.graph.current_node
            if node_id is None:
                raise RuntimeError("topology has nodes but no current node")
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

    def select_global_frontier(self, scores_by_node: dict[int, dict[str, np.ndarray]]) -> str | None:
        return self.graph.select_global(scores_by_node)

    def promote_candidate_frontier(
        self,
        rgb: np.ndarray,
        distances: np.ndarray,
        selection: dict[str, np.ndarray] | None = None,
    ) -> int | None:
        if selection is None:
            scores = self.compass.retrieve(rgb, top_k=1)
            value = np.zeros(12, dtype=np.float32)
            if scores:
                value[:] = float(scores[0][1])
            valid = np.ones(12, dtype=bool)
        else:
            value = np.asarray(selection["fs_scores"], dtype=np.float32).reshape(12)
            valid = np.asarray(selection["valid_mask"], dtype=bool).reshape(12)
        new_id = self.graph.promote_selected(rgb, value, valid,
                                              heading_deg=float(self.config.get("current_heading_deg", 0.0)))
        if new_id is not None:
            self.compass.add_node(new_id, rgb)
        return new_id

    def route_to_parent(self, parent_node: int) -> list[int]:
        """Return the original graph path used for historical-parent recovery."""
        source = self.graph.current_node
        if source is None or source == parent_node:
            return [int(parent_node)]
        adjacency = {node_id: set() for node_id in self.graph.nodes}
        for src, dst, _kind in self.graph.edges:
            adjacency[src].add(dst)
            adjacency[dst].add(src)
        queue = [source]
        previous: dict[int, int | None] = {source: None}
        for node in queue:
            if node == parent_node:
                break
            for nxt in sorted(adjacency[node]):
                if nxt not in previous:
                    previous[nxt] = node
                    queue.append(nxt)
        if parent_node not in previous:
            return []
        path = [int(parent_node)]
        while path[-1] != source:
            predecessor = previous[path[-1]]
            if predecessor is None:
                return []
            path.append(int(predecessor))
        return list(reversed(path))

    def command(self, rgb: np.ndarray, distances: np.ndarray, goal: dict[str, Any], config: dict[str, Any]) -> tuple[float, float]:
        selected = self.graph.frontiers.get(self.graph.selected_frontier or "")
        if selected is None:
            return 0.0, 0.0
        parent_heading = float(self.graph.nodes[selected.parent_node].heading_deg)
        bearing = parent_heading + float(selected.sector * 30.0) - float(config.get("current_heading_deg", 0.0))
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
