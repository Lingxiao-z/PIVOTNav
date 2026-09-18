from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from habitat_gs import HabitatGSAdapter
from modules.Panoramic_Place_Compass.inference import PanoramicPlaceCompass
from modules.Topological_Belief_Cascade.inference import TopologicalBeliefCascade
from modules.Navigable_Curiosity_Field.inference import NavigableCuriosityField


class NavigationSystem:
    def __init__(self, config: dict[str, Any], goal_rgb: np.ndarray, adapter: Any,
                 compass: PanoramicPlaceCompass, topology: TopologicalBeliefCascade,
                 curiosity: NavigableCuriosityField):
        self.config = config
        self.goal_rgb = goal_rgb
        self.adapter = adapter
        self.compass = compass
        self.topology = topology
        self.curiosity = curiosity
        self.step_count = 0

    @classmethod
    def from_habitat_gs(cls, config: dict[str, Any], scene: Path, goal: Path) -> "NavigationSystem":
        adapter = HabitatGSAdapter(scene, goal, config)
        root = Path(config.get("weights_root", "")).expanduser()
        compass = PanoramicPlaceCompass(root, config)
        topology = TopologicalBeliefCascade(config, compass)
        curiosity = NavigableCuriosityField(root, config)
        return cls(config, adapter.goal_rgb(), adapter, compass, topology, curiosity)

    def run(self) -> dict[str, Any]:
        rgb, distances = self.adapter.reset()
        result = {"status": "RUNNING", "steps": 0, "regular_nodes": 0, "candidate_frontier": None}
        while self.step_count < int(self.config.get("max_steps", 1200)):
            self.step_count += 1
            goal = self.compass.goal_evidence(rgb, self.goal_rgb)
            if goal.get("arrival_confirmed", False):
                result.update(status="SUCCESS", steps=self.step_count)
                return result
            node_id = self.topology.observe(rgb, goal)
            if self.topology.needs_exploration(node_id):
                scores = self.curiosity.predict(rgb, self.goal_rgb)
                selection = self.curiosity.select(scores, distances)
                self.topology.select_candidate_frontier(node_id, selection)
            decision = self.topology.command(rgb, distances, goal, self.config)
            rgb, distances, reached = self.adapter.step(decision[0], decision[1])
            if reached:
                self.topology.promote_candidate_frontier(rgb, distances)
            result.update(steps=self.step_count, regular_nodes=self.topology.node_count,
                          candidate_frontier=self.topology.selected_frontier)
        result["status"] = "BUDGET_EXHAUSTED"
        return result


def run_smoke() -> None:
    from modules.Panoramic_Place_Compass.inference import smoke_compass
    from modules.Topological_Belief_Cascade.inference import smoke_topology
    from modules.Navigable_Curiosity_Field.inference import smoke_curiosity

    image = np.zeros((224, 448, 3), dtype=np.uint8)
    assert smoke_compass(image, image)["ok"]
    assert smoke_topology(image)["ok"]
    assert smoke_curiosity(image, image)["ok"]
    print("PIVOTNav smoke test passed")
