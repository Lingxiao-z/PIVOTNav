from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


class HabitatGSAdapter:
    """Small adapter boundary for the existing Habitat-GS environment.

    The simulator-specific construction is intentionally isolated here. A
    Habitat-GS checkout can provide an environment factory through
    `PIVOTNAV_HABITAT_FACTORY=module:function`; the factory receives the scene
    path and must return an object with reset(), step_velocity(v, w), and
    current_erp().
    """

    def __init__(self, scene: Path, goal: Path, config: dict[str, Any]):
        self.scene = scene.resolve()
        self.goal_path = goal.resolve()
        self.config = config
        self.env = self._load_factory()

    def _load_factory(self) -> Any:
        import importlib
        import os

        spec = os.environ.get("PIVOTNAV_HABITAT_FACTORY")
        if not spec or ":" not in spec:
            raise RuntimeError(
                "Set PIVOTNAV_HABITAT_FACTORY=module:function for the local Habitat-GS adapter"
            )
        module_name, function_name = spec.split(":", 1)
        factory = getattr(importlib.import_module(module_name), function_name)
        return factory(self.scene, self.config)

    def goal_rgb(self) -> np.ndarray:
        from PIL import Image

        return np.asarray(Image.open(self.goal_path).convert("RGB"))

    def reset(self) -> tuple[np.ndarray, np.ndarray]:
        result = self.env.reset()
        return self._observation(result)

    def step(self, linear_mps: float, angular_rps: float) -> tuple[np.ndarray, np.ndarray, bool]:
        result = self.env.step_velocity(float(linear_mps), float(angular_rps))
        rgb, distances = self._observation(result)
        reached = bool(result.get("candidate_frontier_reached", False)) if isinstance(result, dict) else False
        return rgb, distances, reached

    @staticmethod
    def _observation(result: Any) -> tuple[np.ndarray, np.ndarray]:
        if not isinstance(result, dict):
            raise TypeError("Habitat-GS factory must return dict observations")
        rgb = np.asarray(result["erp_rgb"])[..., :3].astype(np.uint8)
        distances = np.asarray(result.get("omnitrav_distances", np.full(360, 8.0)), dtype=np.float32)
        if distances.size != 360:
            raise ValueError("omnitrav_distances must contain 360 values")
        return rgb, distances
