from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np


class HabitatGSAdapter:
    """Direct Habitat-Sim adapter for a Habitat-GS task JSON."""

    def __init__(self, task_path: Path, goal: Path, config: dict[str, Any]):
        self.task_path = Path(task_path).resolve()
        self.goal_path = Path(goal).resolve()
        self.config = config
        self.task = self._read_task(self.task_path)
        self.scene_id = str(self.task.get("scene_id", self.task.get("scene", "")))
        if not self.scene_id:
            raise ValueError("task JSON must contain scene_id")
        self._sim = None
        self._agent = None
        self._hs = None
        self._mn = None
        self._curiosity = None
        self._candidate_parent_position: np.ndarray | None = None

    @staticmethod
    def _read_task(path: Path) -> dict[str, Any]:
        if path.is_file():
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("task JSON must contain an object")
            return payload
        return {"scene_id": str(path)}

    def attach_curiosity(self, curiosity: Any) -> None:
        self._curiosity = curiosity

    def goal_rgb(self) -> np.ndarray:
        from PIL import Image

        return np.asarray(Image.open(self.goal_path).convert("RGB"))

    def _scene_root(self) -> Path:
        configured = self.config.get("habitat_root") or os.environ.get("PIVOTNAV_HABITAT_ROOT")
        if not configured:
            raise RuntimeError("Set habitat_root or PIVOTNAV_HABITAT_ROOT to the Habitat-GS checkout")
        return Path(configured).expanduser().resolve()

    def _build_sim(self) -> None:
        import habitat_sim
        import magnum as mn

        root = self._scene_root()
        dataset = root / "data/scene_datasets/gs_scenes/val.scene_dataset_config.json"
        if not dataset.is_file():
            raise FileNotFoundError(dataset)
        self._hs, self._mn = habitat_sim, mn
        sim_cfg = habitat_sim.SimulatorConfiguration()
        sim_cfg.scene_dataset_config_file = str(dataset)
        sim_cfg.scene_id = self.scene_id
        sim_cfg.gpu_device_id = int(self.config.get("habitat_gpu_device_id", 0))
        agent_cfg = habitat_sim.AgentConfiguration()
        sensor = habitat_sim.EquirectangularSensorSpec()
        sensor.uuid = "erp_rgb"
        sensor.sensor_type = habitat_sim.SensorType.COLOR
        sensor.resolution = [int(self.config.get("erp_height", 256)), int(self.config.get("erp_width", 512))]
        sensor.position = mn.Vector3(0.0, float(self.config.get("camera_height_m", 1.5)), 0.0)
        agent_cfg.sensor_specifications = [sensor]
        self._sim = habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg]))
        self._agent = self._sim.get_agent(0)
        start = self.task.get("start_position_gt_audit_only", self.task.get("start_position"))
        if start is not None:
            state = self._agent.get_state()
            state.position = np.asarray(start, dtype=np.float32)
            self._agent.set_state(state, reset_sensors=True)

    def _observe(self) -> tuple[np.ndarray, np.ndarray]:
        if self._sim is None:
            raise RuntimeError("adapter has not been reset")
        observation = self._sim.get_sensor_observations()["erp_rgb"]
        rgb = np.asarray(observation)[..., :3].astype(np.uint8, copy=False)
        if self._curiosity is None:
            raise RuntimeError("OmniTrav provider is not attached")
        distances = self._curiosity.predict_distances(rgb)
        return rgb.copy(), np.asarray(distances, dtype=np.float32).reshape(360)

    def reset(self) -> tuple[np.ndarray, np.ndarray]:
        self._build_sim()
        return self._observe()

    def begin_candidate_frontier(self) -> None:
        if self._agent is None:
            raise RuntimeError("adapter has not been reset")
        self._candidate_parent_position = np.asarray(self._agent.get_state().position, dtype=np.float64).copy()

    def _step_velocity(self, linear_mps: float, angular_rps: float) -> None:
        hs, mn = self._hs, self._mn
        state = self._agent.get_state()
        start = np.asarray(state.position, dtype=np.float64)
        rigid0 = hs.RigidState(
            hs.utils.common.quat_to_magnum(state.rotation),
            mn.Vector3(float(start[0]), float(start[1]), float(start[2])),
        )
        velocity = hs.physics.VelocityControl()
        velocity.controlling_lin_vel = True
        velocity.lin_vel_is_local = True
        velocity.controlling_ang_vel = True
        velocity.ang_vel_is_local = True
        velocity.linear_velocity = mn.Vector3(0.0, 0.0, -float(linear_mps))
        velocity.angular_velocity = mn.Vector3(0.0, float(angular_rps), 0.0)
        rigid1 = velocity.integrate_transform(float(self.config.get("control_dt_s", 0.25)), rigid0)
        filtered = self._sim.step_filter(rigid0.translation, rigid1.translation)
        state.position = np.asarray([float(filtered[0]), float(filtered[1]), float(filtered[2])], dtype=np.float32)
        state.rotation = hs.utils.common.quat_from_magnum(rigid1.rotation)
        self._agent.set_state(state, reset_sensors=True)

    def step(self, linear_mps: float, angular_rps: float) -> tuple[np.ndarray, np.ndarray, bool]:
        self._step_velocity(linear_mps, angular_rps)
        rgb, distances = self._observe()
        reached = False
        if self._candidate_parent_position is not None:
            current = np.asarray(self._agent.get_state().position, dtype=np.float64)
            moved = float(np.linalg.norm((current - self._candidate_parent_position)[[0, 2]]))
            reached = moved >= float(self.config.get("candidate_frontier_distance_m", 3.0))
        return rgb, distances, reached

    def close(self) -> None:
        if self._sim is not None:
            self._sim.close()
            self._sim = None
