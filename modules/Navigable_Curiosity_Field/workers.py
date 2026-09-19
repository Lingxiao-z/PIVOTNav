"""Persistent FS and OmniGuard clients used by the formal runner."""
from __future__ import annotations

import base64
import io
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


CHECKPOINT_SHA = "44aa451546691f35659ce1ecc0d616d67d706217ceb5a8f2ed43cba9b132760f"
DT = 0.25
MAX_V = 0.4
MAX_W = 0.3


def save_rgb(observation: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(observation["rgb"])[..., :3].astype(np.uint8)).save(path)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


class ExpandedWorkerClient:
    def __init__(self, device: str, stderr_path: Path) -> None:
        worker = Path(os.environ.get("PIVOTNAV_FS_WORKER", ""))
        if not worker.is_file():
            raise RuntimeError("PIVOTNAV_FS_WORKER must point to the external FS worker script")
        stderr_path.parent.mkdir(parents=True, exist_ok=True)
        environment = dict(os.environ)
        worker_device = str(device)
        if worker_device.startswith("cuda:"):
            environment["CUDA_VISIBLE_DEVICES"] = worker_device.split(":", 1)[1]
            worker_device = "cuda:0"
        # The formal server worker owns its package/checkpoint paths and only
        # accepts ``--device``. The public worker also accepts an optional
        # package root. Probe the CLI once so the client speaks both contracts
        # without changing the worker's model or preprocessing behavior.
        help_result = subprocess.run(
            [sys.executable, str(worker), "--help"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        supports_package_root = "--package-root" in help_result.stdout
        package_root = os.environ.get("PIVOTNAV_FS_PACKAGE_ROOT")
        if supports_package_root and not package_root:
            raise RuntimeError("PIVOTNAV_FS_PACKAGE_ROOT is required for this FS worker")
        command = [sys.executable, str(worker)]
        if supports_package_root:
            command.extend(["--package-root", package_root])
        command.extend(["--device", worker_device])
        self.stderr = stderr_path.open("w", encoding="utf-8")
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.stderr,
            text=True,
            bufsize=1,
            env=environment,
        )
        ready = self._read()
        if not ready.get("ready"):
            raise RuntimeError(f"FS_WORKER_NOT_READY:{ready}")
        if ready.get("checkpoint_sha256") not in (None, CHECKPOINT_SHA):
            raise RuntimeError(f"FS_CHECKPOINT_MISMATCH:{ready}")
        self.ready = ready
        self.inference_count = 0

    def _read(self) -> dict[str, Any]:
        assert self.process.stdout is not None
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError(f"FS_WORKER_EXITED:{self.process.poll()}")
        value = json.loads(line)
        if value.get("error"):
            raise RuntimeError(value["error"])
        return value

    def infer(self, current_path: Path, goal_path: Path):
        assert self.process.stdin is not None
        self.process.stdin.write(json.dumps({"command": "infer", "current_path": str(current_path), "goal_path": str(goal_path)}) + "\n")
        self.process.stdin.flush()
        value = self._read()
        import numpy as np
        self.inference_count += 1
        return np.asarray(value["fg_probabilities"], dtype=np.float64), np.asarray(value["fs_scores"], dtype=np.float64), float(value["latency_ms"])

    def close(self) -> None:
        try:
            if self.process.poll() is None:
                assert self.process.stdin is not None
                self.process.stdin.write('{"command":"close"}\n')
                self.process.stdin.flush()
                self._read()
        finally:
            if self.process.poll() is None:
                self.process.terminate()
            self.process.wait(timeout=30)
            self.stderr.close()


class OmniGuardClient:
    """In-memory RGB client for the external OmniTrav/OmniGuard worker."""

    def __init__(self, physical_gpu: int, output_dir: Path, stderr_path: Path) -> None:
        root = Path(os.environ.get("PIVOTNAV_OMNIGUARD_ROOT", ""))
        worker = Path(os.environ.get("PIVOTNAV_OMNIGUARD_WORKER", ""))
        checkpoint = Path(os.environ.get("PIVOTNAV_OMNIGUARD_CHECKPOINT", ""))
        if not root.is_dir() or not worker.is_file() or not checkpoint.is_file():
            raise RuntimeError("PIVOTNAV_OMNIGUARD_ROOT/WORKER/CHECKPOINT must be configured")
        environment = dict(os.environ)
        environment["CUDA_VISIBLE_DEVICES"] = str(int(physical_gpu))
        output_dir.mkdir(parents=True, exist_ok=True)
        stderr_path.parent.mkdir(parents=True, exist_ok=True)
        self.stderr = stderr_path.open("w", encoding="utf-8")
        command = [
            os.environ.get("PIVOTNAV_OMNIGUARD_PYTHON", sys.executable),
            str(worker),
            "--repo", str(root),
            "--checkpoint", str(checkpoint),
            "--device", "cuda:0",
            "--output-dir", str(output_dir),
            "--profile", "integration_v5",
            "--overrides-json", json.dumps({
                "esdf": {
                    "preserve_origin_free_after_geometry": True,
                    "robot_radius": 0.1,
                    "safety_margin": 0.0,
                },
                "navigation": {
                    "polar_esdf": {"rollout_length": 0.4, "rollout_num_points": 9}
                },
            }, sort_keys=True),
        ]
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.stderr,
            text=True,
            bufsize=1,
            env=environment,
        )
        self.ready = self._read()
        if not self.ready.get("ready"):
            raise RuntimeError(f"OMNIGUARD_NOT_READY:{self.ready}")
        self.inference_count = 0
        self.model_load_count = 1

    def _read(self) -> dict[str, Any]:
        assert self.process.stdout is not None
        while True:
            line = self.process.stdout.readline()
            if not line:
                raise RuntimeError(f"OMNIGUARD_WORKER_EXITED:{self.process.poll()}")
            try:
                response = json.loads(line)
            except json.JSONDecodeError:
                continue
            if response.get("error"):
                raise RuntimeError(response["error"])
            return response

    def request(self, value: dict[str, Any]) -> dict[str, Any]:
        assert self.process.stdin is not None
        self.process.stdin.write(json.dumps(value) + "\n")
        self.process.stdin.flush()
        return self._read()

    def reset(self) -> None:
        response = self.request({"command": "reset"})
        if not response.get("reset"):
            raise RuntimeError(f"OMNIGUARD_RESET_FAILED:{response}")

    def infer_rgb(
        self,
        rgb: np.ndarray,
        *,
        goal_heading_rad: float,
        goal_distance_m: float = 2.0,
    ) -> dict[str, Any]:
        array = np.asarray(rgb)
        if array.ndim != 3 or array.shape[2] < 3:
            raise ValueError("RGB observation must be HWC with at least three channels")
        buffer = io.BytesIO()
        Image.fromarray(array[..., :3].astype(np.uint8), mode="RGB").save(buffer, format="PNG")
        response = self.request({
            "command": "infer_rgb_png_base64",
            "png_base64": base64.b64encode(buffer.getvalue()).decode("ascii"),
            "goal_heading_rad": float(goal_heading_rad),
            "goal_distance_m": float(goal_distance_m),
            "point_goal_active": True,
        })
        raw = np.asarray(response.get("raw_distance_m"), dtype=np.float32)
        if raw.shape != (360,) or not np.isfinite(raw).all():
            raise RuntimeError(f"INVALID_OMNITRAV_RAW_DIST:{raw.shape}")
        self.inference_count += 1
        return response

    def close(self) -> None:
        try:
            if self.process.poll() is None:
                self.request({"command": "close"})
        finally:
            if self.process.poll() is None:
                self.process.terminate()
            self.process.wait(timeout=30)
            self.stderr.close()
