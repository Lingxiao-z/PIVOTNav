#!/usr/bin/env python3
"""Persistent, in-memory RGB client for the frozen OmniTrav/OmniGuard worker."""
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


OMNIGUARD_ROOT = Path(os.environ.get("PIVOTNAV_OMNIGUARD_ROOT", ""))
WORKER = Path(os.environ.get("PIVOTNAV_OMNIGUARD_WORKER", ""))
CHECKPOINT = Path(os.environ.get("PIVOTNAV_OMNIGUARD_CHECKPOINT", ""))


class OmniGuardClient:
    def __init__(self, physical_gpu: int, output_dir: Path, stderr_path: Path) -> None:
        if not OMNIGUARD_ROOT.is_dir() or not WORKER.is_file() or not CHECKPOINT.is_file():
            raise RuntimeError("PIVOTNAV_OMNIGUARD_ROOT/WORKER/CHECKPOINT must be configured")
        environment = dict(os.environ)
        environment["CUDA_VISIBLE_DEVICES"] = str(int(physical_gpu))
        output_dir.mkdir(parents=True, exist_ok=True)
        stderr_path.parent.mkdir(parents=True, exist_ok=True)
        self.stderr = stderr_path.open("w")
        command = [
            os.environ.get("PIVOTNAV_OMNIGUARD_PYTHON", sys.executable), str(WORKER),
            "--repo", str(OMNIGUARD_ROOT), "--checkpoint", str(CHECKPOINT),
            "--device", "cuda:0", "--output-dir", str(output_dir),
            "--profile", "integration_v5",
            "--overrides-json", json.dumps({
                "esdf": {"preserve_origin_free_after_geometry": True, "robot_radius": 0.1, "safety_margin": 0.0},
                "navigation": {"polar_esdf": {"rollout_length": 0.4, "rollout_num_points": 9}},
            }, sort_keys=True),
        ]
        self.process = subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.stderr,
            text=True, bufsize=1, env=environment,
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

    def infer_rgb(self, rgb: np.ndarray, *, goal_heading_rad: float,
                  goal_distance_m: float = 2.0) -> dict[str, Any]:
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


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    client = OmniGuardClient(args.gpu, args.output / "worker", args.output / "worker.stderr.log")
    try:
        client.reset()
        rgb = np.asarray(Image.open(args.image).convert("RGB"))
        response = client.infer_rgb(rgb, goal_heading_rad=0.0)
        summary = {
            "ready": client.ready, "model_load_count": client.model_load_count,
            "inference_count": client.inference_count,
            "raw_dist_shape": list(np.asarray(response["raw_distance_m"]).shape),
            "linear_velocity_mps": response["linear_velocity_mps"],
            "angular_velocity_rps": response["angular_velocity_rps"],
            "source": response["source"], "per_frame_disk_rgb_roundtrip": False,
        }
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "smoke.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        print(json.dumps(summary))
    finally:
        client.close()


if __name__ == "__main__":
    main()
