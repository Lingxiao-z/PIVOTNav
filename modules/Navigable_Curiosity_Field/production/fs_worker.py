"""Persistent FS worker client used by the formal navigation runner."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any


CHECKPOINT_SHA = "44aa451546691f35659ce1ecc0d616d67d706217ceb5a8f2ed43cba9b132760f"


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
        package_root = os.environ.get("PIVOTNAV_FS_PACKAGE_ROOT")
        if not package_root:
            raise RuntimeError("PIVOTNAV_FS_PACKAGE_ROOT is required")
        self.stderr = stderr_path.open("w", encoding="utf-8")
        self.process = subprocess.Popen(
            [sys.executable, str(worker), "--package-root", package_root, "--device", worker_device],
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
