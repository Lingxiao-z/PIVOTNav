#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image


def load_erp(path: str | Path) -> np.ndarray:
    with Image.open(path) as source:
        return np.asarray(source.convert("RGB").resize((448, 224), Image.Resampling.BILINEAR)).copy()


def emit(value: dict) -> None:
    print(json.dumps(value, sort_keys=True), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    sys.path.insert(0, str(args.package_root.resolve()))
    from panorama_fs.b2_inference import B2FGFSInference

    started = time.perf_counter()
    model = B2FGFSInference(args.package_root, device=args.device)
    emit({"ready": True, "device": args.device, "load_seconds": time.perf_counter() - started})
    for line in sys.stdin:
        try:
            request = json.loads(line)
            command = request.get("command")
            if command == "close":
                emit({"closed": True})
                return
            if command != "infer":
                raise ValueError(f"unsupported command: {command}")
            current = load_erp(request["current_path"])
            goal = load_erp(request["goal_path"])
            started = time.perf_counter()
            output = model.yaw_ensemble({"current_erp_rgb": current, "goal_erp_rgb": goal})
            emit({
                "fg_probabilities": output["fg_probabilities"][0].float().cpu().tolist(),
                "fs_scores": output["fs_scores"][0].float().cpu().tolist(),
                "latency_ms": (time.perf_counter() - started) * 1000.0,
                "input_shape": [224, 448, 3],
                "resize": "PIL_RGB_BILINEAR",
                "yaw_ensemble_sector_shifts": [0, 3, 6, 9],
                "uses_gt_roll_selection": False,
            })
        except Exception as error:
            emit({"error": f"{type(error).__name__}: {error}"})


if __name__ == "__main__":
    main()
