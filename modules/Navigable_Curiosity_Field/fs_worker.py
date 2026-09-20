#!/usr/bin/env python3
"""Persistent JSONL worker for a public NCF checkpoint."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

if __package__:
    from .model import B2FGFSInference
else:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from modules.Navigable_Curiosity_Field.model import B2FGFSInference


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_rgb(path: str | Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB").resize((448, 224), Image.Resampling.BILINEAR)).copy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", type=Path, default=os.environ.get("PIVOTNAV_FS_PACKAGE_ROOT", "."))
    parser.add_argument("--checkpoint", type=Path, default=os.environ.get("PIVOTNAV_FS_CHECKPOINT"))
    parser.add_argument("--dinov2-repo", type=Path, default=os.environ.get("PIVOTNAV_DINOV2_REPO"))
    parser.add_argument("--dinov2-weight", type=Path, default=os.environ.get("PIVOTNAV_DINOV2_WEIGHT"))
    parser.add_argument("--expected-sha256", default=os.environ.get("PIVOTNAV_FS_SHA256"))
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.checkpoint is None or args.dinov2_repo is None or args.dinov2_weight is None:
        parser.error("checkpoint, dinov2-repo, and dinov2-weight are required via CLI or PIVOTNAV_* env")
    checkpoint = args.checkpoint.expanduser().resolve()
    expected = args.expected_sha256 or sha256(checkpoint)
    model = B2FGFSInference(
        args.package_root,
        device=args.device,
        checkpoint_path=checkpoint,
        expected_checkpoint_sha256=expected,
        dinov2_repo=args.dinov2_repo,
        dinov2_weight=args.dinov2_weight,
    )
    print(json.dumps({"ready": True, "device": args.device, "checkpoint_sha256": sha256(checkpoint), "model_load_count": 1}), flush=True)
    for line in os.sys.stdin:
        request = json.loads(line)
        if request.get("command") == "close":
            print(json.dumps({"closed": True}), flush=True)
            return
        if request.get("command") != "infer":
            print(json.dumps({"error": "unsupported command"}), flush=True)
            continue
        started = time.perf_counter()
        output = model.yaw_ensemble({
            "current_erp_rgb": load_rgb(request["current_path"]),
            "goal_erp_rgb": load_rgb(request["goal_path"]),
        })
        print(json.dumps({
            "fg_probabilities": output["fg_probabilities"][0].float().cpu().tolist(),
            "fs_scores": output["fs_scores"][0].float().cpu().tolist(),
            "latency_ms": (time.perf_counter() - started) * 1000.0,
            "model_load_count": 1,
        }), flush=True)


if __name__ == "__main__":
    main()
