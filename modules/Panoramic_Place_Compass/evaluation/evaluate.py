"""Evaluate Compass head checkpoints on the public NPZ contracts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import torch

from ..model.arrival import R36TemporalArrivalHead
from ..model.bearing import R35RelativeTranslationBearingHead, R36BearingStructuralRevision5Head
from ..model.orientation import R34CrossPositionYawHead, R34CrossPositionYawHeadR3
from ..model.relative_bearing import R362PatchBearingHead
from ..training.train import _state


def circular_error_deg(prediction: float | np.ndarray, target: float | np.ndarray):
    return np.abs((np.asarray(prediction) - np.asarray(target) + 180.0) % 360.0 - 180.0)


def evaluate_pair(task: str, data: Path, checkpoint: Path, device: str) -> dict:
    with np.load(data, allow_pickle=False) as archive:
        source = torch.from_numpy(archive["source_tokens"]).to(device).float()
        target = torch.from_numpy(archive["target_tokens"]).to(device).float()
        if task == "yaw":
            model = R34CrossPositionYawHeadR3(R34CrossPositionYawHead()).to(device)
            model.load_state_dict(_state(checkpoint, "track_y", "model"), strict=True)
            with torch.inference_mode():
                source_ring = model.base_head.ring_encoder(source)
                target_ring = model.base_head.ring_encoder(target)
                prediction = model(source, target, source_ring, target_ring)["predicted_yaw_degrees"].cpu().numpy()
            labels = archive["yaw_degrees"]
        elif task == "private-bearing":
            model = R35RelativeTranslationBearingHead().to(device)
            model.load_state_dict(_state(checkpoint, "bearing_head", "model"), strict=True)
            with torch.inference_mode():
                prediction = model(source, target)["bearing_angle_degrees"].cpu().numpy()
            labels = archive["bearing_degrees"]
        elif task == "formal-bearing":
            model = R36BearingStructuralRevision5Head().to(device)
            model.load_state_dict(_state(checkpoint, "bearing_head", "model"), strict=True)
            with torch.inference_mode():
                prediction = model(source, target)["bearing_angle_degrees"].cpu().numpy()
            labels = archive["bearing_degrees"]
        else:
            model = R362PatchBearingHead().to(device)
            model.load_state_dict(_state(checkpoint, "bearing_head", "model"), strict=True)
            with torch.inference_mode():
                prediction = model(source, target)["bearing_angle_degrees"].cpu().numpy()
            labels = archive["bearing_degrees"]
    errors = circular_error_deg(prediction, labels)
    return {
        "samples": len(errors),
        "mae_degrees": float(errors.mean()),
        "median_degrees": float(np.median(errors)),
        "accuracy_at_10_degrees": float((errors <= 10.0).mean()),
    }


def evaluate_arrival(data: Path, checkpoint: Path, device: str) -> dict:
    with np.load(data, allow_pickle=False) as archive:
        features = torch.from_numpy(archive["features"]).to(device).float()
        target = archive["target_arrived"].astype(bool)
        mask = archive["frame_mask"].astype(bool)
    model = R36TemporalArrivalHead().to(device)
    model.load_state_dict(_state(checkpoint, "arrival_temporal_head", "arrival_head", "model"), strict=True)
    with torch.inference_mode():
        prediction = (model(features)["temporal_probability"].cpu().numpy() >= 0.5)
    tp = int((prediction & target & mask).sum())
    fp = int((prediction & ~target & mask).sum())
    fn = int((~prediction & target & mask).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {"frames": int(mask.sum()), "precision": precision, "recall": recall, "f1": 2 * precision * recall / max(precision + recall, 1e-12)}


def evaluate_v7(data: Path, checkpoint: Path) -> dict:
    with np.load(data, allow_pickle=False) as archive:
        features = archive["features"].astype(np.float32)
        target = archive["target_m"].astype(np.float32)
    prediction = joblib.load(checkpoint).predict(features)
    error = np.abs(prediction - target)
    return {"samples": len(error), "mae_m": float(error.mean()), "p90_error_m": float(np.percentile(error, 90))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=("yaw", "private-bearing", "formal-bearing", "r363-bearing", "arrival", "v7"), required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.task in {"yaw", "private-bearing", "formal-bearing", "r363-bearing"}:
        result = evaluate_pair(args.task, args.data, args.checkpoint, args.device)
    elif args.task == "arrival":
        result = evaluate_arrival(args.data, args.checkpoint, args.device)
    else:
        result = evaluate_v7(args.data, args.checkpoint)
    result.update({"task": args.task, "runtime_gt_inputs": []})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
