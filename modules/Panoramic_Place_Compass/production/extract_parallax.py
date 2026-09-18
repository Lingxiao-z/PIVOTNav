from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image


DEFAULT_PROTOCOL = Path(
    "/home/renh/project/pano-navigation/outputs/integration_v2_behavioral/"
    "stage_i5/arrival_parallax_v3_development/protocol.json"
)
PANO = Path("/home/renh/project/pano-navigation")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite(value: float, fallback: float = 0.0) -> float:
    return float(value) if math.isfinite(float(value)) else fallback


def distribution(values: np.ndarray) -> dict[str, float]:
    if not len(values):
        return {"mean": 0.0, "median": 0.0, "p25": 0.0, "p75": 0.0, "p90": 0.0}
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p25": float(np.quantile(values, 0.25)),
        "p75": float(np.quantile(values, 0.75)),
        "p90": float(np.quantile(values, 0.90)),
    }


def sector_features(geometry, query: np.ndarray, target: np.ndarray, yaw: float) -> list[dict]:
    from erp_projection import PerspectiveViewSpec, project
    from lightglue.utils import numpy_image_to_torch, rbd

    sectors = []
    for sector_id, center in enumerate(45.0 * index for index in range(8)):
        query_view, _ = project(
            query,
            PerspectiveViewSpec(-1, (center + yaw) % 360.0, 100.0, 90.0, 256, 256),
        )
        target_view, _ = project(
            target,
            PerspectiveViewSpec(-1, center, 100.0, 90.0, 256, 256),
        )
        with torch.inference_mode():
            query_features = geometry.extractor.extract(
                numpy_image_to_torch(query_view).to(geometry.device), resize=None
            )
            target_features = geometry.extractor.extract(
                numpy_image_to_torch(target_view).to(geometry.device), resize=None
            )
            matched = rbd(geometry.matcher({"image0": query_features, "image1": target_features}))
        matches = matched["matches"].detach().cpu().numpy()
        query_keypoints = rbd(query_features)["keypoints"].detach().cpu().numpy()
        target_keypoints = rbd(target_features)["keypoints"].detach().cpu().numpy()
        p0 = query_keypoints[matches[:, 0]] if len(matches) else np.empty((0, 2))
        p1 = target_keypoints[matches[:, 1]] if len(matches) else np.empty((0, 2))
        essential = essential_mask = homography = homography_mask = None
        if len(matches) >= 5:
            try:
                essential, essential_mask = cv2.findEssentialMat(
                    p0, p1, geometry.K, method=cv2.RANSAC, prob=0.999, threshold=1.5
                )
            except cv2.error:
                pass
        if len(matches) >= 4:
            try:
                homography, homography_mask = cv2.findHomography(p0, p1, cv2.RANSAC, 3.0)
            except cv2.error:
                pass
        if essential_mask is None or len(essential_mask) != len(matches):
            inliers = np.zeros(len(matches), dtype=bool)
        else:
            inliers = essential_mask.reshape(-1).astype(bool)
        selected = inliers if int(inliers.sum()) >= 5 else np.ones(len(matches), dtype=bool)
        flow = np.linalg.norm(p1[selected] - p0[selected], axis=1) if len(matches) else np.empty(0)
        flow_x = np.abs(p1[selected, 0] - p0[selected, 0]) if len(matches) else np.empty(0)
        flow_y = np.abs(p1[selected, 1] - p0[selected, 1]) if len(matches) else np.empty(0)
        homography_scale = 1.0
        if homography is not None and abs(float(homography[2, 2])) > 1e-8:
            normalized = homography / homography[2, 2]
            determinant = float(np.linalg.det(normalized[:2, :2]))
            homography_scale = math.sqrt(abs(determinant)) if math.isfinite(determinant) else 1.0
        homography_ratio = 0.0
        if homography_mask is not None and len(homography_mask) == len(matches):
            homography_ratio = float(homography_mask.reshape(-1).mean())
        sectors.append({
            "sector_id": sector_id,
            "raw_matches": int(len(matches)),
            "essential_inliers": int(inliers.sum()),
            "essential_ratio": float(inliers.mean()) if len(inliers) else 0.0,
            "homography_inlier_ratio": homography_ratio,
            "homography_scale": finite(homography_scale, 1.0),
            "flow": distribution(flow),
            "flow_x": distribution(flow_x),
            "flow_y": distribution(flow_y),
            "runtime_gt_inputs": [],
        })
    return sectors


def causal_features(view_sectors: list[list[dict]], step_m: float) -> dict:
    sector_id = max(
        range(8),
        key=lambda index: sum(view[index]["essential_inliers"] for view in view_sectors),
    )
    common = [view[sector_id] for view in view_sectors]
    flows = [view["flow"]["median"] for view in common]
    deltas = [flows[index] - flows[index + 1] for index in range(2)]
    estimates = []
    for after, delta in zip(flows[1:], deltas):
        estimates.append(step_m * after / delta if delta > 1e-4 else 99.0)
    return {
        "common_sector_id": sector_id,
        "flow_median_sequence": flows,
        "flow_decrease_sequence": deltas,
        "remaining_distance_estimate_sequence_m": [finite(value, 99.0) for value in estimates],
        "remaining_distance_estimate_median_m": float(np.median(estimates)),
        "monotonic_flow_decrease_count": sum(delta > 0.0 for delta in deltas),
        "essential_inlier_sequence": [view["essential_inliers"] for view in common],
        "homography_scale_sequence": [view["homography_scale"] for view in common],
        "runtime_gt_inputs": [],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--device", default="cuda:1")
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    output_root = args.protocol.parent
    output_path = output_root / "parallax_features.jsonl"
    manifest_path = output_root / "extraction_manifest.json"
    if output_path.exists() or manifest_path.exists():
        raise RuntimeError("refuse to overwrite parallax extraction")
    extractor_path = Path(protocol["extractor"]["path"])
    if sha256(extractor_path) != protocol["extractor"]["sha256"]:
        raise RuntimeError("extractor changed after protocol freeze")
    input_path = Path(protocol["input_manifest"]["path"])
    if sha256(input_path) != protocol["input_manifest"]["sha256"]:
        raise RuntimeError("input manifest changed after protocol freeze")
    runtime_path = Path(protocol["frozen_lightglue_runtime"]["path"])
    if sha256(runtime_path) != protocol["frozen_lightglue_runtime"]["sha256"]:
        raise RuntimeError("frozen LightGlue runtime changed")
    sys.path[:0] = [str(PANO / "models/arrival_verifier_frozen")]
    from self_contained_runtime import load_lightglue_geometry

    geometry = load_lightglue_geometry(args.device)
    inputs = [json.loads(line) for line in input_path.read_text().splitlines() if line]
    started = time.perf_counter()
    results = []
    with output_path.open("x") as stream:
        for index, row in enumerate(inputs):
            target_path = Path(row["target"]["path"])
            if sha256(target_path) != row["target"]["sha256"]:
                raise RuntimeError(f"target changed for {row['trial_id']}")
            target = np.asarray(Image.open(target_path).convert("RGB"))
            views = []
            all_sectors = []
            trial_started = time.perf_counter()
            for view in row["views"]:
                path = Path(view["path"])
                if sha256(path) != view["sha256"]:
                    raise RuntimeError(f"view changed for {row['trial_id']} {view['label']}")
                query = np.asarray(Image.open(path).convert("RGB"))
                sectors = sector_features(
                    geometry, query, target, float(view["target_yaw_degrees"])
                )
                all_sectors.append(sectors)
                views.append({"label": view["label"], "sectors": sectors})
            torch.cuda.synchronize(geometry.device)
            result = {
                "trial_id": row["trial_id"],
                "views": views,
                "causal": causal_features(
                    all_sectors, protocol["commanded_forward_distance_per_step_m"]
                ),
                "latency_ms": (time.perf_counter() - trial_started) * 1000.0,
                "runtime_gt_inputs": [],
            }
            stream.write(json.dumps(result, sort_keys=True) + "\n")
            stream.flush()
            results.append(result)
            print(json.dumps({
                "index": index + 1,
                "total": len(inputs),
                "trial_id": row["trial_id"],
                "latency_ms": result["latency_ms"],
            }), flush=True)
    manifest = {
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": sha256(args.protocol),
        "status": "PASS" if len(results) == protocol["expected_trial_count"] else "FAIL",
        "trial_count": len(results),
        "device": args.device,
        "exclusive_gpu_required": False,
        "output": {"path": str(output_path), "sha256": sha256(output_path)},
        "elapsed_seconds": time.perf_counter() - started,
        "runtime_gt_inputs": [],
        "external_processes_modified": [],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
