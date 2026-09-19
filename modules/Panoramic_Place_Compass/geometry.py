"""ERP projection, LightGlue/RANSAC verification, and parallax."""
from __future__ import annotations


# ---------------------------------------------------------------------------
# ERP perspective projection
# ---------------------------------------------------------------------------

#!/usr/bin/env python3
"""Habitat-GS ERP to perspective projection for V3.3.8.

The module intentionally contains no arrival decision logic. It only maps an
ERP RGB image to a deterministic, shared query/reference perspective layout.
"""

import hashlib
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class PerspectiveViewSpec:
    view_index: int
    center_yaw_deg: float
    hfov_deg: float = 100.0
    vfov_deg: float = 90.0
    width: int = 256
    height: int = 256


def layout(view_count: int, hfov_deg: float = 100.0, vfov_deg: float = 90.0,
           width: int = 256, height: int = 256) -> tuple[PerspectiveViewSpec, ...]:
    if view_count not in (8, 12, 16):
        raise ValueError("V3.3.8 comparison requires 8, 12, or 16 views")
    return tuple(PerspectiveViewSpec(i, 360.0 * i / view_count, hfov_deg,
                                     vfov_deg, width, height)
                 for i in range(view_count))


def _mapping(spec: PerspectiveViewSpec, erp_shape: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray]:
    height, width = erp_shape[:2]
    fx = (spec.width / 2.0) / math.tan(math.radians(spec.hfov_deg) / 2.0)
    fy = (spec.height / 2.0) / math.tan(math.radians(spec.vfov_deg) / 2.0)
    cx = (spec.width - 1.0) / 2.0
    cy = (spec.height - 1.0) / 2.0
    u, v = np.meshgrid(np.arange(spec.width, dtype=np.float32),
                       np.arange(spec.height, dtype=np.float32))
    ray_x = (u - cx) / fx
    ray_y = -(v - cy) / fy
    longitude = np.arctan2(ray_x, np.ones_like(ray_x)) + math.radians(spec.center_yaw_deg)
    latitude = np.arctan2(ray_y, np.sqrt(ray_x * ray_x + 1.0))
    map_x = ((longitude / (2.0 * math.pi) + 0.5) * width) % width
    map_y = np.clip((0.5 - latitude / math.pi) * height, 0.0, height - 1.0)
    return map_x.astype(np.float32), map_y.astype(np.float32)


def project(erp_rgb: np.ndarray, spec: PerspectiveViewSpec) -> tuple[np.ndarray, dict]:
    if erp_rgb.ndim != 3 or erp_rgb.shape[2] != 3:
        raise ValueError("ERP must be HxWx3 RGB")
    map_x, map_y = _mapping(spec, erp_rgb.shape)
    view = cv2.remap(erp_rgb, map_x, map_y, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_WRAP)
    width = erp_rgb.shape[1]
    half = spec.hfov_deg / 2.0
    seam_crossing = (spec.center_yaw_deg - half < 0.0 or
                     spec.center_yaw_deg + half >= 360.0)
    metadata = asdict(spec)
    metadata.update({
        "erp_resolution": [int(width), int(erp_rgb.shape[0])],
        "mapping": "gnomonic pinhole ray to ERP longitude/latitude; horizontal modulo wrap",
        "seam_crossing": bool(seam_crossing),
        "map_x_min": float(map_x.min()),
        "map_x_max": float(map_x.max()),
        "map_y_min": float(map_y.min()),
        "map_y_max": float(map_y.max()),
        "view_sha256": hashlib.sha256(view.tobytes()).hexdigest(),
    })
    return view, metadata


def project_layout(erp_rgb: np.ndarray, view_count: int) -> tuple[list[np.ndarray], list[dict]]:
    views, metadata = [], []
    for spec in layout(view_count):
        view, info = project(erp_rgb, spec)
        views.append(view)
        metadata.append(info)
    return views, metadata


def save_layout(erp_path: Path, output_dir: Path, view_count: int) -> dict:
    erp = cv2.cvtColor(cv2.imread(str(erp_path)), cv2.COLOR_BGR2RGB)
    views, metadata = project_layout(erp, view_count)
    output_dir.mkdir(parents=True, exist_ok=True)
    for image, info in zip(views, metadata):
        target = output_dir / f"view_{info['view_index']:02d}.png"
        cv2.imwrite(str(target), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        info["path"] = str(target)
    contact = np.concatenate(views, axis=1)
    contact_path = output_dir / "contact_sheet.png"
    cv2.imwrite(str(contact_path), cv2.cvtColor(contact, cv2.COLOR_RGB2BGR))
    return {"layout": metadata, "contact_sheet": str(contact_path),
            "contact_sheet_sha256": hashlib.sha256(contact.tobytes()).hexdigest()}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("erp")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--views", type=int, choices=(8, 12, 16), default=8)
    args = parser.parse_args()
    print(json.dumps(save_layout(Path(args.erp), args.output_dir, args.views), indent=2))


# ---------------------------------------------------------------------------
# LightGlue and RANSAC geometry
# ---------------------------------------------------------------------------

"""LightGlue/RANSAC geometry used by the final arrival verifier."""

import math
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch



ROOT = Path(__file__).resolve().parent
DEPENDENCY_ROOT = Path(
    os.environ.get("PIVOTNAV_ARRIVAL_DEPENDENCY_ROOT", str(ROOT))
).expanduser().resolve()


def _coverage(points: np.ndarray, size: int = 256, cells: int = 6) -> float:
    if not len(points):
        return 0.0
    occupied = {
        (min(cells - 1, int(x / size * cells)), min(cells - 1, int(y / size * cells)))
        for x, y in points
    }
    return len(occupied) / float(cells * cells)


def _mask(value: np.ndarray | None, count: int) -> np.ndarray:
    if value is None or len(value) != count:
        return np.zeros(count, dtype=bool)
    return value.reshape(-1).astype(bool)


class LightGlueGeometry:
    def __init__(self, device: str) -> None:
        external = DEPENDENCY_ROOT / "frozen_inputs/LightGlue"
        if external.is_dir():
            sys.path.insert(0, str(external))
            from lightglue import LightGlue, SuperPoint
            from lightglue.utils import numpy_image_to_torch, rbd
        else:
            from modules.Panoramic_Place_Compass.third_party.lightglue import LightGlue, SuperPoint
            from modules.Panoramic_Place_Compass.third_party.lightglue.utils import (
                numpy_image_to_torch,
                rbd,
            )

        self.device = torch.device(device)
        self.numpy_image_to_torch = numpy_image_to_torch
        self.rbd = rbd
        os.environ["TORCH_HOME"] = str(DEPENDENCY_ROOT / "frozen_inputs/lightglue/cache/torch")
        self.extractor = SuperPoint(max_num_keypoints=512).eval().to(self.device)
        self.matcher = LightGlue(
            features="superpoint",
            n_layers=9,
            flash=True,
            mp=False,
            depth_confidence=0.95,
            width_confidence=0.99,
            filter_threshold=0.1,
        ).eval().to(self.device)
        focal = 128.0 / math.tan(math.radians(50.0))
        self.camera_matrix = np.array(
            [[focal, 0.0, 127.5], [0.0, focal, 127.5], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        self.K = self.camera_matrix

    def analyze(self, query: np.ndarray, goal: np.ndarray, yaw: float) -> dict:
        sectors = []
        started = time.perf_counter()
        for goal_center in (45.0 * index for index in range(8)):
            query_image, _ = project(
                query,
                PerspectiveViewSpec(-1, (goal_center + yaw) % 360.0, 100.0, 90.0, 256, 256),
            )
            goal_image, _ = project(
                goal,
                PerspectiveViewSpec(-1, goal_center, 100.0, 90.0, 256, 256),
            )
            with torch.inference_mode():
                first = self.extractor.extract(
                    self.numpy_image_to_torch(query_image).to(self.device), resize=None
                )
                second = self.extractor.extract(
                    self.numpy_image_to_torch(goal_image).to(self.device), resize=None
                )
                matched = self.rbd(self.matcher({"image0": first, "image1": second}))

            matches = matched["matches"].detach().cpu().numpy()
            first_keypoints = self.rbd(first)["keypoints"].detach().cpu().numpy()
            second_keypoints = self.rbd(second)["keypoints"].detach().cpu().numpy()
            if len(matches):
                first_points = first_keypoints[matches[:, 0]]
                second_points = second_keypoints[matches[:, 1]]
            else:
                first_points = second_points = np.empty((0, 2), dtype=np.float32)

            homography = fundamental = essential = None
            homography_mask = fundamental_mask = essential_mask = None
            if len(matches) >= 4:
                homography, homography_mask = cv2.findHomography(
                    first_points, second_points, cv2.RANSAC, 3.0
                )
            if len(matches) >= 8:
                fundamental, fundamental_mask = cv2.findFundamentalMat(
                    first_points, second_points, cv2.FM_RANSAC, 1.5, 0.999
                )
            if len(matches) >= 5:
                try:
                    essential, essential_mask = cv2.findEssentialMat(
                        first_points,
                        second_points,
                        self.camera_matrix,
                        method=cv2.RANSAC,
                        prob=0.999,
                        threshold=1.5,
                    )
                except cv2.error:
                    essential = essential_mask = None

            h_mask = _mask(homography_mask, len(matches))
            f_mask = _mask(fundamental_mask, len(matches))
            e_mask = _mask(essential_mask, len(matches))
            e_first, e_second = first_points[e_mask], second_points[e_mask]
            hull = 0.0
            if len(e_first) >= 3:
                hull = min(
                    cv2.contourArea(cv2.convexHull(e_first.astype(np.float32))),
                    cv2.contourArea(cv2.convexHull(e_second.astype(np.float32))),
                ) / (256.0 * 256.0)

            positive_depth = 0.0
            if essential is not None and e_mask.sum() >= 5:
                try:
                    recovered = cv2.recoverPose(
                        np.asarray(essential)[:3],
                        first_points,
                        second_points,
                        self.camera_matrix,
                        mask=e_mask.astype(np.uint8)[:, None],
                    )[0]
                    positive_depth = recovered / max(int(e_mask.sum()), 1)
                except cv2.error:
                    pass

            reprojection = 99.0
            if homography is not None and len(first_points):
                try:
                    projected = cv2.perspectiveTransform(
                        first_points[:, None].astype(float), homography
                    )[:, 0]
                    reprojection = float(
                        np.median(np.linalg.norm(projected - second_points, axis=1))
                    )
                except Exception:
                    pass

            sectors.append({
                "raw": len(matches),
                "e": int(e_mask.sum()),
                "er": float(e_mask.mean()) if len(e_mask) else 0.0,
                "h": float(h_mask.mean()) if len(h_mask) else 0.0,
                "f": float(f_mask.mean()) if len(f_mask) else 0.0,
                "grid": min(_coverage(e_first), _coverage(e_second)),
                "hull": float(hull),
                "hs": min(
                    float(np.ptp(e_first[:, 0])) / 256.0 if len(e_first) else 0.0,
                    float(np.ptp(e_second[:, 0])) / 256.0 if len(e_second) else 0.0,
                ),
                "vs": min(
                    float(np.ptp(e_first[:, 1])) / 256.0 if len(e_first) else 0.0,
                    float(np.ptp(e_second[:, 1])) / 256.0 if len(e_second) else 0.0,
                ),
                "reproj": reprojection,
                "pos": float(positive_depth),
            })

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        best = max(sectors, key=lambda value: value["e"])
        return {
            "e_inliers": best["e"],
            "e_inlier_ratio": best["er"],
            "grid_coverage": best["grid"],
            "hull_area": best["hull"],
            "horizontal_span": best["hs"],
            "vertical_span": best["vs"],
            "reprojection_error": best["reproj"],
            "positive_depth_ratio": best["pos"],
            "supported_sectors": sum(value["e"] >= 8 for value in sectors),
            "h_dominance": best["h"] - max(best["er"], best["f"]),
            "latency_ms": (time.perf_counter() - started) * 1000.0,
            "sectors": sectors,
        }


def load_lightglue_geometry(device: str = "cpu") -> LightGlueGeometry:
    return LightGlueGeometry(device)


# ---------------------------------------------------------------------------
# Dynamic parallax
# ---------------------------------------------------------------------------

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
import torch
from PIL import Image


_protocol_env = os.environ.get("PIVOTNAV_PARALLAX_PROTOCOL")
DEFAULT_PROTOCOL = Path(_protocol_env).expanduser().resolve() if _protocol_env else None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_protocol_path(protocol_path: Path, value: str | os.PathLike[str]) -> Path:
    """Resolve protocol resources relative to the protocol file, not cwd."""
    candidate = Path(value).expanduser()
    return candidate if candidate.is_absolute() else (protocol_path.parent / candidate).resolve()


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
    from modules.Panoramic_Place_Compass.third_party.lightglue.utils import numpy_image_to_torch, rbd

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


EXPECTED_LABELS = ("RESTORE_10", "APPROACH_1", "APPROACH_2")


class DynamicParallaxExtractor:
    """Online adapter around the frozen parallax feature functions."""

    def __init__(
        self,
        *,
        device: str,
        integration_root: Path | None = None,
        frozen_runtime_root: Path | None = None,
        commanded_forward_distance_per_step_m: float = 0.16,
    ) -> None:
        integration_root = Path(integration_root or Path(__file__).resolve().parent).resolve()
        frozen_runtime_root = Path(
            frozen_runtime_root
            or os.environ.get("PIVOTNAV_ARRIVAL_FROZEN_ROOT", integration_root)
        ).resolve()
        sys.path[:0] = [str(integration_root), str(frozen_runtime_root)]

        self.device = str(device)
        self.commanded_forward_distance_per_step_m = float(commanded_forward_distance_per_step_m)
        self.geometry = load_lightglue_geometry(self.device)

    @staticmethod
    def _validate_view(view: Mapping[str, Any], expected_label: str) -> None:
        if view.get("label") != expected_label:
            raise ValueError(
                f"dynamic parallax label mismatch: expected {expected_label}, "
                f"received {view.get('label')}"
            )
        image = np.asarray(view.get("image_rgb"))
        if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
            raise ValueError("dynamic parallax image must be uint8 HxWx3 RGB")
        if not np.isfinite(float(view.get("target_yaw_degrees"))):
            raise ValueError("dynamic parallax target yaw must be finite")

    def extract(
        self,
        *,
        request_id: str,
        generation: int,
        candidate_node_id: str,
        target_hash: str,
        target_rgb: np.ndarray,
        views: Sequence[Mapping[str, Any]],
        commanded_forward_distance_per_step_m: float | None = None,
    ) -> dict[str, Any]:
        if not request_id or not candidate_node_id or generation < 1:
            raise ValueError("dynamic parallax requires a current target identity")
        if len(target_hash) != 64:
            raise ValueError("target_hash must be SHA256")
        target = np.asarray(target_rgb)
        if target.ndim != 3 or target.shape[2] != 3 or target.dtype != np.uint8:
            raise ValueError("dynamic parallax target must be uint8 HxWx3 RGB")
        if len(views) != len(EXPECTED_LABELS):
            raise ValueError("dynamic parallax requires exactly three ordered views")
        for view, expected in zip(views, EXPECTED_LABELS):
            self._validate_view(view, expected)

        step_m = (
            self.commanded_forward_distance_per_step_m
            if commanded_forward_distance_per_step_m is None
            else float(commanded_forward_distance_per_step_m)
        )
        if not np.isfinite(step_m) or step_m <= 0.0:
            raise ValueError("commanded forward distance must be positive and finite")

        started = time.perf_counter()
        all_sectors = []
        output_views = []
        for view in views:
            sectors = sector_features(
                self.geometry,
                np.asarray(view["image_rgb"]),
                target,
                float(view["target_yaw_degrees"]),
            )
            all_sectors.append(sectors)
            output_views.append({"label": view["label"], "sectors": sectors})
        return {
            "request_id": request_id,
            "generation": int(generation),
            "candidate_node_id": candidate_node_id,
            "target_hash": target_hash,
            "views": output_views,
            "causal": causal_features(all_sectors, step_m),
            "latency_ms": (time.perf_counter() - started) * 1000.0,
            "view_labels": list(EXPECTED_LABELS),
            "commanded_forward_distance_per_step_m": step_m,
            "runtime_gt_inputs": [],
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol",
        type=Path,
        default=DEFAULT_PROTOCOL,
        help="External parallax protocol; relative resource paths use its directory",
    )
    parser.add_argument("--device", default="cuda:1")
    args = parser.parse_args()
    if args.protocol is None:
        parser.error("--protocol is required (or set PIVOTNAV_PARALLAX_PROTOCOL)")
    protocol_path = args.protocol.expanduser().resolve()
    protocol = json.loads(protocol_path.read_text())
    output_root = protocol_path.parent
    output_path = output_root / "parallax_features.jsonl"
    manifest_path = output_root / "extraction_manifest.json"
    if output_path.exists() or manifest_path.exists():
        raise RuntimeError("refuse to overwrite parallax extraction")
    extractor_path = resolve_protocol_path(protocol_path, protocol["extractor"]["path"])
    if sha256(extractor_path) != protocol["extractor"]["sha256"]:
        raise RuntimeError("extractor changed after protocol freeze")
    input_path = resolve_protocol_path(protocol_path, protocol["input_manifest"]["path"])
    if sha256(input_path) != protocol["input_manifest"]["sha256"]:
        raise RuntimeError("input manifest changed after protocol freeze")
    runtime_path = resolve_protocol_path(protocol_path, protocol["frozen_lightglue_runtime"]["path"])
    if sha256(runtime_path) != protocol["frozen_lightglue_runtime"]["sha256"]:
        raise RuntimeError("frozen LightGlue runtime changed")
    sys.path[:0] = [str(runtime_path.parent)]

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
        "protocol_sha256": sha256(protocol_path),
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
