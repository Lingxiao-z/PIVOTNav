"""LightGlue/RANSAC geometry used by the final arrival verifier."""
from __future__ import annotations

import math
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from .erp_projection import PerspectiveViewSpec, project


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
