from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from erp_projection import PerspectiveViewSpec, project


class LightGlueRANSACVerifier:
    def __init__(self, weights_root: Path, device: str = "cuda"):
        import torch
        from lightglue import LightGlue, SuperPoint
        from lightglue.utils import numpy_image_to_torch, rbd

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.numpy_image_to_torch = numpy_image_to_torch
        self.rbd = rbd
        self.extractor = SuperPoint(max_num_keypoints=512, weights=str(weights_root / "lightglue/superpoint_v1.pth")).eval().to(self.device)
        self.matcher = LightGlue(features=None, weights=str(weights_root / "lightglue/superpoint_lightglue_v0-1_arxiv-pth"), flash=False, mp=False).eval().to(self.device)

    @staticmethod
    def _view(erp: np.ndarray, center_deg: float) -> np.ndarray:
        spec = PerspectiveViewSpec(view_index=0, center_yaw_deg=float(center_deg) % 360.0, hfov_deg=100.0, vfov_deg=90.0, width=256, height=256)
        return project(np.asarray(erp)[..., :3].astype(np.uint8), spec)[0]

    @staticmethod
    def _coverage(points: np.ndarray) -> float:
        if len(points) == 0:
            return 0.0
        cells = {(min(5, int(p[0] / 256 * 6)), min(5, int(p[1] / 256 * 6))) for p in points}
        return len(cells) / 36.0

    def verify(self, current: np.ndarray, goal: np.ndarray, yaw_deg: float = 0.0) -> dict[str, float | int | bool]:
        best = None
        for center in np.arange(0.0, 360.0, 45.0):
            a_img, b_img = self._view(current, center + yaw_deg), self._view(goal, center)
            with __import__("torch").inference_mode():
                a = self.extractor.extract(self.numpy_image_to_torch(a_img).to(self.device), resize=None)
                b = self.extractor.extract(self.numpy_image_to_torch(b_img).to(self.device), resize=None)
                matches = self.rbd(self.matcher({"image0": a, "image1": b}))
            m = matches["matches"].detach().cpu().numpy()
            ka = self.rbd(a)["keypoints"].detach().cpu().numpy()
            kb = self.rbd(b)["keypoints"].detach().cpu().numpy()
            p0, p1 = (ka[m[:, 0]], kb[m[:, 1]]) if len(m) else (np.empty((0, 2), np.float32), np.empty((0, 2), np.float32))
            inliers = 0
            reproj = 99.0
            if len(m) >= 4:
                h, mask = cv2.findHomography(p0, p1, cv2.RANSAC, 3.0)
                if mask is not None:
                    inlier_mask = mask.reshape(-1).astype(bool)
                    inliers = int(inlier_mask.sum())
                    if h is not None:
                        projected = cv2.perspectiveTransform(p0[:, None].astype(np.float32), h)[:, 0]
                        reproj = float(np.median(np.linalg.norm(projected - p1, axis=1)))
                    coverage = self._coverage(p0[inlier_mask])
                else:
                    coverage = 0.0
            else:
                coverage = 0.0
            item = {"raw_matches": int(len(m)), "inliers": inliers, "grid_coverage": coverage, "reprojection_error_px": reproj}
            if best is None or item["inliers"] > best["inliers"]:
                best = item
        assert best is not None
        best["confirmed"] = bool(best["inliers"] >= 8 and best["grid_coverage"] >= 0.02 and best["reprojection_error_px"] <= 5.0)
        return best
