from __future__ import annotations

import os
import sys
import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from .model import CompassOutput


ROOT = Path(__file__).resolve().parent
for path in (ROOT, ROOT / "pano_vpr_v2", ROOT / "bearing"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def _prepare(rgb: np.ndarray):
    import torch
    from PIL import Image

    value = np.asarray(rgb)[..., :3].astype(np.uint8, copy=False)
    image = Image.fromarray(value, mode="RGB").resize((448, 224), Image.Resampling.BILINEAR)
    return torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1)


class PanoramicPlaceCompass:
    """R36.1 retrieval/bearing/arrival plus LightGlue verification."""

    def __init__(self, weights_root: Path, config: dict[str, Any]):
        self.config = config
        self.weights_root = Path(weights_root).expanduser().resolve()
        self.runtime = None
        self.bearing = None
        self.geometry = None
        self.arrival_state = None
        self._evidence_counter = 0
        self._goal_strong_geometry_streak = 0
        self._goal_last_candidate = False
        self._goal_verification_active = False
        self._goal_verification_frames = 0
        self._goal_reject_cooldown = 0
        self._encoding_cache: dict[str, Any] = {}
        self._goal_encoding: Any | None = None
        self.nodes: dict[int, dict[str, Any]] = {}
        if self.weights_root:
            self._load()

    def _load(self) -> None:
        import torch

        repo_root = ROOT.parent.parent
        dino_checkout = repo_root / "third_party" / "dinov2"
        os.environ.setdefault("PANORAMIC_VPR_DINOV2_CHECKOUT", str(dino_checkout))
        os.environ.setdefault(
            "PANORAMIC_VPR_DINOV2_WEIGHT",
            str(self.weights_root / "dinov2/dinov2_vits14_pretrain.pth"),
        )
        from pano_vpr_v2.r361_modular_inference import load_r361_modular_package

        checkpoint = self.weights_root / "r361/r361_modular.pt"
        self.runtime = load_r361_modular_package(checkpoint, self.config.get("device", "cuda"))
        self.arrival_state = self.runtime.reset_arrival_state()
        try:
            os.environ["PIVOTNAV_R363_CHECKPOINT"] = str(self.weights_root / "r363/r363_bearing.pt")
            from .bearing_runtime import R363BearingRuntime

            self.bearing = R363BearingRuntime(self.runtime, self.config.get("device", "cuda"))
        except Exception as exc:
            raise RuntimeError("R36.3 bearing runtime failed to initialize") from exc
        try:
            from .lightglue_ransac import LightGlueRANSACVerifier

            self.geometry = LightGlueRANSACVerifier(self.weights_root, self.config.get("device", "cuda"))
        except Exception as exc:
            raise RuntimeError("LightGlue/RANSAC verifier failed to initialize") from exc
        self.runtime.eval()

    def encode(self, rgb: np.ndarray) -> Any:
        if self.runtime is None:
            return np.asarray(rgb)
        value = np.ascontiguousarray(np.asarray(rgb)[..., :3].astype(np.uint8, copy=False))
        key = hashlib.sha1(value.tobytes()).hexdigest()
        cached = self._encoding_cache.get(key)
        if cached is None:
            cached = self.runtime.encode_panorama(_prepare(value))
            # Keep a small bounded cache for same-frame goal/control reuse.
            if len(self._encoding_cache) >= 32:
                self._encoding_cache.pop(next(iter(self._encoding_cache)))
            self._encoding_cache[key] = cached
        return cached

    def add_node(self, node_id: int, rgb: np.ndarray) -> None:
        self.nodes[int(node_id)] = {"rgb": np.asarray(rgb)[..., :3].copy(), "encoding": self.encode(rgb)}
        if self.bearing is not None:
            self.bearing.add_node(int(node_id), np.asarray(rgb), _prepare)

    def retrieve(self, rgb: np.ndarray, top_k: int = 8) -> list[tuple[int, float]]:
        if not self.nodes:
            return []
        if self.runtime is None:
            return [(node_id, 0.0) for node_id in sorted(self.nodes)[:top_k]]
        query = self.encode(rgb)
        ids = sorted(self.nodes)
        database = torch_stack([self.nodes[node_id]["encoding"]["global_descriptor"] for node_id in ids], self.runtime.device)
        result = self.runtime.retrieve(query["global_descriptor"], database, min(top_k, len(ids)))
        return [(ids[int(i)], float(s)) for i, s in zip(result["indices"].reshape(-1), result["scores"].reshape(-1))]

    def goal_evidence(self, current_rgb: np.ndarray, goal_rgb: np.ndarray) -> dict[str, Any]:
        if self.runtime is None:
            return {"similarity": 0.0, "bearing_deg": 0.0, "arrival_candidate": False, "arrival_confirmed": False}
        current = self.encode(current_rgb)
        if self._goal_encoding is None:
            self._goal_encoding = self.encode(goal_rgb)
        goal = self._goal_encoding
        pair = self.runtime._pair_outputs(current, goal)
        self._evidence_counter += 1
        arrival = self.runtime.predict_arrival(
            current, goal, f"goal-evidence-{self._evidence_counter}", self.arrival_state,
        )
        self.arrival_state = arrival["temporal_state"]
        similarity = float(self.runtime.retrieve(current["global_descriptor"], goal["global_descriptor"].reshape(1, -1), 1)["scores"].reshape(-1)[0])
        bearing = float(pair["yaw"].get("predicted_yaw_degrees", np.array([0.0])).reshape(-1)[0])
        # Match the frozen server contract: the arrival head is supporting
        # evidence, not a standalone trigger. A goal candidate must first
        # pass the strict VPR similarity gate.
        threshold = float(self.config.get("goal_similarity_threshold", 0.975))
        if self._goal_reject_cooldown > 0:
            self._goal_reject_cooldown -= 1
            return {"similarity": similarity, "bearing_deg": bearing, "arrival_candidate": False,
                    "arrival_confirmed": False, "arrival_probability": float(arrival["temporal_probability"].reshape(-1)[0]),
                    "arrival": arrival, "geometry": None}
        if not self._goal_verification_active and similarity >= threshold:
            self._goal_verification_active = True
            self._goal_verification_frames = 0
            self._goal_strong_geometry_streak = 0
        candidate = bool(self._goal_verification_active)
        confirmed = False
        geometry = None
        if candidate and self.geometry is not None:
            self._goal_verification_frames += 1
            geometry = self.geometry.verify(current_rgb, goal_rgb, bearing)
            arrival_probability = float(arrival["temporal_probability"].reshape(-1)[0])
            min_arrival_probability = float(self.config.get("goal_min_arrival_probability", 0.010))
            strong_geometry = (
                int(geometry.get("inliers", 0)) >= int(self.config.get("goal_min_inliers", 18))
                and float(geometry.get("grid_coverage", 0.0)) >= float(self.config.get("goal_min_grid_coverage", 0.055))
                and float(geometry.get("reprojection_error_px", 99.0)) <= float(self.config.get("goal_max_reprojection_error_px", 3.0))
                and int(geometry.get("supported_sectors", 0)) >= int(self.config.get("goal_min_supported_sectors", 2))
            )
            if strong_geometry and arrival_probability >= min_arrival_probability:
                self._goal_strong_geometry_streak += 1
            else:
                self._goal_strong_geometry_streak = 0
            confirmed = self._goal_strong_geometry_streak >= int(self.config.get("goal_required_consecutive_geometry", 2))
            geometry["strong_geometry"] = strong_geometry
            geometry["strong_geometry_streak"] = self._goal_strong_geometry_streak
            if confirmed:
                self._goal_verification_active = False
            elif self._goal_verification_frames >= int(self.config.get("goal_max_verification_frames", 12)):
                self._goal_verification_active = False
                self._goal_reject_cooldown = int(self.config.get("goal_reject_cooldown_frames", 12))
                self._goal_strong_geometry_streak = 0
        elif not candidate:
            self._goal_strong_geometry_streak = 0
        self._goal_last_candidate = candidate
        return {"similarity": similarity, "bearing_deg": bearing, "arrival_candidate": candidate,
                "arrival_confirmed": confirmed, "arrival_probability": float(arrival["temporal_probability"].reshape(-1)[0]),
                "arrival": arrival, "geometry": geometry}

    def command_bearing(self, current_rgb: np.ndarray, node_id: int) -> float:
        if self.runtime is None:
            return 0.0
        query = self.encode(current_rgb)
        if self.bearing is not None:
            return float(self.bearing.predict_to_node(current_rgb, node_id, _prepare)["bearing_deg"])
        return float(self.runtime.predict_bearing(query, self.nodes[node_id]["encoding"])["bearing_degrees"].reshape(-1)[0])


def torch_stack(values: list[Any], device: Any):
    import torch

    return torch.stack([value.detach().reshape(-1) for value in values]).to(device)


def smoke_compass(current: np.ndarray, goal: np.ndarray) -> dict[str, bool]:
    return {"ok": current.shape == goal.shape and current.ndim == 3 and current.shape[2] == 3}
