from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

from .model import CompassOutput


ROOT = Path(__file__).resolve().parent
IMPL = ROOT / "_impl"
for path in (IMPL, IMPL / "pano_vpr_v2", IMPL / "r363_python"):
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
        self.nodes: dict[int, dict[str, Any]] = {}
        if self.weights_root:
            self._load()

    def _load(self) -> None:
        import torch

        dino_checkout = ROOT.parent / "Navigable_Curiosity_Field" / "_impl" / "dinov2"
        os.environ.setdefault("PANORAMIC_VPR_DINOV2_CHECKOUT", str(dino_checkout))
        os.environ.setdefault(
            "PANORAMIC_VPR_DINOV2_WEIGHT",
            str(self.weights_root / "dinov2/dinov2_vits14_pretrain.pth"),
        )
        from pano_vpr_v2.r361_modular_inference import load_r361_modular_package

        checkpoint = self.weights_root / "r361/r361_modular.pt"
        self.runtime = load_r361_modular_package(checkpoint, self.config.get("device", "cuda"))
        try:
            os.environ["PIVOTNAV_R363_CHECKPOINT"] = str(self.weights_root / "r363/r363_bearing.pt")
            from bearing_runtime import R363BearingRuntime

            self.bearing = R363BearingRuntime(self.runtime, self.config.get("device", "cuda"))
        except Exception:
            self.bearing = None
        try:
            from .lightglue_ransac import LightGlueRANSACVerifier

            self.geometry = LightGlueRANSACVerifier(self.weights_root, self.config.get("device", "cuda"))
        except Exception:
            self.geometry = None
        self.runtime.eval()

    def encode(self, rgb: np.ndarray) -> Any:
        if self.runtime is None:
            return np.asarray(rgb)
        return self.runtime.encode_panorama(_prepare(rgb))

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
        goal = self.encode(goal_rgb)
        pair = self.runtime._pair_outputs(current, goal)
        similarity = float(self.runtime.retrieve(current["global_descriptor"], goal["global_descriptor"].reshape(1, -1), 1)["scores"].reshape(-1)[0])
        bearing = float(pair["yaw"].get("predicted_yaw_degrees", np.array([0.0])).reshape(-1)[0])
        candidate = similarity >= float(self.config.get("goal_similarity_threshold", 0.90))
        confirmed = False
        geometry = None
        if candidate and self.geometry is not None:
            geometry = self.geometry.verify(current_rgb, goal_rgb, bearing)
            confirmed = bool(geometry.get("confirmed", False))
        return {"similarity": similarity, "bearing_deg": bearing, "arrival_candidate": candidate,
                "arrival_confirmed": confirmed, "geometry": geometry}

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
