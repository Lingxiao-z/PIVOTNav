"""Public facade backed by the formal Panoramic Place Compass runtime."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parent


class PanoramicPlaceCompass:
    """Panoramic retrieval, relative bearing, and single-frame goal evidence."""

    def __init__(self, weights_root: Path, config: dict[str, Any]):
        self.config = dict(config)
        self.weights_root = Path(weights_root).expanduser().resolve()
        self.device = str(self.config.get("device", "cuda:0"))
        self.retrieval = None
        self.bearing = None
        self.geometry = None
        self.nodes: dict[int, np.ndarray] = {}
        if str(weights_root) not in {"", "."}:
            self._load()

    def _load(self) -> None:
        from .geometry import load_lightglue_geometry
        from .localization import R361Adapter, R363BearingAdapter

        self.retrieval = R361Adapter(
            ROOT,
            device=self.device,
            checkpoint_path=self.weights_root / "r361/r361_modular.pt",
        )
        self.bearing = R363BearingAdapter(
            ROOT,
            ROOT / "model",
            adapter_root=ROOT,
            device=self.device,
            r361_adapter=self.retrieval,
            checkpoint_path=self.weights_root / "r363/r363_bearing.pt",
        )
        self.geometry = load_lightglue_geometry(self.device)

    def encode(self, rgb: np.ndarray) -> Any:
        return np.asarray(rgb) if self.retrieval is None else self.retrieval.encode_panorama(rgb)

    def add_node(self, node_id: int, rgb: np.ndarray) -> None:
        image = np.asarray(rgb)[..., :3].copy()
        self.nodes[int(node_id)] = image
        if self.bearing is not None:
            self.bearing.encode_node_package_once(int(node_id), image)

    def retrieve(self, rgb: np.ndarray, top_k: int = 8) -> list[tuple[int, float]]:
        if not self.nodes:
            return []
        if self.retrieval is None:
            return [(node_id, 0.0) for node_id in sorted(self.nodes)[:top_k]]
        import torch

        query = self.retrieval.encode_query(rgb)
        node_ids = sorted(self.nodes)
        descriptors = torch.stack([
            self.retrieval.encode_node_once(node_id, self.nodes[node_id])
            for node_id in node_ids
        ])
        result = self.retrieval.retrieve(query, descriptors, top_k=min(top_k, len(node_ids)))
        return [
            (node_ids[int(index)], float(score))
            for index, score in zip(result["indices"].reshape(-1), result["scores"].reshape(-1))
        ]

    def command_bearing(self, current_rgb: np.ndarray, node_id: int) -> float:
        if self.bearing is None:
            return 0.0
        target = self.bearing.encode_node_package_once(node_id, self.nodes[int(node_id)])
        output = self.bearing.predict_bearing_to_encoding(current_rgb, target)
        return float(output["bearing_degrees"].reshape(-1)[0])

    def goal_evidence(self, current_rgb: np.ndarray, goal_rgb: np.ndarray) -> dict[str, Any]:
        """Return diagnostic evidence; final Stop is owned by the formal verifier."""
        if self.retrieval is None:
            return {"similarity": 0.0, "bearing_deg": 0.0, "arrival_candidate": False}
        from .localization import pair_from_encoding

        current = self.retrieval.encode_query(current_rgb)
        goal = self.retrieval.encode_panorama(goal_rgb)
        pair = pair_from_encoding(self.retrieval.runtime, current, goal)
        geometry = self.geometry.analyze(current_rgb, goal_rgb, pair["yaw_degrees"])
        return {
            "similarity": pair["vpr_similarity"],
            "bearing_deg": pair["yaw_degrees"],
            "arrival_probability": pair["arrival_probability"],
            "arrival_candidate": pair["vpr_similarity"] >= float(
                self.config.get("goal_similarity_threshold", 0.975)
            ),
            "geometry": geometry,
            "arrival_confirmed": False,
        }


def smoke_compass(current: np.ndarray, goal: np.ndarray) -> dict[str, bool]:
    return {"ok": current.shape == goal.shape and current.ndim == 3 and current.shape[2] == 3}
