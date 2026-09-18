from __future__ import annotations

import hashlib
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


PANO = Path(__file__).resolve().parent
EXPECTED_LABELS = ("RESTORE_10", "APPROACH_1", "APPROACH_2")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class DynamicParallaxExtractor:
    """Online adapter around the exact frozen v3 parallax feature functions."""

    def __init__(
        self,
        *,
        device: str,
        integration_root: Path | None = None,
        frozen_runtime_root: Path | None = None,
        commanded_forward_distance_per_step_m: float = 0.16,
    ) -> None:
        integration_root = Path(integration_root or PANO).resolve()
        frozen_runtime_root = Path(frozen_runtime_root or os.environ.get("PIVOTNAV_ARRIVAL_FROZEN_ROOT", PANO / "arrival_frozen")).resolve()
        sys.path[:0] = [str(integration_root), str(frozen_runtime_root)]
        from modules.Panoramic_Place_Compass.production.extract_parallax import causal_features, sector_features
        from modules.Panoramic_Place_Compass.production.arrival_frozen.runtime import load_lightglue_geometry

        self.device = str(device)
        self.commanded_forward_distance_per_step_m = float(
            commanded_forward_distance_per_step_m
        )
        self.geometry = load_lightglue_geometry(self.device)
        self._sector_features = sector_features
        self._causal_features = causal_features

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
        yaw = float(view.get("target_yaw_degrees"))
        if not np.isfinite(yaw):
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
        all_sectors: list[list[dict[str, Any]]] = []
        output_views = []
        for view in views:
            sectors = self._sector_features(
                self.geometry,
                np.asarray(view["image_rgb"]),
                target,
                float(view["target_yaw_degrees"]),
            )
            all_sectors.append(sectors)
            output_views.append({"label": view["label"], "sectors": sectors})
        causal = self._causal_features(
            all_sectors, step_m
        )
        return {
            "request_id": request_id,
            "generation": int(generation),
            "candidate_node_id": candidate_node_id,
            "target_hash": target_hash,
            "views": output_views,
            "causal": causal,
            "latency_ms": (time.perf_counter() - started) * 1000.0,
            "view_labels": list(EXPECTED_LABELS),
            "commanded_forward_distance_per_step_m": step_m,
            "runtime_gt_inputs": [],
        }
