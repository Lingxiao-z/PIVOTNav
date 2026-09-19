"""Small public facade for Navigable Curiosity Field.

The formal runner uses the persistent clients in ``workers.py``. This facade
keeps a compact library API for offline FG/FS scoring and deliberately does
not vendor OmniTrav or OmniGuard; those are external runtime dependencies.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parent


class NavigableCuriosityField:
    """Offline curiosity scoring plus validity-aware frontier selection."""

    def __init__(self, weights_root: Path, config: dict[str, Any]):
        self.config = dict(config)
        self._weights_enabled = str(weights_root) not in {"", "."}
        self.weights_root = Path(weights_root).expanduser().resolve()
        self.device = str(self.config.get("device", "cuda"))
        self._predictor = None

    def _load(self) -> None:
        if self._predictor is not None:
            return
        from .model import B2FGFSInference

        package_root = Path(
            self.config.get(
                "fs_package_root",
                os.environ.get("PIVOTNAV_FS_PACKAGE_ROOT", self.weights_root),
            )
        ).expanduser()
        self._predictor = B2FGFSInference(package_root, device=self.device)

    def predict(self, current_rgb: np.ndarray, goal_rgb: np.ndarray) -> dict[str, np.ndarray]:
        if self._predictor is None and not self._weights_enabled:
            return {
                "fg_probability": np.ones(12, dtype=np.float32),
                "fs_scores": np.zeros(12, dtype=np.float32),
            }
        self._load()
        output = self._predictor.yaw_ensemble({
            "current_erp_rgb": current_rgb,
            "goal_erp_rgb": goal_rgb,
        })
        return {
            "fg_probability": output["fg_probabilities"][0].float().cpu().numpy(),
            "fs_scores": output["fs_scores"][0].float().cpu().numpy(),
        }

    def select(
        self,
        scores: dict[str, np.ndarray],
        distances: np.ndarray,
    ) -> dict[str, np.ndarray | int | None]:
        fg = np.asarray(scores["fg_probability"], dtype=np.float32).reshape(12)
        fs = np.asarray(scores["fs_scores"], dtype=np.float32).reshape(12)
        distance = np.asarray(distances, dtype=np.float32).reshape(-1)
        if distance.size != 360:
            raise ValueError("OmniTrav must provide 360 distances")
        valid = fg >= float(self.config.get("fg_threshold", 0.5))
        clearance = np.asarray([
            np.percentile(distance[(i * 30 + np.arange(-10, 11)) % 360], 20)
            for i in range(12)
        ])
        valid &= clearance >= float(self.config.get("frontier_clearance_m", 0.70))
        masked = np.where(valid, fs, -np.inf)
        return {
            "fg_probability": fg,
            "fs_scores": fs,
            "valid_mask": valid,
            "selected_sector": int(np.argmax(masked)) if valid.any() else None,
        }

    def command(self, *_args: Any, **_kwargs: Any) -> tuple[float, float, dict[str, Any]]:
        raise RuntimeError(
            "Online velocity control is provided by the external OmniGuard worker; "
            "configure PIVOTNAV_OMNIGUARD_WORKER and use the formal runner."
        )

    def predict_distances(self, *_args: Any, **_kwargs: Any) -> np.ndarray:
        raise RuntimeError(
            "OmniTrav is an external dependency; use the persistent OmniGuard worker."
        )


def smoke_curiosity(current: np.ndarray, goal: np.ndarray) -> dict[str, bool]:
    field = NavigableCuriosityField(Path(), {})
    scores = field.predict(current, goal)
    selected = field.select(scores, np.full(360, 8.0, dtype=np.float32))
    return {"ok": selected["valid_mask"].shape == (12,)}
