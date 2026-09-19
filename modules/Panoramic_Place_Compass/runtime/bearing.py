from __future__ import annotations

import copy
import hashlib
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


R361_SHA256 = "4090261bcef45f70ba771533d1283a3ebe2b5e7d84c400d829cac25e01d79f4a"
R363_SHA256 = "dd6326d3306b24e8538e42cd5ee0378b118b19e0045fc89aa2567dc7337a8353"
TAIL_PREFIXES = ("blocks.11.", "norm.")


def wrap_degrees(value: float) -> float:
    return float(((float(value) + 180.0) % 360.0) - 180.0)


@dataclass
class EgocentricBearingTracker:
    """Track parent-ERP directions from executed angular commands."""

    heading_from_parent_degrees: float = 0.0
    hop_generation: int = 0

    def reset_for_new_hop(self) -> None:
        self.heading_from_parent_degrees = 0.0
        self.hop_generation += 1

    def apply_executed_angular(self, angular_velocity_rps: float, time_step_s: float) -> None:
        heading_delta = -math.degrees(float(angular_velocity_rps) * float(time_step_s))
        self.heading_from_parent_degrees = wrap_degrees(
            self.heading_from_parent_degrees + heading_delta
        )

    def local_bearing(self, parent_bearing_degrees: float) -> float:
        return wrap_degrees(float(parent_bearing_degrees) - self.heading_from_parent_degrees)

    @staticmethod
    def omnitrav_index_for_local_bearing(local_bearing_degrees: float) -> int:
        return int(round(wrap_degrees(local_bearing_degrees) + 180.0)) % 360

    def parent_sector_for_local(
        self,
        local_sector: int,
        sector_count: int = 12,
        forward_sector: int | None = None,
    ) -> int:
        if sector_count <= 0:
            raise ValueError("sector_count must be positive")
        forward_sector = int(sector_count) // 2 if forward_sector is None else int(forward_sector)
        width = 360.0 / int(sector_count)
        local_angle = wrap_degrees((int(local_sector) - forward_sector) * width)
        parent_angle = wrap_degrees(local_angle + self.heading_from_parent_degrees)
        return (int(round(parent_angle / width)) + forward_sector) % int(sector_count)

    def local_sector_for_parent(
        self,
        parent_sector: int,
        sector_count: int = 12,
        forward_sector: int | None = None,
    ) -> int:
        if sector_count <= 0:
            raise ValueError("sector_count must be positive")
        forward_sector = int(sector_count) // 2 if forward_sector is None else int(forward_sector)
        width = 360.0 / int(sector_count)
        parent_angle = wrap_degrees((int(parent_sector) - forward_sector) * width)
        local_angle = self.local_bearing(parent_angle)
        return (int(round(local_angle / width)) + forward_sector) % int(sector_count)


class R363BearingAdapter:
    """Frozen R36.1 retrieval plus a private R36.3 bearing encoder tail/head."""

    def __init__(
        self,
        r361_package_root: str | Path,
        r363_package_root: str | Path,
        *,
        adapter_root: str | Path,
        device: str = "cuda:0",
        r361_adapter: Any | None = None,
        checkpoint_path: str | Path | None = None,
    ) -> None:
        adapter_root = Path(adapter_root).resolve()
        r363_root = Path(r363_package_root).resolve()
        for path in (str(adapter_root), str(r363_root)):
            if path not in sys.path:
                sys.path.insert(0, path)

        from modules.Panoramic_Place_Compass.runtime.retrieval import R361Adapter
        from modules.Panoramic_Place_Compass.models_vpr.relative_bearing_head import (
            R362BearingHeadConfig,
            R362PatchBearingHead,
        )

        self.device = torch.device(device)
        self.retrieval = r361_adapter or R361Adapter(
            r361_package_root, device=device, precision="fp32"
        )

        checkpoint = Path(checkpoint_path).expanduser().resolve() if checkpoint_path else r363_root / "r363_bearing.pt"
        actual_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        if actual_sha != R363_SHA256:
            raise RuntimeError(
                f"R36.3 checkpoint SHA mismatch: expected {R363_SHA256}, got {actual_sha}"
            )
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if payload.get("schema_version") != "r363_limited_unfreeze_checkpoint_v1":
            raise RuntimeError("unsupported R36.3 checkpoint schema")
        if payload.get("frozen_r361_checkpoint_sha256") != R361_SHA256:
            raise RuntimeError("R36.3 was not trained from the frozen R36.1 package")

        self.bearing_backbone = copy.deepcopy(
            self.retrieval.runtime.encoder_system.backbone
        )
        state = self.bearing_backbone.model.state_dict()
        tail = payload.get("bearing_tail", {})
        expected_tail = {name for name in state if name.startswith(TAIL_PREFIXES)}
        if set(tail) != expected_tail:
            raise RuntimeError(
                f"R36.3 private-tail mismatch: missing={sorted(expected_tail-set(tail))}, "
                f"unexpected={sorted(set(tail)-expected_tail)}"
            )
        state.update(tail)
        self.bearing_backbone.model.load_state_dict(state, strict=True)
        self.bearing_backbone.requires_grad_(False).eval().to(self.device)

        self.bearing_head = R362PatchBearingHead(R362BearingHeadConfig())
        self.bearing_head.load_state_dict(payload["bearing_head"], strict=True)
        self.bearing_head.requires_grad_(False).eval().to(self.device)
        self._private_node_tokens: dict[Any, torch.Tensor] = {}
        self.r363_checkpoint_sha256 = actual_sha

    @property
    def node_encode_count(self) -> int:
        return self.retrieval.node_encode_count

    @property
    def query_encode_count(self) -> int:
        return self.retrieval.query_encode_count

    @staticmethod
    def _normalize(image: torch.Tensor, device: torch.device) -> torch.Tensor:
        value = image.to(device=device, dtype=torch.float32).div_(255.0)
        mean = torch.tensor((0.485, 0.456, 0.406), device=device).view(3, 1, 1)
        std = torch.tensor((0.229, 0.224, 0.225), device=device).view(3, 1, 1)
        return (value - mean) / std

    def _bearing_tokens(self, rgb: np.ndarray) -> torch.Tensor:
        image = self.retrieval.prepare_erp(rgb)
        normalized = self._normalize(image, self.device).unsqueeze(0)
        with torch.inference_mode():
            return self.bearing_backbone.forward_tokens(normalized).float()

    def encode_panorama(self, rgb: np.ndarray) -> dict[str, Any]:
        return self.retrieval.encode_panorama(rgb)

    def encode_node_package_once(self, node_id: Any, rgb: np.ndarray) -> dict[str, Any]:
        encoding = self.retrieval.encode_node_package_once(node_id, rgb)
        if node_id not in self._private_node_tokens:
            self._private_node_tokens[node_id] = self._bearing_tokens(rgb)
        merged = dict(encoding)
        merged["r363_bearing_tokens"] = self._private_node_tokens[node_id]
        return merged

    def encode_bearing_target_once(self, rgb: np.ndarray) -> torch.Tensor:
        """Encode only the private R36.3 target tokens for a goal image."""
        return self._bearing_tokens(rgb)

    def predict_bearing_to_target_tokens(
        self, current_rgb: np.ndarray, target_tokens: torch.Tensor
    ) -> dict[str, Any]:
        source_tokens = self._bearing_tokens(current_rgb)
        with torch.inference_mode():
            output = self.bearing_head(
                source_tokens,
                target_tokens.to(self.device, dtype=torch.float32),
            )
        return {
            **output,
            "bearing_degrees": output["bearing_angle_degrees"].float(),
            "checkpoint_sha256": self.r363_checkpoint_sha256,
        }

    def predict_bearing_to_encoding(
        self, current_rgb: np.ndarray, target_encoding: dict[str, Any]
    ) -> dict[str, Any]:
        if "r363_bearing_tokens" not in target_encoding:
            raise KeyError("target encoding is missing R36.3 private bearing tokens")
        source_tokens = self._bearing_tokens(current_rgb)
        target_tokens = target_encoding["r363_bearing_tokens"].to(
            self.device, dtype=torch.float32
        )
        with torch.inference_mode():
            output = self.bearing_head(source_tokens, target_tokens)
        return {
            **output,
            "bearing_degrees": output["bearing_angle_degrees"].float(),
            "checkpoint_sha256": self.r363_checkpoint_sha256,
        }

    def predict_bearing(self, current_rgb: np.ndarray, target_rgb: np.ndarray) -> dict[str, Any]:
        source_tokens = self._bearing_tokens(current_rgb)
        target_tokens = self._bearing_tokens(target_rgb)
        with torch.inference_mode():
            output = self.bearing_head(source_tokens, target_tokens)
        return {
            **output,
            "bearing_degrees": output["bearing_angle_degrees"].float(),
            "checkpoint_sha256": self.r363_checkpoint_sha256,
        }

    def predict_r361_bearing(
        self, current_rgb: np.ndarray, target_rgb: np.ndarray
    ) -> dict[str, Any]:
        target_encoding = self.retrieval.encode_panorama(target_rgb)
        return self.retrieval.predict_bearing_to_encoding(current_rgb, target_encoding)

    def retrieve(self, *args, **kwargs):
        return self.retrieval.retrieve(*args, **kwargs)

    def reset_episode_cache(self) -> None:
        self.retrieval.reset_episode_cache()
        self._private_node_tokens.clear()

    def metadata(self) -> dict[str, Any]:
        return {
            **self.retrieval.metadata(),
            "bearing_model": "R36.3 private block11/final-norm + patch bearing head",
            "bearing_checkpoint_sha256": self.r363_checkpoint_sha256,
            "retrieval_checkpoint_sha256": R361_SHA256,
            "retrieval_descriptor_modified": False,
        }
