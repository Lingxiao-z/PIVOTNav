"""Panoramic retrieval, relative bearing, and visual evidence."""
from __future__ import annotations


# ---------------------------------------------------------------------------
# Panoramic VPR retrieval
# ---------------------------------------------------------------------------

import os
import sys
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F


class StableR361RetrievalRuntime:
    """Stable retrieval-only adapter for the frozen R36.1 package."""

    def __init__(self, checkpoint: str | Path, device: str = "cuda:0", precision: str = "fp32") -> None:
        self.device = torch.device(device)
        requested = precision.lower()
        self.fallback_events: list[dict[str, str]] = []
        if requested == "bf16":
            self.fallback_events.append({
                "event": "precision_fallback",
                "requested": "bf16",
                "selected": "fp32",
                "reason": "BF16 public path has historical SIGFPE evidence",
            })
            requested = "fp32"
        if requested not in {"fp32", "fp16"}:
            raise ValueError("precision must be fp32, fp16, or bf16")
        if self.device.type == "cpu":
            os.environ["XFORMERS_DISABLED"] = "1"
            self.fallback_events.append({
                "event": "attention_fallback",
                "requested": "xFormers",
                "selected": "PyTorch safe attention",
                "reason": "xFormers attention in this package is CUDA-only",
            })
            requested = "fp32"
        from modules.Panoramic_Place_Compass.model.checkpoint import (
            load_r361_modular_package,
        )

        self.precision = requested
        self.runtime = load_r361_modular_package(checkpoint, self.device)

    def encode_panorama(self, image: torch.Tensor) -> dict[str, Any]:
        images, single = self.runtime._prepare_image(image)
        with torch.inference_mode(), torch.autocast(
            device_type=self.device.type,
            dtype=torch.float16,
            enabled=self.device.type == "cuda" and self.precision == "fp16",
        ):
            output = self.runtime._encode_normalized(images)
        descriptor = output["global_descriptor"].float()
        if not torch.isfinite(descriptor).all():
            raise RuntimeError("retrieval descriptor contains NaN/Inf")
        return {
            "tokens": output["tokens"].float(),
            "global_descriptor": descriptor,
            "descriptor_ring": output["descriptor_ring"].float(),
            "dense_ring": output["dense_ring"].float(),
            "input_was_single": single,
            "precision": self.precision,
            "device": str(self.device),
            "fallback_events": list(self.fallback_events),
        }

    def predict_bearing_from_encodings(
        self,
        source_encoding: dict[str, Any],
        target_encoding: dict[str, Any],
    ) -> dict[str, Any]:
        if "tokens" not in source_encoding or "tokens" not in target_encoding:
            raise TypeError("both encodings must be complete encode_panorama results")
        source_tokens = source_encoding["tokens"].to(self.device, dtype=torch.float32)
        target_tokens = target_encoding["tokens"].to(self.device, dtype=torch.float32)
        with torch.inference_mode():
            output = self.runtime.formal_bearing_head(source_tokens, target_tokens)
        degrees = output["bearing_angle_degrees"].float()
        confidence = output["bearing_confidence"].float()
        valid_probability = output["bearing_valid_probability"].float()
        return {
            "bearing_degrees": degrees,
            "bearing_radians": torch.deg2rad(degrees),
            "confidence": confidence,
            "uncertainty": 1.0 - confidence,
            "valid_probability": valid_probability,
            "valid": valid_probability >= 0.5,
            "angle_range": "[-180, 180)",
            "positive_direction": "OmniGuard frozen coordinate convention",
            "precision": "fp32",
            "checkpoint_sha256": self.runtime.get_model_metadata()["formal_bearing_checkpoint_sha256"],
        }

    def predict_bearing(self, current: torch.Tensor, target_encoding: dict[str, Any]) -> dict[str, Any]:
        return self.predict_bearing_from_encodings(self.encode_panorama(current), target_encoding)

    @staticmethod
    def retrieve(
        query_descriptor: torch.Tensor,
        database_descriptors: torch.Tensor,
        top_k: int,
    ) -> dict[str, torch.Tensor]:
        query = query_descriptor.float()
        database = database_descriptors.float()
        if query.ndim == 1:
            query = query.unsqueeze(0)
        if query.ndim != 2 or database.ndim != 2 or query.shape[1] != database.shape[1]:
            raise ValueError("query/database descriptor shape mismatch")
        similarity = F.normalize(query, dim=-1) @ F.normalize(database, dim=-1).T
        score, index = torch.topk(similarity, k=min(int(top_k), database.shape[0]), dim=-1)
        return {"scores": score, "indices": index}

    def metadata(self) -> dict[str, Any]:
        return {
            "adapter": "StableR361RetrievalRuntime",
            "precision": self.precision,
            "device": str(self.device),
            "fallback_events": list(self.fallback_events),
            "frozen_checkpoint_metadata": self.runtime.get_model_metadata(),
            "training_performed": False,
        }


EXPECTED_SHA256 = "4090261bcef45f70ba771533d1283a3ebe2b5e7d84c400d829cac25e01d79f4a"


class R361Adapter:
    """FP32 retrieval adapter around the preserved R36.1 inference package."""

    def __init__(
        self,
        package_root: str | Path,
        *,
        device: str = "cuda:0",
        precision: str = "fp32",
        checkpoint_path: str | Path | None = None,
    ) -> None:
        root = Path(package_root).resolve()
        python_root = root / "python"
        if str(python_root) not in sys.path:
            sys.path.insert(0, str(python_root))
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        if precision.lower() != "fp32":
            raise ValueError("R36.1 integration retrieval is frozen to FP32; BF16 is forbidden")
        self.root = root
        self.checkpoint = Path(checkpoint_path).expanduser().resolve() if checkpoint_path else root / "r361_modular.pt"
        actual_sha256 = hashlib.sha256(self.checkpoint.read_bytes()).hexdigest()
        if actual_sha256 != EXPECTED_SHA256:
            raise RuntimeError(
                f"R36.1 checkpoint SHA256 mismatch: expected {EXPECTED_SHA256}, got {actual_sha256}"
            )
        self.device = torch.device(device)
        if self.device.type == "cpu":
            os.environ.setdefault("XFORMERS_DISABLED", "1")

        self.retrieval_runtime = StableR361RetrievalRuntime(
            self.checkpoint,
            device=str(self.device),
            precision="fp32",
        )
        # Bearing remains a separate package head; it is not used for retrieval.
        self.runtime = self.retrieval_runtime.runtime
        self._node_descriptor_cache: dict[Any, torch.Tensor] = {}
        self._node_encoding_cache: dict[Any, dict[str, Any]] = {}
        self.node_encode_count = 0
        self.query_encode_count = 0
        self.query_to_node_promotion_count = 0

    @staticmethod
    def prepare_erp(rgb: np.ndarray) -> torch.Tensor:
        array = np.asarray(rgb)
        if array.ndim != 3 or array.shape[2] < 3:
            raise ValueError("ERP must be HWC RGB/RGBA")
        array = np.ascontiguousarray(array[..., :3].astype(np.uint8, copy=False))
        image = Image.fromarray(array, mode="RGB").resize((448, 224), Image.Resampling.BILINEAR)
        return torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1)

    def encode_panorama(self, rgb: np.ndarray) -> dict[str, Any]:
        return self.retrieval_runtime.encode_panorama(self.prepare_erp(rgb))

    def encode_node_once(self, node_id: Any, rgb: np.ndarray) -> torch.Tensor:
        if node_id not in self._node_descriptor_cache:
            self.encode_node_package_once(node_id, rgb)
        return self._node_descriptor_cache[node_id]

    def encode_node_package_once(self, node_id: Any, rgb: np.ndarray) -> dict[str, Any]:
        """Return the package encoding and descriptor from one node encode."""
        if node_id not in self._node_encoding_cache:
            encoding = self.encode_panorama(rgb)
            descriptor = encoding["global_descriptor"].detach()
            if descriptor.ndim == 2 and descriptor.shape[0] == 1:
                descriptor = descriptor[0]
            self._node_encoding_cache[node_id] = encoding
            self._node_descriptor_cache[node_id] = descriptor
            self.node_encode_count += 1
        return self._node_encoding_cache[node_id]

    def encode_query(self, rgb: np.ndarray) -> dict[str, Any]:
        self.query_encode_count += 1
        return self.encode_panorama(rgb)

    def promote_query_encoding_to_node(
        self,
        node_id: Any,
        query_encoding: dict[str, Any],
    ) -> torch.Tensor:
        """Cache an already encoded observation as a new regular node."""
        if node_id in self._node_encoding_cache or node_id in self._node_descriptor_cache:
            raise ValueError(f"node descriptor already cached: {node_id}")
        if "global_descriptor" not in query_encoding:
            raise KeyError("query encoding is missing global_descriptor")
        descriptor = query_encoding["global_descriptor"].detach()
        if descriptor.ndim == 2 and descriptor.shape[0] == 1:
            descriptor = descriptor[0]
        if descriptor.ndim != 1:
            raise ValueError("global descriptor must resolve to one vector")
        self._node_encoding_cache[node_id] = query_encoding
        self._node_descriptor_cache[node_id] = descriptor
        self.query_to_node_promotion_count += 1
        return descriptor

    def reset_episode_cache(self) -> None:
        self._node_descriptor_cache.clear()
        self._node_encoding_cache.clear()
        self.node_encode_count = 0
        self.query_encode_count = 0
        self.query_to_node_promotion_count = 0

    def retrieve(self, query_encoding: dict[str, Any], database_descriptors: torch.Tensor, top_k: int = 5) -> dict[str, torch.Tensor]:
        return self.retrieval_runtime.retrieve(
            query_encoding["global_descriptor"], database_descriptors, top_k
        )

    def predict_bearing(self, current_rgb: np.ndarray, target_rgb: np.ndarray) -> dict[str, Any]:
        return self.runtime.predict_bearing(self.prepare_erp(current_rgb), self.prepare_erp(target_rgb))

    def predict_bearing_to_encoding(self, current_rgb: np.ndarray, target_encoding: dict[str, Any]) -> dict[str, Any]:
        """Predict bearing while reusing a frozen goal-image encoding."""
        # Use the stable runtime's explicit FP32 direction path. Calling the
        # package runtime directly would enable its historical CUDA BF16
        # autocast and would also reject retrieval-only encodings without tokens.
        return self.retrieval_runtime.predict_bearing(self.prepare_erp(current_rgb), target_encoding)

    def get_node_encoding(self, node_id: Any) -> dict[str, Any]:
        """Return a previously cached full node encoding without re-encoding."""
        if node_id not in self._node_encoding_cache:
            raise KeyError(f"node encoding is not cached: {node_id}")
        return self._node_encoding_cache[node_id]

    def predict_bearing_from_encodings(
        self,
        source_encoding: dict[str, Any],
        target_encoding: dict[str, Any],
    ) -> dict[str, Any]:
        """Predict direction from two cached FP32 package encodings."""
        return self.retrieval_runtime.predict_bearing_from_encodings(
            source_encoding, target_encoding
        )

    def metadata(self) -> dict[str, Any]:
        return {
            **self.retrieval_runtime.metadata(),
            "checkpoint_sha256": EXPECTED_SHA256,
            "integration_device": str(self.device),
            "integration_precision": "fp32",
            "input_layout": "HWC RGB uint8 -> CHW RGB uint8",
            "erp_resize": [224, 448],
            "normalization": "package ImageNet MEAN/STD after uint8/255",
            "yaw_range": "[-180,180)",
            "retrieval_integration_approved": True,
            "arrival_navigation_package_approved": False,
            "stop_authority": False,
            "descriptor_cache_policy": "regular node encoded once; query encoded once per observation",
            "query_to_node_promotion_count": self.query_to_node_promotion_count,
            "bearing_head_role": "separate optional direction head; not a retrieval replacement",
            "bearing_precision": "fp32",
        }


# ---------------------------------------------------------------------------
# Relative bearing inference
# ---------------------------------------------------------------------------

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

        from modules.Panoramic_Place_Compass.model.relative_bearing import (
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


# ---------------------------------------------------------------------------
# Shared arrival evidence features
# ---------------------------------------------------------------------------

"""Small online evidence helpers shared by the RGB arrival state machine."""

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def pair_from_encoding(runtime: Any, query: Any, target: Any) -> dict[str, float]:
    """Build the scalar arrival evidence from two cached R36 encodings."""
    import torch
    from torch.nn import functional as F

    with torch.inference_mode():
        pair = runtime._pair_outputs(query, target)
        feature = pair["arrival_feature"].float().unsqueeze(1)
        visual = runtime.arrival_head.visual_encoder(feature)
        probability = torch.sigmoid(runtime.arrival_head.frame_logit(visual).squeeze(-1))[:, 0]
        similarity = F.cosine_similarity(
            query["global_descriptor"].float(),
            target["global_descriptor"].float(),
            dim=-1,
        )
        yaw = pair["yaw"]
    return {
        "arrival_probability": float(probability[0]),
        "vpr_similarity": float(similarity[0]),
        "yaw_degrees": float(yaw["predicted_yaw_degrees"][0]),
        "yaw_confidence": float(yaw["yaw_confidence"][0]),
    }


def wrap_degrees(value: float) -> float:
    return (float(value) + 180.0) % 360.0 - 180.0


def feature_names() -> list[str]:
    names = []
    for view in ("restore", "approach1", "approach2"):
        for field in (
            "raw_matches", "essential_inliers", "essential_ratio",
            "homography_inlier_ratio", "homography_scale", "flow_mean",
            "flow_median", "flow_p25", "flow_p75", "flow_p90",
            "flow_x_median", "flow_y_median",
        ):
            names.append(f"{view}_{field}")
    names.extend([
        "flow_decrease_01", "flow_decrease_12", "distance_estimate_1",
        "distance_estimate_2", "distance_estimate_median",
        "monotonic_flow_decrease_count", "current_arrival_probability",
        "current_vpr_similarity", "current_e_inliers", "current_e_inlier_ratio",
        "current_grid_coverage", "current_hull_area", "current_reprojection_error",
        "current_supported_sectors", "current_h_dominance", "target_top1_fraction",
        "target_top1_current",
    ])
    return names


def make_features(parallax: dict, row: dict) -> np.ndarray:
    common_sector = int(parallax["causal"]["common_sector_id"])
    values = []
    for view in parallax["views"]:
        sector = view["sectors"][common_sector]
        values.extend([
            sector["raw_matches"], sector["essential_inliers"], sector["essential_ratio"],
            sector["homography_inlier_ratio"], sector["homography_scale"],
            sector["flow"]["mean"], sector["flow"]["median"], sector["flow"]["p25"],
            sector["flow"]["p75"], sector["flow"]["p90"], sector["flow_x"]["median"],
            sector["flow_y"]["median"],
        ])
    causal = parallax["causal"]
    estimates = [min(float(value), 10.0) for value in causal["remaining_distance_estimate_sequence_m"]]
    evidence = row["views"][-1]["evidence"]
    top1 = [bool(view["target_top1"]) for view in row["views"]]
    values.extend([
        *causal["flow_decrease_sequence"], *estimates,
        min(float(causal["remaining_distance_estimate_median_m"]), 10.0),
        causal["monotonic_flow_decrease_count"], evidence["arrival_probability"],
        evidence["vpr_similarity"], evidence["e_inliers"], evidence["e_inlier_ratio"],
        evidence["grid_coverage"], evidence["hull_area"], evidence["reprojection_error"],
        evidence["supported_sectors"], evidence["h_dominance"], np.mean(top1), float(top1[-1]),
    ])
    result = np.asarray(values, dtype=np.float32)
    if len(result) != len(feature_names()) or not np.isfinite(result).all():
        raise ValueError(f"invalid parallax feature vector for {row['trial_id']}")
    return result


def hard_safety(row: dict, config: dict) -> tuple[bool, list[str]]:
    by_label = {view["view_label"]: view for view in row["views"]}
    required = ["ORIGINAL", "LEFT_10", "RIGHT_20", "RESTORE_10"]
    reasons = []
    if [view["view_label"] for view in row["views"]][:4] != required:
        reasons.append("missing_rotation_cycle")
    else:
        original = float(by_label["ORIGINAL"]["r361"]["yaw_degrees"])
        errors = [
            abs(wrap_degrees(float(by_label["LEFT_10"]["r361"]["yaw_degrees"]) - original + 10.0)),
            abs(wrap_degrees(float(by_label["RIGHT_20"]["r361"]["yaw_degrees"]) - original - 10.0)),
            abs(wrap_degrees(float(by_label["RESTORE_10"]["r361"]["yaw_degrees"]) - original)),
        ]
        if max(errors) > config["maximum_yaw_cycle_error_degrees"]:
            reasons.append("yaw_cycle_inconsistent")
    evidence = row["views"][-1]["evidence"]
    if int(evidence["e_inliers"]) < config["minimum_current_e_inliers"]:
        reasons.append("insufficient_current_inliers")
    if int(evidence["supported_sectors"]) < config["minimum_current_supported_sectors"]:
        reasons.append("insufficient_current_sector_support")
    return not reasons, reasons


def _view_rgb(view: dict) -> np.ndarray:
    image_rgb = view.get("image_rgb")
    if image_rgb is not None:
        array = np.asarray(image_rgb)
    else:
        image_path = view.get("image_path")
        if not image_path or not Path(image_path).is_file():
            raise RuntimeError("MISSING_RGB_EVIDENCE")
        array = np.asarray(Image.open(image_path).convert("RGB"))
    if array.ndim != 3 or array.shape[-1] < 3:
        raise RuntimeError("INVALID_RGB_EVIDENCE_SHAPE")
    return np.asarray(array[..., :3], dtype=np.float32)


def rgb_change(first: dict, second: dict) -> float:
    first_rgb = _view_rgb(first)
    second_rgb = _view_rgb(second)
    first_aligned = np.roll(
        first_rgb,
        int(round(-float(first["r361"]["yaw_degrees"]) / 360.0 * first_rgb.shape[1])),
        axis=1,
    )
    second_aligned = np.roll(
        second_rgb,
        int(round(-float(second["r361"]["yaw_degrees"]) / 360.0 * second_rgb.shape[1])),
        axis=1,
    )
    return float(np.mean(np.abs(first_aligned - second_aligned)))
