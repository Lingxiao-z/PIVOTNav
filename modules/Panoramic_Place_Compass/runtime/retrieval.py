from __future__ import annotations

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
        from modules.Panoramic_Place_Compass.models_vpr.checkpoint import (
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
