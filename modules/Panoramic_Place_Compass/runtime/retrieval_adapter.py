from __future__ import annotations

import os
import sys
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image


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

        from modules.Panoramic_Place_Compass.runtime.stable_retrieval_runtime import StableR361RetrievalRuntime

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
