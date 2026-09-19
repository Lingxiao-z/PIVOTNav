from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
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
                "reason": "BF16 public path has historical SIGFPE evidence and is not an algorithm comparison path",
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
        from modules.Panoramic_Place_Compass.pano_vpr_v2.r361_modular_inference import (
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
            # Keep the complete FP32 package encoding available for the
            # optional direction head. Retrieval still consumes only the
            # cached global descriptor and never uses the direction output.
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
        """Run the optional direction head without re-encoding either ERP."""
        if not isinstance(source_encoding, dict) or "tokens" not in source_encoding:
            raise TypeError("source_encoding must be a full FP32 encode_panorama result")
        if not isinstance(target_encoding, dict) or "tokens" not in target_encoding:
            raise TypeError("target_encoding must be a full FP32 encode_panorama result")
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
            "positive_direction": "OmniGuard冻结坐标约定",
            "precision": "fp32",
            "checkpoint_sha256": self.runtime.get_model_metadata()["formal_bearing_checkpoint_sha256"],
        }

    def predict_bearing(self, current: torch.Tensor, target_encoding: dict[str, Any]) -> dict[str, Any]:
        """Run the optional R36.1 bearing head with an explicit FP32 path."""
        return self.predict_bearing_from_encodings(
            self.encode_panorama(current), target_encoding
        )

    @staticmethod
    def retrieve(query_descriptor: torch.Tensor, database_descriptors: torch.Tensor, top_k: int) -> dict[str, torch.Tensor]:
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
