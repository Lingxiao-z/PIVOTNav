from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict

import torch
from torch import nn

from .r34_open_set_head import (
    R34OpenSetThresholds,
    R34SetOpenSetHead,
    r34_open_set_decision,
)
from .r34_yaw_head import R34CrossPositionYawHead
from .r34_yaw_head_r3 import R34CrossPositionYawHeadR3
from .system import PanoramicVPRV2System


class R34YCInferenceSystem(nn.Module):
    """Integrated RGB-only retrieval, relative-yaw and open-set Goal Anchor inference."""

    def __init__(
        self,
        encoder_system: PanoramicVPRV2System,
        yaw_head: nn.Module,
        open_set_head: R34SetOpenSetHead,
        thresholds: R34OpenSetThresholds,
    ) -> None:
        super().__init__()
        self.encoder_system = encoder_system
        self.yaw_head = yaw_head
        self.open_set_head = open_set_head
        self.thresholds = thresholds

    @property
    def architecture_record(self) -> Dict[str, Any]:
        return {
            "schema_version": "r34_yc_integrated_inference_v1",
            "encoder": "DINOv2-S/14 ERP backbone with SALAD 2112-dimensional global descriptor",
            "yaw_head": self.yaw_head.architecture_record,
            "open_set_head": self.open_set_head.architecture_record,
            "frozen_thresholds": asdict(self.thresholds),
            "online_inputs": ["RGB ERP"],
            "outputs": [
                "retrieval Top-K",
                "continuous relative yaw and confidence per candidate",
                "Goal Anchor candidate or UNKNOWN",
            ],
        }

    @torch.inference_mode()
    def encode(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        tokens = self.encoder_system.backbone.forward_tokens(images)
        descriptor = self.encoder_system.descriptor_head.forward_tokens(tokens.float())
        if isinstance(self.yaw_head, R34CrossPositionYawHeadR3):
            return {
                "global": descriptor["global"],
                "dense_rings": self.yaw_head.base_head.ring_encoder(tokens),
                "descriptor_rings": descriptor["ring"],
            }
        return {
            "global": descriptor["global"],
            "rings": self.yaw_head.ring_encoder(tokens),
        }

    def effective_thresholds(self, threshold_offset: float | None = None) -> R34OpenSetThresholds:
        offset = self.thresholds.threshold_offset if threshold_offset is None else float(threshold_offset)
        return R34OpenSetThresholds(
            unknown_threshold=self.thresholds.unknown_threshold,
            anchor_threshold=self.thresholds.anchor_threshold,
            anchor_margin=self.thresholds.anchor_margin,
            threshold_offset=offset,
        )

    @torch.inference_mode()
    def match_encoded(
        self,
        query: Dict[str, torch.Tensor],
        gallery: Dict[str, torch.Tensor],
        *,
        top_k: int = 8,
        threshold_offset: float | None = None,
    ) -> Dict[str, torch.Tensor]:
        y_r3 = isinstance(self.yaw_head, R34CrossPositionYawHeadR3)
        expected_encoding = (
            {"global", "dense_rings", "descriptor_rings"}
            if y_r3
            else {"global", "rings"}
        )
        for name, encoding in (("query", query), ("gallery", gallery)):
            if set(encoding) != expected_encoding:
                raise ValueError(
                    f"{name} encoding must contain exactly {sorted(expected_encoding)}"
                )
        retrieval_scores, candidate_indices = self.encoder_system.retrieve_topk(
            query["global"], gallery["global"], top_k
        )
        batch, candidate_count = candidate_indices.shape
        if y_r3:
            query_dense = query["dense_rings"].unsqueeze(1).expand(
                -1, candidate_count, -1, -1, -1
            )
            candidate_dense = gallery["dense_rings"][candidate_indices]
            query_descriptor = query["descriptor_rings"].unsqueeze(1).expand(
                -1, candidate_count, -1, -1
            )
            candidate_descriptor = gallery["descriptor_rings"][candidate_indices]
            yaw = self.yaw_head.forward_rings(
                query_dense.reshape(-1, *query_dense.shape[2:]),
                candidate_dense.reshape(-1, *candidate_dense.shape[2:]),
                query_descriptor.reshape(-1, *query_descriptor.shape[2:]),
                candidate_descriptor.reshape(-1, *candidate_descriptor.shape[2:]),
            )
        else:
            query_rings = query["rings"].unsqueeze(1).expand(-1, candidate_count, -1, -1, -1)
            candidate_rings = gallery["rings"][candidate_indices]
            yaw = self.yaw_head.forward_rings(
                query_rings.reshape(-1, *query_rings.shape[2:]),
                candidate_rings.reshape(-1, *candidate_rings.shape[2:]),
            )
        pair_embeddings = yaw["pair_embedding"].reshape(batch, candidate_count, -1)
        open_set = self.open_set_head(pair_embeddings)
        decision = r34_open_set_decision(open_set, self.effective_thresholds(threshold_offset))
        selected_candidate = decision["selected_candidate_index"]
        safe_selected = selected_candidate.clamp_min(0).unsqueeze(1)
        selected_gallery = candidate_indices.gather(1, safe_selected).squeeze(1)
        selected_gallery = torch.where(
            selected_candidate >= 0,
            selected_gallery,
            torch.full_like(selected_gallery, -1),
        )
        return {
            "candidate_indices": candidate_indices,
            "retrieval_scores": retrieval_scores,
            "relative_yaw_degrees": yaw["predicted_yaw_degrees"].reshape(batch, candidate_count),
            "yaw_confidence": yaw["yaw_confidence"].reshape(batch, candidate_count),
            "node_probabilities": open_set["node_probabilities"],
            "unknown_probability": open_set["unknown_probability"],
            "best_candidate_probability": open_set["best_candidate_probability"],
            "candidate_probability_margin": open_set["candidate_probability_margin"],
            "is_unknown": decision["is_unknown"],
            "selected_candidate_index": selected_candidate,
            "selected_gallery_index": selected_gallery,
            "effective_unknown_threshold": decision["effective_unknown_threshold"],
        }

    def forward(
        self,
        query_images: torch.Tensor,
        gallery_encoding: Dict[str, torch.Tensor],
        *,
        top_k: int = 8,
        threshold_offset: float | None = None,
    ) -> Dict[str, torch.Tensor]:
        return self.match_encoded(
            self.encode(query_images),
            gallery_encoding,
            top_k=top_k,
            threshold_offset=threshold_offset,
        )


def load_r34_yc_checkpoint(
    checkpoint: str | Path,
    device: str | torch.device = "cpu",
) -> tuple[R34YCInferenceSystem, dict[str, Any]]:
    payload = torch.load(Path(checkpoint), map_location="cpu", weights_only=False)
    schema = payload.get("schema_version")
    if schema not in {"r34_yc_integrated_checkpoint_v1", "r34_yc_integrated_checkpoint_v2"}:
        raise RuntimeError("checkpoint is not an R34-YC integrated checkpoint")
    target = torch.device(device)
    encoder = PanoramicVPRV2System().to(target)
    encoder.load_state_dict(payload["system"], strict=True)
    revision = str(payload.get("track_y_revision", "Y-r2"))
    if schema == "r34_yc_integrated_checkpoint_v2":
        if revision != "Y-r3" or "yaw_head_r3" not in payload:
            raise RuntimeError("R34-YC v2 checkpoint must contain a Y-r3 yaw head")
        yaw_head = R34CrossPositionYawHeadR3(R34CrossPositionYawHead()).to(target)
        yaw_head.load_state_dict(payload["yaw_head_r3"], strict=True)
    else:
        yaw_head = R34CrossPositionYawHead().to(target)
        yaw_head.load_state_dict(payload["yaw_head"], strict=True)
    open_set_head = R34SetOpenSetHead().to(target)
    open_set_head.load_state_dict(payload["open_set_head"], strict=True)
    thresholds = R34OpenSetThresholds(**payload["thresholds"])
    model = R34YCInferenceSystem(encoder, yaw_head, open_set_head, thresholds).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    if not all(torch.isfinite(value).all() for value in model.state_dict().values()):
        raise RuntimeError("R34-YC integrated checkpoint contains non-finite weights")
    return model, payload
