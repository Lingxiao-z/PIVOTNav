"""Load the released Panoramic Place Compass checkpoint."""
from __future__ import annotations
import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .arrival import R35MultitaskSystem, R36TemporalArrivalHead
from .bearing import R35RelativeTranslationBearingHead, R36BearingStructuralRevision5Head
from .core import PanoramicVPRV2System
from .orientation import R34CrossPositionYawHead, R34CrossPositionYawHeadR3

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)




CHECKPOINT_SCHEMA = "r361_modular_checkpoint_v1"
MODEL_VERSION = "R36.1 Modular VPR + Bearing + Arrival Package"
REQUIRED_IMAGE_SHAPE = (3, 224, 448)


def sha256_path(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def circular_error_degrees(prediction: float, target: float) -> float:
    return abs((float(prediction) - float(target) + 180.0) % 360.0 - 180.0)


def target_erp_roll_bearing_transform(bearing_degrees: float, roll_degrees: float) -> float:
    """Target ERP yaw changes appearance, not the source-frame translation direction definition."""
    _ = roll_degrees
    return (float(bearing_degrees) + 180.0) % 360.0 - 180.0


@dataclass
class ArrivalTemporalState:
    hidden: torch.Tensor | None = None
    previous_model_temporal_probability: torch.Tensor | None = None
    independent_evidence_ids: tuple[str, ...] = ()
    last_single_frame_probability: torch.Tensor | None = None
    last_model_temporal_probability: torch.Tensor | None = None
    last_temporal_probability: torch.Tensor | None = None
    last_candidate: torch.Tensor | None = None
    last_confirmed: torch.Tensor | None = None


class R361ModularRuntime(nn.Module):
    def __init__(
        self,
        encoder_system: PanoramicVPRV2System,
        track_y: R34CrossPositionYawHeadR3,
        formal_bearing_head: R36BearingStructuralRevision5Head,
        arrival_feature_bearing_head: R35RelativeTranslationBearingHead,
        arrival_head: R36TemporalArrivalHead,
        metadata: dict[str, Any],
    ) -> None:
        super().__init__()
        self.encoder_system = encoder_system
        self.track_y = track_y
        self.formal_bearing_head = formal_bearing_head
        self.arrival_feature_bearing_head = arrival_feature_bearing_head
        self.arrival_head = arrival_head
        self._metadata = metadata

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def _prepare_image(self, image: torch.Tensor) -> tuple[torch.Tensor, bool]:
        if image.ndim == 3:
            image = image.unsqueeze(0)
            single = True
        elif image.ndim == 4:
            single = False
        else:
            raise ValueError("ERP图像必须是[3,224,448]或[B,3,224,448]")
        if tuple(image.shape[-3:]) != REQUIRED_IMAGE_SHAPE:
            raise ValueError(f"ERP图像尺寸必须是{REQUIRED_IMAGE_SHAPE}")
        if image.dtype == torch.uint8:
            value = image.to(self.device, dtype=torch.float32).div_(255.0)
            value = (value - MEAN.to(self.device)) / STD.to(self.device)
        elif image.is_floating_point():
            value = image.to(self.device, dtype=torch.float32)
            if not torch.isfinite(value).all():
                raise ValueError("ERP图像包含NaN/Inf")
            minimum, maximum = float(value.min()), float(value.max())
            if minimum >= 0.0 and maximum <= 1.0:
                value = (value - MEAN.to(self.device)) / STD.to(self.device)
        else:
            raise TypeError("ERP图像只接受uint8或浮点Tensor")
        return value, single

    def _encode_normalized(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        tokens = self.encoder_system.backbone.forward_tokens(images)
        descriptor = self.encoder_system.descriptor_head.forward_tokens(tokens.float())
        base_head = getattr(self.track_y, "base_head", None)
        ring_encoder = getattr(base_head, "ring_encoder", None)
        if not callable(ring_encoder):
            raise RuntimeError("冻结Track Y缺少ring_encoder")
        dense_ring = ring_encoder(tokens)
        return {
            "tokens": tokens,
            "global_descriptor": descriptor["global"],
            "descriptor_ring": descriptor["ring"],
            "dense_ring": dense_ring,
        }

    def encode_panorama(self, image: torch.Tensor) -> dict[str, Any]:
        images, single = self._prepare_image(image)
        with torch.inference_mode(), torch.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16,
            enabled=self.device.type == "cuda",
        ):
            encoded = self._encode_normalized(images)
        return {
            **encoded,
            "input_was_single": single,
            "model_version": MODEL_VERSION,
            "checkpoint_sha256": self._metadata["checkpoint_sha256"],
        }

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
        if database.ndim != 2 or query.ndim != 2 or query.shape[1] != database.shape[1]:
            raise ValueError("query必须是[D]/[B,D]，database必须是[N,D]")
        if not 1 <= int(top_k) <= database.shape[0]:
            raise ValueError("top_k超出database范围")
        similarity = F.normalize(query, dim=-1) @ F.normalize(database, dim=-1).T
        score, index = torch.topk(similarity, k=int(top_k), dim=-1)
        return {"indices": index, "scores": score}

    def _ensure_encoding(self, value: torch.Tensor | dict[str, Any]) -> dict[str, Any]:
        if torch.is_tensor(value):
            return self.encode_panorama(value)
        required = {"tokens", "global_descriptor", "descriptor_ring", "dense_ring"}
        if not isinstance(value, dict) or not required.issubset(value):
            raise TypeError("输入必须是ERP Tensor或encode_panorama返回值")
        return value

    def _track_y(self, source: dict[str, Any], target: dict[str, Any]) -> dict[str, torch.Tensor]:
        forward_rings = getattr(self.track_y, "forward_rings", None)
        if not callable(forward_rings):
            raise RuntimeError("冻结Track Y缺少forward_rings")
        return forward_rings(
            source["dense_ring"],
            target["dense_ring"],
            source["descriptor_ring"],
            target["descriptor_ring"],
        )

    @staticmethod
    def _correlation_summary(scores: torch.Tensor) -> torch.Tensor:
        probability = torch.softmax(scores.float(), dim=-1)
        top2 = torch.topk(probability, k=2, dim=-1).values
        entropy = -(probability * probability.clamp_min(1e-8).log()).sum(dim=-1)
        entropy = entropy / math.log(float(scores.shape[-1]))
        return torch.stack(
            (scores.float().max(-1).values, scores.float().mean(-1), top2[:, 0] - top2[:, 1], entropy),
            dim=-1,
        )

    @staticmethod
    def _summarize_arrival_feature(
        spatial: torch.Tensor, scalar: torch.Tensor
    ) -> torch.Tensor:
        spatial = spatial.float()
        quantiles = torch.quantile(
            spatial,
            torch.tensor([0.25, 0.5, 0.75], device=spatial.device),
            dim=-1,
        ).T
        summary = torch.stack(
            (spatial.mean(-1), spatial.std(-1), spatial.min(-1).values, spatial.max(-1).values),
            dim=-1,
        )
        feature = torch.cat((spatial, scalar.float(), summary, quantiles), dim=-1)
        if feature.shape[-1] != 280:
            raise RuntimeError(f"Arrival冻结特征维度错误: {feature.shape}")
        return feature

    def _pair_outputs(
        self,
        current: torch.Tensor | dict[str, Any],
        target: torch.Tensor | dict[str, Any],
    ) -> dict[str, Any]:
        source = self._ensure_encoding(current)
        goal = self._ensure_encoding(target)
        if source["tokens"].shape[0] != goal["tokens"].shape[0]:
            raise ValueError("current和target batch必须一致")
        with torch.inference_mode(), torch.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16,
            enabled=self.device.type == "cuda",
        ):
            yaw = self._track_y(source, goal)
            private = self.arrival_feature_bearing_head(source["tokens"], goal["tokens"])
            global_similarity = F.cosine_similarity(
                source["global_descriptor"].float(), goal["global_descriptor"].float(), dim=-1
            )
            sector_scores = yaw.get("descriptor_correlation_32")
            if sector_scores is None:
                sector_scores = yaw["scale_scores"].float().mean(dim=1)
            scalar = torch.cat(
                (
                    global_similarity.unsqueeze(-1),
                    self._correlation_summary(sector_scores),
                    torch.stack(
                        (
                            yaw["top1_probability"].float(),
                            yaw["yaw_confidence"].float(),
                            yaw["normalized_entropy"].float(),
                        ),
                        dim=-1,
                    ),
                    torch.stack(
                        (
                            private["bearing_confidence"].float(),
                            private["bearing_valid_probability"].float(),
                            private["normalized_entropy"].float(),
                        ),
                        dim=-1,
                    ),
                    private["local_correlation_summary"].float(),
                    torch.zeros(global_similarity.shape[0], 1, device=self.device),
                    torch.ones(global_similarity.shape[0], 1, device=self.device),
                ),
                dim=-1,
            )
            spatial = torch.cat(
                (yaw["pair_embedding"].float(), private["pair_embedding"].float()), dim=-1
            )
            arrival_feature = self._summarize_arrival_feature(spatial, scalar)
        return {
            "source_encoding": source,
            "target_encoding": goal,
            "yaw": yaw,
            "arrival_private_bearing": private,
            "arrival_feature": arrival_feature,
        }

    def predict_bearing(
        self,
        current_erp: torch.Tensor | dict[str, Any],
        target_erp: torch.Tensor | dict[str, Any],
    ) -> dict[str, Any]:
        source = self._ensure_encoding(current_erp)
        goal = self._ensure_encoding(target_erp)
        if source["tokens"].shape[0] != goal["tokens"].shape[0]:
            raise ValueError("current和target batch必须一致")
        with torch.inference_mode(), torch.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16,
            enabled=self.device.type == "cuda",
        ):
            output = self.formal_bearing_head(source["tokens"], goal["tokens"])
        degrees = output["bearing_angle_degrees"].float()
        radians = torch.deg2rad(degrees)
        confidence = output["bearing_confidence"].float()
        valid_probability = output["bearing_valid_probability"].float()
        return {
            "bearing_degrees": degrees,
            "bearing_radians": radians,
            "confidence": confidence,
            "uncertainty": 1.0 - confidence,
            "valid_probability": valid_probability,
            "valid": valid_probability >= 0.5,
            "angle_range": "[-180, 180)",
            "positive_direction": "OmniGuard冻结坐标约定",
            "model_version": MODEL_VERSION,
            "checkpoint_sha256": self._metadata["formal_bearing_checkpoint_sha256"],
        }

    def reset_arrival_state(self) -> ArrivalTemporalState:
        return ArrivalTemporalState()

    def predict_arrival(
        self,
        current_erp: torch.Tensor | dict[str, Any],
        target_erp: torch.Tensor | dict[str, Any],
        evidence_id: str,
        temporal_state: ArrivalTemporalState | None = None,
    ) -> dict[str, Any]:
        if not isinstance(evidence_id, str) or not evidence_id:
            raise ValueError("evidence_id必须是非空字符串")
        state = temporal_state or self.reset_arrival_state()
        duplicate = evidence_id in state.independent_evidence_ids
        if duplicate:
            return {
                "single_frame_probability": state.last_single_frame_probability,
                "model_temporal_probability": state.last_model_temporal_probability,
                "temporal_probability": state.last_temporal_probability,
                "evidence_count": len(state.independent_evidence_ids),
                "independent_evidence_ids": list(state.independent_evidence_ids),
                "duplicate_evidence_ignored": True,
                "candidate": state.last_candidate,
                "confirmed": state.last_confirmed,
                "first_frame_stop_forbidden": True,
                "temporal_state": state,
                "model_version": MODEL_VERSION,
                "checkpoint_sha256": self._metadata["arrival_head_checkpoint_sha256"],
            }
        pair = self._pair_outputs(current_erp, target_erp)
        feature = pair["arrival_feature"].float().unsqueeze(1)
        with torch.inference_mode():
            visual = self.arrival_head.visual_encoder(feature)
            frame_logit = self.arrival_head.frame_logit(visual).squeeze(-1)
            frame_probability = torch.sigmoid(frame_logit)[:, 0]
            temporal, hidden = self.arrival_head.temporal_encoder(
                torch.cat((visual, frame_probability[:, None, None]), dim=-1),
                state.hidden,
            )
            model_temporal_probability = torch.sigmoid(
                self.arrival_head.temporal_logit(temporal).squeeze(-1)
            )[:, 0]
        if state.previous_model_temporal_probability is None:
            fused = model_temporal_probability
        else:
            fused = 1.0 - (1.0 - state.previous_model_temporal_probability) * (
                1.0 - model_temporal_probability
            )
        evidence_ids = state.independent_evidence_ids + (evidence_id,)
        candidate = model_temporal_probability >= 0.5
        confirmed = (len(evidence_ids) >= 2) & (fused >= 0.75)
        next_state = ArrivalTemporalState(
            hidden=hidden,
            previous_model_temporal_probability=model_temporal_probability,
            independent_evidence_ids=evidence_ids,
            last_single_frame_probability=frame_probability,
            last_model_temporal_probability=model_temporal_probability,
            last_temporal_probability=fused,
            last_candidate=candidate,
            last_confirmed=confirmed,
        )
        return {
            "single_frame_probability": frame_probability,
            "model_temporal_probability": model_temporal_probability,
            "temporal_probability": fused,
            "evidence_count": len(evidence_ids),
            "independent_evidence_ids": list(evidence_ids),
            "duplicate_evidence_ignored": False,
            "candidate": candidate,
            "confirmed": confirmed,
            "first_frame_stop_forbidden": True,
            "temporal_state": next_state,
            "model_version": MODEL_VERSION,
            "checkpoint_sha256": self._metadata["arrival_head_checkpoint_sha256"],
        }

    def get_model_metadata(self) -> dict[str, Any]:
        return dict(self._metadata)


def load_r361_modular_package(
    checkpoint: str | Path,
    device: str | torch.device = "cpu",
) -> R361ModularRuntime:
    path = Path(checkpoint).resolve()
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema_version") != CHECKPOINT_SCHEMA:
        raise RuntimeError("R36.1模块化checkpoint schema不匹配")
    provenance = payload.get("provenance")
    if not isinstance(provenance, dict) or provenance.get("core_internal_gates_passed") is not True:
        raise RuntimeError("R36.1模块化checkpoint未证明核心Internal门槛")
    if provenance.get("test_r32_historical_confirmation_accessed") is not False:
        raise RuntimeError("R36.1模块化checkpoint数据边界非法")
    encoder = PanoramicVPRV2System()
    encoder.load_state_dict(payload["encoder_system"], strict=True)
    track_y = R34CrossPositionYawHeadR3(R34CrossPositionYawHead())
    track_y.load_state_dict(payload["track_y"], strict=True)
    formal_bearing = R36BearingStructuralRevision5Head()
    formal_bearing.load_state_dict(payload["formal_bearing_head"], strict=True)
    private_bearing = R35RelativeTranslationBearingHead()
    private_bearing.load_state_dict(payload["arrival_feature_bearing_head"], strict=True)
    arrival_head = R36TemporalArrivalHead()
    arrival_head.load_state_dict(payload["arrival_head"], strict=True)
    metadata = {
        **payload["metadata"],
        "checkpoint": str(path),
        "checkpoint_sha256": sha256_path(path),
        "schema_version": CHECKPOINT_SCHEMA,
        "model_version": MODEL_VERSION,
    }
    runtime = R361ModularRuntime(
        encoder, track_y, formal_bearing, private_bearing, arrival_head, metadata
    )
    if not all(torch.isfinite(value.float()).all() for value in runtime.state_dict().values()):
        raise RuntimeError("R36.1模块化checkpoint包含NaN/Inf")
    runtime.requires_grad_(False)
    runtime.eval().to(device)
    return runtime
