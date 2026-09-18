from __future__ import annotations

import math
from enum import Enum
from typing import Any, Dict

import torch
from torch import nn
from torch.nn import functional as F

from .r35_bearing_head import R35RelativeTranslationBearingHead
from .r35_open_set_heads import R35CandidateMatchHead, R35SetPresenceHead
from .r35_stage2_revision4 import R35RawPairCandidateVerifier
from .system import PanoramicVPRV2System


class R35TrainingStage(str, Enum):
    BEARING_HEAD = "stage_1_bearing_head"
    OPEN_SET_HEADS = "stage_2_candidate_and_presence_heads"
    LIMITED_JOINT = "stage_3_limited_joint_finetuning"


class R35MultitaskSystem(nn.Module):
    """Shared VPR encoder with frozen Track Y and independent R35 task heads."""

    def __init__(
        self,
        encoder_system: PanoramicVPRV2System,
        frozen_track_y: nn.Module,
        bearing_head: R35RelativeTranslationBearingHead | None = None,
        candidate_match_head: R35CandidateMatchHead | None = None,
        set_presence_head: R35SetPresenceHead | None = None,
        raw_pair_candidate_verifier: R35RawPairCandidateVerifier | None = None,
    ) -> None:
        super().__init__()
        self.encoder_system = encoder_system
        self.track_y = frozen_track_y
        self.bearing_head = bearing_head or R35RelativeTranslationBearingHead()
        self.candidate_match_head = candidate_match_head or R35CandidateMatchHead()
        self.set_presence_head = set_presence_head or R35SetPresenceHead()
        self.raw_pair_candidate_verifier = raw_pair_candidate_verifier
        self._training_stage: R35TrainingStage | None = None
        self._stage3_unfreeze_blocks = 0
        for parameter in self.track_y.parameters():
            parameter.requires_grad_(False)

    @property
    def architecture_record(self) -> Dict[str, Any]:
        return {
            "schema_version": "r35_shared_backbone_multitask_system_v1",
            "shared_encoder": "Track-Y-r3-adapted DINOv2-S/14 plus SALAD descriptor head",
            "heads": {
                "vpr_global_descriptor": "existing frozen-compatible 2112-dimensional SALAD output",
                "relative_yaw": "frozen Track Y-r3",
                "relative_translation_bearing": self.bearing_head.architecture_record,
                "candidate_match": self.candidate_match_head.architecture_record,
                "set_presence_unknown": self.set_presence_head.architecture_record,
                "raw_pair_candidate_verifier": (
                    self.raw_pair_candidate_verifier.__class__.__name__
                    if self.raw_pair_candidate_verifier is not None
                    else None
                ),
            },
            "training_stages": {
                R35TrainingStage.BEARING_HEAD.value: (
                    "shared encoder and Track Y frozen; bearing prediction and calibration layers only; "
                    "the Track-C-only pair embedding adapter remains frozen until limited joint tuning"
                ),
                R35TrainingStage.OPEN_SET_HEADS.value: "shared encoder, Track Y and bearing head frozen; Candidate Match and Set Presence only",
                R35TrainingStage.LIMITED_JOINT.value: "bearing/open-set heads plus only the final one or two DINO blocks; VPR distillation required externally",
            },
            "single_shared_backbone": True,
            "test_time_gt_inputs": False,
        }

    @staticmethod
    def _set_trainable(module: nn.Module, trainable: bool) -> None:
        for parameter in module.parameters():
            parameter.requires_grad_(trainable)

    def _freeze_obsolete_presence_heads(self) -> None:
        evidence_encoder = getattr(self.set_presence_head, "evidence_encoder", None)
        if evidence_encoder is None:
            return
        for name in ("presence_head", "near_wrong_head", "confidence_head"):
            obsolete = getattr(evidence_encoder, name, None)
            if obsolete is not None:
                self._set_trainable(obsolete, False)

    def configure_training_stage(
        self,
        stage: R35TrainingStage | str,
        *,
        stage3_unfreeze_blocks: int = 2,
    ) -> Dict[str, Any]:
        selected = R35TrainingStage(stage)
        if selected is R35TrainingStage.LIMITED_JOINT and stage3_unfreeze_blocks not in (1, 2):
            raise ValueError("Stage 3 may unfreeze only the final one or two DINO blocks")
        self._training_stage = selected
        self._stage3_unfreeze_blocks = 0
        self._set_trainable(self.encoder_system, False)
        self._set_trainable(self.track_y, False)
        self._set_trainable(self.bearing_head, False)
        self._set_trainable(self.candidate_match_head, False)
        self._set_trainable(self.set_presence_head, False)
        if self.raw_pair_candidate_verifier is not None:
            self._set_trainable(self.raw_pair_candidate_verifier, False)
        if selected is R35TrainingStage.BEARING_HEAD:
            self._set_trainable(self.bearing_head, True)
            # pair_embedding is not consumed by any Stage-1 bearing loss. Keeping
            # it trainable would silently apply only AdamW decay and violates the
            # declared DDP trainable scope.
            self._set_trainable(self.bearing_head.pair_embedding, False)
        elif selected is R35TrainingStage.OPEN_SET_HEADS:
            self._set_trainable(self.candidate_match_head, True)
            self._set_trainable(self.set_presence_head, True)
            if self.raw_pair_candidate_verifier is not None:
                self._set_trainable(self.raw_pair_candidate_verifier, True)
        else:
            self.encoder_system.backbone.unfreeze_last_blocks(stage3_unfreeze_blocks)
            self._stage3_unfreeze_blocks = stage3_unfreeze_blocks
            self._set_trainable(self.bearing_head, True)
            self._set_trainable(self.candidate_match_head, True)
            self._set_trainable(self.set_presence_head, True)
            if self.raw_pair_candidate_verifier is not None:
                self._set_trainable(self.raw_pair_candidate_verifier, True)
        self._freeze_obsolete_presence_heads()
        audit = self.trainable_scope_audit()
        if audit["track_y_trainable_parameters"] != 0:
            raise RuntimeError("Track Y must remain frozen in every R35 stage")
        if selected is not R35TrainingStage.LIMITED_JOINT and audit["encoder_trainable_parameters"] != 0:
            raise RuntimeError("shared encoder must be frozen in R35 Stage 1/2")
        return audit

    def trainable_scope_audit(self) -> Dict[str, Any]:
        count = lambda module: sum(
            parameter.numel() for parameter in module.parameters() if parameter.requires_grad
        )
        return {
            "training_stage": self._training_stage.value if self._training_stage else None,
            "stage3_unfreeze_blocks": self._stage3_unfreeze_blocks,
            "encoder_trainable_parameters": count(self.encoder_system),
            "track_y_trainable_parameters": count(self.track_y),
            "bearing_head_trainable_parameters": count(self.bearing_head),
            "candidate_match_trainable_parameters": count(self.candidate_match_head),
            "set_presence_trainable_parameters": count(self.set_presence_head),
            "raw_pair_candidate_verifier_trainable_parameters": (
                count(self.raw_pair_candidate_verifier)
                if self.raw_pair_candidate_verifier is not None
                else 0
            ),
            "total_trainable_parameters": count(self),
        }

    def train(self, mode: bool = True) -> "R35MultitaskSystem":
        super().train(mode)
        self.track_y.eval()
        self.encoder_system.descriptor_head.eval()
        self.encoder_system.matcher.eval()
        if self._training_stage is R35TrainingStage.LIMITED_JOINT:
            self.encoder_system.backbone.train(mode)
        else:
            self.encoder_system.backbone.eval()
        if self._training_stage is R35TrainingStage.BEARING_HEAD:
            self.bearing_head.train(mode)
            self.candidate_match_head.eval()
            self.set_presence_head.eval()
            if self.raw_pair_candidate_verifier is not None:
                self.raw_pair_candidate_verifier.eval()
        elif self._training_stage is R35TrainingStage.OPEN_SET_HEADS:
            self.bearing_head.eval()
            self.candidate_match_head.train(mode)
            self.set_presence_head.train(mode)
            if self.raw_pair_candidate_verifier is not None:
                self.raw_pair_candidate_verifier.train(mode)
        elif self._training_stage is R35TrainingStage.LIMITED_JOINT:
            self.bearing_head.train(mode)
            self.candidate_match_head.train(mode)
            self.set_presence_head.train(mode)
            if self.raw_pair_candidate_verifier is not None:
                self.raw_pair_candidate_verifier.train(mode)
        return self

    def _encode_track_y(
        self,
        tokens: torch.Tensor,
        descriptor_ring: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        base_head = getattr(self.track_y, "base_head", None)
        ring_encoder = getattr(base_head, "ring_encoder", None)
        if callable(ring_encoder):
            return {"dense_ring": ring_encoder(tokens)}
        legacy_encode = getattr(self.track_y, "encode", None)
        if callable(legacy_encode):
            return legacy_encode(tokens, descriptor_ring)
        raise TypeError("Track Y must expose base_head.ring_encoder or the legacy encode interface")

    def _forward_track_y_encoded(
        self,
        source_dense_ring: torch.Tensor,
        target_dense_ring: torch.Tensor,
        source_descriptor_ring: torch.Tensor,
        target_descriptor_ring: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        forward_rings = getattr(self.track_y, "forward_rings", None)
        if callable(forward_rings):
            return forward_rings(
                source_dense_ring,
                target_dense_ring,
                source_descriptor_ring,
                target_descriptor_ring,
            )
        legacy_forward = getattr(self.track_y, "forward_encoded", None)
        if callable(legacy_forward):
            return legacy_forward(
                source_dense_ring,
                target_dense_ring,
                source_descriptor_ring,
                target_descriptor_ring,
            )
        raise TypeError("Track Y must expose forward_rings or the legacy forward_encoded interface")

    def encode(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        tokens = self.encoder_system.backbone.forward_tokens(images)
        descriptor = self.encoder_system.descriptor_head.forward_tokens(tokens.float())
        yaw_encoding = self._encode_track_y(tokens, descriptor["ring"])
        return {
            "tokens": tokens,
            "global": descriptor["global"],
            "descriptor_ring": descriptor["ring"],
            "dense_ring": yaw_encoding["dense_ring"],
        }

    @staticmethod
    def _correlation_summary(scores: torch.Tensor) -> torch.Tensor:
        if scores.ndim != 2 or scores.shape[1] < 2:
            raise ValueError("correlation scores must be [N,L] with L >= 2")
        probability = torch.softmax(scores.float(), dim=-1)
        top2 = torch.topk(probability, k=2, dim=-1).values
        entropy = -(
            probability * probability.clamp_min(1e-8).log()
        ).sum(dim=-1) / math.log(float(scores.shape[-1]))
        return torch.stack(
            (
                scores.float().max(dim=-1).values,
                scores.float().mean(dim=-1),
                top2[:, 0] - top2[:, 1],
                entropy,
            ),
            dim=-1,
        )

    def forward_pairs(
        self,
        source_images: torch.Tensor,
        target_images: torch.Tensor,
    ) -> Dict[str, Any]:
        if source_images.shape != target_images.shape:
            raise ValueError("source and target image batches must have equal shape")
        encoded = self.encode(torch.cat((source_images, target_images), dim=0))
        source = {key: value.chunk(2, dim=0)[0] for key, value in encoded.items()}
        target = {key: value.chunk(2, dim=0)[1] for key, value in encoded.items()}
        yaw = self._forward_track_y_encoded(
            source["dense_ring"],
            target["dense_ring"],
            source["descriptor_ring"],
            target["descriptor_ring"],
        )
        bearing = self.bearing_head(source["tokens"], target["tokens"])
        return {"source": source, "target": target, "yaw": yaw, "bearing": bearing}

    def match_candidate_set(
        self,
        query_images: torch.Tensor,
        candidate_images: torch.Tensor,
        *,
        candidate_mask: torch.Tensor | None = None,
        reciprocal_rank_score: torch.Tensor | None = None,
    ) -> Dict[str, Any]:
        if query_images.ndim != 4 or candidate_images.ndim != 5:
            raise ValueError("query images must be [B,3,H,W] and candidates [B,K,3,H,W]")
        batch, candidate_count = candidate_images.shape[:2]
        if query_images.shape[0] != batch or query_images.shape[1:] != candidate_images.shape[2:]:
            raise ValueError("query and candidate image shapes are incompatible")
        if candidate_mask is None:
            candidate_mask = torch.ones(
                batch, candidate_count, dtype=torch.bool, device=query_images.device
            )
        if candidate_mask.shape != (batch, candidate_count):
            raise ValueError("candidate_mask must be [B,K]")
        if reciprocal_rank_score is None:
            reciprocal_rank_score = torch.zeros(
                batch, candidate_count, device=query_images.device
            )
        if reciprocal_rank_score.shape != (batch, candidate_count):
            raise ValueError("reciprocal_rank_score must be [B,K]")

        flat_candidates = candidate_images.flatten(0, 1)
        encoded = self.encode(torch.cat((query_images, flat_candidates), dim=0))
        query = {key: value[:batch] for key, value in encoded.items()}
        candidate = {
            key: value[batch:].reshape(batch, candidate_count, *value.shape[1:])
            for key, value in encoded.items()
        }
        expand = lambda value: value[:, None].expand(
            batch, candidate_count, *value.shape[1:]
        ).flatten(0, 1)
        candidate_flat = {key: value.flatten(0, 1) for key, value in candidate.items()}
        yaw = self._forward_track_y_encoded(
            expand(query["dense_ring"]),
            candidate_flat["dense_ring"],
            expand(query["descriptor_ring"]),
            candidate_flat["descriptor_ring"],
        )
        bearing = self.bearing_head(
            expand(query["tokens"]),
            candidate_flat["tokens"],
        )
        global_similarity = F.cosine_similarity(
            expand(query["global"]).float(),
            candidate_flat["global"].float(),
            dim=-1,
        )
        sector_scores = yaw.get("descriptor_correlation_32")
        if sector_scores is None:
            scale_scores = yaw["scale_scores"].float()
            sector_scores = scale_scores.mean(dim=1)
        sector_summary = self._correlation_summary(sector_scores)
        yaw_probability = yaw["top1_probability"].float()
        yaw_confidence = yaw["yaw_confidence"].float()
        yaw_entropy = yaw["normalized_entropy"].float()
        bearing_summary = torch.stack(
            (
                bearing["bearing_confidence"].float(),
                bearing["bearing_valid_probability"].float(),
                bearing["normalized_entropy"].float(),
            ),
            dim=-1,
        )
        normalized_rank = (
            torch.arange(candidate_count, device=query_images.device, dtype=torch.float32)
            .view(1, -1)
            .expand(batch, -1)
            / float(max(candidate_count - 1, 1))
        ).reshape(-1)
        scalar_features = torch.cat(
            (
                global_similarity.unsqueeze(-1),
                sector_summary,
                torch.stack((yaw_probability, yaw_confidence, yaw_entropy), dim=-1),
                bearing_summary,
                bearing["local_correlation_summary"].float(),
                normalized_rank.unsqueeze(-1),
                reciprocal_rank_score.reshape(-1, 1).float(),
            ),
            dim=-1,
        ).reshape(batch, candidate_count, -1)
        spatial_pair = torch.cat(
            (yaw["pair_embedding"].float(), bearing["pair_embedding"].float()),
            dim=-1,
        ).reshape(batch, candidate_count, -1)
        candidate_output = self.candidate_match_head(
            spatial_pair,
            scalar_features,
            candidate_mask,
        )
        verifier_output = None
        if self.raw_pair_candidate_verifier is not None:
            verifier_output = self.raw_pair_candidate_verifier(
                spatial_pair, scalar_features, candidate_mask
            )
            presence_output = self.set_presence_head(
                candidate_output,
                global_similarity.reshape(batch, candidate_count),
                yaw_confidence.reshape(batch, candidate_count),
                bearing["bearing_confidence"].reshape(batch, candidate_count),
                normalized_rank.reshape(batch, candidate_count),
                verifier_output,
            )
        else:
            presence_output = self.set_presence_head(
                candidate_output,
                global_similarity.reshape(batch, candidate_count),
                yaw_confidence.reshape(batch, candidate_count),
                bearing["bearing_confidence"].reshape(batch, candidate_count),
                normalized_rank.reshape(batch, candidate_count),
            )
        return {
            "query_encoding": query,
            "candidate_encoding": candidate,
            "yaw": {
                key: value.reshape(batch, candidate_count, *value.shape[1:])
                for key, value in yaw.items()
                if torch.is_tensor(value) and value.shape[0] == batch * candidate_count
            },
            "bearing": {
                key: value.reshape(batch, candidate_count, *value.shape[1:])
                for key, value in bearing.items()
                if torch.is_tensor(value) and value.shape[0] == batch * candidate_count
            },
            "scalar_features": scalar_features,
            "candidate_match": candidate_output,
            "candidate_verifier": verifier_output,
            "set_presence": presence_output,
        }
