from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import nn
from torch.nn import functional as F

from .backbone import DINOv2S14Backbone
from .matcher import MatcherConfig, TopKOpenSetMatcher
from .descriptor import ModelConfig, PanoSaladRingV2Head
from .training_policy import TrainingPhase, configure_training_phase


ENCODING_KEYS = ("global", "ring", "cluster_mass", "dustbin_fraction")


class PanoramicVPRV2System(nn.Module):
    """Unified encoder, scalable retrieval and Top-K open-set verifier."""

    def __init__(
        self,
        model_config: ModelConfig | None = None,
        matcher_config: MatcherConfig | None = None,
        load_backbone: bool = True,
    ):
        super().__init__()
        self.model_config = model_config or ModelConfig()
        self.matcher_config = matcher_config or MatcherConfig(descriptor_dim=self.model_config.global_dim)
        if self.matcher_config.descriptor_dim != self.model_config.global_dim:
            raise ValueError("matcher descriptor_dim must equal model global_dim")
        self.backbone = DINOv2S14Backbone(freeze=True) if load_backbone else None
        self.descriptor_head = PanoSaladRingV2Head(self.model_config)
        self.matcher = TopKOpenSetMatcher(self.matcher_config)
        self._configured_phase: TrainingPhase | None = None

    def encode_tokens(self, tokens: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.descriptor_head.forward_tokens(tokens)

    def encode(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        if self.backbone is None:
            raise RuntimeError("system was constructed without a backbone")
        return self.encode_tokens(self.backbone.forward_tokens(images))

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.encode(images)

    @staticmethod
    def retrieve_topk(query_global: torch.Tensor, gallery_global: torch.Tensor, top_k: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if query_global.ndim != 2 or gallery_global.ndim != 2:
            raise ValueError("query_global and gallery_global must be [B,D] and [N,D]")
        if query_global.shape[1] != gallery_global.shape[1] or gallery_global.shape[0] == 0:
            raise ValueError("invalid gallery descriptor shape")
        similarities = F.normalize(query_global.float(), dim=-1) @ F.normalize(gallery_global.float(), dim=-1).T
        return torch.topk(similarities, k=min(int(top_k), gallery_global.shape[0]), dim=-1)

    @staticmethod
    def gather_topk(gallery: Dict[str, torch.Tensor], indices: torch.Tensor) -> Dict[str, torch.Tensor]:
        if indices.ndim != 2:
            raise ValueError("indices must be [B,K]")
        gathered: Dict[str, torch.Tensor] = {}
        for key in ENCODING_KEYS:
            tensor = gallery[key]
            gathered[key] = tensor[indices]
        return gathered

    def match_gallery(
        self,
        query: Dict[str, torch.Tensor],
        gallery: Dict[str, torch.Tensor],
        top_k: int | None = None,
    ) -> Dict[str, torch.Tensor]:
        k = int(top_k or self.matcher_config.top_k)
        retrieval_scores, indices = self.retrieve_topk(query["global"], gallery["global"], k)
        candidates = self.gather_topk(gallery, indices)
        output = self.matcher(query, candidates)
        output["retrieval_scores"] = retrieval_scores
        output["candidate_indices"] = indices
        output["best_gallery_index"] = indices.gather(1, output["best_candidate_index"].unsqueeze(1)).squeeze(1)
        return output

    def configure_phase(self, phase: TrainingPhase | str, phase_b_unfreeze_blocks: int = 4) -> Dict[str, object]:
        if self.backbone is None:
            raise RuntimeError("cannot configure training phase without a backbone")
        self._configured_phase = TrainingPhase(phase)
        return configure_training_phase(
            self.backbone,
            self.descriptor_head,
            self.matcher,
            self._configured_phase,
            phase_b_unfreeze_blocks=phase_b_unfreeze_blocks,
        )

    def train(self, mode: bool = True) -> "PanoramicVPRV2System":
        super().train(mode)
        if self._configured_phase is None:
            return self
        if self.backbone is not None:
            self.backbone.train(mode and self._configured_phase is TrainingPhase.B)
        self.descriptor_head.train(mode)
        self.matcher.train(mode and self._configured_phase is TrainingPhase.C)
        return self
