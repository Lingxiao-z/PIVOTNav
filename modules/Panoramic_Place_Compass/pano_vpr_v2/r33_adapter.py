from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ResidualAdapterConfig:
    descriptor_dim: int = 2112
    bottleneck_dim: int = 64
    residual_scale: float = 0.1


class ResidualDescriptorAdapter(nn.Module):
    """Small residual adapter used only by the Top-K verifier path.

    Retrieval continues to use the frozen base descriptor. The zero-initialized
    output projection makes the initial revision exactly equivalent to the
    verifier checkpoint from which it is initialized.
    """

    def __init__(self, config: ResidualAdapterConfig | None = None):
        super().__init__()
        self.config = config or ResidualAdapterConfig()
        dim = self.config.descriptor_dim
        bottleneck = self.config.bottleneck_dim
        self.input_norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.down = nn.Linear(dim, bottleneck, bias=False)
        self.activation = nn.GELU()
        self.up = nn.Linear(bottleneck, dim, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, descriptor: torch.Tensor) -> torch.Tensor:
        if descriptor.shape[-1] != self.config.descriptor_dim:
            raise ValueError(
                f"descriptor dimension mismatch: {descriptor.shape[-1]} "
                f"!= {self.config.descriptor_dim}"
            )
        base = F.normalize(descriptor.float(), dim=-1)
        residual = self.up(self.activation(self.down(self.input_norm(base))))
        return F.normalize(base + self.config.residual_scale * residual, dim=-1)

    def audit(self) -> dict[str, object]:
        return {
            "config": asdict(self.config),
            "trainable_parameters": sum(p.numel() for p in self.parameters() if p.requires_grad),
            "total_parameters": sum(p.numel() for p in self.parameters()),
            "zero_initialized_output_projection": bool(torch.count_nonzero(self.up.weight).item() == 0),
        }


def adapt_matcher_inputs(
    adapter: ResidualDescriptorAdapter,
    query: Dict[str, torch.Tensor],
    candidates: Dict[str, torch.Tensor],
) -> tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    adapted_query = dict(query)
    adapted_candidates = dict(candidates)
    adapted_query["global"] = adapter(query["global"])
    adapted_candidates["global"] = adapter(candidates["global"])
    return adapted_query, adapted_candidates


def descriptor_consistency_loss(
    base_query: torch.Tensor,
    adapted_query: torch.Tensor,
    base_candidates: torch.Tensor,
    adapted_candidates: torch.Tensor,
) -> torch.Tensor:
    query_cosine = F.cosine_similarity(base_query.float(), adapted_query.float(), dim=-1)
    candidate_cosine = F.cosine_similarity(base_candidates.float(), adapted_candidates.float(), dim=-1)
    return (1.0 - query_cosine).mean() + (1.0 - candidate_cosine).mean()


class R33R2MatchModule(nn.Module):
    def __init__(self, matcher: nn.Module, adapter: ResidualDescriptorAdapter):
        super().__init__()
        self.matcher = matcher
        self.adapter = adapter

    def forward(
        self,
        query: Dict[str, torch.Tensor],
        candidates: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        adapted_query, adapted_candidates = adapt_matcher_inputs(self.adapter, query, candidates)
        output = self.matcher(adapted_query, adapted_candidates)
        output["adapted_query_global"] = adapted_query["global"]
        output["adapted_candidate_global"] = adapted_candidates["global"]
        return output
