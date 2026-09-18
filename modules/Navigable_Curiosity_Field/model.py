from __future__ import annotations

import torch
from torch import nn


class CuriosityExplorationHead(nn.Module):
    """Unified training scaffold with the internal validity branch."""

    def __init__(self, feature_dim: int = 384):
        super().__init__()
        self.shared = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, feature_dim),
            nn.GELU(),
        )
        self._internal_validity = nn.Linear(feature_dim, 1)
        self.curiosity = nn.Linear(feature_dim * 2, 1)

    def forward(self, source: torch.Tensor, goal: torch.Tensor) -> dict[str, torch.Tensor]:
        source = self.shared(source)
        goal = self.shared(goal)
        return {
            "_internal_validity": self._internal_validity(source).squeeze(-1),
            "scores": self.curiosity(torch.cat((source, goal), dim=-1)).squeeze(-1),
        }
