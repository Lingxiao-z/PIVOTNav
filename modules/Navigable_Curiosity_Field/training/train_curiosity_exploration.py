"""Unified training entry point for the final curiosity decoder."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn

from ..runtime.curiosity.curiosity_model import revised_dino_v3_loss


def train_one_batch(
    model: nn.Module,
    current_tokens: torch.Tensor,
    goal_tokens: torch.Tensor,
    validity_target: torch.Tensor,
    score_target: torch.Tensor,
    score_valid: torch.Tensor,
) -> dict[str, float]:
    """Train FG internally while exposing curiosity score loss as the task."""
    output = model(current_tokens, goal_tokens)
    losses = revised_dino_v3_loss(
        output,
        validity_target,
        score_target,
        score_valid,
    )
    losses["loss_total"].backward()
    return {name: float(value.detach()) for name, value in losses.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.parse_args()
    raise SystemExit(
        "Connect the collected DINO token dataset and optimizer to train_one_batch."
    )


if __name__ == "__main__":
    main()
