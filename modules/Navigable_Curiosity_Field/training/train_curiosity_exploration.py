from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn

from ..model import CuriosityExplorationHead


def train_one_batch(model: nn.Module, source: torch.Tensor, goal: torch.Tensor,
                    validity_target: torch.Tensor, score_target: torch.Tensor) -> float:
    output = model(source, goal)
    loss = nn.functional.binary_cross_entropy_with_logits(
        output["_internal_validity"], validity_target,
    )
    loss = loss + 10.0 * nn.functional.smooth_l1_loss(output["scores"], score_target)
    loss.backward()
    return float(loss.detach())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.parse_args()
    raise SystemExit("The unified training entry point is ready for the collected feature dataset")


if __name__ == "__main__":
    main()
