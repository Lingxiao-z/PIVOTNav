"""Train the released last-block Navigable Curiosity Field model."""
from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from ..model import (
    RevisedDINOv2PanoramaFGFSV3,
    RevisedDINOv2PanoramaFGFSV3LastBlockB2,
    revised_dino_v3_loss,
)
from .dataset import FGFSPairDatasetV2, file_sha256


CHECKPOINT_SCHEMA = "revised_dino_fgfs_v3_last_block_b2_exact_resume_v1"
MODEL_NAME = "revised_dinov2_panorama_fgfs_train_scenes_v3_last_block_b2"
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_batch(
    model: nn.Module,
    current_tokens: torch.Tensor,
    goal_tokens: torch.Tensor,
    validity_target: torch.Tensor,
    score_target: torch.Tensor,
    score_valid: torch.Tensor,
    *,
    cyclic_tokens: tuple[torch.Tensor, torch.Tensor] | None = None,
    fg_alpha: float = 0.5335763888888889,
) -> dict[str, torch.Tensor]:
    output = model(current_tokens, goal_tokens)
    cyclic_output = model(*cyclic_tokens) if cyclic_tokens is not None else None
    return revised_dino_v3_loss(
        output,
        validity_target,
        score_target,
        score_valid,
        fg_alpha=fg_alpha,
        cyclic_output=cyclic_output,
        cyclic_shift=3 if cyclic_output is not None else 0,
    )


def _normalize(images: torch.Tensor, device: torch.device) -> torch.Tensor:
    value = images.to(device=device, dtype=torch.float32, non_blocking=True)
    mean = value.new_tensor(MEAN).view(1, 3, 1, 1)
    std = value.new_tensor(STD).view(1, 3, 1, 1)
    return (value - mean) / std


def _load_backbone(repository: Path, weight: Path, device: torch.device) -> nn.Module:
    if not (repository / "hubconf.py").is_file():
        raise FileNotFoundError(f"DINOv2 checkout is missing hubconf.py: {repository}")
    if not weight.is_file():
        raise FileNotFoundError(f"DINOv2 weight is missing: {weight}")
    return torch.hub.load(
        str(repository),
        "dinov2_vits14",
        source="local",
        pretrained=True,
        weights=str(weight),
    ).to(device)


def _encode_prefix(backbone: nn.Module, images: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        tokens = backbone.prepare_tokens_with_masks(images)
        for block in backbone.blocks[:-1]:
            tokens = block(tokens)
    if tokens.shape[1:] != (513, 384):
        raise RuntimeError(f"unexpected DINOv2 prefix shape: {tuple(tokens.shape)}")
    return tokens.float()


def _infinite(loader: DataLoader) -> Iterator[dict[str, torch.Tensor]]:
    while True:
        yield from loader


def _checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    *,
    step: int,
    args: argparse.Namespace,
    dataset: FGFSPairDatasetV2,
) -> dict:
    return {
        "resume_schema": CHECKPOINT_SCHEMA,
        "step": int(step),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "training_identity": {
            "model_name": MODEL_NAME,
            "label_version": "nts_inspired_habitat_gs_fgfs_v2",
            "dataset_manifest_sha256": dataset.manifest_sha256,
            "dataset_pairs": len(dataset),
            "seed": args.seed,
            "optimizer_steps": args.steps,
            "batch_size": args.batch_size,
            "learning_rates": {
                "last_dino_block": args.backbone_lr,
                "decoder": args.head_lr,
            },
            "unfrozen_dino_blocks": [11],
            "frozen_dino_blocks": list(range(11)),
            "frozen_final_norm": True,
            "dinov2_weight_sha256": file_sha256(args.dinov2_weight),
        },
        "formal_test_accessed": False,
        "test_run_count": 0,
    }


def synthetic_smoke(seed: int = 7) -> dict[str, float | bool]:
    seed_everything(seed)
    model = RevisedDINOv2PanoramaFGFSV3(dropout=0.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    current = torch.randn(1, 16, 32, 384)
    goal = torch.randn(1, 16, 32, 384)
    fg = torch.randint(0, 2, (1, 12)).float()
    fs = torch.rand(1, 12)
    valid = torch.ones(1, 12, dtype=torch.bool)
    losses = train_one_batch(model, current, goal, fg, fs, valid)
    optimizer.zero_grad(set_to_none=True)
    losses["loss_total"].backward()
    finite = bool(torch.isfinite(losses["loss_total"]))
    nonzero_grad = any(
        parameter.grad is not None and float(parameter.grad.abs().sum()) > 0.0
        for parameter in model.parameters()
    )
    optimizer.step()
    return {"loss": float(losses["loss_total"].detach()), "finite": finite, "nonzero_grad": nonzero_grad}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--split", default="train")
    parser.add_argument("--dinov2-repo", type=Path)
    parser.add_argument("--dinov2-weight", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=19000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=2026081502)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--checkpoint-every", type=int, default=1000)
    parser.add_argument("--backbone-lr", type=float, default=3e-6)
    parser.add_argument("--head-lr", type=float, default=3e-4)
    parser.add_argument("--fg-alpha", type=float, default=0.5335763888888889)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if not args.smoke:
        missing = [name for name in ("data_root", "dinov2_repo", "dinov2_weight", "output") if getattr(args, name) is None]
        if missing:
            parser.error("required for training: " + ", ".join(f"--{name.replace('_', '-')}" for name in missing))
    return args


def main() -> None:
    args = parse_args()
    if args.smoke:
        result = synthetic_smoke(args.seed)
        print(json.dumps(result, indent=2, sort_keys=True))
        if not result["finite"] or not result["nonzero_grad"]:
            raise SystemExit(2)
        return

    seed_everything(args.seed)
    device = torch.device(args.device)
    data_root = args.data_root.expanduser().resolve()
    split_root = data_root / "data/formal_v2" / args.split
    if not split_root.is_dir():
        split_root = data_root
    dataset = FGFSPairDatasetV2(split_root)
    if not len(dataset):
        raise RuntimeError(f"no FG/FS pairs found under {split_root}")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
        persistent_workers=args.workers > 0,
    )
    backbone = _load_backbone(args.dinov2_repo.expanduser().resolve(), args.dinov2_weight.expanduser().resolve(), device)
    backbone.requires_grad_(False)
    backbone.blocks[-1].requires_grad_(True)
    decoder = RevisedDINOv2PanoramaFGFSV3().to(device)
    model = RevisedDINOv2PanoramaFGFSV3LastBlockB2(backbone.blocks[-1], backbone.norm, decoder).to(device)
    optimizer = torch.optim.AdamW(
        [
            {"params": model.last_block.parameters(), "lr": args.backbone_lr},
            {"params": model.decoder.parameters(), "lr": args.head_lr},
        ],
        weight_decay=0.01,
    )

    def lr_scale(step: int) -> float:
        if step < args.warmup_steps:
            return float(step + 1) / max(args.warmup_steps, 1)
        progress = (step - args.warmup_steps) / max(args.steps - args.warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_scale)
    start_step = 0
    if args.resume:
        state = torch.load(args.resume, map_location="cpu", weights_only=False)
        if state.get("resume_schema") != CHECKPOINT_SCHEMA:
            raise RuntimeError("resume checkpoint schema mismatch")
        model.load_state_dict(state["model"], strict=True)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_step = int(state["step"])

    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    iterator = _infinite(loader)
    model.train()
    for step in range(start_step + 1, args.steps + 1):
        batch = next(iterator)
        current = _normalize(batch["source"], device)
        goal = _normalize(batch["goal"], device)
        current_roll = torch.roll(current, shifts=112, dims=3)
        goal_roll = torch.roll(goal, shifts=112, dims=3)
        canonical_prefix = _encode_prefix(backbone, torch.cat((current, goal), dim=0))
        rolled_prefix = _encode_prefix(backbone, torch.cat((current_roll, goal_roll), dim=0))
        current_tokens, goal_tokens = canonical_prefix.chunk(2)
        rolled_current, rolled_goal = rolled_prefix.chunk(2)
        optimizer.zero_grad(set_to_none=True)
        context = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if device.type == "cuda" else torch.autocast(device_type="cpu", enabled=False)
        with context:
            losses = train_one_batch(
                model,
                current_tokens,
                goal_tokens,
                batch["fg_labels"].to(device),
                batch["fs_scores"].to(device),
                batch["fs_valid_mask"].to(device),
                cyclic_tokens=(rolled_current, rolled_goal),
                fg_alpha=args.fg_alpha,
            )
        if not torch.isfinite(losses["loss_total"]):
            raise RuntimeError(f"non-finite loss at step {step}")
        losses["loss_total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        scheduler.step()
        if step == args.steps or step % args.checkpoint_every == 0:
            payload = _checkpoint(model, optimizer, scheduler, step=step, args=args, dataset=dataset)
            torch.save(payload, output / f"step_{step:06d}.pt")
            torch.save(payload, output / "last.pt")
        if step == 1 or step % 100 == 0:
            print(json.dumps({"step": step, **{key: float(value.detach()) for key, value in losses.items()}}), flush=True)


if __name__ == "__main__":
    main()
