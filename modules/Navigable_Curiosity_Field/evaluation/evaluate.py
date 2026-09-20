"""Evaluate a trained Navigable Curiosity Field checkpoint."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from ..model import RevisedDINOv2PanoramaFGFSV3, RevisedDINOv2PanoramaFGFSV3LastBlockB2
from ..training.dataset import FGFSPairDatasetV2
from ..training.train_curiosity_exploration import CHECKPOINT_SCHEMA, _encode_prefix, _load_backbone, _normalize
from .metrics import fg_metrics, fs_metrics


def top1_direction(scores: np.ndarray, valid: np.ndarray) -> int | None:
    masked = np.where(np.asarray(valid, dtype=bool), np.asarray(scores, dtype=float), -np.inf)
    return int(np.argmax(masked)) if np.isfinite(masked).any() else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split", default="development")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dinov2-repo", type=Path, required=True)
    parser.add_argument("--dinov2-weight", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-pairs", type=int)
    args = parser.parse_args()

    device = torch.device(args.device)
    root = args.data_root.expanduser().resolve()
    split_root = root / "data/formal_v2" / args.split
    if not split_root.is_dir():
        split_root = root
    source = FGFSPairDatasetV2(split_root)
    dataset = Subset(source, range(min(len(source), args.max_pairs))) if args.max_pairs else source
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers)
    backbone = _load_backbone(args.dinov2_repo.resolve(), args.dinov2_weight.resolve(), device)
    decoder = RevisedDINOv2PanoramaFGFSV3().to(device)
    model = RevisedDINOv2PanoramaFGFSV3LastBlockB2(backbone.blocks[-1], backbone.norm, decoder).to(device)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if payload.get("resume_schema") != CHECKPOINT_SCHEMA:
        raise RuntimeError("checkpoint schema mismatch")
    model.load_state_dict(payload["model"], strict=True)
    model.eval()
    fg_pred, fs_pred, fg_target, fs_target, masks, scenes = [], [], [], [], [], []
    with torch.inference_mode():
        for batch in loader:
            current = _normalize(batch["source"], device)
            goal = _normalize(batch["goal"], device)
            prefix = _encode_prefix(backbone, torch.cat((current, goal), dim=0))
            current_tokens, goal_tokens = prefix.chunk(2)
            output = model(current_tokens, goal_tokens)
            fg_pred.append(output.fg_logits.float().cpu().numpy())
            fs_pred.append(output.fs_scores.float().cpu().numpy())
            fg_target.append(batch["fg_labels"].numpy())
            fs_target.append(batch["fs_scores"].numpy())
            masks.append(batch["fs_valid_mask"].numpy())
            scenes.extend(batch["scene_id"])
    result = {
        "checkpoint": str(args.checkpoint.resolve()),
        "pairs": len(dataset),
        "fg": fg_metrics(np.concatenate(fg_pred), np.concatenate(fg_target), np.asarray(scenes)),
        "fs": fs_metrics(np.concatenate(fs_pred), np.concatenate(fs_target), np.concatenate(masks), np.asarray(scenes)),
        "runtime_gt_inputs": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
