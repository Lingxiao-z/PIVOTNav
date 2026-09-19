#!/usr/bin/env python3
"""Persistent Expanded-v6 FG/FS worker.

The checkpoint is the exact SHA-matched v6 step-19000 artifact.  Inputs are
event snapshots (regular-node ERP plus goal ERP), never per-control frames.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

OLD_PACKAGE = Path(__import__("os").environ.get("PIVOTNAV_FS_PACKAGE_ROOT", ""))
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
CHECKPOINT = Path(__import__("os").environ.get("PIVOTNAV_FS_CHECKPOINT", ""))
EXPECTED_SHA256 = "44aa451546691f35659ce1ecc0d616d67d706217ceb5a8f2ed43cba9b132760f"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_erp(path: str) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB").resize((448, 224), Image.Resampling.BILINEAR)).copy()


class ExpandedV6Inference:
    def __init__(self, device: str) -> None:
        import torch
        from modules.Navigable_Curiosity_Field.runtime.fs.model import (
            RevisedDINOv2PanoramaFGFSV3,
            RevisedDINOv2PanoramaFGFSV3LastBlockB2,
        )
        from modules.Navigable_Curiosity_Field.runtime.fs.inference import normalize_erp
        if sha256(CHECKPOINT) != EXPECTED_SHA256:
            raise RuntimeError("FS_WEIGHT_NOT_AVAILABLE_OR_SHA_MISMATCH")
        self.torch = torch
        self.normalize_erp = normalize_erp
        self.device = torch.device(device)
        package = OLD_PACKAGE
        weight = Path(os.environ.get("PIVOTNAV_DINOV2_WEIGHT", str(package.parent.parent.parent / "weights/dinov2/dinov2_vits14_pretrain.pth")))
        repo = Path(os.environ.get("PIVOTNAV_DINOV2_REPO", str(package.parent.parent.parent / "third_party/dinov2")))
        if str(repo.parent) not in sys.path:
            sys.path.insert(0, str(repo.parent))
        backbone = torch.hub.load(str(repo), "dinov2_vits14", pretrained=True,
                                  weights=str(weight), source="local").to(self.device)
        decoder = RevisedDINOv2PanoramaFGFSV3().to(self.device)
        model = RevisedDINOv2PanoramaFGFSV3LastBlockB2(backbone.blocks[-1], backbone.norm, decoder).to(self.device)
        checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
        if checkpoint.get("step") != 19000:
            raise RuntimeError(f"UNEXPECTED_EXPANDED_V6_STEP:{checkpoint.get('step')}")
        model.load_state_dict(checkpoint["model"], strict=True)
        self.backbone, self.model = backbone, model
        self.eval()

    def eval(self):
        self.backbone.eval(); self.model.eval()
        for parameter in list(self.backbone.parameters()) + list(self.model.parameters()):
            parameter.requires_grad_(False)

    def infer(self, current: np.ndarray, goal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        torch = self.torch
        with torch.inference_mode():
            cur = self.normalize_erp(current, device=self.device)
            tgt = self.normalize_erp(goal, device=self.device)
            batch = torch.cat((cur, tgt), dim=0)
            mean = batch.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
            std = batch.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
            batch = (batch - mean) / std
            tokens = self.backbone.prepare_tokens_with_masks(batch)
            for block in self.backbone.blocks[:-1]:
                tokens = block(tokens)
            out = self.model(tokens[:1], tokens[1:])
            fg = torch.sigmoid(out.fg_logits.float())[0].cpu().numpy()
            fs = out.fs_scores.float()[0].cpu().numpy()
        return fg, fs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-root", type=Path, default=None)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.package_root is not None:
        os.environ["PIVOTNAV_FS_PACKAGE_ROOT"] = str(args.package_root)
    model = ExpandedV6Inference(args.device)
    print(json.dumps({"ready": True, "device": args.device, "checkpoint_sha256": EXPECTED_SHA256,
                      "checkpoint_step": 19000, "model_load_count": 1}), flush=True)
    for line in sys.stdin:
        request = json.loads(line)
        if request.get("command") == "close":
            print(json.dumps({"closed": True}), flush=True); return
        if request.get("command") != "infer":
            print(json.dumps({"error": "unsupported command"}), flush=True); continue
        started = time.perf_counter()
        fg, fs = model.infer(load_erp(request["current_path"]), load_erp(request["goal_path"]))
        print(json.dumps({"fg_probabilities": fg.tolist(), "fs_scores": fs.tolist(),
                          "latency_ms": (time.perf_counter() - started) * 1000.0,
                          "model_load_count": 1}), flush=True)


if __name__ == "__main__":
    main()
