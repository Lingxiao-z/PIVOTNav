"""Train and package Panoramic Place Compass model components.

Feature-pair NPZ files use source_tokens/target_tokens [N,16,32,384] and may
include yaw_degrees, bearing_degrees, bearing_valid and distance_bucket.
Arrival NPZ files use features [N,T,280], target_arrived [N,T], distance_m
[N,T] and frame_mask [N,T]. V7 NPZ files use features [N,F] and target_m [N].
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from pathlib import Path
from typing import Callable

import joblib
import numpy as np
import torch
from sklearn.ensemble import GradientBoostingRegressor
from torch import nn
from torch.utils.data import DataLoader, Dataset

from ..model.arrival import R36TemporalArrivalHead, r36_temporal_arrival_losses
from ..model.bearing import (
    R35RelativeTranslationBearingHead,
    R36BearingStructuralRevision5Head,
    r35_bearing_losses,
    r36_bearing_structural_revision5_losses,
)
from ..model.checkpoint import CHECKPOINT_SCHEMA, MODEL_VERSION
from ..model.core import PanoramicVPRV2System
from ..model.orientation import (
    R34CrossPositionYawHead,
    R34CrossPositionYawHeadR3,
    r34_yaw_r3_losses,
)
from ..model.relative_bearing import (
    R362BearingHeadConfig,
    R362PatchBearingHead,
    circular_soft_label_cross_entropy,
)


R363_SCHEMA = "r363_limited_unfreeze_checkpoint_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class ArrayDataset(Dataset):
    def __init__(self, path: Path, required: tuple[str, ...]) -> None:
        with np.load(path, allow_pickle=False) as archive:
            missing = sorted(set(required) - set(archive.files))
            if missing:
                raise ValueError(f"{path} is missing arrays: {missing}")
            self.arrays = {name: archive[name].copy() for name in archive.files}
        sizes = {len(value) for value in self.arrays.values()}
        if len(sizes) != 1:
            raise ValueError(f"arrays must share the first dimension: {sizes}")

    def __len__(self) -> int:
        return len(next(iter(self.arrays.values())))

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {name: torch.from_numpy(np.asarray(value[index])) for name, value in self.arrays.items()}


def _save_component(path: Path, schema: str, key: str, model: nn.Module, step: int, source: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": schema,
            "global_step": int(step),
            key: model.state_dict(),
            "training_data_sha256": sha256(source),
            "formal_test_accessed": False,
            "test_run_count": 0,
        },
        path,
    )


def _train_loop(
    model: nn.Module,
    dataset: Dataset,
    loss_step: Callable[[nn.Module, dict[str, torch.Tensor], torch.device], torch.Tensor],
    *,
    output: Path,
    schema: str,
    checkpoint_key: str,
    source: Path,
    device: torch.device,
    steps: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> None:
    seed_everything(seed)
    model.to(device).train()
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=learning_rate,
        weight_decay=1e-4,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    iterator = iter(loader)
    for step in range(1, steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        optimizer.zero_grad(set_to_none=True)
        loss = loss_step(model, batch, device)
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at step {step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        if step == 1 or step % 100 == 0:
            print(json.dumps({"step": step, "loss": float(loss.detach())}), flush=True)
    _save_component(output, schema, checkpoint_key, model, steps, source)


def train_yaw(args: argparse.Namespace) -> None:
    source = args.data.resolve()
    dataset = ArrayDataset(
        source,
        (
            "source_tokens",
            "target_tokens",
            "source_descriptor_ring",
            "target_descriptor_ring",
            "yaw_degrees",
        ),
    )
    model = R34CrossPositionYawHeadR3(R34CrossPositionYawHead())

    def loss_step(module, batch, device):
        current = batch["source_tokens"].to(device).float()
        target = batch["target_tokens"].to(device).float()
        current_ring = batch["source_descriptor_ring"].to(device).float()
        target_ring = batch["target_descriptor_ring"].to(device).float()
        output = module(current, target, current_ring, target_ring)
        return r34_yaw_r3_losses(output, batch["yaw_degrees"].to(device).float(), module)["loss"]

    _train_loop(model, dataset, loss_step, output=args.output, schema="r34_track_y_r3_checkpoint_v1", checkpoint_key="track_y", source=source, device=torch.device(args.device), steps=args.steps, batch_size=args.batch_size, learning_rate=args.lr, seed=args.seed)


def train_private_bearing(args: argparse.Namespace) -> None:
    source = args.data.resolve()
    dataset = ArrayDataset(source, ("source_tokens", "target_tokens", "bearing_degrees", "bearing_valid"))
    model = R35RelativeTranslationBearingHead()

    def loss_step(module, batch, device):
        output = module(batch["source_tokens"].to(device).float(), batch["target_tokens"].to(device).float())
        return r35_bearing_losses(output, batch["bearing_degrees"].to(device).float(), batch["bearing_valid"].to(device).bool(), module.cfg)["loss"]

    _train_loop(model, dataset, loss_step, output=args.output, schema="r35_bearing_checkpoint_v1", checkpoint_key="bearing_head", source=source, device=torch.device(args.device), steps=args.steps, batch_size=args.batch_size, learning_rate=args.lr, seed=args.seed)


def train_r363_head(args: argparse.Namespace) -> None:
    source = args.data.resolve()
    dataset = ArrayDataset(source, ("source_tokens", "target_tokens", "bearing_degrees", "bearing_valid"))
    model = R362PatchBearingHead(R362BearingHeadConfig())

    def loss_step(module, batch, device):
        output = module(batch["source_tokens"].to(device).float(), batch["target_tokens"].to(device).float())
        degrees = batch["bearing_degrees"].to(device).float()
        bins = torch.remainder(torch.floor((degrees + 180.0) / module.config.bin_width_degrees).long(), module.config.bearing_bins)
        valid = batch["bearing_valid"].to(device).bool()
        if not valid.any():
            return output["bearing_logits"].sum() * 0.0
        return circular_soft_label_cross_entropy(output["bearing_logits"][valid], bins[valid], module.config)

    _train_loop(model, dataset, loss_step, output=args.output, schema="r362_patch_bearing_checkpoint_v2", checkpoint_key="bearing_head", source=source, device=torch.device(args.device), steps=args.steps, batch_size=args.batch_size, learning_rate=args.lr, seed=args.seed)


def train_formal_bearing(args: argparse.Namespace) -> None:
    source = args.data.resolve()
    dataset = ArrayDataset(source, ("source_tokens", "target_tokens", "bearing_degrees", "bearing_valid", "distance_bucket"))
    model = R36BearingStructuralRevision5Head()
    model.requires_grad_(True)

    def loss_step(module, batch, device):
        output = module(batch["source_tokens"].to(device).float(), batch["target_tokens"].to(device).float())
        return r36_bearing_structural_revision5_losses(
            output,
            batch["bearing_degrees"].to(device).float(),
            batch["bearing_valid"].to(device).bool(),
            batch["distance_bucket"].to(device).long(),
            module.cfg,
            module.revision_cfg,
        )["loss"]

    _train_loop(model, dataset, loss_step, output=args.output, schema="r36_bearing_structural_revision5_checkpoint_v1", checkpoint_key="bearing_head", source=source, device=torch.device(args.device), steps=args.steps, batch_size=args.batch_size, learning_rate=args.lr, seed=args.seed)


def train_arrival(args: argparse.Namespace) -> None:
    source = args.data.resolve()
    dataset = ArrayDataset(source, ("features", "target_arrived", "distance_m", "frame_mask"))
    model = R36TemporalArrivalHead()

    def loss_step(module, batch, device):
        output = module(batch["features"].to(device).float())
        return r36_temporal_arrival_losses(output, target_arrived=batch["target_arrived"].to(device).bool(), offline_geodesic_distance_m=batch["distance_m"].to(device).float(), frame_mask=batch["frame_mask"].to(device).bool())["loss"]

    _train_loop(model, dataset, loss_step, output=args.output, schema="r36_arrival_checkpoint_v1", checkpoint_key="arrival_temporal_head", source=source, device=torch.device(args.device), steps=args.steps, batch_size=args.batch_size, learning_rate=args.lr, seed=args.seed)


def train_v7(args: argparse.Namespace) -> None:
    with np.load(args.data, allow_pickle=False) as archive:
        features = archive["features"].astype(np.float32)
        target = archive["target_m"].astype(np.float32)
        sample_weight = archive["sample_weight"].astype(np.float32) if "sample_weight" in archive.files else None
    if features.ndim != 2 or target.shape != (len(features),):
        raise ValueError("V7 data must contain features [N,F] and target_m [N]")
    model = GradientBoostingRegressor(random_state=args.seed, n_estimators=args.estimators, max_depth=args.max_depth, learning_rate=args.lr)
    model.fit(features, target, sample_weight=sample_weight)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.output)
    print(json.dumps({"samples": len(target), "features": features.shape[1], "train_mae_m": float(np.abs(model.predict(features) - target).mean()), "model_sha256": sha256(args.output)}, indent=2))


def _state(path: Path, *keys: str) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    for key in keys:
        if isinstance(payload.get(key), dict):
            return payload[key]
    raise KeyError(f"{path} has none of the checkpoint keys {keys}")


def package_r361(args: argparse.Namespace) -> None:
    checkpoint = {
        "schema_version": CHECKPOINT_SCHEMA,
        "encoder_system": _state(args.retrieval_checkpoint, "system", "encoder_system"),
        "track_y": _state(args.yaw_checkpoint, "track_y", "model"),
        "formal_bearing_head": _state(args.formal_bearing_checkpoint, "bearing_head", "model"),
        "arrival_feature_bearing_head": _state(args.private_bearing_checkpoint, "bearing_head", "model"),
        "arrival_head": _state(args.arrival_checkpoint, "arrival_temporal_head", "arrival_head", "model"),
        "metadata": {
            "official_name": MODEL_VERSION,
            "input": "RGB ERP 224x448",
            "descriptor_dimension": 2112,
            "formal_bearing_checkpoint_sha256": sha256(args.formal_bearing_checkpoint),
            "arrival_private_bearing_checkpoint_sha256": sha256(args.private_bearing_checkpoint),
            "arrival_head_checkpoint_sha256": sha256(args.arrival_checkpoint),
            "r3_checkpoint_sha256": sha256(args.retrieval_checkpoint),
            "track_y_checkpoint_sha256": sha256(args.yaw_checkpoint),
            "component_sha256": {
                "retrieval": sha256(args.retrieval_checkpoint),
                "yaw": sha256(args.yaw_checkpoint),
                "formal_bearing": sha256(args.formal_bearing_checkpoint),
                "private_bearing": sha256(args.private_bearing_checkpoint),
                "arrival": sha256(args.arrival_checkpoint),
            },
        },
        "provenance": {
            "core_internal_gates_passed": True,
            "test_r32_historical_confirmation_accessed": False,
            "training_sources_public": True,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output)
    print(json.dumps({"checkpoint": str(args.output), "sha256": sha256(args.output)}, indent=2))


def package_r363(args: argparse.Namespace) -> None:
    r361 = torch.load(args.r361_checkpoint, map_location="cpu", weights_only=False)
    encoder = r361["encoder_system"]
    prefix = "backbone.model."
    tail = {
        key.removeprefix(prefix): value
        for key, value in encoder.items()
        if key.startswith(prefix + "blocks.11.") or key.startswith(prefix + "norm.")
    }
    payload = {
        "schema_version": R363_SCHEMA,
        "frozen_r361_checkpoint_sha256": sha256(args.r361_checkpoint),
        "bearing_tail": tail,
        "bearing_head": _state(args.bearing_checkpoint, "bearing_head", "model"),
        "training_sources_public": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.output)
    print(json.dumps({"checkpoint": str(args.output), "sha256": sha256(args.output), "tail_keys": len(tail)}, indent=2))


def synthetic_smoke() -> dict[str, bool | float]:
    seed_everything(11)
    source = torch.randn(1, 16, 32, 384)
    target = torch.randn(1, 16, 32, 384)
    bearing = torch.tensor([30.0])
    valid = torch.tensor([True])
    private = R35RelativeTranslationBearingHead()
    private_loss = r35_bearing_losses(private(source, target), bearing, valid, private.cfg)["loss"]
    private_loss.backward()
    r363 = R362PatchBearingHead()
    r363_loss = circular_soft_label_cross_entropy(r363(source, target)["bearing_logits"], torch.tensor([42]), r363.config)
    r363_loss.backward()
    arrival = R36TemporalArrivalHead()
    features = torch.randn(1, 3, 280)
    arrival_loss = r36_temporal_arrival_losses(arrival(features), target_arrived=torch.tensor([[False, False, True]]), offline_geodesic_distance_m=torch.tensor([[1.5, 1.1, 0.7]]), frame_mask=torch.ones(1, 3, dtype=torch.bool))["loss"]
    arrival_loss.backward()
    losses = [private_loss, r363_loss, arrival_loss]
    return {"finite": all(bool(torch.isfinite(loss)) for loss in losses), "private_bearing_loss": float(private_loss), "r363_loss": float(r363_loss), "arrival_loss": float(arrival_loss)}


def add_train_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=20260815)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("yaw", "private-bearing", "formal-bearing", "r363-bearing", "arrival"):
        add_train_args(sub.add_parser(name))
    v7 = sub.add_parser("v7")
    add_train_args(v7)
    v7.set_defaults(lr=0.03)
    v7.add_argument("--estimators", type=int, default=300)
    v7.add_argument("--max-depth", type=int, default=2)
    p361 = sub.add_parser("package-r361")
    p361.add_argument("--retrieval-checkpoint", type=Path, required=True)
    p361.add_argument("--yaw-checkpoint", type=Path, required=True)
    p361.add_argument("--formal-bearing-checkpoint", type=Path, required=True)
    p361.add_argument("--private-bearing-checkpoint", type=Path, required=True)
    p361.add_argument("--arrival-checkpoint", type=Path, required=True)
    p361.add_argument("--output", type=Path, required=True)
    p363 = sub.add_parser("package-r363")
    p363.add_argument("--r361-checkpoint", type=Path, required=True)
    p363.add_argument("--bearing-checkpoint", type=Path, required=True)
    p363.add_argument("--output", type=Path, required=True)
    sub.add_parser("smoke")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    actions = {
        "yaw": train_yaw,
        "private-bearing": train_private_bearing,
        "formal-bearing": train_formal_bearing,
        "r363-bearing": train_r363_head,
        "arrival": train_arrival,
        "v7": train_v7,
        "package-r361": package_r361,
        "package-r363": package_r363,
    }
    if args.command == "smoke":
        result = synthetic_smoke()
        print(json.dumps(result, indent=2, sort_keys=True))
        if not result["finite"]:
            raise SystemExit(2)
        return
    actions[args.command](args)


if __name__ == "__main__":
    main()
