#!/usr/bin/env python3
from __future__ import annotations

import argparse, hashlib, json, os, sys, tempfile, time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

ROOT = Path(os.environ.get("PIVOTNAV_PPC_TRAINING_ROOT", Path.cwd())).expanduser().resolve()
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from modules.Panoramic_Place_Compass.training.dataset import PlaceShardDataset, v2_place_shards, v2_sample_shards
from modules.Panoramic_Place_Compass.model.core import TrainingPhase, optimizer_parameter_groups
from modules.Panoramic_Place_Compass.model.core import PanoramicVPRV2System
from modules.Panoramic_Place_Compass.training.retrieval.train_phase_a import compute_losses, next_batch_recycling, summary


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def sha_path(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n"); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_torch_save(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def dist_setup() -> tuple[bool, int, int, int]:
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world > 1:
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group("nccl")
    return world > 1, rank, local_rank, world


def unwrap(model):
    return model.module if hasattr(model, "module") else model


def make_loader(args: argparse.Namespace) -> DataLoader:
    place_shards = v2_place_shards(ROOT, args.train_split_tag)
    sample_shards = v2_sample_shards(ROOT, args.train_split_tag)
    if not place_shards or not sample_shards:
        raise RuntimeError(f"missing train shards for {args.train_split_tag}")
    ds = PlaceShardDataset(
        place_shards,
        sample_shards,
        places_per_batch=args.places_per_batch,
        views_per_place=args.views_per_place,
        decoder=args.decoder,
        shuffle=True,
        seed=args.seed,
        include_synthetic_roll=True,
        synthetic_roll_yaws=tuple(float(x) for x in args.synthetic_roll_yaws.split(",") if x.strip()),
        hard_negatives_per_place=args.hard_negatives_per_place,
        hard_negative_probability=args.hard_negative_probability,
    )
    return DataLoader(ds, batch_size=None, num_workers=args.num_workers, pin_memory=True, persistent_workers=args.num_workers > 0, prefetch_factor=args.prefetch_factor if args.num_workers > 0 else None)


def load_checkpoint(system: PanoramicVPRV2System, path: Path, device: torch.device) -> Dict[str, Any]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    if "system" in ckpt:
        system.load_state_dict(ckpt["system"], strict=False)
    elif "head" in ckpt:
        system.descriptor_head.load_state_dict(ckpt["head"], strict=True)
    elif "descriptor_head" in ckpt:
        system.descriptor_head.load_state_dict(ckpt["descriptor_head"], strict=True)
    else:
        raise RuntimeError(f"unsupported checkpoint keys: {sorted(ckpt.keys())}")
    return ckpt


def save_checkpoint(path: Path, system, optimizer, scheduler, scaler, args, step: int, best_metric: float, metrics: Dict[str, Any], epoch_recycles: int, consumed_batches: int) -> None:
    core = unwrap(system)
    payload = {
        "phase": TrainingPhase.B.value,
        "system": core.state_dict(),
        "descriptor_head": core.descriptor_head.state_dict(),
        "matcher": core.matcher.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "global_step": int(step),
        "best_metric": float(best_metric),
        "epoch_recycles": int(epoch_recycles),
        "consumed_batches": int(consumed_batches),
        "args": vars(args),
        "metrics": metrics,
        "created_at": now(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "data_manifest_sha256": sha_path(ROOT / "data/manifests" / f"{args.train_split_tag}_shard_manifest.json"),
        "architecture_validation_sha256": sha_path(ROOT / "runtime/architecture_v1_validation.json"),
    }
    atomic_torch_save(path, payload)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-split-tag", default="train_native_erp224_r3")
    ap.add_argument("--checkpoint", required=True, help="Phase A best/last checkpoint")
    ap.add_argument("--output-name", default="phase_b_native_erp224_r3")
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--places-per-batch", type=int, default=24)
    ap.add_argument("--views-per-place", type=int, default=2)
    ap.add_argument("--synthetic-roll-yaws", default="30,90")
    ap.add_argument("--decoder", default="pil")
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--prefetch-factor", type=int, default=4)
    ap.add_argument("--hard-negatives-per-place", type=int, default=1)
    ap.add_argument("--hard-negative-probability", type=float, default=1.0)
    ap.add_argument("--base-lr", type=float, default=1e-4)
    ap.add_argument("--min-lr", type=float, default=1e-5)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--phase-b-unfreeze-blocks", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.07)
    ap.add_argument("--hard-margin", type=float, default=0.1)
    ap.add_argument("--supcon-weight", type=float, default=1.0)
    ap.add_argument("--retrieval-weight", type=float, default=1.0)
    ap.add_argument("--roll-weight", type=float, default=0.5)
    ap.add_argument("--ring-weight", type=float, default=0.5)
    ap.add_argument("--yaw-weight", type=float, default=0.5)
    ap.add_argument("--hard-weight", type=float, default=0.5)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--amp-init-scale", type=float, default=1024.0)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--checkpoint-every", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20260731)
    ap.add_argument("--resume", default="")
    args = ap.parse_args()

    ddp, rank, local_rank, world = dist_setup()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed + rank); np.random.seed((args.seed + rank) % (2**32 - 1))
    loader = make_loader(args)
    system = PanoramicVPRV2System().to(device)
    load_checkpoint(system, Path(args.checkpoint), device)
    phase_info = system.configure_phase(TrainingPhase.B, phase_b_unfreeze_blocks=args.phase_b_unfreeze_blocks)
    groups = optimizer_parameter_groups(system.backbone, system.descriptor_head, system.matcher, TrainingPhase.B, base_lr=args.base_lr, weight_decay=args.weight_decay)
    optimizer = torch.optim.AdamW(groups)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.steps), eta_min=args.min_lr)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda", init_scale=args.amp_init_scale)
    start_step = 0; consumed_batches = 0; epoch_recycles = 0; best = float("inf"); state: Dict[str, Any] = {}
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        system.load_state_dict(ckpt["system"], strict=False)
        optimizer.load_state_dict(ckpt["optimizer"]); scheduler.load_state_dict(ckpt["scheduler"]); scaler.load_state_dict(ckpt["scaler"])
        start_step = int(ckpt.get("global_step", 0)); consumed_batches = int(ckpt.get("consumed_batches", 0)); epoch_recycles = int(ckpt.get("epoch_recycles", 0)); best = float(ckpt.get("best_metric", best))
        if "torch_rng" in ckpt:
            torch.set_rng_state(ckpt["torch_rng"].cpu() if hasattr(ckpt["torch_rng"], "cpu") else ckpt["torch_rng"])
        if device.type == "cuda" and ckpt.get("cuda_rng"):
            torch.cuda.set_rng_state_all(ckpt["cuda_rng"])
        for _ in range(consumed_batches):
            next_batch_recycling(loader, state)
        state["epoch_recycles"] = epoch_recycles
    model = DDP(system, device_ids=[local_rank], find_unused_parameters=False) if ddp else system
    model.train(True)

    out_dir = ROOT / "outputs/checkpoints" / args.output_name
    metric_path = ROOT / "outputs/metrics" / f"{args.output_name}_train_metrics.jsonl"
    summary_path = ROOT / "outputs/metrics" / f"{args.output_name}_summary.json"
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True); metric_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(ROOT / "outputs/logs" / args.output_name / "train_config.json", {"created_at": now(), "argv": sys.argv[:], "args": vars(args), "phase_info": phase_info, "world_size": world, "note_zh": "Phase B：DINOv2 blocks 8-11 与 final norm 训练；早期 DINO 与 matcher 冻结。"})

    fetch_times=[]; step_times=[]; rows=[]; step = start_step
    while step < args.steps:
        t0 = time.perf_counter(); batch = next_batch_recycling(loader, state); t1 = time.perf_counter()
        images = batch["image"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            out = model(images)
            loss_dict = compute_losses(out, batch, args, device)
        scaler.scale(loss_dict["loss"]).backward(); scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(unwrap(model).parameters(), args.grad_clip)
        scale_before = scaler.get_scale(); scaler.step(optimizer); scaler.update(); optimizer_updated = scaler.get_scale() >= scale_before
        if optimizer_updated: scheduler.step()
        if device.type == "cuda": torch.cuda.synchronize(device)
        step += 1; consumed_batches += 1; epoch_recycles = int(state.get("epoch_recycles", epoch_recycles)); t2 = time.perf_counter(); fetch_times.append(t1-t0); step_times.append(t2-t0)
        row = {k: float(v.detach().cpu()) for k, v in loss_dict.items()}
        row["optimizer_step_skipped"] = not optimizer_updated
        row.update({"created_at": now(), "global_step": step, "rank": rank, "lr_backbone": optimizer.param_groups[0]["lr"], "batch_size": int(images.shape[0]), "data_wait_ratio_running": sum(fetch_times)/max(sum(step_times), 1e-9), "epoch_recycles": epoch_recycles, "hard_negative_place_count": int(batch.get("hard_negative_place_count", 0)), "hard_negative_pair_count_loaded": int(batch.get("hard_negative_pair_count", 0))})
        rows.append(row)
        if rank == 0 and (step == 1 or step % args.log_every == 0):
            print(json.dumps(row, ensure_ascii=False, sort_keys=True), flush=True)
            with metric_path.open("a", encoding="utf-8") as f: f.write(json.dumps(row, ensure_ascii=False, sort_keys=True)+"\n")
        if rank == 0 and (step % args.checkpoint_every == 0 or step == args.steps):
            metric = float(row["loss"])
            save_checkpoint(out_dir / "last.pt", model, optimizer, scheduler, scaler, args, step, min(best, metric), row, epoch_recycles, consumed_batches)
            if metric < best:
                best = metric; save_checkpoint(out_dir / "best.pt", model, optimizer, scheduler, scaler, args, step, best, row, epoch_recycles, consumed_batches)
    if rank == 0:
        payload = {"created_at": now(), "phase": TrainingPhase.B.value, "steps": step, "best_metric": best, "consumed_batches": consumed_batches, "epoch_recycles": epoch_recycles, "last_metrics": rows[-1] if rows else {}, "timing": {"data_wait_ratio": sum(fetch_times)/max(sum(step_times), 1e-9), "step_seconds": summary(step_times)}, "checkpoint_last": str(out_dir/"last.pt"), "checkpoint_best": str(out_dir/"best.pt"), "architecture_validation_sha256": sha_path(ROOT/"runtime/architecture_v1_validation.json")}
        atomic_json(summary_path, payload)
        with (ROOT / "runtime/command_records_v3.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps({"created_at": now(), "step": "train_phase_b_v2", "argv": sys.argv[:], "outputs": [str(summary_path), str(out_dir/"last.pt"), str(out_dir/"best.pt")], "returncode": 0}, ensure_ascii=False, sort_keys=True)+"\n")
        print(json.dumps({"summary": str(summary_path), "last": str(out_dir/"last.pt"), "best": str(out_dir/"best.pt")}, ensure_ascii=False, indent=2))
    if ddp:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
