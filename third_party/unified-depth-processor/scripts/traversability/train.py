"""通行性预测模型训练脚本，支持单卡和多卡 DDP 训练。

Supports scene-level train/val split, azimuth map from camera intrinsics,
decoupled distance/existence loss, multi-camera datasets, multi-dataset
mixed training with per-dataset adaptive resolution, and multi-GPU training
via PyTorch DistributedDataParallel (DDP).

Usage (single GPU):
    python -m scripts.traversability.train \
        --data-root /data/users/renhao/UniNav-DB \
        --datasets DL3DV-1k \
        --splits-jsons /data/users/renhao/UniNav-DB/splits/DL3DV-1k.json \
        --backbone-weights model/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth \
        --epochs 50 --batch-size 16 --lr 1e-4

Usage (multi-GPU, 4 cards):
    torchrun --nproc_per_node=4 -m scripts.traversability.train \
        --data-root /data/users/renhao/UniNav-DB \
        --datasets DL3DV-1k \
        --splits-jsons /data/users/renhao/UniNav-DB/splits/DL3DV-1k.json \
        --backbone-weights model/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth \
        --epochs 50 --batch-size 16 --lr 1e-4

Usage (multi-dataset):
    python -m scripts.traversability.train \
        --data-root /data/users/renhao/UniNav-DB \
        --datasets DL3DV-1k Replica OmniScenes \
        --splits-jsons /data/.../DL3DV-1k.json /data/.../Replica.json /data/.../OmniScenes.json \
        --backbone-weights model/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth \
        --img-size 512 --epochs 50 --batch-size 16 --lr 1e-4

Usage (curated split):
    python -m scripts.traversability.train \
        --data-root /data/users/renhao/UniNav-DB \
        --curated-json /data/users/renhao/UniNav-DB/splits/pinhole_curated.json \
        --backbone-weights model/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth \
        --epochs 50 --batch-size 16 --lr 1e-4
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.amp import autocast, GradScaler

from .model import TraversabilityModel
from .dataset import UniNavDataset, collate_fn
from .losses import total_loss
from .visualize import visualize_sample


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_time(seconds):
    """Format seconds to human readable string."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    m, s = divmod(int(seconds), 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m{s:02d}s"


def _progress_bar(current, total, width=30):
    """Create a simple ASCII progress bar."""
    frac = current / max(total, 1)
    filled = int(width * frac)
    bar = '\u2588' * filled + '\u2591' * (width - filled)
    return f"|{bar}| {current}/{total}"


def _round_robin(loaders):
    """Yield batches from multiple loaders in round-robin order.

    Args:
        loaders: List of DataLoader objects.

    Yields:
        Batches from each loader in turn until all are exhausted.
    """
    iterators = [iter(ld) for ld in loaders]
    active = list(range(len(iterators)))
    while active:
        next_active = []
        for idx in active:
            try:
                yield next(iterators[idx])
                next_active.append(idx)
            except StopIteration:
                pass
        active = next_active


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def build_model(args) -> torch.nn.Module:
    """Instantiate the requested model architecture from CLI args.

    Args:
        args: Parsed argparse namespace.  Relevant fields: model,
            backbone_weights, backbone_type, freeze_backbone,
            unfreeze_last_n, multi_scale.

    Returns:
        Instantiated nn.Module with forward(x, azimuth_map) signature.
    """
    import sys as _sys

    if args.model == 'traversability':
        if not args.backbone_weights:
            print("[ERROR] --backbone-weights is required for --model traversability")
            _sys.exit(1)
        return TraversabilityModel(
            backbone_path=args.backbone_weights,
            backbone_type=args.backbone_type,
            freeze_backbone=args.freeze_backbone,
            unfreeze_last_n=args.unfreeze_last_n,
            num_angles=360,
            use_multi_scale=args.multi_scale,
        )

    elif args.model == 'resnet-fc':
        from .baselines import ResNetFC
        return ResNetFC(
            freeze_backbone=args.freeze_backbone,
            num_angles=360,
        )

    elif args.model == 'dinov3-mlp':
        if not args.backbone_weights:
            print("[ERROR] --backbone-weights is required for --model dinov3-mlp")
            _sys.exit(1)
        from .baselines import DINOv3MLP
        return DINOv3MLP(
            weights_path=args.backbone_weights,
            backbone_type=args.backbone_type,
            freeze_backbone=args.freeze_backbone,
            unfreeze_last_n=args.unfreeze_last_n,
            num_angles=360,
        )

    else:
        print(f"[ERROR] Unknown --model '{args.model}'")
        _sys.exit(1)


# ---------------------------------------------------------------------------
# Training and validation loops
# ---------------------------------------------------------------------------

def _compute_exist_pos_weight(train_datasets, max_samples=2000):
    """Estimate BCE pos_weight from training data by counting FOV vs non-FOV bins.

    Scans up to max_samples frames across all training datasets and counts
    bins in FOV (continuous_has_data=True) vs outside FOV. Returns neg/pos ratio.

    Args:
        train_datasets: List of UniNavDataset objects.
        max_samples: Maximum number of frames to scan.

    Returns:
        float pos_weight = n_negative / n_positive, or 1.0 if no positives found.
    """
    n_pos = 0  # FOV bins (continuous_has_data=True)
    n_neg = 0  # Non-FOV bins (continuous_has_data=False)
    scanned = 0
    per_ds = max(1, max_samples // max(len(train_datasets), 1))

    for ds in train_datasets:
        limit = min(per_ds, len(ds))
        step = max(1, len(ds) // limit)
        for i in range(0, len(ds), step):
            if scanned >= max_samples:
                break
            sample = ds[i]
            continuous_has_data = sample['continuous_has_data'].bool()
            # Count FOV bins as positive, non-FOV bins as negative
            n_pos += int(continuous_has_data.sum().item())
            n_neg += int((~continuous_has_data).sum().item())
            scanned += 1

    if n_pos == 0:
        return 1.0
    return float(n_neg) / float(n_pos)


def train_one_epoch(model, train_loaders, optimizer, scaler, device, accum_steps,
                    dist_weight, exist_pos_weight, smooth_weight, huber_beta,
                    max_grad_norm=1.0, dist_decay='none', dist_decay_scale=20.0, is_main=True,
                    use_azimuth=True):
    """Run one training epoch over all datasets in round-robin order.

    Args:
        model: TraversabilityModel.
        train_loaders: List of DataLoader objects (one per dataset).
        optimizer: Optimizer.
        scaler: GradScaler for AMP.
        device: torch.device.
        accum_steps: Gradient accumulation steps.
        dist_weight: Weight for distance loss.
        exist_pos_weight: BCE pos_weight tensor or None.
        smooth_weight: Weight for smoothness regulariser.
        huber_beta: Huber loss delta.
        max_grad_norm: Max gradient norm for clipping (0 to disable).
        is_main: Whether this is the main process (controls progress output).

    Returns:
        Dict of average loss values.
    """
    model.train()
    running = {'total': 0.0, 'radar_dist': 0.0, 'exist': 0.0, 'smooth': 0.0}
    n_batches = 0
    optimizer.zero_grad()
    t_start = time.time()
    n_total = sum(len(ld) for ld in train_loaders)

    for i, batch in enumerate(_round_robin(train_loaders)):
        images = batch['image'].to(device)
        azimuth_map = batch['azimuth_map'].to(device)
        radar_dist_gt = batch['radar_dist'].to(device)
        continuous_has_data = batch['continuous_has_data'].to(device)
        has_data = batch['has_data'].to(device)

        with autocast(device_type='cuda', dtype=torch.float16):
            _pred_dist, pred_exist, raw_dist = model(
                images, azimuth_map if use_azimuth else None
            )
            losses = total_loss(
                raw_dist, pred_exist,
                radar_dist_gt, continuous_has_data, has_data,
                dist_weight=dist_weight,
                exist_pos_weight=exist_pos_weight,
                smooth_weight=smooth_weight,
                huber_beta=huber_beta,
                dist_decay=dist_decay,
                dist_decay_scale=dist_decay_scale,
            )
            loss = losses['total'] / accum_steps

        scaler.scale(loss).backward()

        if (i + 1) % accum_steps == 0 or (i + 1) == n_total:
            if max_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    (p for p in model.parameters() if p.requires_grad),
                    max_grad_norm,
                )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        for k in running:
            running[k] += losses[k].item()
        n_batches += 1

        # Progress output
        elapsed = time.time() - t_start
        speed = n_batches / elapsed if elapsed > 0 else 0
        eta = (n_total - (i + 1)) / speed if speed > 0 else 0
        avg_loss = running['total'] / n_batches
        avg_dist = running['radar_dist'] / n_batches
        avg_exist = running['exist'] / n_batches
        avg_smooth = running['smooth'] / n_batches
        bar = _progress_bar(i + 1, n_total)
        lr_now = optimizer.param_groups[0]['lr']
        if is_main:
            print(
                f"\r  Train {bar} "
                f"loss={avg_loss:.4f} (dist={avg_dist:.4f} exist={avg_exist:.4f} "
                f"smooth={avg_smooth:.4f}) "
                f"lr={lr_now:.1e} "
                f"[{_format_time(elapsed)}<{_format_time(eta)}, {speed:.1f} bat/s]",
                end='', flush=True,
            )

    if is_main:
        print()
    return {k: v / max(n_batches, 1) for k, v in running.items()}


@torch.no_grad()
def validate(model, val_loaders, device, dist_weight, exist_pos_weight,
             smooth_weight, huber_beta, dist_decay='none', dist_decay_scale=20.0, is_main=True,
             use_azimuth=True):
    """Run validation over all datasets in round-robin order.

    Args:
        model: TraversabilityModel.
        val_loaders: List of DataLoader objects (one per dataset).
        device: torch.device.
        dist_weight: Weight for distance loss.
        exist_pos_weight: BCE pos_weight tensor or None.
        smooth_weight: Weight for smoothness regulariser.
        huber_beta: Huber loss delta.
        is_main: Whether this is the main process (controls progress output).

    Returns:
        Dict of average loss and metric values.
    """
    model.eval()
    running = {'total': 0.0, 'radar_dist': 0.0, 'exist': 0.0, 'smooth': 0.0}
    metric_sum = {'exist_correct': 0, 'exist_total': 0,
                  'dist_mae_sum': 0.0, 'dist_mae_n': 0}
    n_batches = 0
    t_start = time.time()
    n_total = sum(len(ld) for ld in val_loaders)

    for i, batch in enumerate(_round_robin(val_loaders)):
        images = batch['image'].to(device)
        azimuth_map = batch['azimuth_map'].to(device)
        radar_dist_gt = batch['radar_dist'].to(device)
        continuous_has_data = batch['continuous_has_data'].to(device)
        has_data = batch['has_data'].to(device)

        with autocast(device_type='cuda', dtype=torch.float16):
            _pred_dist, pred_exist, raw_dist = model(
                images, azimuth_map if use_azimuth else None
            )
            losses = total_loss(
                raw_dist, pred_exist,
                radar_dist_gt, continuous_has_data, has_data,
                dist_weight=dist_weight,
                exist_pos_weight=exist_pos_weight,
                smooth_weight=smooth_weight,
                huber_beta=huber_beta,
                dist_decay=dist_decay,
                dist_decay_scale=dist_decay_scale,
            )

        for k in running:
            running[k] += losses[k].item()
        n_batches += 1

        # Compute exist_acc and dist_mae in float32
        # Existence accuracy: predict FOV membership (continuous_has_data) on all 360 bins
        exist_gt = continuous_has_data  # (B, 360)
        exist_pred = (pred_exist > 0)  # (B, 360)
        metric_sum['exist_correct'] += (exist_pred == exist_gt).sum().item()
        metric_sum['exist_total'] += exist_gt.numel()

        # Distance MAE: only on bins with actual data
        if continuous_has_data.any():
            has_data_mask = continuous_has_data & has_data
            if has_data_mask.any():
                mae = (raw_dist[has_data_mask].float() - radar_dist_gt[has_data_mask]).abs().sum().item()
                metric_sum['dist_mae_sum'] += mae
                metric_sum['dist_mae_n'] += has_data_mask.sum().item()

        elapsed = time.time() - t_start
        bar = _progress_bar(i + 1, n_total)
        avg_loss = running['total'] / n_batches
        if is_main:
            print(
                f"\r  Val   {bar} loss={avg_loss:.4f} [{_format_time(elapsed)}]",
                end='', flush=True,
            )

    if is_main:
        print()
    result = {k: v / max(n_batches, 1) for k, v in running.items()}
    if metric_sum['exist_total'] > 0:
        result['exist_acc'] = metric_sum['exist_correct'] / metric_sum['exist_total']
    if metric_sum['dist_mae_n'] > 0:
        result['dist_mae'] = metric_sum['dist_mae_sum'] / metric_sum['dist_mae_n']
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Train Traversability Model')
    parser.add_argument('--data-root', required=True,
                        help='UniNav-DB root (e.g., /data/users/renhao/UniNav-DB)')
    parser.add_argument('--curated-json', default=None,
                        help='Path to a curated splits JSON (from gen_curated_splits.py). '
                             'When provided, --datasets and --splits-jsons are ignored.')
    parser.add_argument('--datasets', nargs='+', default=['DL3DV-1k'],
                        help='One or more dataset names')
    parser.add_argument('--scenes', nargs='+', default=None,
                        help='Scene IDs to use (default: all; applied to all datasets)')
    parser.add_argument('--splits-jsons', nargs='+', default=None,
                        help='Splits JSON paths, one per dataset (auto-detected if omitted)')
    parser.add_argument('--model', default='traversability',
                        choices=['traversability', 'resnet-fc', 'dinov3-mlp'],
                        help='Model architecture (default: traversability). '
                             'resnet-fc does not require --backbone-weights.')
    parser.add_argument('--backbone-weights', required=False, default=None,
                        help='Path to DINOv3 pretrained weights '
                             '(required for traversability and dinov3-mlp)')
    parser.add_argument('--backbone-type', default='vitb16',
                        choices=['vits16', 'vitb16', 'vitl16', 'vitl16plus', 'vith16plus'],
                        help='ViT variant to load (default: vitb16)')
    parser.add_argument('--freeze-backbone', action='store_true', default=True,
                        help='Freeze backbone (default: True)')
    parser.add_argument('--unfreeze-last-n', type=int, default=0,
                        help='Unfreeze last N backbone blocks')
    parser.add_argument('--multi-scale', action='store_true', default=True,
                        help='Use multi-scale FPN fusion (default: True)')
    parser.add_argument('--no-multi-scale', dest='multi_scale', action='store_false',
                        help='Disable multi-scale FPN fusion')
    parser.add_argument('--no-azimuth', dest='use_azimuth', action='store_false',
                        default=True,
                        help='Disable azimuth map input (ablation: passes None to model)')
    parser.add_argument('--img-size', type=int, default=512)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--accum-steps', type=int, default=1,
                        help='Gradient accumulation steps')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--backbone-lr', type=float, default=1e-5,
                        help='Backbone learning rate (if unfrozen)')
    parser.add_argument('--weight-decay', type=float, default=0.05)
    parser.add_argument('--val-split', type=float, default=0.1,
                        help='Validation split ratio (by scene)')
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--output-dir', default='checkpoints/traversability',
                        help='Checkpoint output directory')
    parser.add_argument('--clearance-height', type=float, default=0.5,
                        help='Passable-gap threshold in metres (default: 0.5). '
                             'Profile points with height < this value block the robot.')
    parser.add_argument('--dist-weight', type=float, default=2.0,
                        help='Weight for distance Huber loss term (default: 2.0)')
    parser.add_argument('--huber-beta', type=float, default=1.0,
                        help='Huber loss delta in metres (transition from L2 to L1)')
    parser.add_argument('--smooth-weight', type=float, default=0.0,
                        help='Weight for angular TV smoothness regulariser (0 to disable)')
    parser.add_argument('--dist-decay', type=str, default='exp',
                        choices=['none', 'inverse', 'linear', 'exp'],
                        help='Distance-based weighting for Huber loss: none/inverse/linear/exp')
    parser.add_argument('--dist-decay-scale', type=float, default=10.0,
                        help='Scale (metres) for exponential dist decay (default: 10.0)')
    parser.add_argument('--exist-pos-weight', type=float, default=None,
                        help='BCE pos_weight for existence loss (auto if omitted)')
    parser.add_argument('--warmup-epochs', type=int, default=5,
                        help='Linear LR warmup epochs (0 to disable)')
    parser.add_argument('--max-grad-norm', type=float, default=1.0,
                        help='Max gradient norm for clipping (0 to disable)')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--resume', default=None,
                        help='Path to checkpoint (.pth) to resume training from')
    parser.add_argument('--patience', type=int, default=5,
                        help='Early stopping: stop if val loss does not improve for '
                             'this many consecutive epochs (0 to disable, default: 5)')
    parser.add_argument('--viz-every', type=int, default=5,
                        help='Save prediction visualizations every N epochs (0 to disable)')
    parser.add_argument('--viz-samples', type=int, default=10,
                        help='Number of val samples to visualize per epoch')

    args = parser.parse_args()

    # DDP setup: torchrun sets LOCAL_RANK; plain python -m does not.
    local_rank = int(os.environ.get('LOCAL_RANK', -1))
    is_ddp = local_rank >= 0
    is_main = local_rank <= 0  # True for single-GPU or rank-0

    if is_ddp:
        dist.init_process_group('nccl')
        device = torch.device(f'cuda:{local_rank}')
        torch.cuda.set_device(device)
    else:
        device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    import torch.backends.cudnn as cudnn
    cudnn.benchmark = True

    # ---------------------------------------------------------------------------
    # Expand curated JSON into datasets / splits_jsons lists
    # ---------------------------------------------------------------------------
    if args.curated_json is not None:
        curated_path = Path(args.curated_json)
        if not curated_path.exists():
            print(f"[ERROR] Curated JSON not found: {curated_path}")
            import sys; sys.exit(1)
        with open(curated_path, "r", encoding="utf-8") as _fh:
            _curated = json.load(_fh)
        # Write per-dataset temporary splits JSONs into memory by reusing the
        # curated dataset entries directly as splits JSONs (same scene schema).
        # We create lightweight in-memory splits JSON files under a tmp dir.
        import tempfile, os as _os
        _tmp_dir = Path(tempfile.mkdtemp(prefix="curated_splits_"))
        args.datasets = []
        splits_jsons = []
        for _ds_entry in _curated.get("datasets", []):
            _ds_name = _ds_entry["dataset_name"]
            args.datasets.append(_ds_name)
            # Write a minimal splits JSON compatible with UniNavDataset
            _sj = {
                "dataset_name": _ds_name,
                "camera_type":  _ds_entry.get("camera_type", "pinhole"),
                "source":       _ds_entry.get("source", ""),
                "scenes":       _ds_entry.get("scenes", []),
            }
            _sj_path = _tmp_dir / f"{_ds_name}.json"
            with open(_sj_path, "w", encoding="utf-8") as _fh:
                json.dump(_sj, _fh, ensure_ascii=False)
            splits_jsons.append(str(_sj_path))
        print(f"  Curated JSON: {curated_path}")
        print(f"  Expanded to {len(args.datasets)} dataset(s): {args.datasets}")
    else:
        # Auto-detect splits JSONs if not provided
        splits_jsons = args.splits_jsons or []
        if not splits_jsons:
            for ds in args.datasets:
                candidate = Path(args.data_root) / 'splits' / f'{ds}.json'
                if candidate.exists():
                    splits_jsons.append(str(candidate))
                    print(f"  Auto-detected splits JSON: {candidate}")
                else:
                    splits_jsons.append(None)
        # Pad with None if fewer jsons than datasets
        while len(splits_jsons) < len(args.datasets):
            splits_jsons.append(None)

    if is_main:
        print("=" * 70)
        print("Traversability Model Training")
        print("=" * 70)
        print(f"  Data root:         {args.data_root}")
        print(f"  Datasets:          {args.datasets}")
        print(f"  Splits JSONs:      {splits_jsons}")
        print(f"  Scenes:            {args.scenes or 'all'}")
        print(f"  Clearance height:  {args.clearance_height} m")
        print(f"  Image size:        {args.img_size} (long-edge, auto aspect ratio)")
        print(f"  Batch size:        {args.batch_size} x {args.accum_steps} accum"
              f" = {args.batch_size * args.accum_steps} effective")
        print(f"  Epochs:            {args.epochs}")
        print(f"  LR:                {args.lr} (backbone: {args.backbone_lr})")
        print(f"  Weight decay:      {args.weight_decay}")
        print(f"  Val split:         {args.val_split}")
        print(f"  Dist weight:       {args.dist_weight}")
        print(f"  Exist pos_weight:  {args.exist_pos_weight or 'auto'}")
        print(f"  Device:            {device}")
        if is_ddp:
            print(f"  DDP:               {dist.get_world_size()} GPUs")
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(device)
            gpu_mem = torch.cuda.get_device_properties(device).total_memory / 1024**3
            print(f"  GPU:               {gpu_name} ({gpu_mem:.1f} GB)")
        print()

    # ---------------------------------------------------------------------------
    # Per-dataset scene-level train/val split
    # ---------------------------------------------------------------------------

    rng = np.random.RandomState(42)
    train_loaders = []
    train_samplers = []  # DistributedSampler per loader (None in single-GPU mode)
    val_loaders = []
    val_datasets = []  # kept for visualization

    for ds_name, sj in zip(args.datasets, splits_jsons):
        ds_kwargs = dict(
            data_root=args.data_root,
            dataset_name=ds_name,
            splits_json=sj,
            img_size=args.img_size,
            clearance_height=args.clearance_height,
        )

        probe = UniNavDataset(scenes=args.scenes, augment=False, **ds_kwargs)
        all_scene_ids = probe.get_scene_ids()
        print(f"[{ds_name}] samples={len(probe)}, scenes={len(all_scene_ids)}, "
              f"resolution={probe.img_h}x{probe.img_w}")
        del probe

        if not all_scene_ids:
            print(f"[WARN] No samples for dataset {ds_name}, skipping.")
            continue

        scene_order = list(all_scene_ids)
        rng.shuffle(scene_order)
        n_val = max(1, int(len(scene_order) * args.val_split))
        val_scenes = scene_order[:n_val]
        train_scenes = scene_order[n_val:]

        train_ds = UniNavDataset(scenes=train_scenes, augment=True, **ds_kwargs)
        val_ds = UniNavDataset(scenes=val_scenes, augment=False, **ds_kwargs)
        print(f"  train={len(train_ds)} ({len(train_scenes)} scenes), "
              f"val={len(val_ds)} ({len(val_scenes)} scenes)")

        if len(train_ds) == 0:
            print(f"[WARN] No training samples for {ds_name}, skipping.")
            continue

        train_sampler = DistributedSampler(train_ds, shuffle=True) if is_ddp else None
        train_loaders.append(DataLoader(
            train_ds, batch_size=args.batch_size,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            num_workers=args.num_workers,
            pin_memory=True, drop_last=True,
            persistent_workers=args.num_workers > 0,
            prefetch_factor=4 if args.num_workers > 0 else None,
            collate_fn=collate_fn,
        ))
        train_samplers.append(train_sampler)
        if len(val_ds) > 0:
            val_loaders.append(DataLoader(
                val_ds, batch_size=args.batch_size,
                shuffle=False, num_workers=args.num_workers,
                pin_memory=True,
                persistent_workers=args.num_workers > 0,
                prefetch_factor=4 if args.num_workers > 0 else None,
                collate_fn=collate_fn,
            ))
            val_datasets.append(val_ds)

    if not train_loaders:
        print("[ERROR] No training data found across all datasets.")
        return

    # ---------------------------------------------------------------------------
    # Model
    # ---------------------------------------------------------------------------

    model = build_model(args).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if is_main:
        print(f"Total params:     {total_params:,} ({total_params/1e6:.1f}M)")
        print(f"Trainable params: {trainable_params:,} ({trainable_params/1e6:.1f}M)")

    # Conv2d weight gradients can be produced in channels_last layout by autograd,
    # causing a DDP bucket-view stride mismatch warning.  Register a hook to
    # make them contiguous before AllReduce.
    # DISABLED: These hooks cause illegal memory access during DDP initialization
    # for m in model.modules():
    #     if isinstance(m, torch.nn.Conv2d) and m.weight.requires_grad:
    #         m.weight.register_hook(lambda g: g.contiguous() if not g.is_contiguous() else g)

    # ---------------------------------------------------------------------------
    # Optimizer
    # ---------------------------------------------------------------------------

    # Build optimizer from raw (non-DDP) model so backbone param groups work.
    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    head_params = [p for n, p in model.named_parameters()
                   if p.requires_grad and not n.startswith('backbone.')]

    param_groups = [{'params': head_params, 'lr': args.lr}]
    if backbone_params:
        param_groups.append({'params': backbone_params, 'lr': args.backbone_lr})

    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)

    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epochs - args.warmup_epochs, 1), eta_min=1e-6,
    )
    if args.warmup_epochs > 0:
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1e-3, end_factor=1.0,
            total_iters=args.warmup_epochs,
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup, cosine],
            milestones=[args.warmup_epochs],
        )
    else:
        scheduler = cosine

    scaler = GradScaler()

    # Wrap model with DDP after optimizer is built (param references stay valid).
    # gradient_as_bucket_view=True eliminates the stride-mismatch warning from
    # Conv1x1 weights (e.g. MultiScaleFusion lateral projections) by reusing
    # the bucket view directly as the gradient tensor.
    # find_unused_parameters=True is required when backbone is frozen, as DDP
    # needs to know that some parameters won't receive gradients.
    # DISABLED: find_unused_parameters causes illegal memory access during DDP init
    if is_ddp:
        model = DDP(
            model,
            device_ids=[local_rank],
            # gradient_as_bucket_view=True,  # Removed: causes stride mismatch warnings
            find_unused_parameters=False,  # Force False to avoid illegal memory access
        )
    raw_model = model.module if is_ddp else model

    # Existence pos_weight
    exist_pos_weight = None
    if args.exist_pos_weight is not None:
        exist_pos_weight = torch.tensor([args.exist_pos_weight], device=device)
    else:
        # Use cached value from checkpoint if available (skip expensive scan on resume)
        cached_pw = None
        if args.resume is not None:
            try:
                _ckpt_peek = torch.load(args.resume, map_location='cpu')
                cached_pw = _ckpt_peek.get('exist_pos_weight')
            except Exception:
                pass
        if cached_pw is not None:
            exist_pos_weight = torch.tensor([cached_pw], device=device)
            if is_main:
                print(f"  exist_pos_weight: {cached_pw:.2f} (loaded from checkpoint)")
        else:
            train_datasets = [ld.dataset for ld in train_loaders]
            pw = _compute_exist_pos_weight(train_datasets)
            exist_pos_weight = torch.tensor([pw], device=device)
            if is_main:
                print(f"  Auto exist_pos_weight: {pw:.2f} (neg/pos ratio)")

    # Checkpoint dir
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------------------
    # Resume from checkpoint
    # ---------------------------------------------------------------------------

    start_epoch = 1
    best_val_loss = float('inf')
    patience_counter = 0  # consecutive epochs without improvement

    if args.resume is not None:
        ckpt = torch.load(args.resume, map_location=device)
        raw_model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        try:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        except (KeyError, ValueError):
            print("[WARN] Scheduler state incompatible with checkpoint (scheduler type changed); resetting scheduler.")
        if 'scaler_state_dict' in ckpt:
            scaler.load_state_dict(ckpt['scaler_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_val_loss = ckpt.get('best_val_loss', float('inf'))
        patience_counter = ckpt.get('patience_counter', 0)
        if is_main:
            print(f"Resumed from {args.resume} (epoch {ckpt['epoch']}, best_val={best_val_loss:.4f}, patience={patience_counter}/{args.patience})")
        # Warn if key training params differ from checkpoint
        saved_args = ckpt.get('args', {})
        for key in ('clearance_height', 'dist_weight', 'img_size', 'datasets'):
            saved_val = saved_args.get(key.replace('-', '_'))
            cur_val = getattr(args, key.replace('-', '_'), None)
            if saved_val is not None and saved_val != cur_val:
                print(f"[WARN] --{key} changed: checkpoint={saved_val}, current={cur_val}")

    # Fixed val indices for per-epoch visualization (use first val dataset)
    viz_indices = []
    viz_dataset = val_datasets[0] if val_datasets else None
    if args.viz_every > 0 and viz_dataset is not None and len(viz_dataset) > 0:
        n_viz = min(args.viz_samples, len(viz_dataset))
        viz_indices = np.linspace(0, len(viz_dataset) - 1, n_viz, dtype=int).tolist()

    # ---------------------------------------------------------------------------
    # Training loop
    # ---------------------------------------------------------------------------

    history = []

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()

        # Advance DistributedSampler epoch for proper per-epoch shuffling.
        for s in train_samplers:
            if s is not None:
                s.set_epoch(epoch)

        if is_main:
            print(f"\nEpoch {epoch}/{args.epochs}")
            print("-" * 50)

        train_losses = train_one_epoch(
            model, train_loaders, optimizer, scaler, device,
            args.accum_steps, args.dist_weight, exist_pos_weight,
            args.smooth_weight, args.huber_beta, args.max_grad_norm,
            args.dist_decay, args.dist_decay_scale, is_main=is_main,
            use_azimuth=args.use_azimuth,
        )
        val_losses = validate(
            model, val_loaders, device,
            args.dist_weight, exist_pos_weight,
            args.smooth_weight, args.huber_beta,
            args.dist_decay, args.dist_decay_scale, is_main=is_main,
            use_azimuth=args.use_azimuth,
        )

        scheduler.step()

        elapsed = time.time() - t0

        # Epoch summary
        if is_main:
            _sep = '\u2500'
            print(f"\n  {'Metric':<20s} {'Train':>10s} {'Val':>10s}")
            print(f"  {_sep*20} {_sep*10} {_sep*10}")
            print(f"  {'Total Loss':<20s} {train_losses['total']:>10.4f} {val_losses['total']:>10.4f}")
            print(f"  {'Dist Loss':<20s} {train_losses['radar_dist']:>10.4f} {val_losses['radar_dist']:>10.4f}")
            print(f"  {'Exist Loss':<20s} {train_losses['exist']:>10.6f} {val_losses['exist']:>10.6f}")
            if args.smooth_weight > 0:
                print(f"  {'Smooth Loss':<20s} {train_losses['smooth']:>10.4f} {val_losses.get('smooth', 0.0):>10.4f}")
            if 'exist_acc' in val_losses:
                print(f"  {'Exist Acc':<20s} {'':>10s} {val_losses['exist_acc']:>10.4f}")
            if 'dist_mae' in val_losses:
                print(f"  {'Dist MAE (m)':<20s} {'':>10s} {val_losses['dist_mae']:>10.4f}")
            print(f"  LR: {optimizer.param_groups[0]['lr']:.2e} | Time: {_format_time(elapsed)}")

            # Print cache statistics
            for i, loader in enumerate(train_loaders):
                cache_stats = loader.dataset.get_cache_stats()
                if cache_stats.get('enabled'):
                    hit_rate = cache_stats['hit_rate']
                    size_mb = cache_stats['size_mb']
                    num_items = cache_stats['num_items']
                    print(f"  Cache[train_{i}]: hit_rate={hit_rate:.2%}, size={size_mb:.1f}MB, items={num_items}")
            for i, ds in enumerate(val_datasets):
                cache_stats = ds.get_cache_stats()
                if cache_stats.get('enabled'):
                    hit_rate = cache_stats['hit_rate']
                    size_mb = cache_stats['size_mb']
                    num_items = cache_stats['num_items']
                    print(f"  Cache[val_{i}]: hit_rate={hit_rate:.2%}, size={size_mb:.1f}MB, items={num_items}")

        # Save checkpoint
        is_best = val_losses['total'] < best_val_loss
        if is_best:
            best_val_loss = val_losses['total']
            patience_counter = 0
        else:
            patience_counter += 1

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': raw_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'train_losses': train_losses,
            'val_losses': val_losses,
            'best_val_loss': best_val_loss,
            'patience_counter': patience_counter,
            'exist_pos_weight': exist_pos_weight.item(),
            'args': vars(args),
        }

        if is_main:
            torch.save(checkpoint, out_dir / 'latest.pth')
            if is_best:
                torch.save(checkpoint, out_dir / 'best.pth')
                print(f"  * New best model saved (val_loss={best_val_loss:.4f})")
            else:
                patience_str = f'  [{patience_counter}/{args.patience}]' if args.patience > 0 else ''
                print(f"  Best so far: val_loss={best_val_loss:.4f}{patience_str}")

        # Per-epoch visualization (main process only)
        if is_main and args.viz_every > 0 and epoch % args.viz_every == 0 and viz_indices and viz_dataset:
            viz_dir = out_dir / 'viz' / f'epoch_{epoch:04d}'
            viz_dir.mkdir(parents=True, exist_ok=True)
            model.eval()
            with torch.no_grad():
                for vi, idx in enumerate(viz_indices):
                    sample = viz_dataset[idx]
                    image = sample['image'].unsqueeze(0).to(device)
                    azimuth_map = sample['azimuth_map'].unsqueeze(0).to(device)
                    continuous_has_data = sample['continuous_has_data'].numpy()
                    has_data = sample['has_data'].numpy()
                    rgb_path = viz_dataset.samples[idx][0] if idx < len(viz_dataset.samples) else None
                    elev_path = viz_dataset.samples[idx][1] if idx < len(viz_dataset.samples) else None
                    # Reload raw GT from NPZ to avoid interpolation artifacts in visualisation.
                    if elev_path and os.path.exists(elev_path):
                        from .dataset import _extract_radar_dist as _erd
                        _ed = np.load(elev_path, allow_pickle=False)
                        gt_dist = _erd(_ed['profile_data'], _ed['profile_offsets'], args.clearance_height)
                    else:
                        gt_dist = sample['radar_dist'].numpy()
                    pred_dist, pred_exist_logit, _ = model(
                        image, azimuth_map if args.use_azimuth else None
                    )
                    pred_dist = pred_dist[0].cpu().numpy()
                    pred_exist_logit = pred_exist_logit[0].cpu().numpy()
                    rgb_path = viz_dataset.samples[idx][0] if idx < len(viz_dataset.samples) else None
                    elev_path = viz_dataset.samples[idx][1] if idx < len(viz_dataset.samples) else None
                    visualize_sample(
                        sample['image'], gt_dist, pred_dist, pred_exist_logit,
                        continuous_has_data, has_data, vi, viz_dir,
                        rgb_path=rgb_path, elev_path=elev_path,
                    )
            print(f"  Visualizations saved to {viz_dir}")

        history.append({
            'epoch': epoch,
            'train': train_losses,
            'val': val_losses,
        })

        # Early stopping check (broadcast decision from rank-0 in DDP)
        if args.patience > 0 and patience_counter >= args.patience:
            if is_ddp:
                stop_flag = torch.tensor(1, device=device)
                dist.broadcast(stop_flag, src=0)
            if is_main:
                print(f"\n  Early stopping: val loss has not improved for "
                      f"{patience_counter} consecutive epochs.")
            break

    # Final summary
    if is_main:
        stopped_early = args.patience > 0 and patience_counter >= args.patience
        print("\n" + "=" * 70)
        print("Training Complete" + (" (early stopping)" if stopped_early else ""))
        print("=" * 70)
        print(f"Best val_loss: {best_val_loss:.4f}")
        print(f"Checkpoints:   {out_dir}")
        if history:
            print(f"\nLoss curve:")
            print(f"  {'Epoch':>5s}  {'Train':>10s}  {'Val':>10s}  {'Dist(T)':>10s}  {'Exist(T)':>10s}")
            for h in history:
                marker = ' *' if h['val']['total'] == best_val_loss else ''
                print(
                    f"  {h['epoch']:>5d}"
                    f"  {h['train']['total']:>10.4f}"
                    f"  {h['val']['total']:>10.4f}"
                    f"  {h['train']['radar_dist']:>10.4f}"
                    f"  {h['train']['exist']:>10.4f}"
                    f"{marker}"
                )

    if is_ddp:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
