"""可通行性模型评估脚本，支持多指标计算和多模型对比。

Loads one or more trained checkpoints, runs inference over the validation set
(or a specified split), and computes distance/existence metrics. Results are
printed to stdout and optionally saved to JSON and CSV for multi-model comparison.

Metrics computed:
    dist_mae      Mean Absolute Error of distance predictions (m)
    dist_rmse     Root Mean Squared Error (m)
    dist_median   Median AE (m), robust to outliers
    dist_p90      90th-percentile AE (m)
    exist_acc     FOV-membership accuracy over all 360 bins
    exist_prec    Precision: predicted-positive that are truly in FOV
    exist_rec     Recall: true FOV bins correctly predicted
    exist_f1      F1 score

Distance metrics are computed only on bins where both continuous_has_data
and has_data are True (i.e., the bin is within the continuous FOV AND has
actual point cloud data providing a GT distance).

Usage (single checkpoint):
    python -m scripts.traversability.evaluate \\
        --checkpoints path/to/best.pth \\
        --data-root   /data/UniNav-DB \\
        --curated-json /data/UniNav-DB/splits/pinhole_curated.json \\
        --backbone-weights model/dinov3_vitb16_pretrain.pth \\
        --output-json  path/to/eval.json

Usage (multi-model comparison):
    python -m scripts.traversability.evaluate \\
        --checkpoints trav/best.pth resnet/best.pth dino/best.pth \\
        --data-root   /data/UniNav-DB \\
        --curated-json /data/UniNav-DB/splits/pinhole_curated.json \\
        --backbone-weights model/dinov3_vitb16_pretrain.pth \\
        --output-csv   path/to/comparison.csv
"""

import argparse
import csv
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.amp import autocast

from .train import build_model
from .dataset import UniNavDataset, collate_fn


# ---------------------------------------------------------------------------
# DDP Helpers
# ---------------------------------------------------------------------------

def _setup_ddp():
    """Initialize DDP if running under torchrun, otherwise return single-GPU setup.

    Returns:
        (rank, world_size, device) tuple.
    """
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))

        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')
        return rank, world_size, device
    else:
        # Single GPU mode
        return 0, 1, torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _cleanup_ddp():
    """Cleanup DDP process group if initialized."""
    if dist.is_initialized():
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_time(seconds: float) -> str:
    """Format elapsed seconds to a human-readable string."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    m, s = divmod(int(seconds), 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m{s:02d}s"


def _progress_bar(current: int, total: int, width: int = 30) -> str:
    """Return a simple ASCII progress bar string."""
    frac = current / max(total, 1)
    filled = int(width * frac)
    bar = '\u2588' * filled + '\u2591' * (width - filled)
    return f"|{bar}| {current}/{total}"


# ---------------------------------------------------------------------------
# Dataset construction (mirrors curated-JSON logic from train.py)
# ---------------------------------------------------------------------------

def _build_datasets(args_eval, split: str, tmp_dir: Path):
    """Construct UniNavDataset list from curated JSON or explicit dataset args.

    Args:
        args_eval: Parsed CLI namespace with data_root, curated_json, datasets,
            splits_jsons, img_size, clearance_height, scenes.
        split: 'train', 'val', or 'all'.
        tmp_dir: Temporary directory for per-dataset splits JSONs when using
            curated JSON.

    Returns:
        List of UniNavDataset objects.
    """
    if args_eval.curated_json is not None:
        curated_path = Path(args_eval.curated_json)
        if not curated_path.exists():
            print(f"[ERROR] Curated JSON not found: {curated_path}")
            sys.exit(1)
        with open(curated_path, "r", encoding="utf-8") as fh:
            curated = json.load(fh)

        dataset_list = []
        for ds_entry in curated.get("datasets", []):
            ds_name = ds_entry["dataset_name"]
            sj = {
                "dataset_name": ds_name,
                "camera_type":  ds_entry.get("camera_type", "pinhole"),
                "source":       ds_entry.get("source", ""),
                "scenes":       ds_entry.get("scenes", []),
            }
            sj_path = tmp_dir / f"{ds_name}.json"
            with open(sj_path, "w", encoding="utf-8") as fh:
                json.dump(sj, fh, ensure_ascii=False)

            scenes_in_json = [s["scene_id"] for s in sj["scenes"]]
            if split == 'all':
                scenes = scenes_in_json
            else:
                n_val = max(1, int(len(scenes_in_json) * 0.1))
                val_scenes = set(scenes_in_json[-n_val:])
                if split == 'val':
                    scenes = [s for s in scenes_in_json if s in val_scenes]
                else:
                    scenes = [s for s in scenes_in_json if s not in val_scenes]

            if args_eval.scenes:
                scenes = [s for s in scenes if s in args_eval.scenes]

            if not scenes:
                continue

            ds = UniNavDataset(
                data_root=args_eval.data_root,
                dataset_name=ds_name,
                splits_json=str(sj_path),
                scenes=scenes,
                img_size=args_eval.img_size,
                clearance_height=args_eval.clearance_height,
            )
            dataset_list.append(ds)
        return dataset_list

    # Fallback: use --datasets / --splits-jsons
    splits_jsons = args_eval.splits_jsons or []
    datasets = args_eval.datasets or ['DL3DV-1k']
    dataset_list = []
    for i, ds_name in enumerate(datasets):
        sj_path = splits_jsons[i] if i < len(splits_jsons) else None
        if sj_path is None:
            auto = Path(args_eval.data_root) / 'splits' / f'{ds_name}.json'
            if auto.exists():
                sj_path = str(auto)
            else:
                print(f"[WARN] No splits JSON for {ds_name}, skipping.")
                continue

        with open(sj_path, "r", encoding="utf-8") as fh:
            sj = json.load(fh)
        scenes_in_json = [s["scene_id"] for s in sj.get("scenes", [])]
        if split == 'all':
            scenes = scenes_in_json
        else:
            n_val = max(1, int(len(scenes_in_json) * 0.1))
            val_scenes = set(scenes_in_json[-n_val:])
            if split == 'val':
                scenes = [s for s in scenes_in_json if s in val_scenes]
            else:
                scenes = [s for s in scenes_in_json if s not in val_scenes]

        if args_eval.scenes:
            scenes = [s for s in scenes if s in args_eval.scenes]

        if not scenes:
            continue
        ds = UniNavDataset(
            data_root=args_eval.data_root,
            dataset_name=ds_name,
            splits_json=sj_path,
            scenes=scenes,
            img_size=args_eval.img_size,
            clearance_height=args_eval.clearance_height,
        )
        dataset_list.append(ds)
    return dataset_list


# ---------------------------------------------------------------------------
# Single-checkpoint evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_checkpoint(ckpt_path: str, datasets, args_eval, device, rank: int, world_size: int) -> dict:
    """Run full evaluation of one checkpoint over the given datasets.

    Args:
        ckpt_path: Path to .pth checkpoint.
        datasets: List of UniNavDataset objects.
        args_eval: Parsed CLI namespace (for batch_size, num_workers, use_azimuth, save_predictions).
        device: torch.device.
        rank: DDP rank (0 for single GPU).
        world_size: DDP world size (1 for single GPU).

    Returns:
        Dict with checkpoint metadata and computed metrics.
    """
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    saved = ckpt.get('args', {})

    # Build args namespace for build_model, merging saved args with CLI overrides
    model_args = SimpleNamespace(
        model=saved.get('model', 'traversability'),
        backbone_weights=(
            args_eval.backbone_weights
            if args_eval.backbone_weights
            else saved.get('backbone_weights')
        ),
        backbone_type=saved.get('backbone_type', 'vitb16'),
        freeze_backbone=True,
        unfreeze_last_n=0,
        multi_scale=saved.get('multi_scale', True),
    )

    model = build_model(model_args).to(device)
    model.load_state_dict(ckpt['model_state_dict'])

    # Wrap with DDP if multi-GPU
    if world_size > 1:
        model = DDP(model, device_ids=[device.index])

    model.eval()

    epoch = ckpt.get('epoch', '?')
    val_losses = ckpt.get('val_losses', {})

    # Create DataLoaders with DistributedSampler if multi-GPU
    loaders = []
    for ds in datasets:
        sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank, shuffle=False) if world_size > 1 else None
        loader = DataLoader(
            ds,
            batch_size=args_eval.batch_size,
            shuffle=False if sampler else False,
            sampler=sampler,
            num_workers=args_eval.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
            drop_last=False,
        )
        loaders.append(loader)

    total_batches = sum(len(ld) for ld in loaders)

    # Prepare output directories if saving predictions
    if args_eval.save_predictions:
        # Use parent directory name + checkpoint stem for unique model identification
        # e.g., /path/to/traversability_equirect_no_azimuth/best.pth -> traversability_equirect_no_azimuth
        ckpt_path_obj = Path(ckpt_path)
        model_name = ckpt_path_obj.parent.name
        pred_dir = Path(args_eval.data_root) / 'val' / model_name
        gt_dir = Path(args_eval.data_root) / 'val' / 'ground_truth'
        if rank == 0:
            pred_dir.mkdir(parents=True, exist_ok=True)
            gt_dir.mkdir(parents=True, exist_ok=True)
        # Synchronize all ranks after directory creation
        if world_size > 1:
            dist.barrier()
    else:
        pred_dir = None
        gt_dir = None

    # Accumulation arrays
    all_errors = []      # per-bin absolute distance errors (float32)
    sq_errors  = []      # squared errors for RMSE
    exist_tp   = 0       # true positives  (pred=1, gt=1)
    exist_fp   = 0       # false positives (pred=1, gt=0)
    exist_fn   = 0       # false negatives (pred=0, gt=1)
    exist_tn   = 0       # true negatives  (pred=0, gt=0)
    n_samples  = 0
    n_batches  = 0
    t_start    = time.time()

    for loader in loaders:
        for batch in loader:
            images              = batch['image'].to(device)
            azimuth_map         = batch['azimuth_map'].to(device)
            radar_dist_gt       = batch['radar_dist'].to(device)          # (B, 360) float, NaN outside FOV
            continuous_has_data = batch['continuous_has_data'].to(device) # (B, 360) bool
            has_data            = batch['has_data'].to(device)            # (B, 360) bool
            rgb_paths           = batch.get('rgb_path', [])               # List of image paths
            scene_ids           = batch.get('scene_id', [])               # List of scene IDs

            with autocast(device_type='cuda', dtype=torch.float16):
                _ignored, pred_exist_logit, raw_dist = model(
                    images, azimuth_map if args_eval.use_azimuth else None
                )

            # Save predictions and ground truth if requested (all ranks save their own samples)
            if args_eval.save_predictions and pred_dir is not None:
                pred_dist_np = raw_dist.cpu().numpy()  # (B, 360)
                pred_exist_np = (pred_exist_logit > 0).cpu().numpy()  # (B, 360) bool
                gt_dist_np = radar_dist_gt.cpu().numpy()  # (B, 360)
                gt_has_data_np = continuous_has_data.cpu().numpy()  # (B, 360)
                gt_has_data_raw_np = has_data.cpu().numpy()  # (B, 360)

                for i in range(len(rgb_paths)):
                    rgb_path = rgb_paths[i]
                    scene_id = scene_ids[i]
                    frame_name = Path(rgb_path).stem

                    # Save prediction
                    pred_file = pred_dir / f'{frame_name}.npz'
                    np.savez_compressed(
                        pred_file,
                        distance=pred_dist_np[i],
                        has_data=pred_exist_np[i],
                        scene_id=scene_id,
                        rgb_path=rgb_path,
                    )

                    # Save ground truth (only once, check if exists)
                    gt_file = gt_dir / f'{frame_name}.npz'
                    if not gt_file.exists():
                        np.savez_compressed(
                            gt_file,
                            distance=gt_dist_np[i],
                            continuous_has_data=gt_has_data_np[i],
                            has_data=gt_has_data_raw_np[i],
                            scene_id=scene_id,
                            rgb_path=rgb_path,
                        )

            # Distance metrics mask: in FOV AND has actual point cloud data
            dist_mask = (continuous_has_data & has_data)
            if dist_mask.any():
                pred_d = raw_dist[dist_mask].float()
                gt_d   = radar_dist_gt[dist_mask].float()
                err    = (pred_d - gt_d).abs()
                all_errors.append(err.cpu())
                sq_errors.append(err.pow(2).cpu())

            # Existence metrics (over all 360 bins)
            pred_pos = (pred_exist_logit > 0)   # (B, 360) bool
            gt_pos   = continuous_has_data       # (B, 360) bool
            exist_tp += int((pred_pos &  gt_pos).sum().item())
            exist_fp += int((pred_pos & ~gt_pos).sum().item())
            exist_fn += int((~pred_pos &  gt_pos).sum().item())
            exist_tn += int((~pred_pos & ~gt_pos).sum().item())

            n_samples += images.size(0)
            n_batches += 1

            if rank == 0:  # Only print progress on rank 0
                elapsed = time.time() - t_start
                bar = _progress_bar(n_batches, total_batches)
                print(
                    f"\r  {bar} [{_format_time(elapsed)}] {n_samples} samples",
                    end='', flush=True,
                )

    if rank == 0:
        print()

    # Gather results from all ranks if multi-GPU
    if world_size > 1:
        # Concatenate all_errors and sq_errors across ranks
        if all_errors:
            local_errors = torch.cat(all_errors).cpu()
            local_sq_errors = torch.cat(sq_errors).cpu()
        else:
            local_errors = torch.tensor([], dtype=torch.float32)
            local_sq_errors = torch.tensor([], dtype=torch.float32)

        # Use all_gather_object for variable-length tensors
        gathered_errors_list = [None] * world_size
        gathered_sq_errors_list = [None] * world_size
        dist.all_gather_object(gathered_errors_list, local_errors)
        dist.all_gather_object(gathered_sq_errors_list, local_sq_errors)

        # Gather scalar metrics using all_reduce
        local_metrics = torch.tensor([exist_tp, exist_fp, exist_fn, exist_tn, n_samples], dtype=torch.long, device=device)
        dist.all_reduce(local_metrics, op=dist.ReduceOp.SUM)

        if rank == 0:
            all_errors = [e for e in gathered_errors_list if e.numel() > 0]
            sq_errors = [e for e in gathered_sq_errors_list if e.numel() > 0]
            exist_tp = int(local_metrics[0].item())
            exist_fp = int(local_metrics[1].item())
            exist_fn = int(local_metrics[2].item())
            exist_tn = int(local_metrics[3].item())
            n_samples = int(local_metrics[4].item())
        else:
            # Non-rank-0 processes return empty results
            return {}

    # Aggregate distance metrics (only on rank 0 in multi-GPU)
    if all_errors:
        errors = torch.cat(all_errors)
        dist_mae    = errors.mean().item()
        dist_rmse   = errors.pow(2).mean().sqrt().item() if not sq_errors else \
                      torch.cat(sq_errors).mean().sqrt().item()
        dist_median = errors.median().item()
        dist_p90    = errors.quantile(0.9).item()
        n_dist_bins = errors.numel()
    else:
        dist_mae = dist_rmse = dist_median = dist_p90 = float('nan')
        n_dist_bins = 0

    # Existence metrics
    total_exist = exist_tp + exist_fp + exist_fn + exist_tn
    exist_acc   = (exist_tp + exist_tn) / max(total_exist, 1)
    prec_denom  = exist_tp + exist_fp
    rec_denom   = exist_tp + exist_fn
    exist_prec  = exist_tp / prec_denom if prec_denom > 0 else float('nan')
    exist_rec   = exist_tp / rec_denom  if rec_denom  > 0 else float('nan')
    if not (np.isnan(exist_prec) or np.isnan(exist_rec)) and (exist_prec + exist_rec) > 0:
        exist_f1 = 2 * exist_prec * exist_rec / (exist_prec + exist_rec)
    else:
        exist_f1 = float('nan')

    return {
        'checkpoint':    ckpt_path,
        'model':         model_args.model,
        'epoch':         epoch,
        'val_loss_ckpt': val_losses.get('total', None),
        'n_samples':     n_samples,
        'n_dist_bins':   n_dist_bins,
        'dist_mae':      dist_mae,
        'dist_rmse':     dist_rmse,
        'dist_median':   dist_median,
        'dist_p90':      dist_p90,
        'exist_acc':     exist_acc,
        'exist_prec':    exist_prec,
        'exist_rec':     exist_rec,
        'exist_f1':      exist_f1,
    }


# ---------------------------------------------------------------------------
# Result printing
# ---------------------------------------------------------------------------

def _print_results(results: list[dict]) -> None:
    """Print a formatted comparison table to stdout.

    Args:
        results: List of result dicts from evaluate_checkpoint.
    """
    headers = [
        'checkpoint', 'model', 'epoch',
        'dist_mae', 'dist_rmse', 'dist_median', 'dist_p90',
        'exist_acc', 'exist_prec', 'exist_rec', 'exist_f1',
        'n_samples',
    ]
    rows = []
    for r in results:
        rows.append([
            Path(r['checkpoint']).name,
            r['model'],
            str(r['epoch']),
            f"{r['dist_mae']:.4f}"    if not np.isnan(r['dist_mae'])    else 'nan',
            f"{r['dist_rmse']:.4f}"   if not np.isnan(r['dist_rmse'])   else 'nan',
            f"{r['dist_median']:.4f}" if not np.isnan(r['dist_median']) else 'nan',
            f"{r['dist_p90']:.4f}"    if not np.isnan(r['dist_p90'])    else 'nan',
            f"{r['exist_acc']:.4f}"   if not np.isnan(r['exist_acc'])   else 'nan',
            f"{r['exist_prec']:.4f}"  if not np.isnan(r['exist_prec'])  else 'nan',
            f"{r['exist_rec']:.4f}"   if not np.isnan(r['exist_rec'])   else 'nan',
            f"{r['exist_f1']:.4f}"    if not np.isnan(r['exist_f1'])    else 'nan',
            str(r['n_samples']),
        ])

    col_widths = [max(len(h), max((len(row[i]) for row in rows), default=0))
                  for i, h in enumerate(headers)]

    def _fmt_row(cells):
        return '  '.join(c.ljust(w) for c, w in zip(cells, col_widths))

    sep = '  '.join('-' * w for w in col_widths)
    print()
    print(_fmt_row(headers))
    print(sep)
    for row in rows:
        print(_fmt_row(row))
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    """CLI entry point for evaluate.py."""
    parser = argparse.ArgumentParser(
        description='Evaluate traversability model checkpoints.',
    )
    parser.add_argument('--checkpoints', nargs='+', required=True,
                        help='One or more .pth checkpoint paths to evaluate.')
    parser.add_argument('--data-root', required=True,
                        help='UniNav-DB root directory.')
    parser.add_argument('--curated-json', default=None,
                        help='Path to curated splits JSON (same format as train.py). '
                             'When provided, --datasets and --splits-jsons are ignored.')
    parser.add_argument('--datasets', nargs='+', default=['DL3DV-1k'],
                        help='Dataset name(s) to evaluate (default: DL3DV-1k).')
    parser.add_argument('--splits-jsons', nargs='+', default=None,
                        help='Splits JSON paths, one per dataset.')
    parser.add_argument('--scenes', nargs='+', default=None,
                        help='Specific scene IDs to include (default: all).')
    parser.add_argument('--backbone-weights', default=None,
                        help='DINOv3 pretrained weights path '
                             '(overrides path stored in checkpoint args).')
    parser.add_argument('--img-size', type=int, default=384,
                        help='Image size for preprocessing (default: 384).')
    parser.add_argument('--clearance-height', type=float, default=0.5,
                        help='Passable-gap threshold in metres (default: 0.5).')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Inference batch size (default: 16).')
    parser.add_argument('--num-workers', type=int, default=8,
                        help='DataLoader worker count (default: 8).')
    parser.add_argument('--device', default='cuda',
                        help='Compute device: cuda or cpu.')
    parser.add_argument('--split', default='val',
                        choices=['train', 'val', 'all'],
                        help='Which data split to evaluate (default: val).')
    parser.add_argument('--no-azimuth', dest='use_azimuth', action='store_false',
                        default=True,
                        help='Disable azimuth map input to model (ablation).')
    parser.add_argument('--output-json', default=None,
                        help='Write full results list to this JSON file.')
    parser.add_argument('--output-csv', default=None,
                        help='Write per-checkpoint summary row to this CSV file.')
    parser.add_argument('--save-predictions', action='store_true',
                        help='Save per-frame predictions to {data_root}/val/{model_name}/ '
                             'and ground truth to {data_root}/val/ground_truth/.')

    args = parser.parse_args()

    # Setup DDP
    rank, world_size, device = _setup_ddp()

    if rank == 0:
        print(f"Device: {device} | Rank: {rank}/{world_size}")

    # Build datasets once; reused for all checkpoints
    tmp_dir = Path(tempfile.mkdtemp(prefix='eval_splits_'))
    datasets = _build_datasets(args, args.split, tmp_dir)

    if not datasets:
        if rank == 0:
            print("[ERROR] No valid datasets found. Check --data-root and --curated-json.")
        _cleanup_ddp()
        sys.exit(1)

    total_samples = sum(len(ds) for ds in datasets)
    if rank == 0:
        print(f"Split: {args.split} | Datasets: {len(datasets)} | "
              f"Total samples: {total_samples}")

    results = []
    for ckpt_path in args.checkpoints:
        if rank == 0:
            print(f"\nEvaluating: {ckpt_path}")
        r = evaluate_checkpoint(ckpt_path, datasets, args, device, rank, world_size)

        # Only rank 0 has valid results in multi-GPU mode
        if rank == 0:
            results.append(r)
            print(
                f"  dist_mae={r['dist_mae']:.4f}m  dist_rmse={r['dist_rmse']:.4f}m  "
                f"dist_median={r['dist_median']:.4f}m  dist_p90={r['dist_p90']:.4f}m  "
                f"exist_acc={r['exist_acc']:.4f}  exist_f1={r['exist_f1']:.4f}"
            )

    if rank == 0:
        _print_results(results)

        if args.output_json:
            out_path = Path(args.output_json)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, 'w', encoding='utf-8') as fh:
                json.dump(results, fh, indent=2, ensure_ascii=False)
            print(f"JSON results saved to: {out_path}")

        if args.output_csv:
            out_path = Path(args.output_csv)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            fieldnames = [
                'checkpoint', 'model', 'epoch',
                'dist_mae', 'dist_rmse', 'dist_median', 'dist_p90',
                'exist_acc', 'exist_prec', 'exist_rec', 'exist_f1',
                'n_samples', 'n_dist_bins', 'val_loss_ckpt',
            ]
            write_header = not out_path.exists()
            with open(out_path, 'a', newline='', encoding='utf-8') as fh:
                writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction='ignore')
                if write_header:
                    writer.writeheader()
                writer.writerows(results)
            print(f"CSV results appended to: {out_path}")

        if args.save_predictions:
            print(f"Predictions saved to: {Path(args.data_root) / 'val'}")

    _cleanup_ddp()


if __name__ == '__main__':
    main()
