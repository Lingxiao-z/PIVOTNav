"""可通行性模型预测可视化。

Generates comparison plots of model predictions vs ground truth:
  - Radar distance: polar radar plot (pred vs GT)
  - Per-sample grid output as PNG

Usage:
    conda activate da3
    python -m scripts.traversability.visualize \
        --checkpoint checkpoints/traversability_test/best.pth \
        --data-root /data/users/renhao/UniNav-DB \
        --dataset DL3DV-1k \
        --scenes 1 \
        --backbone-weights model/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth \
        --num-samples 8 \
        --output-dir visualizations/traversability
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

from .model import TraversabilityModel
from .dataset import UniNavDataset, _extract_radar_dist


# ImageNet denormalization for display
MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])

# Clearance reference rings (metres) and their display colours
CLEARANCE_RINGS = [0.5, 1.0, 2.0]
CLEARANCE_COLORS = ['#FFD700', '#FF8C00', '#FF1744']


def denormalize_image(tensor: torch.Tensor) -> np.ndarray:
    """Convert normalized image tensor (3, H, W) back to displayable (H, W, 3) uint8."""
    img = tensor.cpu().numpy().transpose(1, 2, 0)
    img = img * STD + MEAN
    img = np.clip(img, 0, 1)
    return (img * 255).astype(np.uint8)


def plot_polar_comparison(ax, angles_rad, gt, pred, continuous_has_data, title, max_r=None, unit='m'):
    """Plot GT vs prediction on a polar axis.

    Args:
        ax: Matplotlib polar axis.
        angles_rad: (360,) angles in radians.
        gt: (360,) ground truth values.
        pred: (360,) predicted values.
        continuous_has_data: (360,) bool, True for valid bins.
        title: Plot title.
        max_r: Maximum radial value (auto if None).
        unit: Unit label.
    """
    # GT: FOV 外设为 NaN
    gt_vis = gt.copy()
    gt_vis[~continuous_has_data] = np.nan

    # Pred: 分为 FOV 内和全部
    pred_fov = pred.copy()
    pred_fov[~continuous_has_data] = np.nan

    # 绘制
    ax.plot(angles_rad, gt_vis, color='#2196F3', linewidth=1.5, label='GT', alpha=0.8)
    ax.plot(angles_rad, pred_fov, color='#FF5722', linewidth=1.5, label='Pred (FOV)', alpha=0.8)
    ax.plot(angles_rad, pred, color='#FF5722', linewidth=0.8, linestyle='--',
            label='Pred (All)', alpha=0.4)

    # Fill between (只在 FOV 内)
    valid = continuous_has_data
    if valid.any():
        ax.fill_between(
            angles_rad, gt_vis, pred_fov,
            where=valid, alpha=0.15, color='#FF5722'
        )

    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)

    if max_r is not None:
        ax.set_rlim(0, max_r)

    ax.set_title(title, fontsize=11, fontweight='bold', pad=12)
    ax.legend(loc='upper right', fontsize=8)

    theta_ring = np.linspace(-np.pi, np.pi, 361)
    for ring_h, color in zip(CLEARANCE_RINGS, CLEARANCE_COLORS):
        if max_r is not None and ring_h <= max_r:
            ax.plot(theta_ring, np.full_like(theta_ring, ring_h),
                    color=color, linestyle='--', linewidth=0.9, alpha=0.7)
            ax.annotate(f'{ring_h}m', xy=(np.pi / 4, ring_h), fontsize=6,
                        color=color, fontweight='bold')


def plot_linear_comparison(ax, angles_deg, gt, pred, continuous_has_data, title, ylabel, ylim=None):
    """Plot GT vs prediction as a linear chart over angle.

    Args:
        ax: Matplotlib axis.
        angles_deg: (360,) angles in degrees.
        gt, pred: (360,) values.
        continuous_has_data: (360,) bool.
        title: Plot title.
        ylabel: Y-axis label.
        ylim: Y-axis limits tuple.
    """
    # GT: FOV 外设为 NaN
    gt_vis = gt.copy()
    gt_vis[~continuous_has_data] = np.nan

    # Pred: 分为 FOV 内和全部
    pred_fov = pred.copy()
    pred_fov[~continuous_has_data] = np.nan

    # 绘制
    ax.plot(angles_deg, gt_vis, color='#2196F3', linewidth=1.2, label='GT', alpha=0.8)
    ax.plot(angles_deg, pred_fov, color='#FF5722', linewidth=1.2, label='Pred (FOV)', alpha=0.8)
    ax.plot(angles_deg, pred, color='#FF5722', linewidth=0.8, linestyle='--',
            label='Pred (All)', alpha=0.4)

    # Fill between (只在 FOV 内)
    valid_idx = np.where(continuous_has_data)[0]
    if len(valid_idx) > 0:
        ax.fill_between(
            angles_deg, gt_vis, pred_fov,
            where=continuous_has_data, alpha=0.15, color='#FF5722'
        )

    fov_transitions = np.diff(continuous_has_data.astype(int))
    starts = np.where(fov_transitions == 1)[0]
    ends = np.where(fov_transitions == -1)[0]
    for s in starts:
        ax.axvline(angles_deg[s], color='gray', linestyle='--', alpha=0.4, linewidth=0.8)
    for e in ends:
        ax.axvline(angles_deg[e], color='gray', linestyle='--', alpha=0.4, linewidth=0.8)

    ax.set_xlabel('Angle (deg)')
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11, fontweight='bold')
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    for ring_h, color in zip(CLEARANCE_RINGS, CLEARANCE_COLORS):
        if ylim is None or ring_h <= ylim[1]:
            ax.axhline(y=ring_h, color=color, linestyle='--', linewidth=0.9,
                        alpha=0.7, label=f'{ring_h}m')


def plot_multi_clearance_radar(ax, angles_rad, profile_data, profile_offsets,
                               continuous_has_data):
    """Plot multi-clearance radar lines on a polar axis.

    Args:
        ax: Matplotlib polar axis.
        angles_rad: (360,) angles in radians.
        profile_data: (N, 2) float32 from elevation NPZ.
        profile_offsets: (361,) int32 CSR row offsets.
        continuous_has_data: (360,) bool, True for valid bins.
    """
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)

    for ring_h, color in zip(CLEARANCE_RINGS, CLEARANCE_COLORS):
        radar_dist = _extract_radar_dist(profile_data, profile_offsets, ring_h)
        finite = np.isfinite(radar_dist) & continuous_has_data
        if finite.any():
            ax.plot(angles_rad[finite], radar_dist[finite],
                    color=color, linewidth=1.2, alpha=0.8, label=f'>{ring_h}m')
            ax.scatter(angles_rad[finite], radar_dist[finite],
                       color=color, s=3, alpha=0.6, edgecolors='none')

    ax.set_title('Multi-Clearance Radar (GT)', fontsize=11, fontweight='bold', pad=12)
    ax.legend(loc='upper right', fontsize=8, title='Clearance')


def visualize_sample(image_tensor, gt_dist, pred_dist, pred_exist_logit,
                     continuous_has_data, has_data, sample_idx, output_dir, rgb_path=None,
                     elev_path=None):
    """Generate a single-sample visualization.

    Layout: 2 cols, 3 rows
        Row 1: RGB image (spans 2 cols)
        Row 2: Radar dist polar | Radar dist linear
        Row 3: Multi-clearance radar (GT) | Error stats
    """
    angles_deg = np.arange(360) - 180.0
    angles_rad = np.deg2rad(angles_deg)

    # Auto-scale: use 95th percentile of valid (finite, non-inf) values, with a floor of 5m
    valid_vals = []
    for arr in (gt_dist, pred_dist):
        v = arr[continuous_has_data & np.isfinite(arr) & (arr < 99.0)]
        if len(v) > 0:
            valid_vals.append(v)
    if valid_vals:
        all_valid = np.concatenate(valid_vals)
        auto_max = max(float(np.percentile(all_valid, 95)) * 1.1, 5.0)
    else:
        auto_max = 20.0

    has_elev = elev_path is not None and Path(elev_path).exists()
    fig = plt.figure(figsize=(14, 12))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.2, 1, 1], hspace=0.3, wspace=0.3)

    # --- Row 1: RGB image (spans 2 cols) ---
    ax_rgb = fig.add_subplot(gs[0, :])
    rgb = denormalize_image(image_tensor)
    ax_rgb.imshow(rgb)
    title = f'Sample {sample_idx}'
    if rgb_path:
        title += f' - {Path(rgb_path).name}'
    ax_rgb.set_title(title, fontsize=13, fontweight='bold')
    ax_rgb.axis('off')

    # --- Row 2 Left: Radar distance polar ---
    ax_dist_polar = fig.add_subplot(gs[1, 0], projection='polar')
    gt_dist_disp = gt_dist.copy()
    gt_dist_disp[np.isinf(gt_dist_disp)] = auto_max  # inf → auto_max for display
    gt_dist_disp = np.clip(gt_dist_disp, 0, auto_max)
    pred_dist_disp = np.clip(pred_dist, 0, auto_max)
    plot_polar_comparison(
        ax_dist_polar, angles_rad, gt_dist_disp, pred_dist_disp, continuous_has_data,
        'Radar Distance (Polar)', max_r=auto_max, unit='m'
    )

    # --- Row 2 Right: Radar distance linear ---
    ax_dist_lin = fig.add_subplot(gs[1, 1])
    gt_dist_lin = gt_dist.copy()
    gt_dist_lin[np.isinf(gt_dist_lin)] = auto_max  # inf → auto_max for display
    plot_linear_comparison(
        ax_dist_lin, angles_deg, gt_dist_lin, pred_dist, continuous_has_data,
        'Radar Distance vs Angle', 'Distance (m)', ylim=(0, auto_max)
    )

    # --- Row 3 Left: Multi-clearance radar (GT) ---
    if has_elev:
        elev_data = np.load(elev_path)
        ax_radar = fig.add_subplot(gs[2, 0], projection='polar')
        plot_multi_clearance_radar(
            ax_radar, angles_rad,
            elev_data['profile_data'], elev_data['profile_offsets'],
            continuous_has_data,
        )
    else:
        # Empty placeholder
        ax_empty = fig.add_subplot(gs[2, 0])
        ax_empty.text(0.5, 0.5, 'Multi-clearance data not available',
                     ha='center', va='center', fontsize=11, color='gray')
        ax_empty.axis('off')

    # --- Row 3 Right: Error statistics ---
    ax_stats = fig.add_subplot(gs[2, 1])
    ax_stats.axis('off')

    # 计算存在性误差（预测 FOV 范围）
    exist_prob = 1 / (1 + np.exp(-pred_exist_logit))
    exist_pred = exist_prob > 0.5

    # GT: continuous_has_data (FOV 范围)
    exist_gt = continuous_has_data
    exist_acc = (exist_gt == exist_pred).mean()

    tp = ((exist_gt == True) & (exist_pred == True)).sum()
    fp = ((exist_gt == False) & (exist_pred == True)).sum()
    fn = ((exist_gt == True) & (exist_pred == False)).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    exist_stats = (
        f"  Accuracy:  {exist_acc:.3f}\n"
        f"  Precision: {precision:.3f}\n"
        f"  Recall:    {recall:.3f}\n"
        f"  TP/FP/FN:  {tp}/{fp}/{fn}"
    )

    # 计算距离误差（只在 has_data=True 的 bin 上）
    fov = continuous_has_data
    if fov.any():
        has_data_mask = fov & has_data
        if has_data_mask.any():
            obs_dist_err = np.abs(pred_dist[has_data_mask] - gt_dist[has_data_mask])
            dist_stats = (
                f"  MAE:    {obs_dist_err.mean():.3f} m\n"
                f"  Median: {np.median(obs_dist_err):.3f} m\n"
                f"  Max:    {obs_dist_err.max():.3f} m\n"
                f"  Bins:   {has_data_mask.sum()}"
            )
        else:
            dist_stats = "  No data in FOV"
    else:
        dist_stats = "  No FOV data"

    stats_text = (
        f"FOV bins: {fov.sum()} / 360\n\n"
        f"Existence Prediction (FOV):\n"
        f"{exist_stats}\n\n"
        f"Distance Error (has_data=True only):\n"
        f"{dist_stats}"
    )

    ax_stats.text(0.05, 0.95, stats_text, transform=ax_stats.transAxes,
                  fontsize=11, verticalalignment='top', fontfamily='monospace',
                  bbox=dict(boxstyle='round,pad=0.8', facecolor='#f0f0f0', alpha=0.9))
    ax_stats.set_title('Error Statistics', fontsize=12, fontweight='bold')

    plt.tight_layout()

    out_path = Path(output_dir) / f'sample_{sample_idx:04d}.png'
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    return out_path


def visualize_summary(all_metrics, output_dir):
    """Generate a summary plot across all evaluated samples.

    Shows distribution of MAE for radar distance.
    """
    if not all_metrics:
        return

    dist_maes = [m['dist_mae'] for m in all_metrics if m['dist_mae'] is not None]

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))

    if dist_maes:
        ax.hist(dist_maes, bins=20, color='#FF5722', alpha=0.7, edgecolor='black')
        ax.axvline(np.mean(dist_maes), color='red', linestyle='--',
                    label=f'Mean: {np.mean(dist_maes):.4f} m')
        ax.set_xlabel('MAE (m)')
        ax.set_ylabel('Count')
        ax.set_title('Radar Distance MAE Distribution\n(obstacle bins only)', fontweight='bold')
        ax.legend()
    else:
        ax.text(0.5, 0.5, 'No obstacle data', ha='center', va='center',
                transform=ax.transAxes, fontsize=14)
        ax.set_title('Radar Distance MAE', fontweight='bold')

    plt.tight_layout()
    out_path = Path(output_dir) / 'summary.png'
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"Summary saved to {out_path}")


@torch.no_grad()
def run_visualization(args):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = UniNavDataset(
        data_root=args.data_root,
        dataset_name=args.dataset,
        scenes=args.scenes,
        splits_json=args.splits_json,
        img_size=args.img_size,
        augment=False,
        clearance_height=args.clearance_height,
    )
    print(f"Dataset: {len(dataset)} samples")

    if len(dataset) == 0:
        print("No samples found.")
        return

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    saved_args = ckpt.get('args', {})
    use_multi_scale = saved_args.get('multi_scale', False)

    model = TraversabilityModel(
        backbone_path=args.backbone_weights,
        freeze_backbone=True,
        num_angles=360,
        use_multi_scale=use_multi_scale,
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    epoch = ckpt.get('epoch', '?')
    val_loss = ckpt.get('val_losses', {}).get('total', '?')
    print(f"Loaded checkpoint: epoch={epoch}, val_loss={val_loss}")

    n = min(args.num_samples, len(dataset))
    if args.sample_indices:
        indices = [int(i) for i in args.sample_indices]
    else:
        indices = np.linspace(0, len(dataset) - 1, n, dtype=int).tolist()

    all_metrics = []

    for count, idx in enumerate(indices):
        sample = dataset[idx]
        image = sample['image'].unsqueeze(0).to(device)
        azimuth_map = sample['azimuth_map'].unsqueeze(0).to(device)
        continuous_has_data = sample['continuous_has_data'].numpy()
        has_data = sample['has_data'].numpy()

        # Model inference: (radar_dist, exist_logit, raw_dist)
        pred_radar_dist, pred_exist_logit, pred_raw_dist = model(image, azimuth_map)
        pred_dist = pred_raw_dist[0].cpu().numpy()  # 使用 raw_dist 与训练评估一致
        pred_exist_logit = pred_exist_logit[0].cpu().numpy()

        rgb_path = dataset.samples[idx][0] if idx < len(dataset.samples) else None
        elev_path = dataset.samples[idx][1] if idx < len(dataset.samples) else None

        # Reload GT from original NPZ (same as subplot 5) to avoid interpolation
        if elev_path and Path(elev_path).exists():
            elev_data = np.load(elev_path)
            gt_dist = _extract_radar_dist(
                elev_data['profile_data'],
                elev_data['profile_offsets'],
                args.clearance_height
            )
        else:
            # Fallback to dataset version if NPZ not available
            gt_dist = sample['radar_dist'].numpy()

        out_path = visualize_sample(
            sample['image'], gt_dist, pred_dist, pred_exist_logit,
            continuous_has_data, has_data, idx, out_dir, rgb_path=rgb_path, elev_path=elev_path,
        )
        print(f"  [{count+1}/{len(indices)}] Sample {idx} -> {out_path}")

        if continuous_has_data.any():
            has_obs = continuous_has_data & (gt_dist < 100.0)
            dist_mae = np.abs(pred_dist[has_obs] - gt_dist[has_obs]).mean() if has_obs.any() else None
            all_metrics.append({'dist_mae': dist_mae, 'idx': idx})

    visualize_summary(all_metrics, out_dir)

    if all_metrics:
        dist_vals = [m['dist_mae'] for m in all_metrics if m['dist_mae'] is not None]
        avg_dist = np.mean(dist_vals) if dist_vals else float('nan')
        print(f"\nOverall metrics ({len(all_metrics)} samples):")
        print(f"  Radar Dist MAE:    {avg_dist:.4f} m")


def main():
    parser = argparse.ArgumentParser(description='Visualize Traversability Model Predictions')
    parser.add_argument('--checkpoint', required=True,
                        help='Path to model checkpoint (.pth)')
    parser.add_argument('--data-root', required=True,
                        help='UniNav-DB root directory')
    parser.add_argument('--dataset', default='DL3DV-1k',
                        help='Dataset name')
    parser.add_argument('--scenes', nargs='+', default=None,
                        help='Scene IDs (default: all)')
    parser.add_argument('--splits-json', default=None,
                        help='Splits JSON path (camera intrinsics + type)')
    parser.add_argument('--backbone-weights', required=True,
                        help='Path to DINOv3 ViT-B/16 weights')
    parser.add_argument('--img-size', type=int, default=384)
    parser.add_argument('--num-samples', type=int, default=8,
                        help='Number of samples to visualize')
    parser.add_argument('--sample-indices', nargs='+', default=None,
                        help='Specific sample indices to visualize')
    parser.add_argument('--output-dir', default='visualizations/traversability',
                        help='Output directory for visualization PNGs')
    parser.add_argument('--clearance-height', type=float, default=0.5,
                        help='Passable-gap threshold in metres (default: 0.5, same as training).')
    parser.add_argument('--device', default='cuda')

    args = parser.parse_args()
    run_visualization(args)


if __name__ == '__main__':
    main()
