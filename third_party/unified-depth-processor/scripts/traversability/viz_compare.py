"""多模型可通行性预测对比可视化。

Multi-model traversability prediction comparison visualization.
Loads ground truth and predictions from multiple models, generates side-by-side
radar plots for visual comparison.

Usage:
    python -m scripts.traversability.viz_compare \
        --val-dir /data/users/renhao/UniNav-DB/val \
        --models traversability_pinhole baseline_resnet_fc_pinhole baseline_dinov3_mlp_pinhole \
        --model-labels "Ours" "ResNet-FC" "DINOv3-MLP" \
        --num-samples 20 \
        --output-dir output/model_comparison
"""

import argparse
import json
from pathlib import Path
from typing import List, Optional, Dict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def load_splits_json(splits_json_path: Path) -> Dict:
    """Load splits JSON and build scene lookup."""
    with open(splits_json_path, 'r') as f:
        splits = json.load(f)

    # Build lookup: (dataset_name, scene_id) -> scene_info
    scene_lookup = {}
    for ds_entry in splits.get('datasets', []):
        dataset_name = ds_entry['dataset_name']
        camera_type = ds_entry.get('camera_type', 'pinhole')
        for scene in ds_entry.get('scenes', []):
            scene_id = scene['scene_id']
            scene_lookup[(dataset_name, scene_id)] = {
                'camera_type': camera_type,
                'intrinsics': scene.get('intrinsics'),
                'resolution': scene.get('resolution'),
            }
    return scene_lookup


def parse_rgb_path(rgb_path: str) -> tuple:
    """Parse rgb_path to extract dataset_name and scene_id.

    Example: ../UniNav-DB/dataset/raw_images/TankAndTemples/Train/00001.jpg
    Returns: ('TankAndTemples', 'Train')
    """
    parts = Path(rgb_path).parts
    # Find 'raw_images' in path
    try:
        idx = parts.index('raw_images')
        dataset_name = parts[idx + 1]
        scene_id = parts[idx + 2]
        return dataset_name, scene_id
    except (ValueError, IndexError):
        return None, None


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_polar_radar(ax, angles_rad, distances, continuous_has_data,
                     color, label, linewidth=1.5, alpha=0.8, linestyle='-'):
    """Plot a single radar distance curve on polar axis.

    Args:
        ax: Matplotlib polar axis.
        angles_rad: (360,) angles in radians.
        distances: (360,) distance values.
        continuous_has_data: (360,) bool mask for valid FOV bins.
        color: Line color.
        label: Legend label.
        linewidth: Line width.
        alpha: Line alpha.
        linestyle: Line style.
    """
    dist_vis = distances.copy()
    dist_vis[~continuous_has_data] = np.nan

    ax.plot(angles_rad, dist_vis, color=color, linewidth=linewidth,
            label=label, alpha=alpha, linestyle=linestyle)


def create_comparison_figure(sample_id: str, rgb_path: str,
                             gt_data: dict, pred_data_list: List[dict],
                             model_labels: List[str], output_path: str,
                             max_dist: Optional[float] = None,
                             show_projection: bool = False,
                             clearance_height: float = 0.5):
    """Generate a comparison figure for one sample across multiple models.

    Layout (without projection):
        - Top row: RGB image (full width)
        - Middle row: Polar radar plot (GT + all models)
        - Bottom row: Linear distance plot (GT + all models)

    Layout (with projection):
        - Top left: RGB image with projected traversability contours
        - Top right: Polar radar plot
        - Bottom: Linear distance plot (full width)

    Args:
        sample_id: Sample identifier (e.g., "00001").
        rgb_path: Path to RGB image.
        gt_data: Ground truth dict with keys: distance, continuous_has_data, has_data.
        pred_data_list: List of prediction dicts (one per model).
        model_labels: List of model names for legend.
        output_path: Output PNG path.
        max_dist: Maximum distance for plot range (auto if None).
        show_projection: Whether to show traversability projection on RGB image.
        clearance_height: Clearance threshold in meters for projection visualization.
    """
    angles_deg = np.arange(360) - 180.0
    angles_rad = np.deg2rad(angles_deg)

    gt_dist = gt_data['distance']
    continuous_has_data = gt_data['continuous_has_data']
    has_data = gt_data['has_data']

    # Auto-scale: 95th percentile of valid GT values, floor at 10m
    valid_gt = gt_dist[continuous_has_data & has_data & np.isfinite(gt_dist)]
    if max_dist is None and len(valid_gt) > 0:
        max_dist = max(10.0, np.percentile(valid_gt, 95))
    elif max_dist is None:
        max_dist = 20.0

    # Color palette for models
    colors = ['#FF5722', '#4CAF50', '#2196F3', '#FF9800', '#9C27B0', '#00BCD4']

    if show_projection:
        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(2, 2, figure=fig, height_ratios=[1.2, 1], hspace=0.3, wspace=0.3)
    else:
        fig = plt.figure(figsize=(14, 10))
        gs = GridSpec(3, 2, figure=fig, height_ratios=[1.2, 1, 1], hspace=0.3, wspace=0.3)

    # ---------------------------------------------------------------------------
    # Row 1: RGB image (with optional projection overlay)
    # ---------------------------------------------------------------------------
    if show_projection:
        ax_rgb = fig.add_subplot(gs[0, 0])
        ax_polar = fig.add_subplot(gs[0, 1], projection='polar')
        ax_linear = fig.add_subplot(gs[1, :])
    else:
        ax_rgb = fig.add_subplot(gs[0, :])
        ax_polar = fig.add_subplot(gs[1, :], projection='polar')
        ax_linear = fig.add_subplot(gs[2, :])

    try:
        import cv2
        img = cv2.imread(rgb_path)
        if img is not None:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            if show_projection:
                # Draw simplified traversability projection as polar contours
                # This is a simplified version - just shows distance rings
                H, W = img_rgb.shape[:2]
                cx, cy = W // 2, H  # Assume camera at bottom center

                # Draw GT contour
                for angle_idx in range(0, 360, 2):  # Sample every 2 degrees
                    if not continuous_has_data[angle_idx]:
                        continue
                    dist = gt_dist[angle_idx]
                    if not np.isfinite(dist) or dist <= 0:
                        continue

                    angle_rad = np.deg2rad(angle_idx - 180)
                    # Simple projection: assume flat ground, scale by image height
                    px_per_meter = H / max_dist  # Rough approximation
                    r_px = dist * px_per_meter
                    x = int(cx + r_px * np.sin(angle_rad))
                    y = int(cy - r_px * np.cos(angle_rad))

                    if 0 <= x < W and 0 <= y < H:
                        cv2.circle(img_rgb, (x, y), 2, (0, 0, 0), -1)  # GT in black

                # Draw model predictions
                for i, (pred_data, label) in enumerate(zip(pred_data_list, model_labels)):
                    pred_dist = pred_data['distance']
                    color_hex = colors[i % len(colors)]
                    # Convert hex to RGB
                    color_rgb = tuple(int(color_hex[j:j+2], 16) for j in (1, 3, 5))

                    for angle_idx in range(0, 360, 2):
                        if not continuous_has_data[angle_idx]:
                            continue
                        dist = pred_dist[angle_idx]
                        if not np.isfinite(dist) or dist <= 0:
                            continue

                        angle_rad = np.deg2rad(angle_idx - 180)
                        r_px = dist * px_per_meter
                        x = int(cx + r_px * np.sin(angle_rad))
                        y = int(cy - r_px * np.cos(angle_rad))

                        if 0 <= x < W and 0 <= y < H:
                            cv2.circle(img_rgb, (x, y), 1, color_rgb, -1)

            ax_rgb.imshow(img_rgb)
            ax_rgb.axis('off')
            title = f'Sample {sample_id}'
            if show_projection:
                title += ' (with Traversability Projection)'
            ax_rgb.set_title(title, fontsize=14, fontweight='bold')
        else:
            ax_rgb.text(0.5, 0.5, f'Image not found:\n{rgb_path}',
                       ha='center', va='center', fontsize=10)
            ax_rgb.axis('off')
    except Exception as e:
        ax_rgb.text(0.5, 0.5, f'Error loading image:\n{e}',
                   ha='center', va='center', fontsize=10)
        ax_rgb.axis('off')

    # ---------------------------------------------------------------------------
    # Polar radar plot
    # ---------------------------------------------------------------------------
    ax_polar.set_theta_zero_location('N')
    ax_polar.set_theta_direction(-1)

    # Plot GT
    plot_polar_radar(ax_polar, angles_rad, gt_dist, continuous_has_data,
                     color='#000000', label='Ground Truth', linewidth=2.0, alpha=0.9)

    # Plot predictions
    for i, (pred_data, label) in enumerate(zip(pred_data_list, model_labels)):
        pred_dist = pred_data['distance']
        color = colors[i % len(colors)]
        plot_polar_radar(ax_polar, angles_rad, pred_dist, continuous_has_data,
                        color=color, label=label, linewidth=1.5, alpha=0.7)

    ax_polar.set_ylim(0, max_dist)
    ax_polar.set_title('Radar Distance Comparison (Polar)', fontsize=12, fontweight='bold', pad=15)
    ax_polar.legend(loc='upper right', fontsize=9, framealpha=0.9)
    ax_polar.grid(True, alpha=0.3)

    # ---------------------------------------------------------------------------
    # Linear distance plot
    # ---------------------------------------------------------------------------

    # Plot GT
    gt_vis = gt_dist.copy()
    gt_vis[~continuous_has_data] = np.nan
    ax_linear.plot(angles_deg, gt_vis, color='#000000', linewidth=2.0,
                  label='Ground Truth', alpha=0.9)

    # Plot predictions
    for i, (pred_data, label) in enumerate(zip(pred_data_list, model_labels)):
        pred_dist = pred_data['distance']
        pred_vis = pred_dist.copy()
        pred_vis[~continuous_has_data] = np.nan
        color = colors[i % len(colors)]
        ax_linear.plot(angles_deg, pred_vis, color=color, linewidth=1.5,
                      label=label, alpha=0.7)

    ax_linear.set_xlim(-180, 180)
    ax_linear.set_ylim(0, max_dist)
    ax_linear.set_xlabel('Azimuth (degrees)', fontsize=11)
    ax_linear.set_ylabel('Distance (m)', fontsize=11)
    ax_linear.set_title('Radar Distance Comparison (Linear)', fontsize=12, fontweight='bold')
    ax_linear.legend(loc='upper right', fontsize=9, framealpha=0.9)
    ax_linear.grid(True, alpha=0.3)
    ax_linear.axvline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)

    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Multi-model traversability comparison visualization')
    parser.add_argument('--val-dir', required=True,
                       help='Path to val directory (e.g., /data/UniNav-DB/val)')
    parser.add_argument('--models', nargs='+', required=True,
                       help='Model directory names (e.g., traversability_pinhole baseline_resnet_fc_pinhole)')
    parser.add_argument('--model-labels', nargs='+', default=None,
                       help='Model labels for legend (default: use directory names)')
    parser.add_argument('--num-samples', type=int, default=20,
                       help='Number of samples to visualize (default: 20)')
    parser.add_argument('--sample-ids', nargs='+', default=None,
                       help='Specific sample IDs to visualize (e.g., 00001 00002)')
    parser.add_argument('--max-dist', type=float, default=None,
                       help='Maximum distance for plot range (auto if omitted)')
    parser.add_argument('--show-projection', action='store_true',
                       help='Show traversability projection overlay on RGB image')
    parser.add_argument('--clearance-height', type=float, default=0.5,
                       help='Clearance threshold in meters (default: 0.5)')
    parser.add_argument('--output-dir', default='output/model_comparison',
                       help='Output directory for comparison figures')

    args = parser.parse_args()

    val_dir = Path(args.val_dir)
    gt_dir = val_dir / 'ground_truth'
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not gt_dir.exists():
        print(f"[ERROR] Ground truth directory not found: {gt_dir}")
        return

    # Check model directories
    model_dirs = []
    for model_name in args.models:
        model_dir = val_dir / model_name
        if not model_dir.exists():
            print(f"[WARN] Model directory not found: {model_dir}, skipping")
        else:
            model_dirs.append(model_dir)

    if not model_dirs:
        print("[ERROR] No valid model directories found")
        return

    # Model labels
    model_labels = args.model_labels if args.model_labels else [d.name for d in model_dirs]
    if len(model_labels) != len(model_dirs):
        print(f"[WARN] Number of labels ({len(model_labels)}) != number of models ({len(model_dirs)})")
        model_labels = [d.name for d in model_dirs]

    # Determine sample IDs
    if args.sample_ids:
        sample_ids = args.sample_ids
    else:
        # Get all GT samples, sort, take first N
        gt_files = sorted(gt_dir.glob('*.npz'))
        sample_ids = [f.stem for f in gt_files[:args.num_samples]]

    if not sample_ids:
        print("[ERROR] No samples found")
        return

    print(f"Generating comparison visualizations for {len(sample_ids)} samples...")
    print(f"  Models: {model_labels}")
    print(f"  Output: {output_dir}")

    for i, sample_id in enumerate(sample_ids, 1):
        gt_path = gt_dir / f'{sample_id}.npz'
        if not gt_path.exists():
            print(f"  [{i}/{len(sample_ids)}] SKIP {sample_id}: GT not found")
            continue

        # Load GT
        gt_npz = np.load(gt_path)
        gt_data = {
            'distance': gt_npz['distance'],
            'continuous_has_data': gt_npz['continuous_has_data'],
            'has_data': gt_npz['has_data'],
        }
        rgb_path = str(gt_npz['rgb_path'])

        # Load predictions
        pred_data_list = []
        for model_dir in model_dirs:
            pred_path = model_dir / f'{sample_id}.npz'
            if not pred_path.exists():
                print(f"  [{i}/{len(sample_ids)}] WARN {sample_id}: prediction not found in {model_dir.name}")
                # Use dummy data (all zeros)
                pred_data_list.append({
                    'distance': np.zeros(360),
                    'has_data': np.zeros(360, dtype=bool),
                })
            else:
                pred_npz = np.load(pred_path)
                pred_data_list.append({
                    'distance': pred_npz['distance'],
                    'has_data': pred_npz['has_data'],
                })

        # Generate figure
        output_path = output_dir / f'{sample_id}.png'
        try:
            create_comparison_figure(
                sample_id, rgb_path, gt_data, pred_data_list,
                model_labels, str(output_path), max_dist=args.max_dist,
                show_projection=args.show_projection,
                clearance_height=args.clearance_height
            )
            print(f"  [{i}/{len(sample_ids)}] OK {sample_id} -> {output_path.name}")
        except Exception as e:
            print(f"  [{i}/{len(sample_ids)}] ERROR {sample_id}: {e}")

    print(f"\nDone. Saved {len(sample_ids)} figures to {output_dir}")


if __name__ == '__main__':
    main()

