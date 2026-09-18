"""多模型可通行性预测对比可视化（带几何投影）。

Multi-model traversability comparison with geometry-aware projection.
Projects model predictions onto RGB images using depth maps and camera intrinsics.

Usage:
    python -m scripts.traversability.viz_compare_projection \
        --val-dir /data1/renhao/datasets/UniNav-DB/val \
        --data-root /data1/renhao/datasets/UniNav-DB/dataset \
        --splits-json /data1/renhao/datasets/UniNav-DB/splits/pinhole_curated.json \
        --models traversability_pinhole baseline_resnet_fc_pinhole \
        --model-labels "Ours" "Baseline" \
        --num-samples 5 \
        --output-dir output/model_comparison_proj
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL import Image, ImageDraw
from matplotlib.colors import to_rgb


# ---------------------------------------------------------------------------
# Scene metadata helpers
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


def parse_rgb_path(rgb_path: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Parse rgb_path to extract dataset_name, scene_id, and frame_stem.

    Example: ../UniNav-DB/dataset/raw_images/TankAndTemples/Train/00001.jpg
    Returns: ('TankAndTemples', 'Train', '00001')
    """
    parts = Path(rgb_path).parts
    try:
        idx = parts.index('raw_images')
        dataset_name = parts[idx + 1]
        scene_id = parts[idx + 2]
        frame_stem = Path(parts[idx + 3]).stem
        return dataset_name, scene_id, frame_stem
    except (ValueError, IndexError):
        return None, None, None


# ---------------------------------------------------------------------------
# Depth and camera projection
# ---------------------------------------------------------------------------

def load_depth_png(depth_path: Path) -> Optional[np.ndarray]:
    """Load RGBA-packed float32 depth PNG."""
    if not depth_path.exists():
        return None
    arr = np.array(Image.open(depth_path))
    if arr.ndim == 3 and arr.shape[2] == 4 and arr.dtype == np.uint8:
        return arr.view(np.float32).reshape(arr.shape[0], arr.shape[1])
    return None


def load_depth_meta(meta_path: Path) -> Dict:
    """Load depth_meta NPZ."""
    if not meta_path.exists():
        return {}
    data = np.load(meta_path, allow_pickle=False)
    return {k: data[k] for k in data.files}


def build_3d_points(depth: np.ndarray, camera_type: str, intrinsics: Optional[Dict],
                   meta: Dict) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Build 3D point cloud from depth map.

    Returns:
        (points_3d, valid_mask) where points_3d is (H, W, 3) in camera frame.
    """
    h, w = depth.shape
    valid = np.isfinite(depth) & (depth > 0)

    x_map = np.full((h, w), np.nan, dtype=np.float32)
    y_map = np.full((h, w), np.nan, dtype=np.float32)
    z_map = np.full((h, w), np.nan, dtype=np.float32)

    u, v = np.meshgrid(np.arange(w), np.arange(h))

    if camera_type == 'pinhole':
        if intrinsics:
            fx = float(intrinsics['fx'])
            fy = float(intrinsics['fy'])
            cx = float(intrinsics['cx'])
            cy = float(intrinsics['cy'])
        elif 'intrinsics' in meta:
            K = meta['intrinsics']
            fx, fy = float(K[0, 0]), float(K[1, 1])
            cx, cy = float(K[0, 2]), float(K[1, 2])
        else:
            return None, None

        z = depth[valid]
        x_map[valid] = (u[valid] - cx) * z / fx
        y_map[valid] = (v[valid] - cy) * z / fy
        z_map[valid] = z

    elif camera_type == 'equirectangular':
        az = (u.astype(np.float32) / w - 0.5) * (2 * np.pi)
        el = (0.5 - v.astype(np.float32) / h) * np.pi
        x_map[valid] = (np.cos(el) * np.sin(az) * depth)[valid]
        y_map[valid] = (-np.sin(el) * depth)[valid]
        z_map[valid] = (np.cos(el) * np.cos(az) * depth)[valid]

    elif camera_type == 'fisheye':
        # Try to load rays
        if 'rays' in meta:
            rays = np.asarray(meta['rays'], dtype=np.float32)
        elif 'rays_lr' in meta:
            rays_lr = np.asarray(meta['rays_lr'], dtype=np.float32)
            # Upsample rays_lr to full resolution
            from PIL import Image as PILImage
            rays = np.stack([
                np.array(PILImage.fromarray(rays_lr[:, :, i]).resize((w, h), PILImage.BILINEAR))
                for i in range(3)
            ], axis=-1).astype(np.float32)
        else:
            return None, None

        if rays.shape[:2] != (h, w) or rays.shape[2] != 3:
            return None, None

        # Normalize rays
        norm = np.linalg.norm(rays, axis=2, keepdims=True)
        rays = rays / np.clip(norm, 1e-8, None)
        rays = np.nan_to_num(rays, nan=0.0)

        valid_r = valid & np.isfinite(rays).all(axis=2)
        x_map[valid_r] = rays[:, :, 0][valid_r] * depth[valid_r]
        y_map[valid_r] = rays[:, :, 1][valid_r] * depth[valid_r]
        z_map[valid_r] = rays[:, :, 2][valid_r] * depth[valid_r]
        valid = valid_r

    else:
        return None, None

    valid &= np.isfinite(x_map) & np.isfinite(y_map) & np.isfinite(z_map)
    if not valid.any():
        return None, None

    points_3d = np.stack([x_map, y_map, z_map], axis=-1)
    return points_3d, valid


# ---------------------------------------------------------------------------
# Traversability projection
# ---------------------------------------------------------------------------

def project_radar_to_image(radar_dist: np.ndarray, continuous_has_data: np.ndarray,
                          points_3d: np.ndarray, valid_mask: np.ndarray,
                          max_dist: float = 20.0, ground_plane: Optional[Tuple] = None) -> Tuple[List[Tuple[int, int]], np.ndarray]:
    """Project radar distance curve onto image using 3D points.

    Args:
        radar_dist: (360,) distance array in meters.
        continuous_has_data: (360,) bool mask for valid FOV.
        points_3d: (H, W, 3) 3D points in camera frame.
        valid_mask: (H, W) bool mask for valid depth pixels.
        max_dist: Maximum distance to consider.
        ground_plane: Optional (normal, d) tuple for RANSAC ground plane projection.

    Returns:
        (projected_uv_list, angles_rad) where projected_uv_list contains (u, v) tuples.
    """
    h, w = points_3d.shape[:2]
    n_bins = len(radar_dist)

    # Build angular bins from 3D points
    x = points_3d[:, :, 0][valid_mask]
    y = points_3d[:, :, 1][valid_mask]
    z = points_3d[:, :, 2][valid_mask]

    # If ground plane is provided, compute height above ground
    if ground_plane is not None:
        normal, d = ground_plane
        # Height above ground plane: h = dot(point, normal) - d
        heights = x * normal[0] + y * normal[1] + z * normal[2] - d
    else:
        heights = None

    # Compute azimuth angle (around Y-axis, 0 = forward/+Z)
    azimuth = np.arctan2(x, z)  # Range: [-pi, pi]
    dist_3d = np.sqrt(x**2 + z**2)

    # Bin assignment: azimuth [-pi, pi] -> bin [0, 359]
    bin_idx = ((azimuth + np.pi) / (2 * np.pi) * n_bins).astype(int)
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)

    # Get pixel coordinates
    v_coords, u_coords = np.where(valid_mask)
    v_coords = v_coords.astype(np.float32)
    u_coords = u_coords.astype(np.float32)

    # Build candidate pixels for each bin
    candidate_bins = {}
    for b in range(n_bins):
        mask_b = (bin_idx == b)
        if mask_b.any():
            if heights is not None:
                # Store (v, u, dist, height) for ground plane projection
                candidate_bins[b] = (v_coords[mask_b], u_coords[mask_b], dist_3d[mask_b], heights[mask_b])
            else:
                candidate_bins[b] = (v_coords[mask_b], u_coords[mask_b], dist_3d[mask_b], None)

    # Sample points along radar curve
    projected_uv = []
    angles_rad = np.linspace(-np.pi, np.pi, n_bins, endpoint=False)

    for b in range(n_bins):
        if not continuous_has_data[b]:
            continue
        dist_target = radar_dist[b]
        if not np.isfinite(dist_target) or dist_target <= 0 or dist_target > max_dist:
            continue

        if b not in candidate_bins:
            continue

        bin_data = candidate_bins[b]
        bv, bu, bd = bin_data[0], bin_data[1], bin_data[2]
        bh = bin_data[3] if len(bin_data) > 3 else None

        if len(bd) == 0:
            continue

        # If ground plane is available, find pixel closest to ground (height ≈ 0)
        if bh is not None:
            # Filter pixels near ground (height within ±0.2m)
            ground_mask = np.abs(bh) < 0.2
            if ground_mask.any():
                # Among ground pixels, find closest in distance
                ground_dists = bd[ground_mask]
                ground_v = bv[ground_mask]
                ground_u = bu[ground_mask]
                diffs = np.abs(ground_dists - dist_target)
                best_idx = np.argmin(diffs)
                if diffs[best_idx] < 1.0:  # Within 1m tolerance
                    u = int(np.round(ground_u[best_idx]))
                    v = int(np.round(ground_v[best_idx]))
                    if 0 <= u < w and 0 <= v < h:
                        projected_uv.append((u, v))
                continue

        # Fallback: find closest pixel in distance (no ground plane)
        diffs = np.abs(bd - dist_target)
        best_idx = np.argmin(diffs)
        if diffs[best_idx] < 1.0:  # Within 1m tolerance
            u = int(np.round(bu[best_idx]))
            v = int(np.round(bv[best_idx]))
            if 0 <= u < w and 0 <= v < h:
                projected_uv.append((u, v))

    return projected_uv, angles_rad


def draw_contour_on_image(img_rgb: np.ndarray, projected_uv: List[Tuple[int, int]],
                         color: Tuple[int, int, int], linewidth: int = 2, alpha: float = 1.0,
                         break_on_large_jump: bool = False, jump_threshold: float = 0.5):
    """Draw contour line on RGB image with transparency support.

    Args:
        img_rgb: RGB image array
        projected_uv: List of (u, v) pixel coordinates
        color: RGB color tuple
        linewidth: Line width in pixels
        alpha: Transparency (1.0 = opaque)
        break_on_large_jump: If True, break line when consecutive points are far apart
        jump_threshold: Fraction of image width to consider as a "large jump"
    """
    import cv2
    if len(projected_uv) < 2:
        return img_rgb

    # Create overlay for transparency
    overlay = img_rgb.copy()

    if break_on_large_jump:
        # For equirectangular, break line at large horizontal jumps (left-right wrap)
        h, w = img_rgb.shape[:2]
        threshold_px = w * jump_threshold

        segments = []
        current_segment = [projected_uv[0]]

        for i in range(1, len(projected_uv)):
            u_prev, v_prev = projected_uv[i-1]
            u_curr, v_curr = projected_uv[i]

            # Check for large horizontal jump
            if abs(u_curr - u_prev) > threshold_px:
                # Save current segment and start new one
                if len(current_segment) >= 2:
                    segments.append(current_segment)
                current_segment = [projected_uv[i]]
            else:
                current_segment.append(projected_uv[i])

        # Add last segment
        if len(current_segment) >= 2:
            segments.append(current_segment)

        # Draw each segment separately
        for segment in segments:
            pts = np.array(segment, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(overlay, [pts], isClosed=False, color=color,
                         thickness=linewidth, lineType=cv2.LINE_AA)
    else:
        # Normal drawing without breaking
        pts = np.array(projected_uv, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(overlay, [pts], isClosed=False, color=color,
                     thickness=linewidth, lineType=cv2.LINE_AA)

    # Blend with alpha
    cv2.addWeighted(overlay, alpha, img_rgb, 1 - alpha, 0, img_rgb)

    return img_rgb


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_polar_radar(ax, angles_rad, distances, continuous_has_data,
                     color, label, linewidth=1.5, alpha=0.8, linestyle='-'):
    """Plot radar distance curve on polar axis.

    For equirectangular cameras, breaks the line at -180/+180 boundary to avoid
    connecting the leftmost and rightmost points.
    """
    dist_vis = distances.copy()
    dist_vis[~continuous_has_data] = np.nan

    # Check if we need to break at boundaries (for equirectangular)
    # Detect large angle jumps (> 180 degrees) which indicate wrapping
    angles_deg = np.rad2deg(angles_rad)
    angle_diffs = np.abs(np.diff(angles_deg))

    # If there's a large jump (> 180 degrees), split the plot
    if np.any(angle_diffs > 180):
        # Find the split point
        split_idx = np.where(angle_diffs > 180)[0][0] + 1

        # Plot first segment
        ax.plot(angles_rad[:split_idx], dist_vis[:split_idx],
                color=color, linewidth=linewidth, label=label,
                alpha=alpha, linestyle=linestyle)

        # Plot second segment (no label to avoid duplicate in legend)
        ax.plot(angles_rad[split_idx:], dist_vis[split_idx:],
                color=color, linewidth=linewidth,
                alpha=alpha, linestyle=linestyle)
    else:
        # Normal plot without splitting
        ax.plot(angles_rad, dist_vis, color=color, linewidth=linewidth,
                label=label, alpha=alpha, linestyle=linestyle)


def create_comparison_figure_with_projection(
    sample_id: str,
    rgb_path: str,
    depth_path: Path,
    meta_path: Path,
    gt_data: Dict,
    pred_data_list: List[Dict],
    model_labels: List[str],
    output_path: str,
    camera_type: str,
    intrinsics: Optional[Dict],
    data_root: Path,
    max_dist: float = 20.0,
):
    """Generate comparison figure with geometric projection."""
    angles_deg = np.arange(360) - 180.0
    angles_rad = np.deg2rad(angles_deg)

    gt_dist = gt_data['distance']
    continuous_has_data = gt_data['continuous_has_data']

    # Auto-compute max_dist to cover all GT data
    valid_gt_dist = gt_dist[continuous_has_data & np.isfinite(gt_dist)]
    if len(valid_gt_dist) > 0:
        gt_max = np.max(valid_gt_dist)
        # Add 10% margin to ensure all data is visible
        max_dist = np.ceil(gt_max * 1.1)
        print(f"  INFO: GT max distance = {gt_max:.2f}m, using max_dist = {max_dist:.2f}m")
    else:
        max_dist = 20.0
        print(f"  WARN: No valid GT data, using default max_dist = {max_dist}m")

    # Color palette: High saturation colors
    # GT will be bright green
    colors_hex = ['#2196F3', '#FF0000', '#FFD700', '#FF9800', '#9C27B0']  # Blue, Red, Yellow, Orange, Purple
    colors_rgb = [tuple(int(c[i:i+2], 16) for i in (1, 3, 5)) for c in colors_hex]
    gt_color_hex = '#00FF00'  # Bright green for GT

    # Load depth and build 3D points
    depth = load_depth_png(depth_path)
    meta = load_depth_meta(meta_path)

    if depth is None:
        print(f"  WARN: Depth not found for {sample_id}, skipping projection")
        return

    points_3d, valid_mask = build_3d_points(depth, camera_type, intrinsics, meta)
    if points_3d is None:
        print(f"  WARN: Failed to build 3D points for {sample_id}")
        return

    # Load ground mask and fit RANSAC plane
    from scripts.utils.pointcloud import load_masks, _fit_ground_plane_ransac

    # Construct mask path
    dataset_name, scene_id, _ = parse_rgb_path(rgb_path)
    mask_path = data_root / 'masks' / dataset_name / scene_id / f'{sample_id}.npz'

    ground_plane = None
    if mask_path.exists():
        masks = load_masks(str(mask_path.parent), sample_id)
        ground_mask = masks.get('ground_raw')  # Use raw mask for RANSAC
        if ground_mask is not None and ground_mask.any():
            ground_pts = points_3d[ground_mask & valid_mask]
            if len(ground_pts) > 100:
                ground_plane = _fit_ground_plane_ransac(ground_pts)
                if ground_plane is not None:
                    print(f"  INFO: RANSAC ground plane fitted with {len(ground_pts)} points")

    # Create figure (3 columns: RGB projection | Polar | Linear)
    # Width ratios: 1 : 1 : 16/9
    fig = plt.figure(figsize=(28, 8))
    gs = GridSpec(1, 3, figure=fig, wspace=0.25, width_ratios=[1, 1, 16/9])

    # ---------------------------------------------------------------------------
    # Left: RGB with projection
    # ---------------------------------------------------------------------------
    ax_rgb = fig.add_subplot(gs[0, 0])

    import cv2
    img = cv2.imread(rgb_path)
    if img is None:
        ax_rgb.text(0.5, 0.5, f'Image not found', ha='center', va='center')
        ax_rgb.axis('off')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).copy()

    # Determine if we need to break lines at boundaries (for equirectangular)
    break_lines = (camera_type == 'equirectangular')

    # Project GT (Bright Green, no transparency on RGB)
    gt_color_rgb = tuple(int(gt_color_hex[i:i+2], 16) for i in (1, 3, 5))
    gt_uv, _ = project_radar_to_image(gt_dist, continuous_has_data, points_3d, valid_mask, max_dist, ground_plane)
    draw_contour_on_image(img_rgb, gt_uv, gt_color_rgb, linewidth=12, alpha=1.0, break_on_large_jump=break_lines)

    # Project predictions
    for i, (pred_data, label) in enumerate(zip(pred_data_list, model_labels)):
        pred_dist = pred_data['distance']
        pred_uv, _ = project_radar_to_image(pred_dist, continuous_has_data, points_3d, valid_mask, max_dist, ground_plane)
        color_rgb = colors_rgb[i % len(colors_rgb)]
        draw_contour_on_image(img_rgb, pred_uv, color_rgb, linewidth=9, alpha=1.0, break_on_large_jump=break_lines)

    ax_rgb.imshow(img_rgb)
    ax_rgb.axis('off')
    ax_rgb.set_title(f'Sample {sample_id} - Traversability Projection', fontsize=32, fontweight='bold')

    # ---------------------------------------------------------------------------
    # Center: Polar radar
    # ---------------------------------------------------------------------------
    ax_polar = fig.add_subplot(gs[0, 1], projection='polar')
    ax_polar.set_theta_zero_location('N')
    ax_polar.set_theta_direction(-1)

    # Set theta range based on camera type
    if camera_type == 'pinhole':
        # For pinhole, use FOV range from continuous_has_data
        valid_angles = angles_deg[continuous_has_data]
        if len(valid_angles) > 0:
            theta_min = max(-90, np.floor(valid_angles.min()))
            theta_max = min(90, np.ceil(valid_angles.max()))
        else:
            theta_min, theta_max = -90, 90
        ax_polar.set_thetamin(theta_min)
        ax_polar.set_thetamax(theta_max)
    else:
        # For fisheye and equirectangular, use full 360 degrees
        ax_polar.set_thetamin(-180)
        ax_polar.set_thetamax(180)

    plot_polar_radar(ax_polar, angles_rad, gt_dist, continuous_has_data,
                     gt_color_hex, 'Ground Truth', linewidth=9.0, alpha=0.7)

    for i, (pred_data, label) in enumerate(zip(pred_data_list, model_labels)):
        pred_dist = pred_data['distance']
        color = colors_hex[i % len(colors_hex)]
        plot_polar_radar(ax_polar, angles_rad, pred_dist, continuous_has_data,
                        color, label, linewidth=7.5, alpha=0.7)

    ax_polar.set_ylim(0, max_dist)
    ax_polar.set_title('Polar Radar', fontsize=61, fontweight='bold', pad=15)
    ax_polar.legend(loc='upper right', fontsize=45, framealpha=0.9)
    ax_polar.grid(True, alpha=0.3)
    ax_polar.tick_params(axis='both', labelsize=27)

    # ---------------------------------------------------------------------------
    # Right: Linear plot (auto-range based on data)
    # ---------------------------------------------------------------------------
    ax_linear = fig.add_subplot(gs[0, 2])

    # Compute adaptive x-range to cover all GT data
    valid_angles = angles_deg[continuous_has_data]
    if len(valid_angles) > 0:
        xlim_min = np.floor(valid_angles.min())
        xlim_max = np.ceil(valid_angles.max())
        # Add small margin
        margin = (xlim_max - xlim_min) * 0.05
        xlim_min = max(-180, xlim_min - margin)
        xlim_max = min(180, xlim_max + margin)
    else:
        xlim_min, xlim_max = -90, 90

    gt_vis = gt_dist.copy()
    gt_vis[~continuous_has_data] = np.nan
    ax_linear.plot(angles_deg, gt_vis, color=gt_color_hex, linewidth=9.0,
                  label='Ground Truth', alpha=0.7)

    for i, (pred_data, label) in enumerate(zip(pred_data_list, model_labels)):
        pred_dist = pred_data['distance']
        pred_vis = pred_dist.copy()
        pred_vis[~continuous_has_data] = np.nan
        color = colors_hex[i % len(colors_hex)]
        ax_linear.plot(angles_deg, pred_vis, color=color, linewidth=7.5,
                      label=label, alpha=0.7)

    ax_linear.set_xlim(xlim_min, xlim_max)
    ax_linear.set_ylim(0, max_dist)
    ax_linear.set_xlabel('Azimuth (degrees)', fontsize=56)
    ax_linear.set_ylabel('Distance (m)', fontsize=56)
    ax_linear.set_title('Linear Distance Plot', fontsize=61, fontweight='bold')
    ax_linear.legend(loc='upper right', fontsize=45, framealpha=0.9)
    ax_linear.grid(True, alpha=0.3)
    ax_linear.tick_params(axis='both', labelsize=31)
    ax_linear.axvline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)

    # Save combined figure
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    # ---------------------------------------------------------------------------
    # Save individual panels without legends
    # ---------------------------------------------------------------------------
    output_dir = Path(output_path).parent
    sample_stem = Path(output_path).stem

    # 1. Save RGB projection only
    fig_rgb = plt.figure(figsize=(8, 8))
    ax = fig_rgb.add_subplot(111)
    ax.imshow(img_rgb)
    ax.axis('off')
    plt.savefig(output_dir / f'{sample_stem}_rgb.png', dpi=150, bbox_inches='tight', pad_inches=0)
    plt.close(fig_rgb)

    # 2. Save Polar plot only (no legend, camera-type aware range)
    fig_polar = plt.figure(figsize=(8, 8))
    ax = fig_polar.add_subplot(111, projection='polar')
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)

    # Set theta range based on camera type
    if camera_type == 'pinhole':
        # For pinhole, use FOV range from continuous_has_data
        valid_angles = angles_deg[continuous_has_data]
        if len(valid_angles) > 0:
            theta_min = max(-90, np.floor(valid_angles.min()))
            theta_max = min(90, np.ceil(valid_angles.max()))
        else:
            theta_min, theta_max = -90, 90
        ax.set_thetamin(theta_min)
        ax.set_thetamax(theta_max)
    else:
        # For fisheye and equirectangular, use full 360 degrees
        ax.set_thetamin(-180)
        ax.set_thetamax(180)

    plot_polar_radar(ax, angles_rad, gt_dist, continuous_has_data,
                     gt_color_hex, 'Ground Truth', linewidth=9.0, alpha=0.7)
    for i, (pred_data, label) in enumerate(zip(pred_data_list, model_labels)):
        pred_dist = pred_data['distance']
        color = colors_hex[i % len(colors_hex)]
        plot_polar_radar(ax, angles_rad, pred_dist, continuous_has_data,
                        color, label, linewidth=7.5, alpha=0.7)

    ax.set_ylim(0, max_dist)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', labelsize=27)
    plt.savefig(output_dir / f'{sample_stem}_polar.png', dpi=150, bbox_inches='tight')
    plt.close(fig_polar)

    # 3. Save Linear plot only (no legend)
    fig_linear = plt.figure(figsize=(14.22, 8))  # 16:9 ratio
    ax = fig_linear.add_subplot(111)

    gt_vis = gt_dist.copy()
    gt_vis[~continuous_has_data] = np.nan
    ax.plot(angles_deg, gt_vis, color=gt_color_hex, linewidth=9.0, alpha=0.7)

    for i, (pred_data, label) in enumerate(zip(pred_data_list, model_labels)):
        pred_dist = pred_data['distance']
        pred_vis = pred_dist.copy()
        pred_vis[~continuous_has_data] = np.nan
        color = colors_hex[i % len(colors_hex)]
        ax.plot(angles_deg, pred_vis, color=color, linewidth=7.5, alpha=0.7)

    ax.set_xlim(xlim_min, xlim_max)
    ax.set_ylim(0, max_dist)
    ax.set_xlabel('Azimuth (degrees)', fontsize=56)
    ax.set_ylabel('Distance (m)', fontsize=56)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', labelsize=31)
    ax.axvline(0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    plt.savefig(output_dir / f'{sample_stem}_linear.png', dpi=150, bbox_inches='tight')
    plt.close(fig_linear)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Multi-model traversability comparison with projection')
    parser.add_argument('--val-dir', required=True,
                       help='Path to val directory (e.g., /data/UniNav-DB/val)')
    parser.add_argument('--data-root', required=True,
                       help='Path to dataset root (e.g., /data/UniNav-DB/dataset)')
    parser.add_argument('--splits-json', required=True,
                       help='Path to splits JSON (e.g., /data/UniNav-DB/splits/pinhole_curated.json)')
    parser.add_argument('--models', nargs='+', required=True,
                       help='Model directory names')
    parser.add_argument('--model-labels', nargs='+', default=None,
                       help='Model labels for legend')
    parser.add_argument('--num-samples', type=int, default=10,
                       help='Number of samples to visualize (default: 10)')
    parser.add_argument('--sample-ids', nargs='+', default=None,
                       help='Specific sample IDs')
    parser.add_argument('--max-dist', type=float, default=20.0,
                       help='Maximum distance for plots (default: 20.0)')
    parser.add_argument('--output-dir', default='output/model_comparison_proj',
                       help='Output directory')

    args = parser.parse_args()

    val_dir = Path(args.val_dir)
    data_root = Path(args.data_root)
    gt_dir = val_dir / 'ground_truth'
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load splits
    scene_lookup = load_splits_json(Path(args.splits_json))

    # Model directories
    model_dirs = [val_dir / m for m in args.models if (val_dir / m).exists()]
    model_labels = args.model_labels if args.model_labels else [d.name for d in model_dirs]

    # Sample IDs
    if args.sample_ids:
        sample_ids = args.sample_ids
    else:
        all_samples = sorted([f.stem for f in gt_dir.glob('*.npz')])
        sample_ids = all_samples[:args.num_samples]

    print(f"Generating {len(sample_ids)} comparison figures with projection...")
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
        rgb_path_rel = str(gt_npz['rgb_path'])

        # Parse scene info
        dataset_name, scene_id, frame_stem = parse_rgb_path(rgb_path_rel)
        if not dataset_name or not scene_id:
            print(f"  [{i}/{len(sample_ids)}] SKIP {sample_id}: Cannot parse rgb_path")
            continue

        scene_info = scene_lookup.get((dataset_name, scene_id))
        if not scene_info:
            print(f"  [{i}/{len(sample_ids)}] SKIP {sample_id}: Scene not in splits")
            continue

        camera_type = scene_info['camera_type']
        intrinsics = scene_info['intrinsics']

        # Build paths
        rgb_path = str(data_root / 'raw_images' / dataset_name / scene_id / f'{frame_stem}.jpg')
        if not Path(rgb_path).exists():
            rgb_path = str(data_root / 'raw_images' / dataset_name / scene_id / f'{frame_stem}.png')

        depth_path = data_root / 'depth' / dataset_name / scene_id / f'{frame_stem}.png'
        meta_path = data_root / 'depth_meta' / dataset_name / scene_id / f'{frame_stem}.npz'

        # Load predictions
        pred_data_list = []
        for model_dir in model_dirs:
            pred_path = model_dir / f'{sample_id}.npz'
            if pred_path.exists():
                pred_npz = np.load(pred_path)
                pred_data_list.append({
                    'distance': pred_npz['distance'],
                    'has_data': pred_npz['has_data'],
                })
            else:
                pred_data_list.append({
                    'distance': np.zeros(360),
                    'has_data': np.zeros(360, dtype=bool),
                })

        # Generate figure
        output_path = output_dir / f'{sample_id}.png'
        try:
            create_comparison_figure_with_projection(
                sample_id, rgb_path, depth_path, meta_path,
                gt_data, pred_data_list, model_labels, str(output_path),
                camera_type, intrinsics, data_root, max_dist=args.max_dist
            )
            print(f"  [{i}/{len(sample_ids)}] OK {sample_id}")
        except Exception as e:
            print(f"  [{i}/{len(sample_ids)}] ERROR {sample_id}: {e}")

    print(f"\nDone. Saved to {output_dir}")


if __name__ == '__main__':
    main()
