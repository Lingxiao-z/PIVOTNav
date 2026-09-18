"""生成带有可穿越性投影叠加的视频。

Generate video with traversability projection overlay.
Supports using low-resolution images for inference and high-resolution images/video for composition.

Usage:
    # Basic usage with image directory
    python -m scripts.traversability.generate_video \
        --image-dir /path/to/frames \
        --depth-dir /path/to/depth \
        --checkpoint checkpoints/best.pth \
        --splits-json splits/fisheye_curated.json \
        --dataset-name KITTI-360 \
        --scene-id 2013_05_28_drive_0000_sync \
        --backbone-weights model/dinov3_vitb16_pretrain.pth \
        --output-video output/demo.mp4

    # With HD image directory
    python -m scripts.traversability.generate_video \
        --image-dir /path/to/low_res_frames \
        --hd-image-dir /path/to/hd_frames \
        --depth-dir /path/to/depth \
        --checkpoint checkpoints/best.pth \
        --splits-json splits/fisheye_curated.json \
        --dataset-name KITTI-360 \
        --scene-id 2013_05_28_drive_0000_sync \
        --backbone-weights model/dinov3_vitb16_pretrain.pth \
        --output-video output/demo_hd.mp4

    # With HD video file
    python -m scripts.traversability.generate_video \
        --image-dir /path/to/low_res_frames \
        --hd-video tmp/video.mp4 \
        --depth-dir /path/to/depth \
        --checkpoint checkpoints/best.pth \
        --splits-json splits/fisheye_curated.json \
        --dataset-name KITTI-360 \
        --scene-id 2013_05_28_drive_0000_sync \
        --backbone-weights model/dinov3_vitb16_pretrain.pth \
        --output-video output/demo_hd.mp4
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.traversability.train import build_model


# ---------------------------------------------------------------------------
# Utility Functions (reused from existing scripts)
# ---------------------------------------------------------------------------

def natural_sort_key(s: str) -> List:
    """Natural sorting key for filenames with numbers."""
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(r'([0-9]+)', s)]


def load_splits_json(splits_json_path: Path, dataset_name: str, scene_id: str) -> Dict:
    """Load camera parameters from splits JSON."""
    with open(splits_json_path, 'r') as f:
        splits = json.load(f)

    # Check if this is a single-dataset JSON (e.g., Pano3D_gibson.json)
    if 'dataset_name' in splits and splits['dataset_name'] == dataset_name:
        camera_type = splits.get('camera_type', 'pinhole')
        for scene in splits.get('scenes', []):
            if scene['scene_id'] == scene_id:
                return {
                    'camera_type': camera_type,
                    'intrinsics': scene.get('intrinsics'),
                    'resolution': scene.get('resolution'),
                }

    # Otherwise, check multi-dataset format (e.g., pinhole_curated.json)
    for ds_entry in splits.get('datasets', []):
        if ds_entry['dataset_name'] == dataset_name:
            camera_type = ds_entry.get('camera_type', 'pinhole')
            for scene in ds_entry.get('scenes', []):
                if scene['scene_id'] == scene_id:
                    return {
                        'camera_type': camera_type,
                        'intrinsics': scene.get('intrinsics'),
                        'resolution': scene.get('resolution'),
                    }

    raise ValueError(f"Scene {dataset_name}/{scene_id} not found in splits JSON")


def load_depth_png(depth_path: str) -> Optional[np.ndarray]:
    """Load RGBA-packed float32 depth PNG (same as viz_compare_projection.py)."""
    from PIL import Image
    if not os.path.exists(depth_path):
        return None
    arr = np.array(Image.open(depth_path))
    if arr.ndim == 3 and arr.shape[2] == 4 and arr.dtype == np.uint8:
        return arr.view(np.float32).reshape(arr.shape[0], arr.shape[1])
    return None


def load_depth_meta(meta_path: str) -> Optional[Dict]:
    """Load depth_meta NPZ (same as viz_compare_projection.py)."""
    if not os.path.exists(meta_path):
        return {}
    data = np.load(meta_path, allow_pickle=False)
    return {k: data[k] for k in data.files}


def build_3d_points(depth: np.ndarray, camera_type: str,
                   intrinsics: Optional[Dict], meta: Optional[Dict]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Build 3D point cloud from depth map."""
    from scripts.traversability.viz_compare_projection import build_3d_points as _build_3d_points
    return _build_3d_points(depth, camera_type, intrinsics, meta)


def project_radar_to_image(radar_dist: np.ndarray, continuous_has_data: np.ndarray,
                          points_3d: np.ndarray, valid_mask: np.ndarray,
                          max_dist: float = 20.0, ground_plane: Optional[Tuple] = None) -> Tuple[List[Tuple[int, int]], np.ndarray]:
    """Project radar distance curve onto image."""
    from scripts.traversability.viz_compare_projection import project_radar_to_image as _project
    return _project(radar_dist, continuous_has_data, points_3d, valid_mask, max_dist, ground_plane)


def draw_contour_on_image(img_rgb: np.ndarray, projected_uv: List[Tuple[int, int]],
                         color: Tuple[int, int, int], linewidth: int = 2, alpha: float = 1.0,
                         break_on_large_jump: bool = False, jump_threshold: float = 0.5):
    """Draw contour line on RGB image."""
    from scripts.traversability.viz_compare_projection import draw_contour_on_image as _draw
    return _draw(img_rgb, projected_uv, color, linewidth, alpha, break_on_large_jump, jump_threshold)


def generate_azimuth_map(image_shape: Tuple[int, int], camera_type: str,
                        intrinsics: Optional[Dict], meta: Optional[Dict] = None) -> np.ndarray:
    """Generate azimuth map for the image."""
    h, w = image_shape

    if camera_type == 'pinhole':
        # Pinhole camera: use intrinsics
        fx = intrinsics['fx']
        cx = intrinsics['cx']

        # Create pixel grid
        u = np.arange(w, dtype=np.float32)
        azimuth_map = np.arctan2(u - cx, fx)  # (W,)
        azimuth_map = np.tile(azimuth_map[None, :], (h, 1))  # (H, W)

    elif camera_type == 'fisheye':
        # Fisheye: use rays from meta
        if meta and 'rays' in meta:
            rays = np.array(meta['rays'], dtype=np.float32).reshape(h, w, 3)
            azimuth_map = np.arctan2(rays[:, :, 0], rays[:, :, 2])
        elif meta and 'rays_lr' in meta:
            # Upsample rays_lr to full resolution
            from PIL import Image as PILImage
            rays_lr = np.asarray(meta['rays_lr'], dtype=np.float32)
            rays = np.stack([
                np.array(PILImage.fromarray(rays_lr[:, :, i]).resize((w, h), PILImage.BILINEAR))
                for i in range(3)
            ], axis=-1).astype(np.float32)
            azimuth_map = np.arctan2(rays[:, :, 0], rays[:, :, 2])
        else:
            raise ValueError("Fisheye camera requires 'rays' or 'rays_lr' in meta")

    elif camera_type == 'equirectangular':
        # Equirectangular: linear mapping
        u = np.arange(w, dtype=np.float32)
        azimuth_map = (u / w) * 2 * np.pi - np.pi  # [-π, π]
        azimuth_map = np.tile(azimuth_map[None, :], (h, 1))

    else:
        raise ValueError(f"Unknown camera type: {camera_type}")

    return azimuth_map.astype(np.float32)


def preprocess_image(img_rgb: np.ndarray) -> torch.Tensor:
    """Preprocess RGB image for model input."""
    # Normalize to [0, 1]
    img = img_rgb.astype(np.float32) / 255.0

    # ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img - mean) / std

    # Convert to tensor (C, H, W)
    img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0)

    return img_tensor


def plot_polar_radar(ax, angles_rad, distances, continuous_has_data,
                     color, linewidth=1.5, alpha=0.8):
    """Plot radar distance curve on polar axis."""
    dist_vis = distances.copy()
    dist_vis[~continuous_has_data] = np.nan

    angles_deg = np.rad2deg(angles_rad)
    angle_diffs = np.abs(np.diff(angles_deg))

    if np.any(angle_diffs > 180):
        split_idx = np.where(angle_diffs > 180)[0][0] + 1
        ax.plot(angles_rad[:split_idx], dist_vis[:split_idx],
                color=color, linewidth=linewidth, alpha=alpha)
        ax.plot(angles_rad[split_idx:], dist_vis[split_idx:],
                color=color, linewidth=linewidth, alpha=alpha)
    else:
        ax.plot(angles_rad, dist_vis, color=color, linewidth=linewidth, alpha=alpha)


def load_model_from_checkpoint(checkpoint_path: str, backbone_weights: str, device: torch.device):
    """Load traversability model from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    saved = ckpt.get('args', {})

    # Reconstruct model args
    model_args = SimpleNamespace(
        model=saved.get('model', 'traversability'),
        backbone_weights=backbone_weights,
        backbone_type=saved.get('backbone_type', 'vitb16'),
        freeze_backbone=True,
        unfreeze_last_n=0,
        multi_scale=saved.get('multi_scale', True),
    )

    model = build_model(model_args).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    print(f"Loaded model from {checkpoint_path}")
    print(f"  Backbone: {model_args.backbone_type}")
    print(f"  Multi-scale: {model_args.multi_scale}")

    return model


# ---------------------------------------------------------------------------
# Main Video Generation
# ---------------------------------------------------------------------------

def generate_video(args):
    """Main video generation function."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load camera parameters
    scene_info = load_splits_json(Path(args.splits_json), args.dataset_name, args.scene_id)
    camera_type = scene_info['camera_type']
    intrinsics = scene_info['intrinsics']
    print(f"Camera type: {camera_type}")
    print(f"Intrinsics: {intrinsics}")

    # Load model
    model = load_model_from_checkpoint(args.checkpoint, args.backbone_weights, device)

    # Collect image files for inference
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.PNG']:
        image_files.extend(Path(args.image_dir).glob(ext))
    image_files = sorted(image_files, key=lambda x: natural_sort_key(str(x)))

    if len(image_files) == 0:
        print(f"No images found in {args.image_dir}")
        return

    # Store total count for alignment before applying max_frames
    total_image_count = len(image_files)

    if args.max_frames:
        image_files = image_files[:args.max_frames]

    # Actual number of images to process
    num_images_to_process = len(image_files)

    print(f"Processing {num_images_to_process} frames (total in dataset: {total_image_count})")

    # Determine HD image source
    hd_video_cap = None
    hd_video_total_frames = 0

    if args.hd_video:
        hd_video_cap = cv2.VideoCapture(args.hd_video)
        if not hd_video_cap.isOpened():
            print(f"[ERROR] Failed to open HD video: {args.hd_video}")
            return
        hd_video_total_frames = int(hd_video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Using HD video from {args.hd_video} ({hd_video_total_frames} frames)")
        print(f"Alignment: {num_images_to_process} inferences will be mapped to {hd_video_total_frames} video frames")
        use_hd = True
        hd_image_dir = None
    elif args.hd_image_dir:
        hd_image_dir = Path(args.hd_image_dir)
        use_hd = True
        print(f"Using HD images from {hd_image_dir}")
    else:
        hd_image_dir = Path(args.image_dir)
        use_hd = False

    # Parse color
    color_hex = args.color
    if color_hex.startswith('#'):
        color_hex = color_hex[1:]
    color_rgb = tuple(int(color_hex[i:i+2], 16) for i in (0, 2, 4))

    # Determine if we need to break lines (for equirectangular)
    break_lines = (camera_type == 'equirectangular')

    # Initialize video writers
    video_writer_lr = None
    video_writer_hd = None

    # Calculate frame rate for low-res video
    if args.output_lr_video and hd_video_cap:
        video_fps = hd_video_cap.get(cv2.CAP_PROP_FPS)
        fps_lr = video_fps * num_images_to_process / hd_video_total_frames
        print(f"Low-res video FPS: {fps_lr:.2f} (scaled from {video_fps:.2f})")
    else:
        fps_lr = args.fps

    # Decide processing strategy
    if args.output_hd_video and hd_video_cap:
        # Need to cache inference results for HD video generation
        print("Caching inference results for HD video generation...")
        inference_cache = []
    else:
        inference_cache = None

    # Radar video cache
    if args.output_radar_video:
        print("Caching inference results for radar video generation...")
        radar_cache = []
    else:
        radar_cache = None

    # Process frames
    for frame_idx, img_path in enumerate(tqdm(image_files, desc="Processing frames")):
        frame_name = img_path.stem

        # Load RGB image for inference
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            print(f"Warning: Failed to load {img_path}, skipping")
            continue

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]

        # For LR video, use inference image directly
        lr_rgb = img_rgb
        h_lr, w_lr = h, w

        # For HD video caching, load corresponding HD frame if needed
        if inference_cache is not None and hd_video_cap:
            video_frame_idx = int(frame_idx * hd_video_total_frames / num_images_to_process)
            hd_video_cap.set(cv2.CAP_PROP_POS_FRAMES, video_frame_idx)
            ret, hd_bgr = hd_video_cap.read()
            if ret:
                if args.flip_vertical:
                    hd_bgr = cv2.flip(hd_bgr, 0)
                hd_rgb = cv2.cvtColor(hd_bgr, cv2.COLOR_BGR2RGB)
                h_hd, w_hd = hd_rgb.shape[:2]
            else:
                hd_rgb = img_rgb
                h_hd, w_hd = h, w
        else:
            hd_rgb = None
            h_hd, w_hd = h, w

        # Load depth and meta
        depth_path = Path(args.depth_dir) / f"{frame_name}.png"
        meta = {}
        if args.meta_dir:
            meta_path = Path(args.meta_dir) / f"{frame_name}.npz"
            meta = load_depth_meta(str(meta_path))
        elif camera_type in ['fisheye', 'equirectangular']:
            # For fisheye/equirect, try to load meta from default location
            if args.data_root:
                meta_path = Path(args.data_root) / 'depth_meta' / args.dataset_name / args.scene_id / f"{frame_name}.npz"
                if meta_path.exists():
                    meta = load_depth_meta(str(meta_path))

        depth = load_depth_png(str(depth_path))
        if depth is None:
            print(f"Warning: No depth for {frame_name}, skipping")
            continue

        # Build 3D points
        points_3d, valid_mask = build_3d_points(depth, camera_type, intrinsics, meta)
        if points_3d is None:
            print(f"Warning: Failed to build 3D points for {frame_name}, skipping")
            continue

        # Generate azimuth map
        azimuth_map = generate_azimuth_map((h, w), camera_type, intrinsics, meta)

        # Preprocess image
        image_tensor = preprocess_image(img_rgb).to(device)
        azimuth_tensor = torch.from_numpy(azimuth_map).unsqueeze(0).unsqueeze(0).to(device)

        # Model inference
        with torch.no_grad():
            pred_dist, pred_exist_logit, _ = model(image_tensor, azimuth_tensor)

        pred_dist = pred_dist[0].cpu().numpy()  # (360,)
        pred_exist_logit = pred_exist_logit[0].cpu().numpy()  # (360,)

        # Determine FOV mask from exist_logit (sigmoid > 0.5)
        continuous_has_data = (torch.sigmoid(torch.from_numpy(pred_exist_logit)).numpy() > 0.5)

        # Compute max distance
        valid_dist = pred_dist[continuous_has_data & np.isfinite(pred_dist)]
        if len(valid_dist) > 0:
            max_dist = np.ceil(np.max(valid_dist) * 1.1)
        else:
            max_dist = 20.0

        # Optional: Load ground mask and fit RANSAC plane
        ground_plane = None
        if args.data_root:
            from scripts.utils.pointcloud import load_masks, _fit_ground_plane_ransac
            mask_path = Path(args.data_root) / 'masks' / args.dataset_name / args.scene_id / f'{frame_name}.npz'
            if mask_path.exists():
                masks = load_masks(str(mask_path.parent), frame_name)
                ground_mask = masks.get('ground_raw')
                if ground_mask is not None and ground_mask.any():
                    ground_pts = points_3d[ground_mask & valid_mask]
                    if len(ground_pts) > 100:
                        ground_plane = _fit_ground_plane_ransac(ground_pts)

        # Project to image
        projected_uv, _ = project_radar_to_image(
            pred_dist, continuous_has_data, points_3d, valid_mask,
            max_dist, ground_plane
        )

        # Draw on LR image for LR video
        if args.output_lr_video:
            lr_rgb_copy = lr_rgb.copy()
            draw_contour_on_image(lr_rgb_copy, projected_uv, color_rgb,
                                 linewidth=args.linewidth, alpha=args.alpha,
                                 break_on_large_jump=break_lines)

        # Cache for HD video if needed
        if inference_cache is not None and hd_rgb is not None:
            # Scale projected coordinates to HD resolution
            if h_hd != h or w_hd != w:
                scale_x = w_hd / w
                scale_y = h_hd / h
                projected_uv_hd = [(int(u * scale_x), int(v * scale_y)) for u, v in projected_uv]
            else:
                projected_uv_hd = projected_uv

            # Calculate video frame range for this inference
            start_frame = int(frame_idx * hd_video_total_frames / num_images_to_process)
            end_frame = int((frame_idx + 1) * hd_video_total_frames / num_images_to_process)

            inference_cache.append({
                'projected_uv_hd': projected_uv_hd,
                'w_hd': w_hd,
                'h_hd': h_hd,
                'start_frame': start_frame,
                'end_frame': end_frame
            })

        # Cache for radar video if needed
        if radar_cache is not None:
            radar_cache.append({
                'pred_dist': pred_dist.copy(),
                'continuous_has_data': continuous_has_data.copy(),
                'max_dist': max_dist
            })

        # Initialize LR video writer on first frame
        if video_writer_lr is None and args.output_lr_video:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer_lr = cv2.VideoWriter(args.output_lr_video, fourcc, fps_lr, (w_lr, h_lr))
            print(f"Initialized LR video writer: {w_lr}x{h_lr} @ {fps_lr:.2f}fps")

        # Write to low-res video
        if video_writer_lr:
            frame_bgr = cv2.cvtColor(lr_rgb_copy, cv2.COLOR_RGB2BGR)
            video_writer_lr.write(frame_bgr)

    # Release LR video writer
    if video_writer_lr is not None:
        video_writer_lr.release()
        print(f"\nLow-res video saved to {args.output_lr_video}")

    # Generate HD video if requested
    if args.output_hd_video and hd_video_cap and inference_cache:
        num_cached_inferences = len(inference_cache)
        print(f"\nGenerating HD video from {hd_video_total_frames} frames...")
        print(f"Using {num_cached_inferences} cached inference results")

        hd_video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to start

        video_fps = hd_video_cap.get(cv2.CAP_PROP_FPS)
        w_hd = inference_cache[0]['w_hd']
        h_hd = inference_cache[0]['h_hd']

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer_hd = cv2.VideoWriter(args.output_hd_video, fourcc, video_fps, (w_hd, h_hd))
        print(f"Initialized HD video writer: {w_hd}x{h_hd} @ {video_fps:.2f}fps")

        for video_frame_idx in tqdm(range(hd_video_total_frames), desc="Writing HD video"):
            # Read video frame
            ret, hd_bgr = hd_video_cap.read()
            if not ret:
                break

            if args.flip_vertical:
                hd_bgr = cv2.flip(hd_bgr, 0)
            hd_rgb = cv2.cvtColor(hd_bgr, cv2.COLOR_BGR2RGB)

            # Find if this frame has a corresponding inference result
            found_inference = False
            for cached in inference_cache:
                if cached['start_frame'] <= video_frame_idx < cached['end_frame']:
                    # Draw cached projection on this frame
                    draw_contour_on_image(hd_rgb, cached['projected_uv_hd'], color_rgb,
                                         linewidth=args.linewidth, alpha=args.alpha,
                                         break_on_large_jump=break_lines)
                    found_inference = True
                    break

            # If no inference found, keep original frame (no drawing)
            frame_bgr = cv2.cvtColor(hd_rgb, cv2.COLOR_RGB2BGR)
            video_writer_hd.write(frame_bgr)

        video_writer_hd.release()
        print(f"High-res video saved to {args.output_hd_video}")

    # Generate radar video if requested
    if args.output_radar_video and radar_cache:
        print(f"\nGenerating radar polar view video from {len(radar_cache)} frames...")

        # Fixed parameters based on camera type
        global_max_dist = 10.0
        if camera_type in ['equirectangular', 'fisheye']:
            theta_min, theta_max = -180, 180
            print(f"Fixed display range: distance={global_max_dist}m, angle=[{theta_min}, {theta_max}] degrees (360° view)")
        else:
            theta_min, theta_max = -50, 50
            print(f"Fixed display range: distance={global_max_dist}m, angle=[{theta_min}, {theta_max}] degrees")

        # Prepare angles
        angles_deg = np.arange(360) - 180.0
        angles_rad = np.deg2rad(angles_deg)

        # Video parameters
        radar_size = 800
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer_radar = cv2.VideoWriter(args.output_radar_video, fourcc, args.fps, (radar_size, radar_size))
        print(f"Initialized radar video writer: {radar_size}x{radar_size} @ {args.fps}fps")

        for cached in tqdm(radar_cache, desc="Writing radar video"):
            # Create polar plot
            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(111, projection='polar')
            ax.set_theta_zero_location('N')
            ax.set_theta_direction(-1)
            ax.set_thetamin(theta_min)
            ax.set_thetamax(theta_max)

            # Plot prediction (linewidth reduced to half: 7.5 -> 3.75)
            plot_polar_radar(ax, angles_rad, cached['pred_dist'], cached['continuous_has_data'],
                           args.color, linewidth=3.75, alpha=0.7)

            ax.set_ylim(0, global_max_dist)
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='both', labelsize=14)

            # Convert to image
            fig.canvas.draw()
            img_array = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
            img_array = img_array.reshape(fig.canvas.get_width_height()[::-1] + (4,))
            img_array = img_array[:, :, :3]  # Remove alpha channel

            # Resize to target size
            img_resized = cv2.resize(img_array, (radar_size, radar_size))
            img_bgr = cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR)

            video_writer_radar.write(img_bgr)
            plt.close(fig)

        video_writer_radar.release()
        print(f"Radar video saved to {args.output_radar_video}")

    if video_writer_lr is None and not (args.output_hd_video and inference_cache):
        print("\nNo frames were processed")

    # Release HD video capture if used
    if hd_video_cap is not None:
        hd_video_cap.release()


def main():
    parser = argparse.ArgumentParser(description='Generate video with traversability projection')

    # Input/output paths
    parser.add_argument('--image-dir', required=True, help='Directory containing sequential image frames for inference')
    parser.add_argument('--hd-image-dir', help='Directory containing high-resolution images for video composition (optional, uses --image-dir if not provided)')
    parser.add_argument('--hd-video', help='High-resolution video file for video composition (alternative to --hd-image-dir)')
    parser.add_argument('--depth-dir', required=True, help='Directory containing depth maps (same naming as images)')
    parser.add_argument('--meta-dir', help='Directory containing depth meta JSON files (optional, uses splits JSON if not provided)')
    parser.add_argument('--output-video', help='Output video path (low-res version, deprecated, use --output-lr-video)')
    parser.add_argument('--output-hd-video', help='Output HD video path (original frame rate, multi-frame per inference)')
    parser.add_argument('--output-lr-video', help='Output low-res video path (scaled frame rate)')
    parser.add_argument('--output-radar-video', help='Output radar polar view video path')

    # Model and camera parameters
    parser.add_argument('--checkpoint', required=True, help='Path to model checkpoint (.pth)')
    parser.add_argument('--splits-json', required=True, help='Path to splits JSON with camera parameters')
    parser.add_argument('--dataset-name', required=True, help='Dataset name (e.g., KITTI-360)')
    parser.add_argument('--scene-id', required=True, help='Scene ID (e.g., 2013_05_28_drive_0000_sync)')
    parser.add_argument('--backbone-weights', required=True, help='Path to DINOv3 backbone weights')

    # Optional parameters
    parser.add_argument('--data-root', help='Data root for mask loading (optional, for ground plane)')
    parser.add_argument('--fps', type=int, default=3, help='Video frame rate (default: 3)')
    parser.add_argument('--max-frames', type=int, default=500, help='Maximum number of frames to process (default: 500)')

    # Visualization parameters
    parser.add_argument('--color', default='#2196F3', help='Line color in hex (default: #2196F3 blue)')
    parser.add_argument('--linewidth', type=int, default=2, help='Line width in pixels (default: 2)')
    parser.add_argument('--alpha', type=float, default=1.0, help='Line transparency 0-1 (default: 1.0)')
    parser.add_argument('--flip-vertical', action='store_true', help='Flip video frames vertically (for upside-down videos)')

    args = parser.parse_args()

    # Backward compatibility
    if args.output_video and not args.output_lr_video:
        args.output_lr_video = args.output_video

    # Validate output paths
    if not args.output_hd_video and not args.output_lr_video and not args.output_radar_video:
        print("[ERROR] At least one output video must be provided")
        return

    # Create output directories
    if args.output_hd_video:
        os.makedirs(os.path.dirname(args.output_hd_video) or '.', exist_ok=True)
    if args.output_lr_video:
        os.makedirs(os.path.dirname(args.output_lr_video) or '.', exist_ok=True)
    if args.output_radar_video:
        os.makedirs(os.path.dirname(args.output_radar_video) or '.', exist_ok=True)

    generate_video(args)


if __name__ == '__main__':
    main()
