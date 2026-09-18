"""UniNav-DB 可通行性训练数据集。

Loads RGB images and corresponding elevation ground truth from per-frame
NPZ files produced by run_elevation.py.  Computes a per-pixel azimuth map
from camera intrinsics so the model can learn angle-to-spatial mapping.

Directory structure:
    dataset/raw_images/{DATASET}/{SCENE}/frame_XXXXX.png
    dataset/elevation/{DATASET}/{SCENE}/frame_XXXXX.npz

Each elevation NPZ contains:
    profile_data    (N, 2) float32  — (dist, height) pairs in CSR order
    profile_offsets (361,) int32    — CSR row offsets for 360 bins
    has_data        (360,) uint8    — 1 if bin has point cloud data

The clearance_height parameter sets the robot passable-gap threshold:
profile points with height < clearance_height represent locations where
the robot cannot pass.  The first such point (nearest) per bin gives the
obstacle distance.
"""

import json
import os
from collections import OrderedDict
from pathlib import Path
from typing import Dict, FrozenSet, List, Optional


_EMPTY_FROZENSET: FrozenSet[str] = frozenset()

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms as T


# ---------------------------------------------------------------------------
# Custom transforms
# ---------------------------------------------------------------------------

def _pil_to_tensor(pic):
    """Convert PIL Image to tensor, avoiding torch.from_numpy type issues."""
    img = np.array(pic, dtype=np.float32).transpose(2, 0, 1) / 255.0
    return torch.tensor(img, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Cache configuration
# ---------------------------------------------------------------------------

_CACHE_DIR = os.environ.get('UNINAV_CACHE_DIR', './tmp/uninav_cache')
_CACHE_SIZE_GB = float(os.environ.get('UNINAV_CACHE_SIZE_GB', '50'))
_CACHE_ENABLED = os.environ.get('UNINAV_CACHE_ENABLED', '1') == '1'
_CACHE_IMAGES = os.environ.get('UNINAV_CACHE_IMAGES', '0') == '1'


# ---------------------------------------------------------------------------
# Cache infrastructure
# ---------------------------------------------------------------------------

class CachedSample:
    """Cached ground truth and azimuth map for one sample."""
    __slots__ = ('radar_dist', 'has_data', 'continuous_has_data', 'azimuth_map', 'image')

    def __init__(self, radar_dist, has_data, continuous_has_data, azimuth_map, image=None):
        self.radar_dist = radar_dist
        self.has_data = has_data
        self.continuous_has_data = continuous_has_data
        self.azimuth_map = azimuth_map
        self.image = image

    def size_bytes(self) -> int:
        """Estimate memory footprint."""
        size = (
            self.radar_dist.nbytes +
            self.has_data.nbytes +
            self.continuous_has_data.nbytes +
            self.azimuth_map.nbytes
        )
        if self.image is not None:
            size += self.image.nbytes
        return size


class LRUCache:
    """LRU cache with size limit in bytes."""

    def __init__(self, max_size_bytes: int):
        self.max_size_bytes = max_size_bytes
        self.cache: OrderedDict[int, CachedSample] = OrderedDict()
        self.current_size = 0
        self.hits = 0
        self.misses = 0

    def get(self, key: int) -> Optional[CachedSample]:
        if key in self.cache:
            self.cache.move_to_end(key)
            self.hits += 1
            return self.cache[key]
        self.misses += 1
        return None

    def put(self, key: int, value: CachedSample):
        if key in self.cache:
            self.current_size -= self.cache[key].size_bytes()
            del self.cache[key]

        value_size = value.size_bytes()
        while self.current_size + value_size > self.max_size_bytes and self.cache:
            _, evicted = self.cache.popitem(last=False)
            self.current_size -= evicted.size_bytes()

        self.cache[key] = value
        self.current_size += value_size

    def stats(self) -> dict:
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0.0
        return {
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate,
            'size_mb': self.current_size / (1024**2),
            'num_items': len(self.cache),
        }


# ---------------------------------------------------------------------------
# Fisheye ray loader
# ---------------------------------------------------------------------------

def _load_fisheye_rays_from_meta(meta_path: str) -> Optional[np.ndarray]:
    """Load fisheye unit-direction rays from a single depth_meta NPZ file.

    Tries NPZ keys 'rays_lr' (low-res float16) then 'rays' (full-res float32).
    L2-normalises and zeros out NaN/Inf values.

    Args:
        meta_path: Path to a frame-level depth_meta NPZ sidecar.

    Returns:
        float32 array of shape (H, W, 3), or None if no rays key is present or
        the file cannot be read.
    """
    try:
        with np.load(meta_path, allow_pickle=False) as data:
            if 'rays_lr' in data:
                rays = np.asarray(data['rays_lr'], dtype=np.float32)
            elif 'rays' in data:
                rays = np.asarray(data['rays'], dtype=np.float32)
            else:
                return None
    except Exception:
        return None
    if rays.ndim != 3 or rays.shape[2] != 3:
        return None
    norm = np.linalg.norm(rays, axis=2, keepdims=True)
    rays = rays / np.clip(norm, 1e-8, None)
    return np.nan_to_num(rays, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)


# ---------------------------------------------------------------------------
# Azimuth map computation
# ---------------------------------------------------------------------------

def compute_azimuth_map(
    width: int,
    camera_type: str,
    intrinsics: Optional[Dict] = None,
) -> np.ndarray:
    """Compute per-column horizontal azimuth angle from camera model.

    Azimuth only varies along the horizontal axis, so only one row is
    computed and returned as (1, W).  The caller is responsible for
    expanding to full image height if needed.

    The azimuth follows the project convention: 0 = forward (camera Z-axis),
    positive = right, negative = left, range approximately [-pi, pi].

    Args:
        width:       Image width in pixels.
        camera_type: One of 'pinhole', 'fisheye', 'equirectangular'.
        intrinsics:  Camera intrinsics dict.  Required keys depend on type:
            pinhole:         {'fx', 'cx'}
            fisheye:         {'hfov', 'cx'}
            equirectangular: not needed (may be None)

    For fisheye cameras, this function provides a per-column fallback only.
    The accurate per-pixel 2D azimuth is computed in UniNavDataset.__getitem__
    by loading per-frame rays from depth_meta NPZ files.

    Returns:
        (1, W) float32 azimuth in radians.
    """
    u = np.arange(width, dtype=np.float32)

    if camera_type == 'pinhole':
        fx = float(intrinsics['fx'])
        cx = float(intrinsics['cx'])
        # fx is always positive; arctan is faster than arctan2 here
        azimuth_row = np.arctan((u - cx) / fx)  # (W,)

    elif camera_type == 'equirectangular':
        # Precompute scalar step to avoid per-element division
        step = 2.0 * np.pi / width
        azimuth_row = (u - width * 0.5) * step  # [-pi, pi]

    elif camera_type == 'fisheye':
        # Per-column fallback only (ignores vertical displacement from principal point).
        # Used when depth_meta rays are unavailable.
        hfov = float(intrinsics['hfov'])
        cx = float(intrinsics['cx'])
        azimuth_row = (u - cx) / cx * (hfov / 2.0)

    else:
        # Fallback: assume pinhole-like with FOV = 90 deg
        half_w = width / 2.0
        azimuth_row = np.arctan((u - half_w) / half_w)

    return azimuth_row[np.newaxis, :]  # (1, W)


# ---------------------------------------------------------------------------
# Radar distance extraction from CSR profiles
# ---------------------------------------------------------------------------

def _extract_radar_dist(
    profile_data: np.ndarray,
    profile_offsets: np.ndarray,
    clearance_height: float = 0.0,
) -> np.ndarray:
    """Extract nearest obstacle distance per bin from CSR profile data.

    Each bin stores (dist, height) pairs sorted by distance (near to far).
    Height represents the passable gap at that distance.  The first point
    where height < clearance_height is the obstacle location for the robot.

    Fully vectorized implementation without Python loops.

    Args:
        profile_data:    (N, 2) float32 — (dist, height) pairs.
        profile_offsets: (361,) int32 — CSR row offsets.
        clearance_height: Passable-gap threshold (metres).  Points with
            height < this value are treated as blocking obstacles.

    Returns:
        (360,) float32 — nearest obstacle distance per bin; inf if none.
    """
    n_bins = len(profile_offsets) - 1
    radar_dist = np.full(n_bins, np.inf, dtype=np.float32)

    if len(profile_data) == 0:
        return radar_dist

    dists = profile_data[:, 0]
    heights = profile_data[:, 1]
    blocked = heights < clearance_height  # (N,) bool

    starts = profile_offsets[:-1].astype(int)  # (360,)
    ends = profile_offsets[1:].astype(int)      # (360,)

    # Create bin_id array using repeat: each bin ID repeated (end-start) times
    bin_lengths = ends - starts  # (360,) number of points per bin
    bin_ids = np.repeat(np.arange(n_bins), bin_lengths)  # (N,) bin ID for each point

    # Compute local index within each bin
    local_indices = np.arange(len(profile_data)) - starts[bin_ids]  # (N,)

    # Filter to blocked points only
    blocked_mask = blocked
    if not blocked_mask.any():
        return radar_dist

    blocked_bins = bin_ids[blocked_mask]
    blocked_local_idx = local_indices[blocked_mask]
    blocked_dists = dists[blocked_mask]

    # For each bin, find the blocked point with minimum local_idx (= first blocked)
    # Use pandas-style groupby logic with numpy
    sort_idx = np.lexsort((blocked_local_idx, blocked_bins))  # Sort by (bin, local_idx)
    sorted_bins = blocked_bins[sort_idx]
    sorted_dists = blocked_dists[sort_idx]

    # Find first occurrence of each bin (after sorting by bin then local_idx)
    _, first_idx = np.unique(sorted_bins, return_index=True)

    # Assign distances to corresponding bins
    unique_bins = sorted_bins[first_idx]
    first_blocked_dists = sorted_dists[first_idx]
    radar_dist[unique_bins] = first_blocked_dists

    return radar_dist


def _interpolate_radar_dist(
    radar_dist: np.ndarray,
    continuous_has_data: np.ndarray,
    has_data: np.ndarray,
    max_dist: float = 100.0,
) -> np.ndarray:
    """Fill missing radar distances within FOV and clamp passable directions to max_dist.

    Two cases are handled differently:
    - Case A (has_data=True, inf/nan): sensor had hits in this bin but none below
      clearance_height -> path is genuinely passable -> set to max_dist.
    - Case B (has_data=False, within FOV span): no sensor data in this bin despite
      being within the overall FOV arc -> missing data -> linearly interpolate from
      the nearest finite neighbours on each side.

    Args:
        radar_dist: (360,) float32, raw distances (inf = no obstacle or no data).
        continuous_has_data: (360,) bool, True = within FOV span (min to max valid bin).
        has_data: (360,) bool, True = at least one point-cloud hit in this bin (any height).
        max_dist: fallback distance (m) used when no finite neighbours exist.

    Returns:
        (360,) float32, processed radar distances.
    """
    result = radar_dist.copy()
    n = len(result)

    # Case A: sensor had data but all points above clearance -> passable, clamp to max_dist
    passable = continuous_has_data & has_data & (np.isinf(result) | np.isnan(result))
    result[passable] = max_dist

    # Case B: no sensor data within FOV span -> genuinely missing -> interpolate
    needs_interp = continuous_has_data & ~has_data & (np.isinf(result) | np.isnan(result))

    if not needs_interp.any():
        return result

    # Step 3: 找到所有有效数据点 (FOV内且有限值)
    valid = continuous_has_data & np.isfinite(result)

    if not valid.any():
        # 如果FOV内完全没有有效数据，全部设为 max_dist
        result[continuous_has_data] = max_dist
        return result

    # Step 4: 对需要插值的点进行处理 (循环数组，处理周期性)
    for i in np.where(needs_interp)[0]:
        if not continuous_has_data[i]:
            continue

        # 向左查找最近的有效点
        left_idx = None
        left_dist = 0
        for offset in range(1, n):
            idx = (i - offset) % n
            if valid[idx]:
                left_idx = idx
                left_dist = offset
                break

        # 向右查找最近的有效点
        right_idx = None
        right_dist = 0
        for offset in range(1, n):
            idx = (i + offset) % n
            if valid[idx]:
                right_idx = idx
                right_dist = offset
                break

        # 根据找到的有效点进行插值
        if left_idx is not None and right_idx is not None:
            # 两侧都有有效点: 线性插值
            left_val = result[left_idx]
            right_val = result[right_idx]
            total_dist = left_dist + right_dist
            weight_right = left_dist / total_dist
            result[i] = left_val * (1 - weight_right) + right_val * weight_right
        elif left_idx is not None:
            # 只有左侧有效点: 使用左侧值
            result[i] = result[left_idx]
        elif right_idx is not None:
            # 只有右侧有效点: 使用右侧值
            result[i] = result[right_idx]
        else:
            # 没有任何有效点: 使用 max_dist
            result[i] = max_dist

    return result


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class UniNavDataset(Dataset):
    """Dataset that pairs RGB images with elevation ground truth.

    Args:
        data_root:        Root path (e.g., /data/users/renhao/UniNav-DB).
        dataset_name:     Dataset name (e.g., 'DL3DV-1k').
        scenes:           List of scene IDs to include (e.g., ['1', '2']).
                          If None, include all scenes with elevation data.
        splits_json:      Path to splits JSON for camera intrinsics and
                          camera_type.  When provided, each sample gets an
                          azimuth map computed from the scene intrinsics.
        img_size:         Long-edge target size.  The actual (img_h, img_w) is
                          derived from the dataset's native aspect ratio, scaled
                          so the long edge equals img_size (never upscaled),
                          then rounded to the nearest multiple of 16.
        augment:          Enable training augmentation (ColorJitter).
        clearance_height: Passable-gap threshold in metres (default 0.0).
                          Profile points with height < this value are
                          treated as blocking obstacles.
    """

    def __init__(self,
                 data_root: str,
                 dataset_name: str,
                 scenes: Optional[List[str]] = None,
                 splits_json: Optional[str] = None,
                 img_size: int = 384,
                 augment: bool = False,
                 clearance_height: float = 0.5):
        self.data_root = Path(data_root)
        self.dataset_name = dataset_name
        self.augment = augment
        self.clearance_height = clearance_height

        # Load splits JSON for camera info
        self.camera_type = 'pinhole'  # default fallback
        self.scene_intrinsics: Dict[str, Dict] = {}
        self.scene_resolutions: Dict[str, list] = {}
        self._azimuth_cache: Dict[tuple, np.ndarray] = {}
        # Maps scene_id -> frozenset of frame stems excluded by gen_curated_splits.
        self.scene_excluded: Dict[str, FrozenSet[str]] = {}

        complete_scene_ids: Optional[set] = None  # None = no filter

        if splits_json is not None and Path(splits_json).exists():
            with open(splits_json, 'r', encoding='utf-8') as f:
                splits = json.load(f)
            self.camera_type = splits.get('camera_type', 'pinhole')
            complete_scene_ids = set()
            for sc in splits.get('scenes', []):
                sid = str(sc['scene_id'])
                intr = sc.get('intrinsics')
                if intr is not None:
                    self.scene_intrinsics[sid] = intr
                res = sc.get('resolution')
                if res is not None:
                    self.scene_resolutions[sid] = res  # [W, H]
                # Curated JSON scenes have no status field (already filtered
                # upstream); treat empty status as approved.  For standard
                # splits JSONs only check the six boolean stage keys so that
                # integer fields (masks_corrupt, elevation_corrupt) do not
                # incorrectly reject scenes with value 0.
                status = sc.get('status', {})
                if not status or all(v > 0 for v in status.values()):
                    complete_scene_ids.add(sid)
                # Load per-frame exclusion list produced by gen_curated_splits.
                excl = sc.get('excluded_frames')
                if excl:
                    self.scene_excluded[sid] = frozenset(excl)

        # Store img_size for per-scene resolution computation
        self.img_size = img_size

        # Compute per-scene target resolutions
        # Each scene may have different native resolution
        self.scene_target_resolutions: Dict[str, tuple] = {}  # scene_id -> (img_h, img_w)

        if self.scene_resolutions:
            for scene_id, (orig_w, orig_h) in self.scene_resolutions.items():
                scale = min(img_size / max(orig_w, orig_h), 1.0)
                img_h = max(16, round(orig_h * scale / 16) * 16)
                img_w = max(16, round(orig_w * scale / 16) * 16)
                self.scene_target_resolutions[scene_id] = (img_h, img_w)

        # Fallback: use first scene's resolution or square img_size
        if self.scene_target_resolutions:
            self.img_h, self.img_w = next(iter(self.scene_target_resolutions.values()))
        else:
            self.img_h = img_size
            self.img_w = img_size

        rgb_base = self.data_root / 'dataset' / 'raw_images' / dataset_name
        elev_base = self.data_root / 'dataset' / 'elevation' / dataset_name
        meta_base = self.data_root / 'dataset' / 'depth_meta' / dataset_name

        # Collect (rgb_path, elevation_path, scene_id, meta_path) 4-tuples.
        # meta_path points to the depth_meta NPZ sidecar for fisheye ray loading.
        self.samples: List[tuple] = []

        if scenes is None:
            if elev_base.is_dir():
                scenes = sorted([
                    d.name for d in elev_base.iterdir()
                    if d.is_dir() and not d.name.startswith('.')
                ])
            else:
                scenes = []

        # Filter to only scenes that passed all pipeline stages
        if complete_scene_ids is not None:
            before = len(scenes)
            scenes = [s for s in scenes if s in complete_scene_ids]
            skipped = before - len(scenes)
            if skipped > 0:
                print(f"[WARN] Skipped {skipped} incomplete scene(s) based on splits status.")

        for scene_id in scenes:
            scene_elev_dir = elev_base / scene_id
            scene_rgb_dir = rgb_base / scene_id

            if not scene_elev_dir.is_dir():
                continue

            scene_excl = self.scene_excluded.get(scene_id, _EMPTY_FROZENSET)
            for npz_file in sorted(scene_elev_dir.glob('*.npz')):
                frame_name = npz_file.stem

                # Skip frames flagged as outliers by gen_curated_splits.
                if frame_name in scene_excl:
                    continue

                rgb_path = None
                for ext in ('.png', '.jpg', '.jpeg'):
                    candidate = scene_rgb_dir / f'{frame_name}{ext}'
                    if candidate.exists():
                        rgb_path = candidate
                        break

                if rgb_path is not None:
                    meta_path = meta_base / scene_id / f'{frame_name}.npz'
                    self.samples.append(
                        (str(rgb_path), str(npz_file), str(scene_id), str(meta_path))
                    )

        # Image transforms
        # Note: Resize will be applied per-sample based on scene resolution
        if augment:
            self.base_transform = T.Compose([
                T.ColorJitter(
                    brightness=0.2, contrast=0.2,
                    saturation=0.2, hue=0.05,
                ),
                T.RandomGrayscale(p=0.1),
                T.Lambda(_pil_to_tensor),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.base_transform = T.Compose([
                T.Lambda(_pil_to_tensor),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
            ])

        # Initialize caches (only memory cache for ground truth data)
        if _CACHE_ENABLED:
            self._memory_cache = LRUCache(int(_CACHE_SIZE_GB * 1024**3))
        else:
            self._memory_cache = None

    def get_scene_ids(self) -> List[str]:
        """Return sorted unique scene IDs present in this dataset."""
        return sorted(set(s[2] for s in self.samples))

    def __len__(self) -> int:
        return len(self.samples)

    def get_cache_stats(self) -> dict:
        """Return cache statistics."""
        if self._memory_cache is None:
            return {'enabled': False}
        stats = self._memory_cache.stats()
        stats['enabled'] = True
        return stats

    def __getitem__(self, idx: int) -> dict:
        rgb_path, elev_path, scene_id, meta_path = self.samples[idx]

        # Get target resolution for this scene
        if scene_id in self.scene_target_resolutions:
            img_h, img_w = self.scene_target_resolutions[scene_id]
        else:
            img_h, img_w = self.img_h, self.img_w

        # Check memory cache for ground truth data
        cached = None
        if self._memory_cache is not None:
            cached = self._memory_cache.get(idx)

        # If cache hit and image is cached, return immediately
        if cached is not None and cached.image is not None:
            return {
                'image': cached.image.clone(),
                'radar_dist': torch.as_tensor(cached.radar_dist),
                'has_data': torch.as_tensor(cached.has_data),
                'continuous_has_data': torch.as_tensor(cached.continuous_has_data),
                'azimuth_map': torch.as_tensor(cached.azimuth_map),
                'rgb_path': rgb_path,
                'scene_id': scene_id,
            }

        # Load and process RGB image
        img = Image.open(rgb_path).convert('RGB')
        orig_w, orig_h = img.size
        img = T.Resize((img_h, img_w))(img)
        img_tensor = self.base_transform(img)  # (3, img_h, img_w)

        # If ground truth is cached (but not image), return with loaded image
        if cached is not None:
            return {
                'image': img_tensor,
                'radar_dist': torch.as_tensor(cached.radar_dist),
                'has_data': torch.as_tensor(cached.has_data),
                'continuous_has_data': torch.as_tensor(cached.continuous_has_data),
                'azimuth_map': torch.as_tensor(cached.azimuth_map),
                'rgb_path': rgb_path,
                'scene_id': scene_id,
            }

        # Cache miss: load and process ground truth data
        data = None
        for attempt in range(len(self.samples)):
            try:
                data = np.load(elev_path)
                break
            except Exception as e:
                print(f"[WARN] Skipping corrupt elevation file {elev_path}: {e}")
                idx = (idx + 1) % len(self.samples)
                rgb_path, elev_path, scene_id, meta_path = self.samples[idx]
        if data is None:
            raise RuntimeError(f"[ERROR] All {len(self.samples)} samples are corrupt.")

        profile_data    = data['profile_data']                        # (N, 2)
        profile_offsets = data['profile_offsets']                     # (361,)

        # Compute radar_dist with clearance filtering
        radar_dist = _extract_radar_dist(
            profile_data, profile_offsets, self.clearance_height,
        )

        # For fisheye cameras, redefine has_data as bins that have a measured obstacle
        # below clearance_height (i.e. radar_dist is finite).  The NPZ has_data only
        # indicates "any point cloud hit" which is unreliable for fisheye depth models
        # whose inferred depth can extend outside the valid circular region, spuriously
        # marking bins as having data.  Using isfinite(radar_dist) ensures has_data
        # reflects genuine obstacle presence at the configured clearance level.
        # For pinhole/equirect, keep the original NPZ value which correctly tracks FOV.
        if self.camera_type == 'fisheye':
            has_data = np.isfinite(radar_dist)
        else:
            has_data = data['has_data'].astype(bool)               # (360,)

        # Step 1: Make has_data continuous (fill gaps between first and last valid bin)
        valid_bins = np.where(has_data)[0]
        if len(valid_bins) > 0:
            fov_start = valid_bins.min()
            fov_end = valid_bins.max()
            continuous_has_data = np.zeros(360, dtype=bool)
            continuous_has_data[fov_start:fov_end+1] = True
        else:
            continuous_has_data = has_data.copy()

        # Step 2: Clamp passable bins to max_dist; interpolate truly missing bins
        radar_dist_processed = _interpolate_radar_dist(
            radar_dist, continuous_has_data, has_data, max_dist=100.0
        )

        # Step 3: Set outside FOV to NaN
        radar_dist_processed[~continuous_has_data] = np.nan

        # Compute azimuth map.
        # Fisheye: rays are model-inferred and vary per frame, so load per-frame from
        # depth_meta NPZ and compute arctan2(ray_x, ray_z) — true 2D, no caching.
        # Pinhole / equirectangular: azimuth is purely geometric (constant per scene),
        # so compute once from intrinsics and cache per (scene_id, img_h, img_w).
        if self.camera_type == 'fisheye':
            rays = _load_fisheye_rays_from_meta(meta_path)
            if rays is not None:
                # True per-pixel azimuth: horizontal angle of each ray from forward (Z) axis.
                azimuth_2d = np.arctan2(rays[:, :, 0], rays[:, :, 2])  # (H_rays, W_rays)
                az_t = torch.as_tensor(azimuth_2d).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
                az_t = F.interpolate(az_t, size=(img_h, img_w), mode='bilinear', align_corners=False)
                az_tensor = az_t.squeeze(0)  # (1, img_h, img_w)
            else:
                # Fallback: per-column approximation when depth_meta rays are unavailable.
                print(f"[WARN] No rays found at {meta_path}, using column-only azimuth.")
                intrinsics = self.scene_intrinsics.get(scene_id)
                azimuth_row = compute_azimuth_map(orig_w, self.camera_type, intrinsics)
                az_1d = torch.as_tensor(azimuth_row).unsqueeze(0).unsqueeze(0)
                az_1d = F.interpolate(az_1d, size=(1, img_w), mode='bilinear', align_corners=False)
                az_tensor = az_1d.squeeze(0).expand(1, img_h, img_w).contiguous()
        else:
            cache_key = (scene_id, img_h, img_w)
            if cache_key not in self._azimuth_cache:
                intrinsics = self.scene_intrinsics.get(scene_id)
                azimuth_row = compute_azimuth_map(orig_w, self.camera_type, intrinsics)
                az_1d = torch.as_tensor(azimuth_row).unsqueeze(0).unsqueeze(0)  # (1,1,1,orig_w)
                az_1d = F.interpolate(az_1d, size=(1, img_w), mode='bilinear', align_corners=False)
                self._azimuth_cache[cache_key] = az_1d.squeeze(0).expand(1, img_h, img_w).contiguous()
            az_tensor = self._azimuth_cache[cache_key]

        # Store in memory cache
        if self._memory_cache is not None:
            # Optionally cache image tensor if enabled
            cached_image = img_tensor.clone() if _CACHE_IMAGES else None
            cached_sample = CachedSample(
                radar_dist=radar_dist_processed.copy(),
                has_data=has_data.copy(),
                continuous_has_data=continuous_has_data.copy(),
                azimuth_map=az_tensor.numpy().copy(),
                image=cached_image,
            )
            self._memory_cache.put(idx, cached_sample)

        return {
            'image':       img_tensor,
            'radar_dist':  torch.as_tensor(radar_dist_processed),
            'has_data':    torch.as_tensor(has_data),
            'continuous_has_data': torch.as_tensor(continuous_has_data),
            'azimuth_map': az_tensor,
            'rgb_path':    rgb_path,
            'scene_id':    scene_id,
        }


def collate_fn(batch):
    """Custom collate function to handle variable-sized images.
    
    Pads all images and azimuth maps to the maximum size in the batch.
    """
    import torch.nn.functional as F
    
    # Find max dimensions in batch
    max_h = max(sample['image'].shape[1] for sample in batch)
    max_w = max(sample['image'].shape[2] for sample in batch)
    
    # Pad images and azimuth maps
    padded_batch = []
    for sample in batch:
        img = sample['image']
        azimuth = sample['azimuth_map']
        
        # Pad to max size (pad: left, right, top, bottom)
        pad_h = max_h - img.shape[1]
        pad_w = max_w - img.shape[2]
        
        img_padded = F.pad(img, (0, pad_w, 0, pad_h), mode='constant', value=0)
        azimuth_padded = F.pad(azimuth, (0, pad_w, 0, pad_h), mode='constant', value=0)
        
        padded_batch.append({
            'image': img_padded,
            'radar_dist': sample['radar_dist'],
            'has_data': sample['has_data'],
            'continuous_has_data': sample['continuous_has_data'],
            'azimuth_map': azimuth_padded,
            'rgb_path': sample.get('rgb_path', ''),
            'scene_id': sample.get('scene_id', ''),
        })
    
    # Stack into batch
    return {
        'image': torch.stack([s['image'] for s in padded_batch]),
        'radar_dist': torch.stack([s['radar_dist'] for s in padded_batch]),
        'has_data': torch.stack([s['has_data'] for s in padded_batch]),
        'continuous_has_data': torch.stack([s['continuous_has_data'] for s in padded_batch]),
        'azimuth_map': torch.stack([s['azimuth_map'] for s in padded_batch]),
        'rgb_path': [s['rgb_path'] for s in padded_batch],
        'scene_id': [s['scene_id'] for s in padded_batch],
    }
