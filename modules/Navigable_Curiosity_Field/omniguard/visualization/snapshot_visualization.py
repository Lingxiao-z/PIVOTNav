from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from ..geometry import signed_model_degrees
from ..models.depth_runtime import VisualizationDepthRuntime, resolve_visualization_camera_model
from ..models.unik3d_runtime import load_rays_npz
from .geometry_overlay import azimuth_to_rgb, build_pixel_geometry
from .mpl_style import figure_size_from_pixels, import_pyplot, paper_rc, save_figure


ALGORITHM_LAYER_ROOT = Path(__file__).resolve().parents[5]
NAVIGATION_ROOT = ALGORITHM_LAYER_ROOT.parent
RL_ROOT = NAVIGATION_ROOT.parent
DEFAULT_VISUALIZATION_CONFIG_PATH = ALGORITHM_LAYER_ROOT / "configs" / "omniguard_visualization.yaml"
DEFAULT_SAM3_MODEL_PATH = RL_ROOT / "checkpoints" / "sam3.pt"
_MAX_RANSAC_POINTS = 50_000

_ALLOWED_FIRST_DEBUG_COMPONENTS = {
    "input",
    "azimuth_depth",
    "traversability_projection",
    "traversability_360bins",
    "traversability_overlay",
    "esdf_maps",
}

_DEFAULT_VISUALIZATION_CONFIG: dict[str, Any] = {
    "output": {
        "save_subplots": True,
        "save_subplot_pdf": False,
        "dpi": 160,
        "subplot_width": 960,
        "subplot_height": 720,
        "first_debug_width": 1800,
        "first_debug_tile_height": 620,
    },
    "traversability": {
        "traversable_distance_m": 0.7,
        "probability_threshold": 0.5,
        "overlay_distance_percentile": 95.0,
        "distance_axis_percentile": 100.0,
        "distance_axis_scale": 1.05,
        "distance_axis_min_m": 5.0,
        "distance_axis_max_m": 0.0,
        "projection_tolerance_m": 1.0,
        "projection_line_width_px": 10,
        "projection_mode": "sam3",
        "projection_ground_height_tolerance_m": 0.2,
        "projection_min_distance_m": 8.0,
        "projection_use_ground_plane": True,
        "projection_save_ground_intermediates": True,
        "projection_ground_min_points": 100,
        "projection_ground_reuse_cache": True,
        "projection_ground_fit_mask_key": "ground_raw",
        "projection_sam3_model_path": str(DEFAULT_SAM3_MODEL_PATH),
        "projection_sam3_device": "cuda",
        "projection_sam3_conf": 0.25,
        "projection_ground_prompts": ["ground", "floor", "road"],
        "projection_obstacle_prompts": ["obstacle", "object", "wall", "barrier", "pole", "person", "car"],
    },
    "esdf": {
        "config_path": str(
            ALGORITHM_LAYER_ROOT / "OmniGuard" / "deployment" / "config" / "esdf_guidance.yaml"
        ),
        "radar_axis_percentile": 100.0,
        "radar_axis_scale": 1.05,
        "radar_axis_min_m": 5.0,
        "esdf_color_percentile": 98.0,
    },
    "depth": {
        "enabled": True,
        "save_npz": True,
        "save_visualization": True,
        "colorbar_pad": 0.02,
        "colorbar_fraction": 0.045,
    },
    "first_esdf_debug": {
        "enabled": True,
        "columns": 1,
        "include": [
            "input",
            "azimuth_depth",
            "traversability_projection",
            "traversability_360bins",
            "traversability_overlay",
            "esdf_maps",
        ],
    },
}
_DEPTH_RUNTIME_CACHE: dict[int, VisualizationDepthRuntime] = {}


def _deep_merge(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_visualization_chain_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = Path(path).expanduser() if path else DEFAULT_VISUALIZATION_CONFIG_PATH
    loaded: dict[str, Any] = {}
    if config_path.is_file():
        with config_path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
        if not isinstance(payload, dict):
            raise TypeError(f"Visualization config root must be a mapping: {config_path}")
        loaded = payload
    config = _deep_merge(_DEFAULT_VISUALIZATION_CONFIG, loaded)
    config["config_path"] = str(config_path)
    include = []
    for item in config.get("first_esdf_debug", {}).get("include", []):
        key = str(item).strip()
        if key in _ALLOWED_FIRST_DEBUG_COMPONENTS:
            include.append(key)
    if not include:
        include = list(_DEFAULT_VISUALIZATION_CONFIG["first_esdf_debug"]["include"])
    config.setdefault("first_esdf_debug", {})["include"] = include
    return config


def apply_visualization_runtime_overrides(runtime_config: dict[str, Any], visualization_config: dict[str, Any]) -> None:
    runtime_config["visualization_chain"] = copy.deepcopy(visualization_config)
    traversable = visualization_config.get("traversability", {}).get("traversable_distance_m")
    if traversable is not None:
        runtime_config.setdefault("controllers", {}).setdefault("radar", {})["traversable_distance_m"] = float(traversable)
    if not bool(visualization_config.get("depth", {}).get("enabled", True)):
        runtime_config.setdefault("visualization", {}).setdefault("depth", {})["enabled"] = False


def visualization_config_from_runtime(runtime_config: dict[str, Any]) -> dict[str, Any]:
    value = runtime_config.get("visualization_chain")
    if isinstance(value, dict):
        return _deep_merge(_DEFAULT_VISUALIZATION_CONFIG, value)
    return load_visualization_chain_config()


def _figure_rc() -> dict[str, Any]:
    return paper_rc(
        title_fontsize=15.0,
        base_fontsize=10.0,
        legend_fontsize=8.0,
        tick_fontsize=8.0,
        axes_labelsize=10.0,
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=str)


def _save_image(path: Path, image_bgr: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), np.asarray(image_bgr, dtype=np.uint8)):
        raise IOError(f"Failed to write image: {path}")
    return str(path)


def _colorbar_layout(visualization_config: dict[str, Any]) -> tuple[float, float]:
    depth_cfg = visualization_config.get("depth", {})
    fraction = float(depth_cfg.get("colorbar_fraction", 0.045))
    pad = float(depth_cfg.get("colorbar_pad", 0.02))
    return float(np.clip(fraction, 0.015, 0.12)), float(np.clip(pad, 0.0, 0.2))


def _distance_axis_limit(
    arrays: list[np.ndarray],
    *,
    mask: np.ndarray | None,
    percentile: float,
    scale: float,
    minimum: float,
    maximum: float | None = None,
) -> float:
    valid_parts: list[np.ndarray] = []
    for array in arrays:
        values = np.asarray(array, dtype=np.float32).reshape(-1)
        if mask is not None and mask.shape == values.shape:
            values = values[mask]
        values = values[np.isfinite(values) & (values > 0.0) & (values < 99.0)]
        if values.size:
            valid_parts.append(values)
    if valid_parts:
        joined = np.concatenate(valid_parts)
        farthest = float(np.nanmax(joined))
        limit = float(np.nanpercentile(joined, float(percentile))) * float(scale)
        limit = max(limit, farthest * 1.03)
    else:
        limit = float(minimum)
    limit = max(limit, float(minimum), 1e-3)
    if maximum is not None and maximum > 0.0:
        limit = min(limit, float(maximum))
    return limit


def _ensure_save_parents(*paths: str | Path | None) -> None:
    for value in paths:
        if value is not None:
            Path(value).parent.mkdir(parents=True, exist_ok=True)


def _visualization_intrinsics(frame_bgr: np.ndarray, output: Any) -> dict[str, Any]:
    height, width = frame_bgr.shape[:2]
    intrinsics = dict(getattr(output, "camera_intrinsics", {}) or {})
    camera_model = str(getattr(output, "camera_model", "pinhole"))
    if camera_model == "fisheye" and "rays" not in intrinsics:
        geometry = getattr(output, "camera_geometry", {}) or {}
        rays_source = geometry.get("rays_source")
        if rays_source:
            intrinsics["rays"] = load_rays_npz(str(rays_source), target_hw=(height, width))
    return intrinsics


def _observed_angle_mask(output: Any, n_bins: int) -> np.ndarray:
    mask = getattr(output, "observed_angle_mask", None)
    if mask is not None:
        observed = np.asarray(mask, dtype=bool).reshape(-1)
        if observed.shape == (n_bins,):
            return observed
    return np.ones((n_bins,), dtype=bool)


def _depth_runtime(config: dict[str, Any]) -> VisualizationDepthRuntime:
    key = id(config)
    runtime = _DEPTH_RUNTIME_CACHE.get(key)
    if runtime is None:
        runtime = VisualizationDepthRuntime(config)
        _DEPTH_RUNTIME_CACHE[key] = runtime
    return runtime


def infer_visualization_depth(
    *,
    frame_bgr: np.ndarray,
    output: Any,
    config: dict[str, Any],
    camera_info: dict[str, Any] | None,
    cache_dir: str | Path,
) -> dict[str, Any]:
    depth_cfg = visualization_config_from_runtime(config).get("depth", {})
    runtime_depth_cfg = dict(config.get("visualization", {}).get("depth", {}) or {})
    if not bool(depth_cfg.get("enabled", True)) or not bool(runtime_depth_cfg.get("enabled", True)):
        return {"depth_m": None, "backend": "disabled", "rays": None, "error": "depth disabled"}
    try:
        intrinsics = _visualization_intrinsics(frame_bgr, output)
        camera_model = resolve_visualization_camera_model(
            config,
            camera_info,
            output_camera_model=str(getattr(output, "camera_model", "")),
        )
        result = _depth_runtime(config).infer(
            frame_bgr=frame_bgr,
            camera_model=camera_model,
            intrinsics=intrinsics,
            camera_info=camera_info,
            cache_dir=cache_dir,
        )
        return {
            "depth_m": np.asarray(result.depth_m, dtype=np.float32),
            "backend": str(result.backend),
            "camera_model": str(result.camera_model),
            "rays": result.rays,
            "depth_cache_path": result.depth_cache_path,
            "rays_cache_path": result.rays_cache_path,
            "error": None,
        }
    except Exception as error:
        return {
            "depth_m": None,
            "backend": "unavailable",
            "camera_model": str(getattr(output, "camera_model", "")),
            "rays": None,
            "depth_cache_path": None,
            "rays_cache_path": None,
            "error": f"{type(error).__name__}: {error}",
        }


def save_depth_data_artifacts(
    *,
    output_dir: str | Path,
    frame_stem: str,
    depth_result: dict[str, Any],
    camera_info: dict[str, Any] | None,
    output: Any,
    visualization_config: dict[str, Any],
) -> dict[str, Any]:
    depth_dir = Path(output_dir)
    depth_dir.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, Any] = {
        "backend": depth_result.get("backend"),
        "camera_model": depth_result.get("camera_model"),
        "depth_cache_path": depth_result.get("depth_cache_path"),
        "rays_cache_path": depth_result.get("rays_cache_path"),
        "error": depth_result.get("error"),
    }
    depth_m = depth_result.get("depth_m")
    if depth_m is not None and bool(visualization_config.get("depth", {}).get("save_npz", True)):
        npz_path = depth_dir / f"{frame_stem}_depth.npz"
        np.savez_compressed(
            npz_path,
            depth_m=np.asarray(depth_m, dtype=np.float32),
            backend=np.array(str(depth_result.get("backend", ""))),
            camera_model=np.array(str(depth_result.get("camera_model", ""))),
        )
        artifacts["npz"] = str(npz_path)
        artifacts["depth_shape"] = list(np.asarray(depth_m).shape)
        artifacts["depth_min_m"] = float(np.nanmin(depth_m))
        artifacts["depth_max_m"] = float(np.nanmax(depth_m))
    intrinsics_path = depth_dir / f"{frame_stem}_camera_intrinsics.json"
    _write_json(
        intrinsics_path,
        {
            "camera_info": camera_info,
            "camera_intrinsics": getattr(output, "camera_intrinsics", {}),
            "camera_geometry": getattr(output, "camera_geometry", {}),
        },
    )
    artifacts["camera_intrinsics"] = str(intrinsics_path)
    metadata_path = depth_dir / f"{frame_stem}_depth_metadata.json"
    _write_json(metadata_path, artifacts)
    artifacts["metadata"] = str(metadata_path)
    return artifacts


def _plot_depth(axis: Any, depth_result: dict[str, Any], visualization_config: dict[str, Any]) -> Any | None:
    depth_m = depth_result.get("depth_m")
    axis.axis("off")
    if depth_m is None:
        axis.text(
            0.5,
            0.5,
            str(depth_result.get("error") or "depth unavailable"),
            ha="center",
            va="center",
            transform=axis.transAxes,
            wrap=True,
            color="#991b1b",
        )
        axis.set_title("Depth")
        return None
    depth = np.asarray(depth_m, dtype=np.float32)
    valid = np.isfinite(depth) & (depth > 0)
    if not valid.any():
        axis.text(0.5, 0.5, "depth has no valid pixels", ha="center", va="center", transform=axis.transAxes)
        axis.set_title("Depth")
        return None
    vmax = max(float(np.nanpercentile(depth[valid], 95.0)), 1e-3)
    image = axis.imshow(np.clip(depth, 0.0, vmax), cmap="magma_r", vmin=0.0, vmax=vmax)
    axis.set_title(f"Depth ({depth_result.get('backend', 'unknown')})")
    return image


def _azimuth_rgb(frame_bgr: np.ndarray, output: Any) -> np.ndarray:
    height, width = frame_bgr.shape[:2]
    intrinsics = _visualization_intrinsics(frame_bgr, output)
    geom = build_pixel_geometry(
        image_hw=(height, width),
        camera_model=str(getattr(output, "camera_model", "pinhole")),
        intrinsics=intrinsics,
    )
    return azimuth_to_rgb(geom["azimuth_map"], geom["valid_mask"])


def save_azimuth_depth_panel(
    *,
    path: str | Path,
    frame_bgr: np.ndarray,
    output: Any,
    depth_result: dict[str, Any],
    visualization_config: dict[str, Any],
    pdf_path: str | Path | None = None,
) -> str:
    path = Path(path)
    _ensure_save_parents(path, pdf_path)
    _, plt = import_pyplot()
    dpi = int(visualization_config.get("output", {}).get("dpi", 160))
    cbar_fraction, cbar_pad = _colorbar_layout(visualization_config)
    with plt.rc_context(rc=_figure_rc()):
        fig = plt.figure(figsize=figure_size_from_pixels(1400, 620, dpi), dpi=dpi, constrained_layout=True)
        grid = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, cbar_fraction], wspace=cbar_pad)
        ax_az = fig.add_subplot(grid[0, 0])
        ax_depth = fig.add_subplot(grid[0, 1])
        ax_cbar = fig.add_subplot(grid[0, 2])
        ax_az.imshow(_azimuth_rgb(frame_bgr, output))
        ax_az.set_title(f"Azimuth Map ({getattr(output, 'camera_model', 'camera')})")
        ax_az.axis("off")
        image = _plot_depth(ax_depth, depth_result, visualization_config)
        if image is not None:
            cbar = fig.colorbar(image, cax=ax_cbar)
            cbar.set_label("Depth (m)", labelpad=3.0)
        else:
            ax_cbar.axis("off")
        try:
            save_figure(fig, png_path=path, pdf_path=pdf_path, dpi=dpi, pad_inches=0.03)
        finally:
            plt.close(fig)
    return str(path)


def save_depth_visualization(
    *,
    path: str | Path,
    depth_result: dict[str, Any],
    visualization_config: dict[str, Any],
    pdf_path: str | Path | None = None,
) -> str:
    path = Path(path)
    _ensure_save_parents(path, pdf_path)
    _, plt = import_pyplot()
    dpi = int(visualization_config.get("output", {}).get("dpi", 160))
    cbar_fraction, cbar_pad = _colorbar_layout(visualization_config)
    with plt.rc_context(rc=_figure_rc()):
        fig = plt.figure(figsize=figure_size_from_pixels(900, 620, dpi), dpi=dpi, constrained_layout=True)
        grid = fig.add_gridspec(1, 2, width_ratios=[1.0, cbar_fraction], wspace=cbar_pad)
        ax_depth = fig.add_subplot(grid[0, 0])
        ax_cbar = fig.add_subplot(grid[0, 1])
        image = _plot_depth(ax_depth, depth_result, visualization_config)
        if image is not None:
            cbar = fig.colorbar(image, cax=ax_cbar)
            cbar.set_label("Depth (m)", labelpad=3.0)
        else:
            ax_cbar.axis("off")
        try:
            save_figure(fig, png_path=path, pdf_path=pdf_path, dpi=dpi, pad_inches=0.03)
        finally:
            plt.close(fig)
    return str(path)


def save_traversability_360bins_plot(
    *,
    path: str | Path,
    output: Any,
    config: dict[str, Any],
    visualization_config: dict[str, Any],
    pdf_path: str | Path | None = None,
) -> str:
    path = Path(path)
    _ensure_save_parents(path, pdf_path)
    _, plt = import_pyplot()
    dpi = int(visualization_config.get("output", {}).get("dpi", 160))
    effective = np.asarray(output.effective_distance_m, dtype=np.float32)
    raw = np.asarray(output.raw_distance_m, dtype=np.float32)
    combined = np.asarray(output.combined_distance_m, dtype=np.float32)
    probability = np.asarray(output.exist_probability, dtype=np.float32)
    angles = signed_model_degrees(int(effective.shape[0]))
    observed = _observed_angle_mask(output, int(effective.shape[0]))
    traversable = float(config["controllers"]["radar"]["traversable_distance_m"])
    trav_cfg = visualization_config.get("traversability", {})
    display_max = _distance_axis_limit(
        [raw, combined, effective],
        mask=observed,
        percentile=float(trav_cfg.get("distance_axis_percentile", 100.0)),
        scale=float(trav_cfg.get("distance_axis_scale", 1.05)),
        minimum=max(float(trav_cfg.get("distance_axis_min_m", 5.0)), traversable * 1.2),
        maximum=float(trav_cfg.get("distance_axis_max_m", 0.0)),
    )
    with plt.rc_context(rc=_figure_rc()):
        fig, axes = plt.subplots(
            2,
            1,
            figsize=figure_size_from_pixels(1400, 860, dpi),
            dpi=dpi,
            constrained_layout=True,
        )
        for axis in axes:
            if observed.any():
                obs_angles = angles[observed]
                axis.axvspan(float(obs_angles.min()), float(obs_angles.max()), color="#dbeafe", alpha=0.45, lw=0)
        axes[0].plot(angles, raw, label="raw", linewidth=1.25, color="#7a90a6")
        axes[0].plot(angles, combined, label="combined", linewidth=1.25, color="#8b5cf6")
        axes[0].plot(angles, effective, label="effective", linewidth=1.7, color="#ea580c")
        axes[0].axhline(traversable, color="#15803d", linestyle="--", linewidth=1.0, label="traversable")
        axes[0].set_ylim(0.0, display_max)
        axes[0].set_ylabel("distance (m)")
        axes[0].set_title("Traversability 360-bin Distance")
        axes[0].grid(True, alpha=0.3, linestyle="--")
        axes[0].legend(loc="upper right", frameon=False, ncol=4)
        axes[1].plot(angles, probability, color="#7B1FA2", linewidth=1.4)
        axes[1].axhline(
            float(visualization_config["traversability"].get("probability_threshold", 0.5)),
            color="gray",
            linestyle="--",
            linewidth=1.0,
        )
        axes[1].set_xlabel("angle (deg)")
        axes[1].set_ylabel("exist probability")
        axes[1].set_ylim(-0.02, 1.02)
        axes[1].grid(True, alpha=0.3, linestyle="--")
        try:
            save_figure(fig, png_path=path, pdf_path=pdf_path, dpi=dpi, pad_inches=0.04)
        finally:
            plt.close(fig)
    return str(path)


def render_traversability_bar_overlay(
    *,
    frame_bgr: np.ndarray,
    output: Any,
    config: dict[str, Any],
    visualization_config: dict[str, Any],
) -> np.ndarray:
    overlay = np.asarray(frame_bgr, dtype=np.uint8).copy()
    height, width = overlay.shape[:2]
    effective = np.asarray(output.effective_distance_m, dtype=np.float32)
    probability = np.asarray(output.exist_probability, dtype=np.float32)
    angles = signed_model_degrees(int(effective.shape[0]))
    observed = _observed_angle_mask(output, int(effective.shape[0]))
    visible_angles = angles[observed]
    if visible_angles.size == 0:
        return overlay
    min_angle = float(visible_angles.min())
    max_angle = float(visible_angles.max())
    if max_angle <= min_angle:
        min_angle, max_angle = -180.0, 180.0
    percentile = float(visualization_config["traversability"].get("overlay_distance_percentile", 95.0))
    valid_dist = effective[observed & np.isfinite(effective)]
    max_dist = max(float(np.nanpercentile(valid_dist, percentile)) if valid_dist.size else 1.0, 1.0)
    for angle, distance, prob, is_observed in zip(angles, effective, probability, observed):
        if not bool(is_observed):
            continue
        norm = (float(angle) - min_angle) / max(max_angle - min_angle, 1e-6)
        x = int(round(np.clip(norm, 0.0, 1.0) * max(width - 1, 1)))
        distance_ratio = float(np.clip(distance / max_dist, 0.0, 1.0))
        color = (
            int(round(255 * (1.0 - distance_ratio))),
            int(round(210 * distance_ratio)),
            int(round(255 * (1.0 - float(prob)))),
        )
        line_h = int(round((0.2 + 0.8 * float(prob)) * height * 0.35))
        cv2.line(overlay, (x, height - 1), (x, max(0, height - 1 - line_h)), color, 2, cv2.LINE_AA)
    return overlay


def save_traversability_bar_overlay(
    *,
    path: str | Path,
    frame_bgr: np.ndarray,
    output: Any,
    config: dict[str, Any],
    visualization_config: dict[str, Any],
) -> str:
    image = render_traversability_bar_overlay(
        frame_bgr=frame_bgr,
        output=output,
        config=config,
        visualization_config=visualization_config,
    )
    return _save_image(Path(path), image)


def _projection_camera_type(camera_model: str) -> str:
    value = str(camera_model).strip().lower()
    if value in {"pano", "panorama", "equirect", "equirectangular"}:
        return "equirectangular"
    return value


def _projection_camera_slug(camera_model: str) -> str:
    camera_type = _projection_camera_type(camera_model)
    return "pano" if camera_type == "equirectangular" else camera_type


def _resize_depth_to_frame(depth_m: np.ndarray, frame_bgr: np.ndarray) -> np.ndarray:
    height, width = frame_bgr.shape[:2]
    depth = np.asarray(depth_m, dtype=np.float32)
    if depth.shape == (height, width):
        return depth
    return cv2.resize(depth, (width, height), interpolation=cv2.INTER_NEAREST).astype(np.float32, copy=False)


def _build_unified_depth_points(
    *,
    depth_m: np.ndarray,
    camera_model: str,
    intrinsics: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Mirror unified-depth-processor's build_3d_points convention."""
    depth = np.asarray(depth_m, dtype=np.float32)
    height, width = depth.shape
    valid = np.isfinite(depth) & (depth > 0.0)
    x_map = np.full((height, width), np.nan, dtype=np.float32)
    y_map = np.full((height, width), np.nan, dtype=np.float32)
    z_map = np.full((height, width), np.nan, dtype=np.float32)
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    camera_type = _projection_camera_type(camera_model)

    if camera_type == "pinhole":
        required = ("fx", "fy", "cx", "cy")
        missing = [key for key in required if intrinsics.get(key) is None]
        if missing:
            raise RuntimeError(f"Pinhole projection requires intrinsics: missing {missing}.")
        fx = float(intrinsics["fx"])
        fy = float(intrinsics["fy"])
        cx = float(intrinsics["cx"])
        cy = float(intrinsics["cy"])
        z = depth[valid]
        x_map[valid] = (u[valid] - cx) * z / max(fx, 1e-6)
        y_map[valid] = (v[valid] - cy) * z / max(fy, 1e-6)
        z_map[valid] = z

    elif camera_type == "equirectangular":
        azimuth = (u.astype(np.float32) / max(width, 1) - 0.5) * (2.0 * np.pi)
        elevation = (0.5 - v.astype(np.float32) / max(height, 1)) * np.pi
        x_map[valid] = (np.cos(elevation) * np.sin(azimuth) * depth)[valid]
        y_map[valid] = (-np.sin(elevation) * depth)[valid]
        z_map[valid] = (np.cos(elevation) * np.cos(azimuth) * depth)[valid]

    elif camera_type == "fisheye":
        rays_payload = intrinsics.get("rays")
        if rays_payload is None and intrinsics.get("rays_lr") is not None:
            rays_lr = np.asarray(intrinsics["rays_lr"], dtype=np.float32)
            rays_payload = np.stack(
                [
                    cv2.resize(rays_lr[:, :, channel], (width, height), interpolation=cv2.INTER_LINEAR)
                    for channel in range(3)
                ],
                axis=-1,
            )
        if rays_payload is None:
            raise RuntimeError("Fisheye projection requires rays or rays_lr.")
        rays = np.asarray(rays_payload, dtype=np.float32)
        if rays.shape[:2] != (height, width) or rays.ndim != 3 or rays.shape[2] != 3:
            rays = cv2.resize(rays, (width, height), interpolation=cv2.INTER_LINEAR)
        if rays.shape[:2] != (height, width) or rays.ndim != 3 or rays.shape[2] != 3:
            raise RuntimeError(f"Fisheye rays shape {rays.shape} does not match depth {(height, width)}.")
        norm = np.linalg.norm(rays, axis=2, keepdims=True)
        rays = rays / np.clip(norm, 1e-8, None)
        rays = np.nan_to_num(rays, nan=0.0, posinf=0.0, neginf=0.0)
        valid_rays = valid & np.isfinite(rays).all(axis=2)
        x_map[valid_rays] = rays[:, :, 0][valid_rays] * depth[valid_rays]
        y_map[valid_rays] = rays[:, :, 1][valid_rays] * depth[valid_rays]
        z_map[valid_rays] = rays[:, :, 2][valid_rays] * depth[valid_rays]
        valid = valid_rays

    else:
        raise ValueError(f"Unsupported camera model for projection: {camera_model}")

    valid &= np.isfinite(x_map) & np.isfinite(y_map) & np.isfinite(z_map)
    if not valid.any():
        raise RuntimeError("Depth projection produced no valid RGB pixels.")
    return np.stack([x_map, y_map, z_map], axis=-1), valid.astype(bool, copy=False)


def _fit_ground_plane_ransac(
    points: np.ndarray,
    *,
    n_iter: int = 100,
    dist_threshold: float = 0.05,
    min_inliers: int = 30,
) -> tuple[np.ndarray, float] | None:
    """Same RANSAC ground-plane fit used by unified-depth-processor."""
    points = np.asarray(points, dtype=np.float32)
    if points.shape[0] < 3:
        return None
    rng = np.random.default_rng(42)
    if points.shape[0] > _MAX_RANSAC_POINTS:
        fit_points = points[rng.choice(points.shape[0], _MAX_RANSAC_POINTS, replace=False)]
    else:
        fit_points = points
    sample_count = int(fit_points.shape[0])
    if sample_count < 3:
        return None

    idx = rng.integers(0, sample_count, size=(n_iter, 3))
    idx[:, 1] = (idx[:, 0] + 1 + rng.integers(0, sample_count - 1, size=n_iter)) % sample_count
    idx[:, 2] = (idx[:, 0] + 1 + rng.integers(0, sample_count - 2, size=n_iter)) % sample_count
    p0, p1, p2 = fit_points[idx[:, 0]], fit_points[idx[:, 1]], fit_points[idx[:, 2]]
    normals = np.cross(p1 - p0, p2 - p0)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    valid = norms[:, 0] > 1e-8
    if not valid.any():
        return None

    normals[valid] /= norms[valid]
    ds = np.einsum("ij,ij->i", normals, p0)
    valid_idx = np.where(valid)[0]
    distances = np.abs(fit_points @ normals[valid_idx].T - ds[valid_idx])
    inlier_counts = (distances < float(dist_threshold)).sum(axis=0)
    best_local = int(np.argmax(inlier_counts))
    if int(inlier_counts[best_local]) < int(min_inliers):
        return None

    best_normal = normals[valid_idx[best_local]]
    best_d = float(ds[valid_idx[best_local]])
    inlier_mask = np.abs(points @ best_normal - best_d) < float(dist_threshold)
    inlier_points = points[inlier_mask]
    if inlier_points.shape[0] < int(min_inliers):
        return None
    centroid = inlier_points.mean(axis=0)
    _u, _s, vt = np.linalg.svd(inlier_points - centroid, full_matrices=False)
    normal = vt[-1].astype(np.float32)
    d = float(np.dot(normal, centroid))
    if normal[1] > 0.0:
        normal, d = -normal, -d
    return normal.astype(np.float32, copy=False), d


def _load_npz_dict(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def _save_ground_debug_image(path: Path, mask: np.ndarray) -> str:
    image = (np.asarray(mask, dtype=np.uint8) * 255)
    return _save_image(path, cv2.cvtColor(image, cv2.COLOR_GRAY2BGR))


def _save_ground_debug_overlay(path: Path, frame_bgr: np.ndarray, mask: np.ndarray) -> str:
    overlay = np.asarray(frame_bgr, dtype=np.uint8).copy()
    ground = np.asarray(mask, dtype=bool)
    color = np.zeros_like(overlay)
    color[:, :] = (70, 190, 70)
    overlay[ground] = cv2.addWeighted(overlay[ground], 0.45, color[ground], 0.55, 0.0)
    return _save_image(path, overlay)


def _merge_sam_masks(masks: list[Any], shape: tuple[int, int]) -> np.ndarray:
    merged = np.zeros(shape, dtype=bool)
    for mask in masks:
        if mask is None:
            continue
        array = np.asarray(mask)
        if array.ndim > 2:
            array = np.squeeze(array)
        if array.shape != shape:
            array = cv2.resize(array.astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
        merged |= array.astype(bool)
    return merged


def _run_local_sam3_segmentation(
    *,
    frame_bgr: np.ndarray,
    image_path: Path,
    model_path: Path,
    ground_prompts: list[str],
    obstacle_prompts: list[str],
    conf: float,
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Self-contained copy of the OmniTrav SAM3 prompt flow."""
    if not model_path.is_file():
        raise FileNotFoundError(f"SAM3 model not found: {model_path}")
    from ultralytics.models.sam import SAM3SemanticPredictor
    import torch

    resolved_device = str(device).strip().lower() or "cpu"
    if resolved_device in {"auto", "cuda"}:
        resolved_device = "cuda" if torch.cuda.is_available() else "cpu"

    overrides = {
        "conf": float(conf),
        "task": "segment",
        "mode": "predict",
        "model": str(model_path.resolve()),
        "imgsz": 644,
        "save": False,
        "verbose": False,
        "device": resolved_device,
    }
    feature_extractor = SAM3SemanticPredictor(overrides=overrides)
    mask_decoder = SAM3SemanticPredictor(overrides=overrides)
    try:
        mask_decoder.setup_model()
        feature_extractor.set_image(str(image_path))
        features = feature_extractor.features
        src_shape = frame_bgr.shape[:2]

        def segment(prompts: list[str]) -> list[np.ndarray | None]:
            if not prompts:
                return []
            masks_tensor, _ = mask_decoder.inference_features(features, src_shape=src_shape, text=prompts)
            if masks_tensor is None or len(masks_tensor) == 0:
                return [None] * len(prompts)
            masks_np = masks_tensor.cpu().numpy()
            if masks_np.ndim == 2:
                masks_np = masks_np[np.newaxis, ...]
            return [masks_np[index] > 0.5 if index < len(masks_np) else None for index in range(len(prompts))]

        ground_raw = _merge_sam_masks(segment(ground_prompts), src_shape)
        obstacle_raw = _merge_sam_masks(segment(obstacle_prompts), src_shape)
        return ground_raw, obstacle_raw
    finally:
        del feature_extractor
        del mask_decoder
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def _scale_morph_iters(height: int, width: int, dilate_base: int = 7, erode_base: int = 3) -> tuple[int, int]:
    ref_pixels = 640.0 * 480.0
    scale = ((float(height) * float(width)) / ref_pixels) ** 0.5
    dilate = max(1, int(round(float(dilate_base) * scale)))
    erode = max(1, min(int(round(float(erode_base) * scale)), max(dilate - 1, 1)))
    return dilate, erode


def _refine_ground_masks(
    *,
    ground_raw: np.ndarray,
    obstacle_raw: np.ndarray,
    camera_model: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Self-contained copy of OmniTrav's mask refinement policy."""
    ground = np.asarray(ground_raw, dtype=bool)
    obstacle = np.asarray(obstacle_raw, dtype=bool)
    obstacle_clean = obstacle & ~ground
    height, width = ground.shape
    dilate_iter, erode_iter = _scale_morph_iters(height, width)
    kernel = np.ones((3, 3), dtype=np.uint8)
    ground_dilated = cv2.dilate(ground.astype(np.uint8), kernel, iterations=dilate_iter).astype(bool)
    ground_morphed = cv2.erode(
        ground_dilated.astype(np.uint8),
        kernel,
        iterations=erode_iter,
        borderType=cv2.BORDER_CONSTANT,
        borderValue=1,
    ).astype(bool)
    ground_refined = ground_morphed & ~obstacle_clean
    obstacle_refined = ~ground_refined

    camera_type = _projection_camera_type(camera_model)
    if camera_type == "fisheye":
        cy, cx = height / 2.0, width / 2.0
        radius = min(height, width) / 2.0
        ys, xs = np.ogrid[:height, :width]
        outside_circle = ((ys - cy) ** 2 + (xs - cx) ** 2) > radius**2
        ground_refined |= outside_circle
        obstacle_refined &= ~outside_circle
    elif camera_type == "equirectangular":
        cutoff = int(height * 0.85)
        ground_refined[cutoff:, :] = True
        obstacle_refined[cutoff:, :] = False

    return ground_refined.astype(bool), obstacle_refined.astype(bool)


def _mask_overlay(frame_bgr: np.ndarray, ground: np.ndarray, obstacle: np.ndarray | None = None) -> np.ndarray:
    overlay = np.asarray(frame_bgr, dtype=np.uint8).copy()
    ground_mask = np.asarray(ground, dtype=bool)
    obstacle_mask = np.asarray(obstacle, dtype=bool) if obstacle is not None else np.zeros(ground_mask.shape, dtype=bool)
    ground_color = np.zeros_like(overlay)
    obstacle_color = np.zeros_like(overlay)
    ground_color[:, :] = (70, 190, 70)
    obstacle_color[:, :] = (40, 40, 230)
    overlay[ground_mask] = cv2.addWeighted(overlay[ground_mask], 0.5, ground_color[ground_mask], 0.5, 0.0)
    overlay[obstacle_mask] = cv2.addWeighted(
        overlay[obstacle_mask],
        0.55,
        obstacle_color[obstacle_mask],
        0.45,
        0.0,
    )
    return overlay


def _save_ground_segmentation_panel(
    *,
    path: Path,
    frame_bgr: np.ndarray,
    ground_raw: np.ndarray,
    obstacle_raw: np.ndarray,
    ground: np.ndarray,
    obstacle: np.ndarray,
) -> str:
    h, w = frame_bgr.shape[:2]
    panel_h = h
    panel_w = w * 3
    canvas = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
    canvas[:, 0:w] = frame_bgr
    canvas[:, w : 2 * w] = _mask_overlay(frame_bgr, ground_raw, obstacle_raw)
    canvas[:, 2 * w : 3 * w] = _mask_overlay(frame_bgr, ground, obstacle)
    return _save_image(path, canvas)


def _generate_projection_ground_masks(
    *,
    frame_bgr: np.ndarray,
    output: Any,
    visualization_config: dict[str, Any],
    frame_stem: str,
    ground_dir: str | Path | None,
) -> tuple[dict[str, np.ndarray] | None, dict[str, Any]]:
    trav_cfg = visualization_config.get("traversability", {})
    metadata: dict[str, Any] = {
        "generator": "omniguard_local_sam3",
        "cache_used": False,
    }
    if ground_dir is None:
        metadata["reason"] = "ground output dir unavailable"
        return None, metadata

    ground_root = Path(ground_dir)
    ground_root.mkdir(parents=True, exist_ok=True)
    masks_npz = ground_root / f"{frame_stem}_ground_masks.npz"
    reuse_cache = bool(trav_cfg.get("projection_ground_reuse_cache", True))
    if reuse_cache and masks_npz.is_file():
        mask_data = _load_npz_dict(masks_npz)
        required = {"ground_raw", "obstacle_raw", "ground", "obstacle"}
        if required.issubset(mask_data):
            metadata["cache_used"] = True
            metadata["mask_npz"] = str(masks_npz)
            return {key: np.asarray(mask_data[key], dtype=bool) for key in required}, metadata

    image_path = ground_root / f"{frame_stem}_sam3_input.png"
    _save_image(image_path, frame_bgr)
    model_path = Path(str(trav_cfg.get("projection_sam3_model_path") or DEFAULT_SAM3_MODEL_PATH)).expanduser()
    ground_prompts = [str(item) for item in trav_cfg.get("projection_ground_prompts", [])]
    obstacle_prompts = [str(item) for item in trav_cfg.get("projection_obstacle_prompts", [])]
    metadata.update(
        {
            "sam3_model_path": str(model_path),
            "sam3_device": str(trav_cfg.get("projection_sam3_device", "cpu")),
            "ground_prompts": ground_prompts,
            "obstacle_prompts": obstacle_prompts,
            "sam3_input": str(image_path),
        }
    )

    ground_raw, obstacle_raw = _run_local_sam3_segmentation(
        frame_bgr=frame_bgr,
        image_path=image_path,
        model_path=model_path,
        ground_prompts=ground_prompts,
        obstacle_prompts=obstacle_prompts,
        conf=float(trav_cfg.get("projection_sam3_conf", 0.25)),
        device=str(trav_cfg.get("projection_sam3_device", "cpu")),
    )
    ground, obstacle = _refine_ground_masks(
        ground_raw=ground_raw,
        obstacle_raw=obstacle_raw,
        camera_model=str(getattr(output, "camera_model", "pinhole")),
    )
    np.savez_compressed(
        masks_npz,
        ground_raw=ground_raw.astype(np.uint8),
        obstacle_raw=obstacle_raw.astype(np.uint8),
        ground=ground.astype(np.uint8),
        obstacle=obstacle.astype(np.uint8),
    )
    metadata["mask_npz"] = str(masks_npz)
    metadata["ground_raw_pixels"] = int(np.count_nonzero(ground_raw))
    metadata["obstacle_raw_pixels"] = int(np.count_nonzero(obstacle_raw))
    metadata["ground_pixels"] = int(np.count_nonzero(ground))
    metadata["obstacle_pixels"] = int(np.count_nonzero(obstacle))
    _save_ground_debug_image(ground_root / f"{frame_stem}_ground_raw.png", ground_raw)
    _save_ground_debug_image(ground_root / f"{frame_stem}_obstacle_raw.png", obstacle_raw)
    _save_ground_debug_image(ground_root / f"{frame_stem}_ground_refined.png", ground)
    _save_ground_debug_image(ground_root / f"{frame_stem}_obstacle_refined.png", obstacle)
    metadata["ground_overlay_png"] = _save_ground_debug_overlay(ground_root / f"{frame_stem}_ground_overlay.png", frame_bgr, ground)
    metadata["sam3_overlay_png"] = _save_ground_segmentation_panel(
        path=ground_root / f"{frame_stem}_sam3_ground_debug.png",
        frame_bgr=frame_bgr,
        ground_raw=ground_raw,
        obstacle_raw=obstacle_raw,
        ground=ground,
        obstacle=obstacle,
    )
    return {
        "ground_raw": ground_raw,
        "obstacle_raw": obstacle_raw,
        "ground": ground,
        "obstacle": obstacle,
    }, metadata


def _resolve_projection_ground_plane(
    *,
    frame_bgr: np.ndarray,
    output: Any,
    points_3d: np.ndarray,
    valid_mask: np.ndarray,
    visualization_config: dict[str, Any],
    frame_stem: str,
    ground_dir: str | Path | None,
) -> tuple[tuple[np.ndarray, float] | None, dict[str, Any]]:
    trav_cfg = visualization_config.get("traversability", {})
    projection_mode = str(trav_cfg.get("projection_mode", "sam3")).strip().lower()
    if projection_mode in {"legacy", "none", "depth", "no_ground", "original"}:
        projection_mode = "original"
    elif projection_mode in {"sam", "sam3_ground", "ground", "ground_plane"}:
        projection_mode = "sam3"
    metadata: dict[str, Any] = {
        "mode": projection_mode,
        "enabled": projection_mode == "sam3" and bool(trav_cfg.get("projection_use_ground_plane", True)),
        "ground_plane_fitted": False,
        "ground_points": 0,
    }
    if projection_mode != "sam3":
        metadata["reason"] = "original projection mode"
        if ground_dir is not None:
            ground_root = Path(ground_dir)
            ground_root.mkdir(parents=True, exist_ok=True)
            _write_json(ground_root / f"{frame_stem}_projection_ground.json", metadata)
        return None, metadata
    if not metadata["enabled"]:
        metadata["reason"] = "disabled"
        return None, metadata

    try:
        mask_data, generated_meta = _generate_projection_ground_masks(
            frame_bgr=frame_bgr,
            output=output,
            visualization_config=visualization_config,
            frame_stem=frame_stem,
            ground_dir=ground_dir,
        )
        metadata.update(generated_meta)
    except Exception as error:
        metadata["reason"] = f"ground generation failed: {type(error).__name__}: {error}"
        if ground_dir is not None:
            _write_json(Path(ground_dir) / f"{frame_stem}_projection_ground.json", metadata)
        return None, metadata

    if mask_data is None:
        metadata.setdefault("reason", "ground generation unavailable")
        if ground_dir is not None:
            _write_json(Path(ground_dir) / f"{frame_stem}_projection_ground.json", metadata)
        return None, metadata

    preferred_key = str(trav_cfg.get("projection_ground_fit_mask_key", "ground_raw"))
    key_candidates = [preferred_key, "ground_raw", "ground"]
    ground = None
    ground_key = None
    for key in key_candidates:
        if key in mask_data:
            ground = np.asarray(mask_data[key], dtype=bool)
            ground_key = key
            break
    metadata["ground_mask_key"] = ground_key
    if ground is None:
        metadata["reason"] = "ground key missing"
        if ground_dir is not None:
            _write_json(Path(ground_dir) / f"{frame_stem}_projection_ground.json", metadata)
        return None, metadata
    if ground.shape != valid_mask.shape:
        metadata["reason"] = f"ground mask shape {ground.shape} != depth shape {valid_mask.shape}"
        if ground_dir is not None:
            _write_json(Path(ground_dir) / f"{frame_stem}_projection_ground.json", metadata)
        return None, metadata

    ground_valid = ground & valid_mask
    ground_points = points_3d[ground_valid]
    ground_points = ground_points[np.isfinite(ground_points).all(axis=1)]
    metadata["ground_points"] = int(ground_points.shape[0])
    metadata["ground_pixel_count"] = int(np.count_nonzero(ground))
    min_points = int(trav_cfg.get("projection_ground_min_points", 100))
    if ground_points.shape[0] <= min_points:
        metadata["reason"] = f"not enough ground points <= {min_points}"
        if ground_dir is not None:
            _write_json(Path(ground_dir) / f"{frame_stem}_projection_ground.json", metadata)
        return None, metadata

    plane = _fit_ground_plane_ransac(ground_points)
    if plane is None:
        metadata["reason"] = "ransac failed"
    else:
        normal, d = plane
        metadata["ground_plane_fitted"] = True
        metadata["plane_normal"] = [float(value) for value in normal]
        metadata["plane_d"] = float(d)

    if bool(trav_cfg.get("projection_save_ground_intermediates", True)) and ground_dir is not None:
        ground_root = Path(ground_dir)
        ground_root.mkdir(parents=True, exist_ok=True)
        metadata["ground_mask_png"] = _save_ground_debug_image(ground_root / f"{frame_stem}_ground_mask.png", ground)
        metadata["projection_ground_overlay_png"] = _save_ground_debug_overlay(
            ground_root / f"{frame_stem}_ground_overlay.png",
            frame_bgr,
            ground,
        )
        npz_path = ground_root / f"{frame_stem}_ground_projection_inputs.npz"
        np.savez_compressed(
            npz_path,
            ground_mask=ground.astype(np.uint8),
            valid_depth_mask=valid_mask.astype(np.uint8),
            ground_valid_mask=ground_valid.astype(np.uint8),
            plane_normal=np.asarray(metadata.get("plane_normal", [np.nan, np.nan, np.nan]), dtype=np.float32),
            plane_d=np.asarray(metadata.get("plane_d", np.nan), dtype=np.float32),
        )
        metadata["ground_npz"] = str(npz_path)
        json_path = ground_root / f"{frame_stem}_projection_ground.json"
        _write_json(json_path, metadata)
        metadata["ground_metadata_json"] = str(json_path)

    return plane, metadata


def _project_radar_to_image_unified(
    *,
    radar_dist: np.ndarray,
    continuous_has_data: np.ndarray,
    points_3d: np.ndarray,
    valid_mask: np.ndarray,
    max_dist: float,
    distance_tolerance_m: float,
    ground_height_tolerance_m: float,
    ground_plane: tuple[np.ndarray, float] | None,
) -> list[tuple[int, int]]:
    """Mirror unified-depth-processor's project_radar_to_image behavior."""
    height, width = points_3d.shape[:2]
    distances = np.asarray(radar_dist, dtype=np.float32).reshape(-1)
    observed = np.asarray(continuous_has_data, dtype=bool).reshape(-1)
    n_bins = int(distances.shape[0])
    if observed.shape != (n_bins,):
        observed = np.ones((n_bins,), dtype=bool)

    x = points_3d[:, :, 0][valid_mask]
    y = points_3d[:, :, 1][valid_mask]
    z = points_3d[:, :, 2][valid_mask]
    if ground_plane is not None:
        normal, d = ground_plane
        heights = x * float(normal[0]) + y * float(normal[1]) + z * float(normal[2]) - float(d)
    else:
        heights = None

    azimuth = np.arctan2(x, z)
    dist_3d = np.sqrt(x**2 + z**2)
    bin_idx = ((azimuth + np.pi) / (2.0 * np.pi) * n_bins).astype(np.int32)
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)
    v_coords, u_coords = np.where(valid_mask)
    v_coords = v_coords.astype(np.float32)
    u_coords = u_coords.astype(np.float32)

    candidate_bins: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]] = {}
    for bin_id in range(n_bins):
        mask = bin_idx == bin_id
        if not mask.any():
            continue
        candidate_bins[bin_id] = (
            v_coords[mask],
            u_coords[mask],
            dist_3d[mask],
            heights[mask] if heights is not None else None,
        )

    projected_uv: list[tuple[int, int]] = []
    for bin_id in range(n_bins):
        if not bool(observed[bin_id]):
            continue
        target = float(distances[bin_id])
        if not np.isfinite(target) or target <= 0.0 or target > float(max_dist):
            continue
        if bin_id not in candidate_bins:
            continue

        bin_v, bin_u, bin_dist, bin_height = candidate_bins[bin_id]
        if bin_dist.size == 0:
            continue

        if bin_height is not None:
            ground_candidates = np.abs(bin_height) < float(ground_height_tolerance_m)
            if ground_candidates.any():
                ground_dists = bin_dist[ground_candidates]
                diffs = np.abs(ground_dists - target)
                best = int(np.argmin(diffs))
                if float(diffs[best]) < float(distance_tolerance_m):
                    u = int(np.round(bin_u[ground_candidates][best]))
                    v = int(np.round(bin_v[ground_candidates][best]))
                    if 0 <= u < width and 0 <= v < height:
                        projected_uv.append((u, v))
                continue

        diffs = np.abs(bin_dist - target)
        best = int(np.argmin(diffs))
        if float(diffs[best]) < float(distance_tolerance_m):
            u = int(np.round(bin_u[best]))
            v = int(np.round(bin_v[best]))
            if 0 <= u < width and 0 <= v < height:
                projected_uv.append((u, v))
    return projected_uv


def _draw_unified_contour(
    img_rgb: np.ndarray,
    projected_uv: list[tuple[int, int]],
    *,
    color: tuple[int, int, int] = (0, 255, 0),
    linewidth: int = 10,
    alpha: float = 1.0,
    break_on_large_jump: bool = False,
    jump_threshold: float = 0.5,
) -> np.ndarray:
    if len(projected_uv) < 2:
        return img_rgb
    overlay = img_rgb.copy()
    if break_on_large_jump:
        _h, w = img_rgb.shape[:2]
        threshold_px = float(w) * float(jump_threshold)
        segments: list[list[tuple[int, int]]] = []
        current = [projected_uv[0]]
        for index in range(1, len(projected_uv)):
            u_prev, _v_prev = projected_uv[index - 1]
            u_curr, _v_curr = projected_uv[index]
            if abs(float(u_curr - u_prev)) > threshold_px:
                if len(current) >= 2:
                    segments.append(current)
                current = [projected_uv[index]]
            else:
                current.append(projected_uv[index])
        if len(current) >= 2:
            segments.append(current)
    else:
        segments = [projected_uv]
    for segment in segments:
        pts = np.asarray(segment, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(overlay, [pts], isClosed=False, color=color, thickness=int(linewidth), lineType=cv2.LINE_AA)
    cv2.addWeighted(overlay, float(alpha), img_rgb, 1.0 - float(alpha), 0, img_rgb)
    return img_rgb


def render_traversability_projection(
    *,
    frame_bgr: np.ndarray,
    output: Any,
    config: dict[str, Any],
    depth_result: dict[str, Any],
    visualization_config: dict[str, Any],
    frame_stem: str = "frame_000001",
    ground_output_dir: str | Path | None = None,
) -> np.ndarray:
    depth_m = depth_result.get("depth_m")
    if depth_m is None:
        image = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        cv2.putText(
            image,
            "Depth unavailable for projection",
            (24, 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        return image[:, :, ::-1].copy()

    intrinsics = _visualization_intrinsics(frame_bgr, output)
    if str(getattr(output, "camera_model", "")) == "fisheye" and depth_result.get("rays") is not None:
        intrinsics["rays"] = depth_result["rays"]
    depth = _resize_depth_to_frame(np.asarray(depth_m, dtype=np.float32), frame_bgr)
    points_3d, valid_mask = _build_unified_depth_points(
        depth_m=depth,
        camera_model=str(getattr(output, "camera_model", "pinhole")),
        intrinsics=intrinsics,
    )
    distance = np.asarray(output.effective_distance_m, dtype=np.float32)
    observed = _observed_angle_mask(output, int(distance.shape[0]))
    valid_dist = distance[observed & np.isfinite(distance) & (distance > 0.0)]
    trav_cfg = visualization_config.get("traversability", {})
    max_dist = max(
        float(trav_cfg.get("projection_min_distance_m", 8.0)),
        float(np.nanmax(valid_dist)) if valid_dist.size else 0.0,
    )
    ground_plane, ground_meta = _resolve_projection_ground_plane(
        frame_bgr=frame_bgr,
        output=output,
        points_3d=points_3d,
        valid_mask=valid_mask,
        visualization_config=visualization_config,
        frame_stem=frame_stem,
        ground_dir=ground_output_dir,
    )
    projected_uv = _project_radar_to_image_unified(
        radar_dist=distance,
        continuous_has_data=observed,
        points_3d=points_3d,
        valid_mask=valid_mask,
        max_dist=max_dist,
        distance_tolerance_m=float(trav_cfg.get("projection_tolerance_m", 1.0)),
        ground_height_tolerance_m=float(trav_cfg.get("projection_ground_height_tolerance_m", 0.2)),
        ground_plane=ground_plane,
    )
    if ground_output_dir is not None and bool(trav_cfg.get("projection_save_ground_intermediates", True)):
        metadata = dict(ground_meta)
        metadata.update(
            {
                "projection_max_dist_m": float(max_dist),
                "projection_tolerance_m": float(trav_cfg.get("projection_tolerance_m", 1.0)),
                "projection_ground_height_tolerance_m": float(
                    trav_cfg.get("projection_ground_height_tolerance_m", 0.2)
                ),
                "projected_points": int(len(projected_uv)),
                "valid_depth_pixels": int(np.count_nonzero(valid_mask)),
            }
        )
        _write_json(Path(ground_output_dir) / f"{frame_stem}_projection_metadata.json", metadata)
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).copy()
    _draw_unified_contour(
        rgb,
        projected_uv,
        linewidth=int(visualization_config["traversability"].get("projection_line_width_px", 10)),
        break_on_large_jump=_projection_camera_type(str(getattr(output, "camera_model", ""))) == "equirectangular",
    )
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def save_traversability_projection(
    *,
    path: str | Path,
    frame_bgr: np.ndarray,
    output: Any,
    config: dict[str, Any],
    depth_result: dict[str, Any],
    visualization_config: dict[str, Any],
    frame_stem: str = "frame_000001",
    ground_output_dir: str | Path | None = None,
) -> str:
    image = render_traversability_projection(
        frame_bgr=frame_bgr,
        output=output,
        config=config,
        depth_result=depth_result,
        visualization_config=visualization_config,
        frame_stem=frame_stem,
        ground_output_dir=ground_output_dir,
    )
    return _save_image(Path(path), image)


def save_esdf_maps_panel(
    *,
    path: str | Path,
    output: Any,
    config: dict[str, Any],
    grid_spec: Any,
    occupancy_grid: np.ndarray,
    esdf_grid_m: np.ndarray,
    visualization_config: dict[str, Any],
    pdf_path: str | Path | None = None,
) -> str:
    path = Path(path)
    _ensure_save_parents(path, pdf_path)
    _, plt = import_pyplot()
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.patches import Patch

    dpi = int(visualization_config.get("output", {}).get("dpi", 160))
    cbar_fraction, cbar_pad = _colorbar_layout(visualization_config)
    extent = [
        float(grid_spec.x_min_m),
        float(grid_spec.x_max_m),
        float(grid_spec.y_min_m),
        float(grid_spec.y_max_m),
    ]
    occupancy = np.asarray(occupancy_grid, dtype=np.int8)
    esdf = np.asarray(esdf_grid_m, dtype=np.float32)
    angles = signed_model_degrees(int(output.effective_distance_m.shape[0]))
    observed = _observed_angle_mask(output, int(output.effective_distance_m.shape[0]))
    distance = np.asarray(output.effective_distance_m, dtype=np.float32)
    raw_distance = np.asarray(getattr(output, "raw_distance_m", distance), dtype=np.float32)
    combined_distance = np.asarray(getattr(output, "combined_distance_m", distance), dtype=np.float32)
    esdf_cfg = visualization_config.get("esdf", {})
    radar_max = _distance_axis_limit(
        [raw_distance, combined_distance, distance],
        mask=observed,
        percentile=float(esdf_cfg.get("radar_axis_percentile", 100.0)),
        scale=float(esdf_cfg.get("radar_axis_scale", 1.05)),
        minimum=float(esdf_cfg.get("radar_axis_min_m", 5.0)),
        maximum=float(esdf_cfg.get("radar_axis_max_m", 0.0)),
    )
    with plt.rc_context(rc=_figure_rc()):
        fig = plt.figure(figsize=figure_size_from_pixels(1700, 650, dpi), dpi=dpi, constrained_layout=True)
        grid = fig.add_gridspec(1, 4, width_ratios=[1.05, 1.0, 1.0, cbar_fraction], wspace=max(cbar_pad, 0.02))
        ax_radar = fig.add_subplot(grid[0, 0])
        ax_occ = fig.add_subplot(grid[0, 1])
        ax_esdf = fig.add_subplot(grid[0, 2])
        ax_cbar = fig.add_subplot(grid[0, 3])

        ax_radar.plot(angles, distance, color="#1d4ed8", linewidth=1.6)
        if observed.any():
            obs_angles = angles[observed]
            ax_radar.axvspan(float(obs_angles.min()), float(obs_angles.max()), color="#dbeafe", alpha=0.45, lw=0)
        ax_radar.set_title("Radar Distance")
        ax_radar.set_xlabel("angle (deg)")
        ax_radar.set_ylabel("distance (m)")
        ax_radar.set_ylim(0.0, radar_max)
        ax_radar.grid(True, alpha=0.3, linestyle="--")

        occ_display = np.zeros_like(occupancy, dtype=np.int8)
        occ_display[occupancy == 0] = 1
        occ_display[occupancy == 1] = 2
        occ_cmap = ListedColormap(["#6b7280", "#f8fafc", "#111827"])
        occ_norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], occ_cmap.N)
        ax_occ.imshow(occ_display, origin="lower", extent=extent, cmap=occ_cmap, norm=occ_norm, interpolation="nearest", aspect="equal")
        ax_occ.scatter([0.0], [0.0], s=32.0, c="#ef4444", edgecolors="white", linewidths=0.8, zorder=4)
        ax_occ.set_title("Occupancy Grid")
        ax_occ.set_xlabel("x (m)")
        ax_occ.set_ylabel("y (m)")
        ax_occ.set_xlim(extent[0], extent[1])
        ax_occ.set_ylim(extent[2], extent[3])
        ax_occ.legend(
            handles=[
                Patch(facecolor="#f8fafc", edgecolor="#9ca3af", label="Free"),
                Patch(facecolor="#111827", label="Occupied"),
                Patch(facecolor="#6b7280", label="Unknown"),
            ],
            loc="upper right",
            frameon=True,
            framealpha=0.85,
            fontsize=7.5,
        )

        finite_esdf = esdf[np.isfinite(esdf)]
        if finite_esdf.size:
            vmax = float(np.nanpercentile(finite_esdf, float(esdf_cfg.get("esdf_color_percentile", 98.0))))
        else:
            vmax = 1.0
        vmax = max(vmax, 1e-6)
        image = ax_esdf.imshow(esdf, origin="lower", extent=extent, cmap="viridis", vmin=0.0, vmax=vmax, interpolation="nearest", aspect="equal")
        occupied_overlay = np.ma.masked_where(occupancy != 1, np.ones_like(occupancy, dtype=np.float32))
        ax_esdf.imshow(occupied_overlay, origin="lower", extent=extent, cmap=ListedColormap(["#111827"]), alpha=0.75, aspect="equal")
        ax_esdf.scatter([0.0], [0.0], s=32.0, c="#ef4444", edgecolors="white", linewidths=0.8, zorder=4)
        ax_esdf.set_title("ESDF")
        ax_esdf.set_xlabel("x (m)")
        ax_esdf.set_ylabel("y (m)")
        ax_esdf.set_xlim(extent[0], extent[1])
        ax_esdf.set_ylim(extent[2], extent[3])
        cbar = fig.colorbar(image, cax=ax_cbar)
        cbar.set_label("Clearance (m)", labelpad=3.0)
        try:
            save_figure(fig, png_path=path, pdf_path=pdf_path, dpi=dpi, pad_inches=0.04)
        finally:
            plt.close(fig)
    return str(path)


def save_first_debug_collage(
    *,
    path: str | Path,
    component_paths: dict[str, str],
    visualization_config: dict[str, Any],
    pdf_path: str | Path | None = None,
) -> str:
    path = Path(path)
    _ensure_save_parents(path, pdf_path)
    first_cfg = visualization_config.get("first_esdf_debug", {})
    include = [key for key in first_cfg.get("include", []) if key in component_paths]
    if not include:
        return ""
    columns = max(1, int(first_cfg.get("columns", 1)))
    rows = int(np.ceil(len(include) / float(columns)))
    width = int(visualization_config.get("output", {}).get("first_debug_width", 1800))
    tile_height = int(visualization_config.get("output", {}).get("first_debug_tile_height", 620))
    dpi = int(visualization_config.get("output", {}).get("dpi", 160))
    _, plt = import_pyplot()
    with plt.rc_context(rc=_figure_rc()):
        fig, axes = plt.subplots(
            rows,
            columns,
            figsize=figure_size_from_pixels(width, max(tile_height * rows, 320), dpi),
            dpi=dpi,
            constrained_layout=True,
        )
        axes_arr = np.asarray(axes).reshape(-1)
        for axis in axes_arr:
            axis.axis("off")
        for axis, key in zip(axes_arr, include):
            image = cv2.imread(component_paths[key], cv2.IMREAD_COLOR)
            if image is None:
                axis.text(0.5, 0.5, f"Missing image\n{key}", ha="center", va="center", transform=axis.transAxes)
                continue
            axis.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            axis.set_title(key.replace("_", " ").title(), pad=5.0)
            axis.axis("off")
        try:
            save_figure(fig, png_path=path, pdf_path=pdf_path, dpi=dpi, pad_inches=0.04)
        finally:
            plt.close(fig)
    return str(path)


def save_snapshot_visualization_set(
    *,
    output_dir: str | Path,
    frame_stem: str,
    frame_bgr: np.ndarray,
    output: Any,
    config: dict[str, Any],
    camera_info: dict[str, Any] | None,
    grid_spec: Any,
    occupancy_grid: np.ndarray,
    esdf_grid_m: np.ndarray,
    visualization_config: dict[str, Any] | None = None,
    save_first_debug: bool = True,
    depth_result: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], list[dict[str, str]], dict[str, Any]]:
    root = Path(output_dir)
    vis_cfg = copy.deepcopy(visualization_config or visualization_config_from_runtime(config))
    paths: dict[str, Any] = {"main_images": {}}
    errors: list[dict[str, str]] = []
    output_cfg = vis_cfg.get("output", {})
    save_pdf = bool(output_cfg.get("save_subplot_pdf", False))
    if depth_result is None:
        depth_result = infer_visualization_depth(
            frame_bgr=frame_bgr,
            output=output,
            config=config,
            camera_info=camera_info,
            cache_dir=root / "runtime_depth",
        )

    def capture(name: str, callback: Any) -> None:
        try:
            result = callback()
            if result:
                paths["main_images"][name] = str(result)
        except Exception as error:
            errors.append({"name": name, "error": f"{type(error).__name__}: {error}"})

    def pdf_path_for(name: str) -> Path | None:
        return root / f"{frame_stem}_{name}.pdf" if save_pdf else None

    input_path = root / f"{frame_stem}_input.png"
    capture("input", lambda: _save_image(input_path, frame_bgr))
    depth_dir = root / "depth"
    paths["depth"] = save_depth_data_artifacts(
        output_dir=depth_dir,
        frame_stem=frame_stem,
        depth_result=depth_result,
        camera_info=camera_info,
        output=output,
        visualization_config=vis_cfg,
    )
    if bool(vis_cfg.get("depth", {}).get("save_visualization", True)):
        capture(
            "depth",
            lambda: save_depth_visualization(
                path=root / f"{frame_stem}_depth.png",
                depth_result=depth_result,
                visualization_config=vis_cfg,
                pdf_path=pdf_path_for("depth"),
            ),
        )
    capture(
        "azimuth_depth",
        lambda: save_azimuth_depth_panel(
            path=root / f"{frame_stem}_azimuth_depth.png",
            frame_bgr=frame_bgr,
            output=output,
            depth_result=depth_result,
            visualization_config=vis_cfg,
            pdf_path=pdf_path_for("azimuth_depth"),
        ),
    )
    trav_dir = root / "traversability"
    ground_dir = root / "ground"
    capture(
        "traversability_projection",
        lambda: save_traversability_projection(
            path=root / f"{frame_stem}_traversability_projection.png",
            frame_bgr=frame_bgr,
            output=output,
            config=config,
            depth_result=depth_result,
            visualization_config=vis_cfg,
            frame_stem=frame_stem,
            ground_output_dir=ground_dir,
        ),
    )
    if ground_dir.is_dir():
        paths["ground"] = {"dir": str(ground_dir)}
    capture(
        "traversability_360bins",
        lambda: save_traversability_360bins_plot(
            path=root / f"{frame_stem}_traversability_360bins.png",
            output=output,
            config=config,
            visualization_config=vis_cfg,
            pdf_path=pdf_path_for("traversability_360bins"),
        ),
    )
    capture(
        "traversability_overlay",
        lambda: save_traversability_bar_overlay(
            path=root / f"{frame_stem}_traversability_overlay.png",
            frame_bgr=frame_bgr,
            output=output,
            config=config,
            visualization_config=vis_cfg,
        ),
    )
    esdf_dir = root / "esdf"
    capture(
        "esdf_maps",
        lambda: save_esdf_maps_panel(
            path=root / f"{frame_stem}_esdf_maps.png",
            output=output,
            config=config,
            grid_spec=grid_spec,
            occupancy_grid=occupancy_grid,
            esdf_grid_m=esdf_grid_m,
            visualization_config=vis_cfg,
            pdf_path=pdf_path_for("esdf_maps"),
        ),
    )

    if save_first_debug and bool(vis_cfg.get("first_esdf_debug", {}).get("enabled", True)):
        debug_path = root / f"{frame_stem}_first_esdf_debug.png"
        try:
            result = save_first_debug_collage(
                path=debug_path,
                component_paths=paths["main_images"],
                visualization_config=vis_cfg,
                pdf_path=pdf_path_for("first_esdf_debug"),
            )
            if result:
                paths["first_esdf_debug"] = result
        except Exception as error:
            errors.append({"name": "first_esdf_debug", "error": f"{type(error).__name__}: {error}"})
    return paths, errors, depth_result
