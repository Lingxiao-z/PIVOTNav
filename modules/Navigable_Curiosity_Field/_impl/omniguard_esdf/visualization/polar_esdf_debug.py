from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from ..geometry import model_heading_rad_to_body_xy
from .mpl_style import figure_size_from_pixels, figure_to_bgr_array, import_pyplot, paper_rc, save_figure


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _paper_style_rc(title_fontsize: float = 20.0, legend_fontsize: float = 10.0) -> dict[str, Any]:
    return paper_rc(
        title_fontsize=title_fontsize,
        base_fontsize=12.0,
        legend_fontsize=legend_fontsize,
        tick_fontsize=11.0,
        axes_labelsize=13.0,
    )


def _grid_extent(grid_meta: dict[str, Any]) -> tuple[float, float, float, float]:
    x_min = _safe_float(grid_meta.get("x_min_m"), -1.0)
    x_max = _safe_float(grid_meta.get("x_max_m"), 8.0)
    y_min = _safe_float(grid_meta.get("y_min_m"), -4.0)
    y_max = _safe_float(grid_meta.get("y_max_m"), 4.0)
    return x_min, x_max, y_min, y_max


def _build_distances_figure(debug_payload: dict[str, Any], *, width: int, height: int, dpi: int) -> Any:
    _, plt = import_pyplot()
    raw = np.asarray(debug_payload.get("raw_distance_m", []), dtype=np.float32)
    smoothed = np.asarray(debug_payload.get("smoothed_distance_m", []), dtype=np.float32)
    observed = np.asarray(debug_payload.get("observed_angle_mask", []), dtype=np.uint8).astype(bool)
    angles_deg = np.linspace(-180.0, 180.0, raw.shape[0], endpoint=False, dtype=np.float32) if raw.size else np.asarray([])
    figure, axes = plt.subplots(figsize=figure_size_from_pixels(width, height, dpi), dpi=dpi, constrained_layout=True)
    axes.set_title("Distances", loc="center", pad=10.0)
    if raw.size == 0:
        axes.text(0.5, 0.5, "No data", ha="center", va="center", transform=axes.transAxes)
        axes.axis("off")
        return figure

    if observed.size == raw.size and np.any(observed):
        observed_angles = angles_deg[observed]
        axes.axvspan(float(observed_angles.min()), float(observed_angles.max()), color="#dbe9f6", alpha=0.7, lw=0.0)
    axes.plot(angles_deg, raw, label="Raw", color="#7a90a6", linewidth=2.3)
    if smoothed.size == raw.size:
        axes.plot(angles_deg, smoothed, label="Smoothed", color="#1f77b4", linewidth=2.6)

    theta_goal = debug_payload.get("theta_goal_deg")
    theta_best = debug_payload.get("theta_best_deg")
    if theta_goal is not None:
        axes.axvline(float(theta_goal), color="#d97706", linestyle="--", linewidth=2.0, label="Goal Heading")
    if theta_best is not None:
        axes.axvline(float(theta_best), color="#0f766e", linestyle="-.", linewidth=2.0, label="Best Heading")

    axes.set_xlim(-180.0, 180.0)
    axes.set_xlabel("Heading (deg)")
    axes.set_ylabel("Distance (m)")
    axes.grid(True, alpha=0.24, linestyle="--", linewidth=0.8)
    axes.legend(loc="upper right", frameon=False, fontsize=10.0, ncol=2)
    return figure


def _build_occupancy_figure(debug_payload: dict[str, Any], *, width: int, height: int, dpi: int) -> Any:
    _, plt = import_pyplot()
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    occupancy_grid = np.asarray(debug_payload.get("occupancy_grid", []), dtype=np.int8)
    grid_meta = dict(debug_payload.get("grid_meta", {}))
    extent = _grid_extent(grid_meta)
    figure, axes = plt.subplots(figsize=figure_size_from_pixels(width, height, dpi), dpi=dpi, constrained_layout=True)
    axes.set_title("Occupancy", loc="center", pad=10.0)
    if occupancy_grid.size == 0:
        axes.text(0.5, 0.5, "No data", ha="center", va="center", transform=axes.transAxes)
        axes.axis("off")
        return figure

    occupancy_display = np.zeros_like(occupancy_grid, dtype=np.int8)
    occupancy_display[occupancy_grid == 0] = 1
    occupancy_display[occupancy_grid == 1] = 2
    cmap = ListedColormap(["#7a7a7a", "#f4f4f4", "#202020"])
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)
    axes.imshow(
        occupancy_display,
        origin="lower",
        extent=extent,
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
        aspect="equal",
    )

    boundary_points = np.asarray(debug_payload.get("boundary_points_local_m", []), dtype=np.float32).reshape(-1, 2)
    if boundary_points.size:
        axes.scatter(boundary_points[:, 0], boundary_points[:, 1], s=10.0, c="#d94801", marker="o", linewidths=0.0)
    axes.scatter([0.0], [0.0], s=30.0, c="black", marker="o", zorder=5)
    axes.set_xlim(extent[0], extent[1])
    axes.set_ylim(extent[2], extent[3])
    axes.set_xlabel("x (m)")
    axes.set_ylabel("y (m)")
    legend_handles = [
        Patch(facecolor="#f4f4f4", edgecolor="none", label="Free"),
        Patch(facecolor="#202020", edgecolor="none", label="Occupied"),
        Patch(facecolor="#7a7a7a", edgecolor="none", label="Unknown"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#d94801", markersize=6, label="Boundary Hits"),
    ]
    axes.legend(handles=legend_handles, loc="upper right", frameon=False, fontsize=9.5, ncol=2)
    return figure


def _build_esdf_figure(debug_payload: dict[str, Any], *, width: int, height: int, dpi: int) -> Any:
    _, plt = import_pyplot()
    from matplotlib.colors import ListedColormap
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    esdf_grid = np.asarray(debug_payload.get("esdf_grid_m", []), dtype=np.float32)
    occupancy_grid = np.asarray(debug_payload.get("occupancy_grid", []), dtype=np.int8)
    rollout_points = debug_payload.get("rollout_points_local_m", [])
    candidate_scores = np.asarray(debug_payload.get("candidate_scores", []), dtype=np.float32)
    grid_meta = dict(debug_payload.get("grid_meta", {}))
    extent = _grid_extent(grid_meta)
    figure, axes = plt.subplots(figsize=figure_size_from_pixels(width, height, dpi), dpi=dpi, constrained_layout=True)
    axes.set_title("Local ESDF", loc="center", pad=10.0)
    if esdf_grid.size == 0:
        axes.text(0.5, 0.5, "No data", ha="center", va="center", transform=axes.transAxes)
        axes.axis("off")
        return figure

    free_mask = occupancy_grid == 0
    esdf_free = np.ma.masked_where(~free_mask, esdf_grid)
    vmax = max(1e-6, float(np.nanmax(esdf_grid)))
    image = axes.imshow(
        esdf_free,
        origin="lower",
        extent=extent,
        cmap="RdYlBu",
        vmin=0.0,
        vmax=vmax,
        interpolation="nearest",
        aspect="equal",
    )
    colorbar = figure.colorbar(image, ax=axes, fraction=0.046, pad=0.03)
    colorbar.set_label("Clearance (m)")

    if occupancy_grid.shape == esdf_grid.shape:
        occupied_overlay = np.ma.masked_where(occupancy_grid != 1, np.ones_like(occupancy_grid, dtype=np.float32))
        unknown_overlay = np.ma.masked_where(occupancy_grid != -1, np.ones_like(occupancy_grid, dtype=np.float32))
        axes.imshow(unknown_overlay, origin="lower", extent=extent, cmap=ListedColormap(["#9a9a9a"]), alpha=0.92, aspect="equal")
        axes.imshow(occupied_overlay, origin="lower", extent=extent, cmap=ListedColormap(["#111111"]), alpha=0.96, aspect="equal")

    theta_goal = debug_payload.get("theta_goal_deg")
    theta_best = debug_payload.get("theta_best_deg")
    if theta_goal is not None:
        goal_x, goal_y = model_heading_rad_to_body_xy(math.radians(float(theta_goal)), 1.5)
        axes.annotate("", xy=(goal_x, goal_y), xytext=(0.0, 0.0), arrowprops={"arrowstyle": "->", "lw": 2.2, "color": "#d97706"})
    if theta_best is not None:
        best_x, best_y = model_heading_rad_to_body_xy(math.radians(float(theta_best)), 1.5)
        axes.annotate("", xy=(best_x, best_y), xytext=(0.0, 0.0), arrowprops={"arrowstyle": "->", "lw": 2.2, "color": "#0f766e"})

    if rollout_points and candidate_scores.size == len(rollout_points) and np.any(np.isfinite(candidate_scores)):
        best_index = int(np.argmax(np.where(np.isfinite(candidate_scores), candidate_scores, -np.inf)))
        best_rollout = np.asarray(rollout_points[best_index], dtype=np.float32).reshape(-1, 2)
        if best_rollout.size:
            axes.plot(best_rollout[:, 0], best_rollout[:, 1], color="white", linewidth=2.2, label="Best Rollout")

    axes.scatter([0.0], [0.0], s=28.0, c="black", marker="o", zorder=5)
    axes.set_xlim(extent[0], extent[1])
    axes.set_ylim(extent[2], extent[3])
    axes.set_xlabel("x (m)")
    axes.set_ylabel("y (m)")
    legend_handles = [
        Patch(facecolor="#111111", edgecolor="none", label="Occupied"),
        Patch(facecolor="#9a9a9a", edgecolor="none", label="Unknown"),
        Line2D([0], [0], color="#d97706", linewidth=2.2, label="Goal Heading"),
        Line2D([0], [0], color="#0f766e", linewidth=2.2, label="Best Heading"),
        Line2D([0], [0], color="white", linewidth=2.2, label="Best Rollout"),
    ]
    axes.legend(handles=legend_handles, loc="upper right", frameon=False, fontsize=9.5, ncol=2)
    return figure


def _build_candidate_scores_figure(debug_payload: dict[str, Any], *, width: int, height: int, dpi: int) -> Any:
    _, plt = import_pyplot()
    from matplotlib.patches import Patch

    candidate_scores = np.asarray(debug_payload.get("candidate_scores", []), dtype=np.float32)
    candidate_min_esdf = np.asarray(debug_payload.get("candidate_min_esdf_m", []), dtype=np.float32)
    figure, axes = plt.subplots(figsize=figure_size_from_pixels(width, height, dpi), dpi=dpi, constrained_layout=True)
    axes.set_title("Candidate Scores", loc="center", pad=10.0)
    if candidate_scores.size == 0:
        axes.text(0.5, 0.5, "No data", ha="center", va="center", transform=axes.transAxes)
        axes.axis("off")
        return figure

    candidate_indices = np.arange(candidate_scores.size, dtype=np.float32)
    best_index = int(np.argmax(np.where(np.isfinite(candidate_scores), candidate_scores, -np.inf)))
    colors: list[str] = []
    for index, score in enumerate(candidate_scores):
        if not math.isfinite(float(score)):
            colors.append("#b0b0b0")
        elif index < candidate_min_esdf.shape[0] and float(candidate_min_esdf[index]) <= 0.2:
            colors.append("#c44e52")
        elif index == best_index:
            colors.append("#dd8452")
        else:
            colors.append("#4c72b0")
    axes.bar(candidate_indices, candidate_scores, width=0.82, color=colors, edgecolor="black", linewidth=0.4)
    axes.axhline(0.0, color="#555555", linewidth=1.0)
    axes.set_xlabel("Candidate index")
    axes.set_ylabel("Score")
    axes.grid(True, axis="y", alpha=0.24, linestyle="--", linewidth=0.8)
    axes.legend(
        handles=[
            Patch(facecolor="#4c72b0", edgecolor="black", linewidth=0.4, label="Candidate"),
            Patch(facecolor="#dd8452", edgecolor="black", linewidth=0.4, label="Best"),
            Patch(facecolor="#c44e52", edgecolor="black", linewidth=0.4, label="Low Clearance"),
            Patch(facecolor="#b0b0b0", edgecolor="black", linewidth=0.4, label="Invalid"),
        ],
        loc="upper right",
        frameon=False,
        fontsize=9.5,
        ncol=2,
    )
    return figure


def render_polar_esdf_debug_subplots(debug_payload: dict[str, Any] | None, *, width: int, height: int) -> dict[str, np.ndarray]:
    if not debug_payload:
        empty = np.full((max(int(height), 240), max(int(width), 320), 3), 245, dtype=np.uint8)
        return {"distances": empty.copy(), "occupancy": empty.copy(), "esdf": empty.copy(), "candidate_scores": empty.copy()}

    _, plt = import_pyplot()
    dpi = 160
    images: dict[str, np.ndarray] = {}
    builders = {
        "distances": _build_distances_figure,
        "occupancy": _build_occupancy_figure,
        "esdf": _build_esdf_figure,
        "candidate_scores": _build_candidate_scores_figure,
    }
    with plt.rc_context(rc=_paper_style_rc()):
        for stem, builder in builders.items():
            figure = builder(debug_payload, width=int(width), height=int(height), dpi=dpi)
            try:
                images[stem] = figure_to_bgr_array(figure)
            finally:
                plt.close(figure)
    return images


def save_polar_esdf_subplot_artifacts(
    debug_payload: dict[str, Any] | None,
    *,
    output_dir: str | Path,
    width: int,
    height: int,
    png_dirname: str = "esdf_debug_subplots_png",
    pdf_dirname: str = "esdf_debug_subplots_pdf",
    pdf_dpi: int = 150,
) -> dict[str, dict[str, str]]:
    if not debug_payload:
        return {}

    _, plt = import_pyplot()
    builders = {
        "distances": _build_distances_figure,
        "occupancy": _build_occupancy_figure,
        "esdf": _build_esdf_figure,
        "candidate_scores": _build_candidate_scores_figure,
    }
    root_dir = Path(output_dir)
    png_dir = root_dir / png_dirname
    pdf_dir = root_dir / pdf_dirname
    png_dir.mkdir(parents=True, exist_ok=True)
    pdf_dir.mkdir(parents=True, exist_ok=True)

    artifacts: dict[str, dict[str, str]] = {}
    dpi = max(120, int(pdf_dpi))
    with plt.rc_context(rc=_paper_style_rc()):
        for stem, builder in builders.items():
            png_path = png_dir / f"{stem}.png"
            pdf_path = pdf_dir / f"{stem}.pdf"
            figure = builder(debug_payload, width=int(width), height=int(height), dpi=dpi)
            try:
                save_figure(figure, png_path=png_path, pdf_path=pdf_path, dpi=dpi, pad_inches=0.05)
            finally:
                plt.close(figure)
            artifacts[stem] = {"png_path": str(png_path), "pdf_path": str(pdf_path)}
    return artifacts


def _build_polar_esdf_panel_figure(debug_payload: dict[str, Any] | None, *, width: int, panel_height: int, dpi: int = 160) -> Any:
    matplotlib, plt = import_pyplot()
    width = max(int(width), 640)
    panel_height = max(int(panel_height), 480)
    subplot_width = max((width - 48) // 2, 320)
    subplot_height = max((panel_height - 120) // 2, 240)
    subplot_images = render_polar_esdf_debug_subplots(debug_payload, width=subplot_width, height=subplot_height)

    with matplotlib.rc_context(rc=_paper_style_rc()):
        figure = plt.figure(figsize=figure_size_from_pixels(width, panel_height, dpi), dpi=dpi, constrained_layout=True)
        grid = figure.add_gridspec(3, 2, height_ratios=[0.2, 1.0, 1.0], hspace=0.04, wspace=0.04)
        ax_status = figure.add_subplot(grid[0, :])
        ax_status.axis("off")
        status_text = (
            f"state={debug_payload.get('state', 'n/a') if debug_payload else 'n/a'}    "
            f"theta_g={_safe_float(None if debug_payload is None else debug_payload.get('theta_goal_deg'), 0.0):+.1f} deg    "
            f"theta_best={_safe_float(None if debug_payload is None else debug_payload.get('theta_best_deg'), 0.0):+.1f} deg    "
            f"front_esdf={_safe_float(None if debug_payload is None else debug_payload.get('front_min_esdf_m'), 0.0):.2f} m"
        )
        ax_status.text(0.5, 0.72, "Polar ESDF Debug Panel", ha="center", va="center", transform=ax_status.transAxes)
        ax_status.text(
            0.5,
            0.20,
            status_text,
            ha="center",
            va="center",
            transform=ax_status.transAxes,
            fontsize=12.0,
            bbox=dict(boxstyle="round,pad=0.45", facecolor="#f5f5f5", edgecolor="#d0d0d0", alpha=0.96),
        )

        axes = [
            figure.add_subplot(grid[1, 0]),
            figure.add_subplot(grid[1, 1]),
            figure.add_subplot(grid[2, 0]),
            figure.add_subplot(grid[2, 1]),
        ]
        ordered_stems = ["distances", "occupancy", "esdf", "candidate_scores"]
        for axis, stem in zip(axes, ordered_stems):
            image_bgr = subplot_images.get(stem)
            if image_bgr is not None:
                axis.imshow(image_bgr[:, :, ::-1])
            axis.axis("off")
        return figure


def save_polar_esdf_panel_artifacts(
    debug_payload: dict[str, Any] | None,
    *,
    png_path: str | Path | None,
    pdf_path: str | Path | None,
    width: int,
    panel_height: int,
    dpi: int = 160,
) -> None:
    if not debug_payload:
        return
    _, plt = import_pyplot()
    figure = _build_polar_esdf_panel_figure(debug_payload, width=width, panel_height=panel_height, dpi=dpi)
    try:
        save_figure(figure, png_path=png_path, pdf_path=pdf_path, dpi=dpi, pad_inches=0.04)
    finally:
        plt.close(figure)


def render_polar_esdf_debug_panel(debug_payload: dict[str, Any] | None, *, width: int, panel_height: int = 320) -> np.ndarray:
    if not debug_payload:
        return np.full((panel_height, width, 3), 245, dtype=np.uint8)
    _, plt = import_pyplot()
    figure = _build_polar_esdf_panel_figure(debug_payload, width=width, panel_height=panel_height, dpi=160)
    try:
        return figure_to_bgr_array(figure)
    finally:
        plt.close(figure)


def save_polar_esdf_debug_artifacts(
    debug_payload: dict[str, Any] | None,
    *,
    output_dir: str | Path,
    frame_id: int,
    panel_width: int,
    panel_height: int = 320,
    save_panel_image: bool = True,
) -> dict[str, str]:
    if not debug_payload:
        return {}

    root_dir = Path(output_dir)
    arrays_dir = root_dir / "polar_esdf_arrays"
    panels_dir = root_dir / "polar_esdf_panels"
    arrays_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "raw_distance_m": np.asarray(debug_payload.get("raw_distance_m", []), dtype=np.float32),
        "ema_distance_m": np.asarray(debug_payload.get("ema_distance_m", []), dtype=np.float32),
        "smoothed_distance_m": np.asarray(debug_payload.get("smoothed_distance_m", []), dtype=np.float32),
        "observed_angle_mask": np.asarray(debug_payload.get("observed_angle_mask", []), dtype=np.uint8),
        "candidate_angles_deg": np.asarray(debug_payload.get("candidate_angles_deg", []), dtype=np.float32),
        "candidate_scores": np.asarray(debug_payload.get("candidate_scores", []), dtype=np.float32),
        "candidate_mean_esdf_m": np.asarray(debug_payload.get("candidate_mean_esdf_m", []), dtype=np.float32),
        "candidate_min_esdf_m": np.asarray(debug_payload.get("candidate_min_esdf_m", []), dtype=np.float32),
        "boundary_points_local_m": np.asarray(debug_payload.get("boundary_points_local_m", []), dtype=np.float32),
        "occupancy_grid": np.asarray(debug_payload.get("occupancy_grid", []), dtype=np.int8),
        "esdf_grid_m": np.asarray(debug_payload.get("esdf_grid_m", []), dtype=np.float32),
        "theta_goal_deg": np.asarray([_safe_float(debug_payload.get("theta_goal_deg"), 0.0)], dtype=np.float32),
        "theta_best_deg": np.asarray([_safe_float(debug_payload.get("theta_best_deg"), 0.0)], dtype=np.float32),
        "front_min_esdf_m": np.asarray([_safe_float(debug_payload.get("front_min_esdf_m"), 0.0)], dtype=np.float32),
        "path_min_esdf_m": np.asarray([_safe_float(debug_payload.get("path_min_esdf_m"), 0.0)], dtype=np.float32),
        "grid_meta_json": np.asarray([json.dumps(debug_payload.get("grid_meta", {}), ensure_ascii=False)], dtype="<U512"),
    }
    array_path = arrays_dir / f"frame_{int(frame_id):06d}.npz"
    np.savez_compressed(array_path, **payload)

    result = {"array_path": str(array_path)}
    if save_panel_image:
        panels_dir.mkdir(parents=True, exist_ok=True)
        panel_path = panels_dir / f"frame_{int(frame_id):06d}.png"
        save_polar_esdf_panel_artifacts(
            debug_payload,
            png_path=panel_path,
            pdf_path=None,
            width=int(panel_width),
            panel_height=int(panel_height),
        )
        result["panel_path"] = str(panel_path)
    return result
