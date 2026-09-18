from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from ..geometry import model_heading_rad_to_body_xy
from .mpl_style import import_pyplot, paper_rc, save_figure


def _grid_extent(grid_spec: Any) -> list[float]:
    return [
        float(grid_spec.x_min_m),
        float(grid_spec.x_max_m),
        float(grid_spec.y_min_m),
        float(grid_spec.y_max_m),
    ]


def _finite_scores(scores: np.ndarray) -> np.ndarray:
    if scores.size == 0:
        return scores.astype(np.float32, copy=True)
    finite_mask = np.isfinite(scores)
    if not np.any(finite_mask):
        return np.zeros_like(scores, dtype=np.float32)
    min_score = float(np.min(scores[finite_mask]))
    fallback = min_score - max(1.0, abs(min_score) * 0.1)
    return np.where(finite_mask, scores, fallback).astype(np.float32)


def _score_norm(scores: np.ndarray) -> tuple[np.ndarray, float, float]:
    display_scores = _finite_scores(scores)
    if display_scores.size == 0:
        return display_scores, 0.0, 1.0
    vmin = float(np.min(display_scores))
    vmax = float(np.max(display_scores))
    if abs(vmax - vmin) < 1e-6:
        vmax = vmin + 1.0
    normalized = (display_scores - vmin) / (vmax - vmin)
    return normalized.astype(np.float32), vmin, vmax


def _plot_esdf_base(
    axis: Any,
    *,
    grid_spec: Any,
    occupancy_grid: np.ndarray,
    esdf_grid_m: np.ndarray,
    title: str,
    colorbar: bool,
) -> Any:
    from matplotlib.colors import ListedColormap

    image = axis.imshow(
        np.asarray(esdf_grid_m, dtype=np.float32),
        origin="lower",
        extent=_grid_extent(grid_spec),
        cmap="viridis",
        interpolation="nearest",
        aspect="equal",
        vmin=0.0,
        vmax=max(1e-6, float(np.nanmax(esdf_grid_m))),
    )
    occupancy = np.asarray(occupancy_grid, dtype=np.int8)
    unknown_overlay = np.ma.masked_where(occupancy != -1, np.ones_like(occupancy, dtype=np.float32))
    occupied_overlay = np.ma.masked_where(occupancy != 1, np.ones_like(occupancy, dtype=np.float32))
    axis.imshow(
        unknown_overlay,
        origin="lower",
        extent=_grid_extent(grid_spec),
        cmap=ListedColormap(["#6b7280"]),
        interpolation="nearest",
        aspect="equal",
        alpha=0.70,
    )
    axis.imshow(
        occupied_overlay,
        origin="lower",
        extent=_grid_extent(grid_spec),
        cmap=ListedColormap(["#111827"]),
        interpolation="nearest",
        aspect="equal",
        alpha=0.88,
    )
    axis.scatter([0.0], [0.0], marker="^", s=56, color="#f9fafb", edgecolor="#111827", linewidth=0.8, zorder=10)
    axis.set_title(title, pad=9.0)
    axis.set_xlabel("x (m)")
    axis.set_ylabel("y (m)")
    axis.grid(False)
    axis.set_xlim(float(grid_spec.x_min_m), float(grid_spec.x_max_m))
    axis.set_ylim(float(grid_spec.y_min_m), float(grid_spec.y_max_m))
    if colorbar:
        cb = axis.figure.colorbar(image, ax=axis, fraction=0.046, pad=0.03)
        cb.set_label("Clearance (m)", labelpad=4.0)
        cb.ax.tick_params(labelsize=7)
    return image


def _plot_direction_panel(
    axis: Any,
    *,
    grid_spec: Any,
    occupancy_grid: np.ndarray,
    esdf_grid_m: np.ndarray,
    scoring: Any,
    theta_goal_rad: float,
    best_theta_rad: float | None,
) -> None:
    from matplotlib import cm
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from matplotlib.lines import Line2D

    scores = np.asarray(getattr(scoring, "scores", []), dtype=np.float32)
    score_values, score_min, score_max = _score_norm(scores)
    cmap = cm.get_cmap("coolwarm")
    _plot_esdf_base(
        axis,
        grid_spec=grid_spec,
        occupancy_grid=occupancy_grid,
        esdf_grid_m=esdf_grid_m,
        title="Rule Directions on ESDF",
        colorbar=False,
    )

    rollouts = list(getattr(scoring, "rollout_points_local_m", []) or [])
    for index, points in enumerate(rollouts):
        rollout = np.asarray(points, dtype=np.float32).reshape(-1, 2)
        if rollout.size == 0:
            continue
        color = cmap(float(score_values[index])) if index < score_values.size else "#94a3b8"
        axis.plot(rollout[:, 0], rollout[:, 1], color=color, linewidth=1.2, alpha=0.64, zorder=3)

    best_index = getattr(scoring, "best_index", None)
    if best_index is not None and 0 <= int(best_index) < len(rollouts):
        best_rollout = np.asarray(rollouts[int(best_index)], dtype=np.float32).reshape(-1, 2)
        if best_rollout.size:
            axis.plot(best_rollout[:, 0], best_rollout[:, 1], color="#10b981", linewidth=3.0, alpha=0.98, zorder=6)

    goal_x, goal_y = model_heading_rad_to_body_xy(float(theta_goal_rad), 1.35)
    axis.annotate(
        "",
        xy=(goal_x, goal_y),
        xytext=(0.0, 0.0),
        arrowprops={"arrowstyle": "->", "lw": 2.5, "color": "#f59e0b", "linestyle": "--"},
        zorder=8,
    )
    if best_theta_rad is not None:
        best_x, best_y = model_heading_rad_to_body_xy(float(best_theta_rad), 1.65)
        axis.annotate(
            "",
            xy=(best_x, best_y),
            xytext=(0.0, 0.0),
            arrowprops={"arrowstyle": "->", "lw": 3.0, "color": "#10b981"},
            zorder=9,
        )

    norm = Normalize(vmin=score_min, vmax=score_max)
    colorbar = axis.figure.colorbar(
        ScalarMappable(norm=norm, cmap=cmap),
        ax=axis,
        fraction=0.046,
        pad=0.03,
    )
    colorbar.set_label("Score", labelpad=4.0)
    colorbar.ax.tick_params(labelsize=7)
    axis.legend(
        handles=[
            Line2D([0], [0], color="#94a3b8", linewidth=1.6, label="Candidates"),
            Line2D([0], [0], color="#f59e0b", linewidth=2.5, linestyle="--", label="Goal"),
            Line2D([0], [0], color="#10b981", linewidth=3.0, label="Selected"),
        ],
        loc="upper right",
        frameon=True,
        framealpha=0.86,
        borderpad=0.35,
        handlelength=1.5,
    )


def _plot_score_bars(axis: Any, *, scoring: Any) -> None:
    from matplotlib.patches import Patch

    scores = np.asarray(getattr(scoring, "scores", []), dtype=np.float32)
    min_esdf = np.asarray(getattr(scoring, "min_esdf_m", []), dtype=np.float32)
    if scores.size == 0:
        axis.text(0.5, 0.5, "No candidate scores", ha="center", va="center", transform=axis.transAxes)
        axis.set_axis_off()
        return

    display_scores = _finite_scores(scores)
    best_index = getattr(scoring, "best_index", None)
    candidate_indices = np.arange(scores.size, dtype=np.float32)
    colors: list[str] = []
    for index, score in enumerate(scores):
        if best_index is not None and index == int(best_index):
            colors.append("#10b981")
        elif not math.isfinite(float(score)):
            colors.append("#9ca3af")
        elif index < min_esdf.size and float(min_esdf[index]) <= 0.2:
            colors.append("#ef4444")
        else:
            colors.append("#2563eb")

    axis.bar(candidate_indices, display_scores, width=0.82, color=colors, edgecolor="#111827", linewidth=0.45)
    axis.axhline(0.0, color="#374151", linewidth=1.0)
    if best_index is not None and 0 <= int(best_index) < scores.size:
        axis.axvline(float(best_index), color="#065f46", linestyle="--", linewidth=1.5)
    axis.set_title("Candidate Score Bars", pad=9.0)
    axis.set_xlabel("Candidate index")
    axis.set_ylabel("Score")
    axis.grid(True, axis="y", alpha=0.24, linestyle="--", linewidth=0.7)
    axis.legend(
        handles=[
            Patch(facecolor="#2563eb", edgecolor="#111827", linewidth=0.45, label="Candidate"),
            Patch(facecolor="#10b981", edgecolor="#111827", linewidth=0.45, label="Selected"),
            Patch(facecolor="#ef4444", edgecolor="#111827", linewidth=0.45, label="Low clearance"),
            Patch(facecolor="#9ca3af", edgecolor="#111827", linewidth=0.45, label="Invalid"),
        ],
        loc="upper right",
        frameon=False,
        fontsize=8.0,
        ncol=2,
    )


def save_rule_avoidance_debug_panel(
    *,
    path: str | Path,
    grid_spec: Any,
    occupancy_grid: np.ndarray,
    esdf_grid_m: np.ndarray,
    scoring: Any,
    theta_goal_rad: float,
    best_theta_rad: float | None,
    best_score: float | None = None,
    path_min_esdf_m: float | None = None,
    path_mean_esdf_m: float | None = None,
    linear_velocity_mps: float | None = None,
    angular_velocity_rps: float | None = None,
    dpi: int = 180,
) -> None:
    _, plt = import_pyplot()
    with plt.rc_context(
        rc=paper_rc(
            title_fontsize=13.0,
            base_fontsize=9.0,
            legend_fontsize=8.0,
            tick_fontsize=8.0,
            axes_labelsize=9.0,
        )
    ):
        figure, axes = plt.subplots(1, 3, figsize=(18.0, 5.2), dpi=dpi, constrained_layout=True)
        _plot_esdf_base(
            axes[0],
            grid_spec=grid_spec,
            occupancy_grid=np.asarray(occupancy_grid, dtype=np.int8),
            esdf_grid_m=np.asarray(esdf_grid_m, dtype=np.float32),
            title="ESDF Clearance",
            colorbar=True,
        )
        _plot_direction_panel(
            axes[1],
            grid_spec=grid_spec,
            occupancy_grid=np.asarray(occupancy_grid, dtype=np.int8),
            esdf_grid_m=np.asarray(esdf_grid_m, dtype=np.float32),
            scoring=scoring,
            theta_goal_rad=float(theta_goal_rad),
            best_theta_rad=best_theta_rad,
        )
        _plot_score_bars(axes[2], scoring=scoring)

        summary = (
            f"selected={math.degrees(best_theta_rad):+.1f} deg" if best_theta_rad is not None else "selected=n/a"
        )
        if best_score is not None:
            summary += f"    score={best_score:.3f}"
        if path_min_esdf_m is not None:
            summary += f"    min_esdf={path_min_esdf_m:.2f} m"
        if path_mean_esdf_m is not None:
            summary += f"    mean_esdf={path_mean_esdf_m:.2f} m"
        if linear_velocity_mps is not None and angular_velocity_rps is not None:
            summary += f"    cmd=({linear_velocity_mps:.2f} m/s, {angular_velocity_rps:.2f} rad/s)"
        figure.suptitle(f"OmniGuard Rule Avoidance Debug | {summary}", y=1.02, fontsize=14.0)

        try:
            save_figure(figure, png_path=path, pdf_path=None, dpi=dpi, pad_inches=0.04)
        finally:
            plt.close(figure)
