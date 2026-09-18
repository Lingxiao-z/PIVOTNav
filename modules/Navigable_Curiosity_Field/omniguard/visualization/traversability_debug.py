from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..geometry import signed_model_degrees
from ..ros_utils.camera_utils import sanitize_bgr_frame
from .mpl_style import (
    ensure_matplotlib_cache_dir,
    figure_size_from_pixels,
    figure_to_bgr_array,
    import_pyplot,
    paper_rc,
    save_figure,
)


ensure_matplotlib_cache_dir()


def _panel_rc() -> dict[str, Any]:
    return paper_rc(
        title_fontsize=18.0,
        base_fontsize=11.0,
        legend_fontsize=10.0,
        tick_fontsize=10.0,
        axes_labelsize=12.0,
    )


def _angle_to_pixel(angle_deg: float, width: int, hfov_deg: float) -> int:
    norm = np.clip((angle_deg + hfov_deg * 0.5) / max(hfov_deg, 1e-6), 0.0, 1.0)
    return int(round(norm * max(width - 1, 1)))


def render_traversability_overlay(
    frame_bgr: np.ndarray,
    effective_distance_m: np.ndarray,
    exist_probability: np.ndarray,
    config: dict[str, Any],
    camera_info: dict[str, Any] | None = None,
) -> np.ndarray:
    frame = sanitize_bgr_frame(frame_bgr, context="traversability_overlay")
    overlay = frame.copy()
    height, width = overlay.shape[:2]
    hfov_deg = float(config["model"]["camera"].get("hfov_deg", 90.0))
    max_dist = max(
        float(config["controllers"]["navigation"]["polar_esdf"]["max_range"]),
        float(np.max(np.asarray(effective_distance_m, dtype=np.float32))),
        1.0,
    )
    angles_deg = signed_model_degrees(int(effective_distance_m.shape[0]))
    for angle_deg, distance_m, prob in zip(angles_deg, effective_distance_m, exist_probability):
        if abs(float(angle_deg)) > hfov_deg * 0.5 + 1e-3:
            continue
        x = _angle_to_pixel(float(angle_deg), width, hfov_deg)
        distance_ratio = float(np.clip(distance_m / max_dist, 0.0, 1.0))
        color = (
            int(round(255 * (1.0 - distance_ratio))),
            int(round(180 * distance_ratio)),
            int(round(255 * (1.0 - float(prob)))),
        )
        line_height = int(round((0.2 + 0.8 * float(prob)) * height * 0.35))
        cv2.line(overlay, (x, height - 1), (x, max(0, height - 1 - line_height)), color, 2)

    cv2.putText(
        overlay,
        "Traversability Overlay",
        (20, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return overlay


def save_traversability_csv(
    path: str | Path,
    *,
    raw_distance_m: np.ndarray,
    combined_distance_m: np.ndarray,
    effective_distance_m: np.ndarray,
    exist_logit: np.ndarray,
    exist_probability: np.ndarray,
    traversable_distance_m: float,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    angles_deg = signed_model_degrees(int(effective_distance_m.shape[0]))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "angle_deg",
                "raw_distance_m",
                "combined_distance_m",
                "effective_distance_m",
                "exist_logit",
                "exist_probability",
                "is_traversable",
            ]
        )
        for values in zip(
            angles_deg,
            raw_distance_m,
            combined_distance_m,
            effective_distance_m,
            exist_logit,
            exist_probability,
        ):
            angle_deg, raw_d, combined_d, effective_d, logit, prob = values
            writer.writerow(
                [
                    float(angle_deg),
                    float(raw_d),
                    float(combined_d),
                    float(effective_d),
                    float(logit),
                    float(prob),
                    int(float(effective_d) >= float(traversable_distance_m)),
                ]
            )


def _build_distance_figure(
    *,
    effective_distance_m: np.ndarray,
    raw_distance_m: np.ndarray,
    combined_distance_m: np.ndarray,
    exist_probability: np.ndarray,
    config: dict[str, Any],
    width: int,
    height: int,
    dpi: int,
) -> Any:
    _, plt = import_pyplot()
    angles_deg = signed_model_degrees(int(effective_distance_m.shape[0]))
    figure, axes = plt.subplots(
        2,
        1,
        figsize=figure_size_from_pixels(width, height, dpi),
        dpi=dpi,
        constrained_layout=True,
    )
    traversable = float(config["controllers"]["radar"]["traversable_distance_m"])
    axes[0].plot(angles_deg, raw_distance_m, label="Raw", color="#7a90a6", linewidth=1.8)
    axes[0].plot(angles_deg, combined_distance_m, label="Combined", color="#8b5cf6", linewidth=1.8)
    axes[0].plot(angles_deg, effective_distance_m, label="Effective", color="#ea580c", linewidth=2.0)
    axes[0].axhline(traversable, color="#15803d", linestyle="--", linewidth=1.1, label="Traversable")
    axes[0].set_xlabel("Heading (deg)")
    axes[0].set_ylabel("Distance (m)")
    axes[0].set_title("Distance Profile")
    axes[0].grid(True, alpha=0.25, linestyle="--")
    axes[0].legend(loc="upper right", frameon=False)

    axes[1].plot(angles_deg, exist_probability, color="#7B1FA2", linewidth=1.8)
    axes[1].axhline(0.5, color="#555555", linestyle="--", linewidth=1.0)
    axes[1].set_xlabel("Heading (deg)")
    axes[1].set_ylabel("Prob")
    axes[1].set_ylim(-0.02, 1.02)
    axes[1].set_title("Exist Probability")
    axes[1].grid(True, alpha=0.25, linestyle="--")
    return figure


def _build_summary_figure(
    *,
    frame_bgr: np.ndarray,
    effective_distance_m: np.ndarray,
    raw_distance_m: np.ndarray,
    combined_distance_m: np.ndarray,
    exist_probability: np.ndarray,
    config: dict[str, Any],
    width: int,
    height: int,
    dpi: int,
) -> Any:
    _, plt = import_pyplot()
    overlay = render_traversability_overlay(
        frame_bgr=frame_bgr,
        effective_distance_m=effective_distance_m,
        exist_probability=exist_probability,
        config=config,
    )
    figure = plt.figure(figsize=figure_size_from_pixels(width, height, dpi), dpi=dpi, constrained_layout=True)
    grid = figure.add_gridspec(2, 2, hspace=0.04, wspace=0.04)
    ax_input = figure.add_subplot(grid[0, 0])
    ax_overlay = figure.add_subplot(grid[0, 1])
    ax_dist = figure.add_subplot(grid[1, :])

    ax_input.imshow(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    ax_input.set_title("Input")
    ax_input.axis("off")

    ax_overlay.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    ax_overlay.set_title("Overlay")
    ax_overlay.axis("off")

    angles_deg = signed_model_degrees(int(effective_distance_m.shape[0]))
    traversable = float(config["controllers"]["radar"]["traversable_distance_m"])
    ax_dist.plot(angles_deg, raw_distance_m, label="Raw", color="#7a90a6", linewidth=1.8)
    ax_dist.plot(angles_deg, combined_distance_m, label="Combined", color="#8b5cf6", linewidth=1.8)
    ax_dist.plot(angles_deg, effective_distance_m, label="Effective", color="#ea580c", linewidth=2.0)
    ax_dist.plot(angles_deg, exist_probability, label="Exist Prob", color="#7B1FA2", linewidth=1.6)
    ax_dist.axhline(traversable, color="#15803d", linestyle="--", linewidth=1.1, label="Traversable")
    ax_dist.set_xlabel("Heading (deg)")
    ax_dist.set_ylabel("Distance / Prob")
    ax_dist.set_title("Traversability Summary")
    ax_dist.grid(True, alpha=0.25, linestyle="--")
    ax_dist.legend(loc="upper right", frameon=False, ncol=3)
    return figure


def save_traversability_panel_subplot_artifacts(
    *,
    output_dir: str | Path,
    frame_bgr: np.ndarray,
    effective_distance_m: np.ndarray,
    raw_distance_m: np.ndarray,
    combined_distance_m: np.ndarray,
    exist_probability: np.ndarray,
    config: dict[str, Any],
    camera_info: dict[str, Any] | None = None,
    azimuth_range_deg: tuple[float, float] | None = None,
    frame_label: str = "",
    camera_info_source_label: str = "",
    camera_info_json_path: str | None = None,
    checkpoint_hfov_deg: float | None = None,
    device_label: str = "",
) -> dict[str, dict[str, str]]:
    del camera_info, azimuth_range_deg, frame_label, camera_info_source_label, camera_info_json_path, checkpoint_hfov_deg, device_label
    root_dir = Path(output_dir)
    png_dir = root_dir / "png"
    pdf_dir = root_dir / "pdf"
    png_dir.mkdir(parents=True, exist_ok=True)
    pdf_dir.mkdir(parents=True, exist_ok=True)

    _, plt = import_pyplot()
    artifacts: dict[str, dict[str, str]] = {}
    with plt.rc_context(rc=_panel_rc()):
        panels = {
            "summary": _build_summary_figure(
                frame_bgr=frame_bgr,
                effective_distance_m=effective_distance_m,
                raw_distance_m=raw_distance_m,
                combined_distance_m=combined_distance_m,
                exist_probability=exist_probability,
                config=config,
                width=1280,
                height=960,
                dpi=160,
            ),
            "distance_probability": _build_distance_figure(
                effective_distance_m=effective_distance_m,
                raw_distance_m=raw_distance_m,
                combined_distance_m=combined_distance_m,
                exist_probability=exist_probability,
                config=config,
                width=1280,
                height=960,
                dpi=160,
            ),
        }
        for stem, figure in panels.items():
            png_path = png_dir / f"{stem}.png"
            pdf_path = pdf_dir / f"{stem}.pdf"
            try:
                save_figure(figure, png_path=png_path, pdf_path=pdf_path, dpi=160, pad_inches=0.04)
            finally:
                plt.close(figure)
            artifacts[stem] = {"png_path": str(png_path), "pdf_path": str(pdf_path)}
    return artifacts


def save_traversability_panel_artifacts(
    *,
    png_path: str | Path | None,
    pdf_path: str | Path | None,
    frame_bgr: np.ndarray,
    effective_distance_m: np.ndarray,
    raw_distance_m: np.ndarray,
    combined_distance_m: np.ndarray,
    exist_probability: np.ndarray,
    config: dict[str, Any],
    camera_info: dict[str, Any] | None = None,
    azimuth_range_deg: tuple[float, float] | None = None,
    frame_label: str = "",
    camera_info_source_label: str = "",
    camera_info_json_path: str | None = None,
    checkpoint_hfov_deg: float | None = None,
    device_label: str = "",
) -> None:
    del camera_info, azimuth_range_deg, frame_label, camera_info_source_label, camera_info_json_path, checkpoint_hfov_deg, device_label
    _, plt = import_pyplot()
    with plt.rc_context(rc=_panel_rc()):
        figure = _build_summary_figure(
            frame_bgr=frame_bgr,
            effective_distance_m=effective_distance_m,
            raw_distance_m=raw_distance_m,
            combined_distance_m=combined_distance_m,
            exist_probability=exist_probability,
            config=config,
            width=1600,
            height=1200,
            dpi=160,
        )
        try:
            save_figure(figure, png_path=png_path, pdf_path=pdf_path, dpi=160, pad_inches=0.04)
        finally:
            plt.close(figure)
