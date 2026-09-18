from __future__ import annotations

import os
import tempfile
import warnings
from pathlib import Path
from typing import Any

import numpy as np


def ensure_matplotlib_cache_dir() -> None:
    cache_dir = Path(tempfile.gettempdir()) / "matplotlib"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))


ensure_matplotlib_cache_dir()
warnings.filterwarnings(
    "ignore",
    message="Unable to import Axes3D.*",
    category=UserWarning,
    module=r"matplotlib\.projections",
)


def import_pyplot() -> tuple[Any, Any]:
    ensure_matplotlib_cache_dir()
    import matplotlib

    matplotlib.use("Agg", force=True)
    from matplotlib import pyplot as plt

    return matplotlib, plt


def paper_rc(
    *,
    title_fontsize: float = 20.0,
    base_fontsize: float = 12.0,
    legend_fontsize: float = 12.0,
    tick_fontsize: float = 11.0,
    axes_labelsize: float = 13.0,
) -> dict[str, Any]:
    return {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman No9 L", "DejaVu Serif"],
        "font.size": float(base_fontsize),
        "axes.titlesize": float(title_fontsize),
        "axes.titleweight": "regular",
        "axes.labelsize": float(axes_labelsize),
        "xtick.labelsize": float(tick_fontsize),
        "ytick.labelsize": float(tick_fontsize),
        "legend.fontsize": float(legend_fontsize),
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    }


def figure_size_from_pixels(width_px: int, height_px: int, dpi: int) -> tuple[float, float]:
    dpi = max(int(dpi), 1)
    return max(int(width_px), 1) / float(dpi), max(int(height_px), 1) / float(dpi)


def figure_to_rgb_array(figure: Any) -> np.ndarray:
    ensure_matplotlib_cache_dir()
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    canvas = FigureCanvasAgg(figure)
    canvas.draw()
    rgba = np.asarray(canvas.buffer_rgba())
    return rgba[:, :, :3].copy()


def figure_to_bgr_array(figure: Any) -> np.ndarray:
    rgb = figure_to_rgb_array(figure)
    return rgb[:, :, ::-1].copy()


def save_figure(
    figure: Any,
    *,
    png_path: str | Path | None = None,
    pdf_path: str | Path | None = None,
    dpi: int = 160,
    bbox_inches: str = "tight",
    pad_inches: float = 0.05,
) -> None:
    if png_path is not None:
        figure.savefig(
            png_path,
            format="png",
            dpi=dpi,
            bbox_inches=bbox_inches,
            pad_inches=pad_inches,
            facecolor="white",
        )
    if pdf_path is not None:
        figure.savefig(
            pdf_path,
            format="pdf",
            dpi=dpi,
            bbox_inches=bbox_inches,
            pad_inches=pad_inches,
            facecolor="white",
        )
