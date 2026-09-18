from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .snapshot_visualization import (
    save_first_debug_collage,
    save_snapshot_visualization_set,
    visualization_config_from_runtime,
)


def save_first_esdf_debug_panel(
    *,
    path: str | Path,
    frame_bgr: np.ndarray,
    output: Any,
    config: dict[str, Any],
    grid_spec: Any,
    occupancy_grid: np.ndarray,
    esdf_grid_m: np.ndarray,
    camera_info: dict[str, Any] | None = None,
    dpi: int = 180,
) -> None:
    """Save the configured first-frame debug collage via the unified visualization chain."""
    del dpi
    panel_path = Path(path)
    visualization_config = visualization_config_from_runtime(config)
    if not bool(visualization_config.get("first_esdf_debug", {}).get("enabled", True)):
        return
    paths, errors, _depth_result = save_snapshot_visualization_set(
        output_dir=panel_path.parent,
        frame_stem=panel_path.stem,
        frame_bgr=frame_bgr,
        output=output,
        config=config,
        camera_info=camera_info,
        grid_spec=grid_spec,
        occupancy_grid=occupancy_grid,
        esdf_grid_m=esdf_grid_m,
        visualization_config=visualization_config,
        save_first_debug=False,
    )
    save_first_debug_collage(
        path=panel_path,
        component_paths=dict(paths.get("main_images", {})),
        visualization_config=visualization_config,
    )
    if errors:
        error_path = panel_path.with_suffix(".errors.json")
        error_path.write_text(json.dumps(errors, ensure_ascii=False, indent=2), encoding="utf-8")
