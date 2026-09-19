from __future__ import annotations

import importlib
import os
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent


def load_lightglue_geometry(device: str = "cpu"):
    """Load the exact V3.3.12 geometry class against this frozen package."""
    runtime = importlib.import_module(
        "modules.Panoramic_Place_Compass.runtime.arrival_frozen.v3312_visual_runtime"
    )
    runtime.V3310 = Path(
        os.environ.get("PIVOTNAV_ARRIVAL_DEPENDENCY_ROOT", str(PACKAGE_ROOT))
    ).expanduser().resolve()
    return runtime.LightGlueGeometry(device)
