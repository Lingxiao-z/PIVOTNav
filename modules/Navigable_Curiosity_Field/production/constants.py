from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


DT = 0.25
MAX_V = 0.4
MAX_W = 0.3


def save_rgb(observation: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(observation["rgb"])[..., :3].astype(np.uint8)).save(path)
