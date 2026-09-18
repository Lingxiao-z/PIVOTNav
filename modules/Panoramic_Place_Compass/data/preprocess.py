"""Validate and resize ERP samples used by the Compass training pipeline."""

from pathlib import Path
from PIL import Image


def preprocess_directory(source: Path, destination: Path, size: tuple[int, int] = (448, 224)) -> int:
    destination.mkdir(parents=True, exist_ok=True)
    count = 0
    for path in sorted(source.glob("*.png")):
        Image.open(path).convert("RGB").resize(size, Image.Resampling.BILINEAR).save(destination / path.name)
        count += 1
    return count
