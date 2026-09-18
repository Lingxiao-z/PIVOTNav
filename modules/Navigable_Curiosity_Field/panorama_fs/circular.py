from __future__ import annotations

import numpy as np


def roll_source(erp: np.ndarray, labels: np.ndarray, sectors: int) -> tuple[np.ndarray, np.ndarray]:
    """Roll Source ERP and labels together by an integer sector count."""
    width = erp.shape[-2]
    if width % 12:
        raise ValueError("ERP width must be divisible by 12 for exact sector rolls")
    return np.roll(erp, sectors * (width // 12), axis=-2), np.roll(labels, sectors, axis=-1)


def roll_goal(erp: np.ndarray, sectors: int) -> np.ndarray:
    """Roll Goal ERP independently; Source-coordinate labels do not move."""
    width = erp.shape[-2]
    if width % 12:
        raise ValueError("ERP width must be divisible by 12 for exact sector rolls")
    return np.roll(erp, sectors * (width // 12), axis=-2)


def circular_relative_indices(n: int = 12) -> np.ndarray:
    idx = np.arange(n)
    delta = idx[None, :] - idx[:, None]
    return (delta + n // 2) % n - n // 2


def direction_cosine_matrix(n: int = 12) -> np.ndarray:
    return np.cos(2.0 * np.pi * circular_relative_indices(n) / n).astype(np.float32)

