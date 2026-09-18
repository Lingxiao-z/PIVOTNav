from __future__ import annotations

import cv2
import numpy as np


def sanitize_bgr_frame(frame_bgr: np.ndarray, context: str = "frame") -> np.ndarray:
    if frame_bgr is None:
        raise ValueError(f"{context}: frame is None")

    frame_np = np.asarray(frame_bgr)
    if frame_np.ndim == 2:
        frame_np = cv2.cvtColor(frame_np, cv2.COLOR_GRAY2BGR)
    elif frame_np.ndim != 3:
        raise ValueError(f"{context}: unexpected ndim={frame_np.ndim}, shape={frame_np.shape}")

    if frame_np.shape[2] == 1:
        frame_np = np.repeat(frame_np, 3, axis=2)
    elif frame_np.shape[2] != 3:
        raise ValueError(f"{context}: expected 3 channels, got shape={frame_np.shape}")

    if frame_np.dtype != np.uint8:
        frame_np = np.clip(frame_np, 0, 255).astype(np.uint8)

    return np.ascontiguousarray(frame_np)
