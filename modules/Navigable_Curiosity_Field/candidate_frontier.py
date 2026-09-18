from __future__ import annotations

import numpy as np


def sector_centers() -> np.ndarray:
    return np.arange(12, dtype=np.float32) * 30.0


def select_candidate_frontier(fs_scores: np.ndarray, valid_mask: np.ndarray) -> int | None:
    scores = np.asarray(fs_scores, dtype=np.float32).reshape(12)
    valid = np.asarray(valid_mask, dtype=bool).reshape(12)
    if not valid.any():
        return None
    return int(np.argmax(np.where(valid, scores, -np.inf)))
