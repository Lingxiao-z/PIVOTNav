from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch


def _numpy_rng_safe_globals() -> list[type | Any]:
    """Globals needed by NumPy RandomState payloads in exact-resume checkpoints."""
    try:
        reconstruct = np._core.multiarray._reconstruct
    except AttributeError:  # NumPy 1.x
        reconstruct = np.core.multiarray._reconstruct
    return [reconstruct, np.ndarray, np.dtype, type(np.dtype(np.uint32))]


def load_model_state_dict(
    checkpoint: str | Path, *, map_location: str | torch.device = "cpu",
) -> Mapping[str, torch.Tensor]:
    """Load only a model state from a trusted, hash-verified exact-resume checkpoint."""
    with torch.serialization.safe_globals(_numpy_rng_safe_globals()):
        payload = torch.load(checkpoint, map_location=map_location, weights_only=True)
    if not isinstance(payload, Mapping) or not isinstance(payload.get("model"), Mapping):
        raise RuntimeError("checkpoint has no model state dictionary")
    state = payload["model"]
    if not state or any(not isinstance(name, str) or not torch.is_tensor(value) for name, value in state.items()):
        raise RuntimeError("checkpoint model state is empty or contains non-tensor values")
    return state
