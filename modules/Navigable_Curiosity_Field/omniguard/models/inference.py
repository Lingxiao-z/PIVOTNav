from __future__ import annotations

from .upstream_check_inference import (
    ModelMetadata,
    ModelOutput,
    UpstreamTraversabilityInference,
)


class TraversabilityInference(UpstreamTraversabilityInference):
    """Compatibility alias for the shared runtime inference class."""


__all__ = [
    "ModelMetadata",
    "ModelOutput",
    "TraversabilityInference",
]
