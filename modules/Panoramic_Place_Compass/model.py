"""Public model metadata for the Panoramic Place Compass."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CompassOutput:
    node_id: int | None
    similarity: float
    bearing_deg: float
    arrival_candidate: bool
    arrival_confirmed: bool
