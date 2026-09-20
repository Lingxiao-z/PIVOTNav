from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict

SCHEMA_VERSION = '2.0.0'


@dataclass(frozen=True)
class PanoramicVPREvidenceV2:
    candidate_id: str
    normalized_similarity: float
    rank: int
    top1_top2_margin: float
    yaw_shift_degrees: float
    yaw_alignment_confidence: float
    ring_peak_score: float
    ring_peak_second_margin: float
    topk_entropy: float
    open_set_probability: float
    candidate_uncertainty: float
    backend_name: str = 'Pano-SALAD-Ring-V2'
    backend_version: str = 'unknown'
    descriptor_is_backend_private: bool = True
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def build_vpr_evidence(**kwargs: Any) -> Dict[str, Any]:
    open_set_probability = float(kwargs.pop('open_set_probability', 0.0))
    open_set_probability = max(0.0, min(1.0, open_set_probability))
    kwargs.setdefault('candidate_uncertainty', open_set_probability)
    kwargs['open_set_probability'] = open_set_probability
    return PanoramicVPREvidenceV2(**kwargs).to_dict()
