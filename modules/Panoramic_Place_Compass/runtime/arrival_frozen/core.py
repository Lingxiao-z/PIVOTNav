from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping


class Decision(str, Enum):
    CONFIRM = "CONFIRM"
    UNCERTAIN_NEAR_BOUNDARY = "UNCERTAIN_NEAR_BOUNDARY"
    REJECT = "REJECT"


FORBIDDEN_RUNTIME_TOKENS = {
    "gt", "distance", "pose", "position", "navmesh", "geodesic",
    "collision", "success", "spl", "scene", "label",
}

ALLOWED_EVIDENCE_FIELDS = {
    "arrival_probability", "vpr_similarity", "candidate_margin",
    "lightglue_second_margin", "e_inliers", "e_inlier_ratio",
    "grid_coverage", "hull_area", "horizontal_span", "vertical_span",
    "reprojection_error", "positive_depth_ratio", "supported_sectors",
    "h_dominance", "yaw_consistent", "candidate_stable",
    "multiview_consistency", "repeated_texture_risk",
    "through_wall_risk", "vggt_veto",
}


def validate_online_evidence(values: Mapping[str, Any]) -> None:
    unknown = set(values) - ALLOWED_EVIDENCE_FIELDS
    forbidden = set()
    for key in values:
        normalized = key.lower().replace("-", "_")
        parts = set(normalized.split("_"))
        if parts & FORBIDDEN_RUNTIME_TOKENS:
            forbidden.add(key)
    if unknown or forbidden:
        raise ValueError(f"non-visual runtime evidence: unknown={sorted(unknown)} forbidden={sorted(forbidden)}")


@dataclass(frozen=True)
class Evidence:
    arrival_probability: float
    vpr_similarity: float
    candidate_margin: float
    lightglue_second_margin: float
    e_inliers: int
    e_inlier_ratio: float
    grid_coverage: float
    hull_area: float
    horizontal_span: float
    vertical_span: float
    reprojection_error: float
    positive_depth_ratio: float
    supported_sectors: int
    h_dominance: float
    yaw_consistent: bool
    candidate_stable: bool
    multiview_consistency: float
    repeated_texture_risk: bool = False
    through_wall_risk: bool = False
    vggt_veto: bool = False

    @classmethod
    def from_runtime(cls, values: Mapping[str, Any]) -> "Evidence":
        validate_online_evidence(values)
        return cls(**values)


@dataclass(frozen=True)
class Thresholds:
    candidate_probability: float = 0.001
    uncertain_probability: float = 0.003
    confirm_probability: float = 0.010
    uncertain_inliers: int = 8
    confirm_inliers: int = 18
    confirm_inlier_ratio: float = 0.36
    confirm_grid_coverage: float = 0.055
    confirm_hull_area: float = 0.025
    confirm_horizontal_span: float = 0.28
    confirm_vertical_span: float = 0.16
    max_reprojection_error: float = 3.0
    min_positive_depth_ratio: float = 0.45
    min_supported_sectors: int = 2
    max_h_dominance: float = 0.32
    min_candidate_margin: float = 0.025
    min_lightglue_second_margin: float = 4.0
    min_multiview_consistency: float = 0.62
    min_delta_score: float = -0.0005
    min_delta_geometry: float = -0.08


def geometry_strength(e: Evidence) -> float:
    return (
        min(e.e_inliers / 24.0, 1.0) * 0.25
        + min(e.e_inlier_ratio / 0.55, 1.0) * 0.20
        + min(e.grid_coverage / 0.12, 1.0) * 0.20
        + min(e.hull_area / 0.08, 1.0) * 0.10
        + min(e.horizontal_span / 0.55, 1.0) * 0.10
        + min(e.vertical_span / 0.35, 1.0) * 0.10
        + min(e.supported_sectors / 4.0, 1.0) * 0.05
    )


def absolute_decision(e: Evidence, t: Thresholds) -> Decision:
    hard_risk = (
        e.vggt_veto or e.repeated_texture_risk or e.through_wall_risk
        or not e.candidate_stable or not e.yaw_consistent
        or e.h_dominance > t.max_h_dominance
    )
    strong = (
        e.arrival_probability >= t.confirm_probability
        and e.candidate_margin >= t.min_candidate_margin
        and e.lightglue_second_margin >= t.min_lightglue_second_margin
        and e.e_inliers >= t.confirm_inliers
        and e.e_inlier_ratio >= t.confirm_inlier_ratio
        and e.grid_coverage >= t.confirm_grid_coverage
        and e.hull_area >= t.confirm_hull_area
        and e.horizontal_span >= t.confirm_horizontal_span
        and e.vertical_span >= t.confirm_vertical_span
        and e.reprojection_error <= t.max_reprojection_error
        and e.positive_depth_ratio >= t.min_positive_depth_ratio
        and e.supported_sectors >= t.min_supported_sectors
        and e.multiview_consistency >= t.min_multiview_consistency
    )
    if strong and not hard_risk:
        return Decision.CONFIRM
    plausible = (
        e.arrival_probability >= t.uncertain_probability
        and e.e_inliers >= t.uncertain_inliers
        and e.vpr_similarity > 0.60
        and not e.vggt_veto
    )
    return Decision.UNCERTAIN_NEAR_BOUNDARY if plausible else Decision.REJECT


def delta_decision(history: list[Evidence], t: Thresholds) -> Decision:
    current = history[-1]
    absolute = absolute_decision(current, t)
    if len(history) == 1:
        return absolute
    previous = history[-2]
    delta_score = current.arrival_probability - previous.arrival_probability
    delta_geometry = geometry_strength(current) - geometry_strength(previous)
    if (
        current.vggt_veto or not current.candidate_stable
        or delta_score < t.min_delta_score and delta_geometry < t.min_delta_geometry
    ):
        return Decision.REJECT
    stable_views = sum(
        1 for item in history
        if item.candidate_stable and item.yaw_consistent and not item.repeated_texture_risk
    )
    if absolute == Decision.CONFIRM and stable_views >= 2:
        return Decision.CONFIRM
    return Decision.UNCERTAIN_NEAR_BOUNDARY


@dataclass
class BoundaryVerifier:
    thresholds: Thresholds = field(default_factory=Thresholds)
    max_micro_actions: int = 3
    nominal_step_m: float = 0.15
    max_nominal_distance_m: float = 0.6
    timeout_s: float = 10.0
    evidence: list[Evidence] = field(default_factory=list)
    micro_actions: int = 0
    started_at_s: float | None = None
    state: Decision = Decision.REJECT

    def begin(self, evidence: Evidence, timestamp_s: float) -> Decision:
        self.started_at_s = timestamp_s
        self.evidence = [evidence]
        self.micro_actions = 0
        self.state = absolute_decision(evidence, self.thresholds)
        return self.state

    def can_approach(self, *, timestamp_s: float, traversable: bool) -> bool:
        if self.started_at_s is None or self.state != Decision.UNCERTAIN_NEAR_BOUNDARY:
            return False
        if not traversable or timestamp_s - self.started_at_s >= self.timeout_s:
            self.state = Decision.REJECT
            return False
        return (
            self.micro_actions < self.max_micro_actions
            and (self.micro_actions + 1) * self.nominal_step_m <= self.max_nominal_distance_m
        )

    def after_approach(self, evidence: Evidence, timestamp_s: float) -> Decision:
        if self.started_at_s is None:
            raise RuntimeError("begin must be called first")
        self.micro_actions += 1
        self.evidence.append(evidence)
        if timestamp_s - self.started_at_s >= self.timeout_s:
            self.state = Decision.REJECT
        else:
            self.state = delta_decision(self.evidence, self.thresholds)
            if self.state == Decision.UNCERTAIN_NEAR_BOUNDARY and self.micro_actions >= self.max_micro_actions:
                self.state = Decision.REJECT
        return self.state


def direct_target_contract() -> dict[str, bool]:
    return {
        "target_known_before_verification": True,
        "target_forced_into_arrival_evaluation": True,
        "ann_used_as_arrival_gate": False,
        "bpl_used_as_arrival_gate": False,
        "bpl_can_affirm_stop": False,
    }
