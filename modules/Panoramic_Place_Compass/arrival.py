"""Online visual arrival state machine and typed commit gate."""
from __future__ import annotations


# ---------------------------------------------------------------------------
# Arrival evidence decisions
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Typed arrival transaction gate
# ---------------------------------------------------------------------------

"""Final-confirmed transaction bridge between the verifier and topology graph."""

from dataclasses import dataclass
from typing import Any, Mapping

from modules.Topological_Belief_Cascade.topology import (
    CandidateFrontierNavigator,
    GOAL_FINAL_CONFIRMED,
    CANDIDATE_FRONTIER_FINAL_CONFIRMED,
    HISTORICAL_NODE_FINAL_CONFIRMED,
    assert_stop_semantics,
)


@dataclass(frozen=True)
class ArrivalRequest:
    generation: int
    target_kind: str
    candidate_frontier_id: str | None = None
    existing_node_id: int | None = None
    final_goal: bool = False


class ArrivalGate:
    """Apply typed arrival transactions with a single Goal Stop authority.

    ``goal`` is the only target that can authorize Stop. Candidate-frontier and historical
    node transactions mutate topology only after a final confirmation and are
    emitted with distinct event types for downstream audits.
    """

    schema = "pivotnav_candidate_frontier_arrival_gate_v1"

    def __init__(self, policy: CandidateFrontierNavigator) -> None:
        self.policy = policy
        self.generation = 0
        self.active: ArrivalRequest | None = None
        self.audit: list[dict[str, Any]] = []
        self.stop_authorized = False

    def begin(self, *, target_kind: str, candidate_frontier_id: str | None = None,
              existing_node_id: int | None = None, final_goal: bool = False) -> ArrivalRequest:
        # Accept the old serialized target token only at this compatibility boundary.
        if target_kind == "ghost":
            target_kind = "candidate_frontier"
        if target_kind not in {"goal", "candidate_frontier", "existing"}:
            raise ValueError(f"unsupported arrival target kind: {target_kind}")
        if target_kind != "goal" and final_goal:
            raise ValueError("only a goal request may set final_goal=true")
        self.generation += 1
        self.active = ArrivalRequest(self.generation, target_kind, candidate_frontier_id, existing_node_id, bool(final_goal))
        self.stop_authorized = False
        self.audit.append({"event": "arrival_generation_started", "generation": self.generation,
                           "target_kind": target_kind, "candidate_frontier_id": candidate_frontier_id,
                           "existing_node_id": existing_node_id, "event_type": {
                               "goal": GOAL_FINAL_CONFIRMED,
                               "candidate_frontier": CANDIDATE_FRONTIER_FINAL_CONFIRMED,
                               "existing": HISTORICAL_NODE_FINAL_CONFIRMED,
                           }[target_kind], "final_confirmed": False,
                           "node_switched": False, "bpl_updated": False, "stop_authorized": False})
        return self.active

    def apply(self, *, generation: int, final_confirmed: bool, step: int,
              evidence: Mapping[str, Any] | None = None) -> bool:
        request = self.active
        if request is None or int(generation) != request.generation:
            self.audit.append({"event": "arrival_result_rejected", "reason": "stale_generation",
                               "generation": generation, "final_confirmed": bool(final_confirmed),
                               "node_switched": False, "bpl_updated": False, "stop_authorized": False})
            return False
        if not final_confirmed:
            self.audit.append({"event": "arrival_result_rejected", "reason": "not_final_confirmed",
                               "generation": generation, "step": int(step), "final_confirmed": False,
                               "node_switched": False, "bpl_updated": False, "stop_authorized": False})
            return False
        if request.target_kind == "candidate_frontier":
            node_id = self.policy.on_final_confirmed(step=step, target_kind="candidate_frontier", candidate_frontier_id=request.candidate_frontier_id,
                                                     reason="frozen_arrival_verifier_final_confirmed")
        elif request.target_kind == "existing":
            node_id = self.policy.on_final_confirmed(step=step, target_kind="existing",
                                                     existing_node_id=request.existing_node_id,
                                                     reason="frozen_arrival_verifier_final_confirmed")
        elif request.target_kind == "goal":
            # Goal confirmation authorizes the final Stop but does not invent
            # a formal topology node or BPL transition.
            node_id = None
        else:
            raise ValueError(f"unsupported arrival target kind: {request.target_kind}")
        is_goal = request.target_kind == "goal"
        self.stop_authorized = bool(is_goal and request.final_goal)
        event_type = {
            "goal": GOAL_FINAL_CONFIRMED,
            "candidate_frontier": CANDIDATE_FRONTIER_FINAL_CONFIRMED,
            "existing": HISTORICAL_NODE_FINAL_CONFIRMED,
        }[request.target_kind]
        applied = {"event": "arrival_result_applied", "generation": generation, "step": int(step),
                           "event_type": event_type, "node_id": node_id, "final_confirmed": True,
                           "node_switched": not is_goal,
                           "bpl_updated": not is_goal,
                           "stop_authorized": self.stop_authorized,
                           "evidence": dict(evidence or {})}
        assert_stop_semantics(applied)
        self.audit.append(applied)
        if not is_goal:
            assert not self.stop_authorized
        if is_goal:
            assert self.stop_authorized and event_type == GOAL_FINAL_CONFIRMED
        return True


# ---------------------------------------------------------------------------
# Multi-view arrival sequence
# ---------------------------------------------------------------------------

import hashlib
import json
import time
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Mapping


class ArrivalState(str, Enum):
    NAVIGATING = "NAVIGATING"
    CANDIDATE = "CANDIDATE"
    HOLD = "HOLD"
    REOBSERVE = "REOBSERVE"
    CONFIRMED = "CONFIRMED"
    REJECTED = "REJECTED"
    TIMEOUT = "TIMEOUT"
    RECOVERY = "RECOVERY"


@dataclass(frozen=True)
class TargetIdentity:
    request_id: str
    generation: int
    candidate_node_id: str
    target_hash: str
    final_goal: bool


@dataclass(frozen=True)
class FrameAssessment:
    identity: TargetIdentity
    view_index: int
    predicted_distance_m: float
    safe: bool
    safety_reasons: tuple[str, ...] = ()
    rgb_change_from_previous: float | None = None
    inference_latency_ms: float = 0.0
    evidence: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ArrivalEvent:
    identity: TargetIdentity
    state: ArrivalState
    final_confirmed: bool
    evidence: Mapping[str, Any] = field(default_factory=dict)


class HardenedConfirmedEventGate:
    """Single transaction authority for node, Event-BPL, and final Stop."""

    def __init__(self) -> None:
        self.active: TargetIdentity | None = None
        self.applied: set[tuple[str, int, str, str]] = set()
        self.audit: list[dict[str, Any]] = []

    @staticmethod
    def _key(identity: TargetIdentity) -> tuple[str, int, str, str]:
        return (
            identity.request_id,
            identity.generation,
            identity.candidate_node_id,
            identity.target_hash,
        )

    def begin(self, identity: TargetIdentity) -> None:
        if not identity.request_id or not identity.candidate_node_id:
            raise ValueError("request and candidate node identities must be non-empty")
        if len(identity.target_hash) != 64:
            raise ValueError("target_hash must be SHA256")
        self.active = identity
        self.audit.append(
            {
                "event": "arrival_request_started",
                **identity.__dict__,
                "node_switched": False,
                "bpl_updated": False,
                "stop_authorized": False,
            }
        )

    def apply(
        self,
        event: ArrivalEvent,
        *,
        on_final_confirmed_transaction: Callable[
            [TargetIdentity, Mapping[str, Any]], Mapping[str, Any] | None
        ],
    ) -> bool:
        reason = None
        if self.active is None or event.identity != self.active:
            reason = "stale_or_target_identity_mismatch"
        elif self._key(event.identity) in self.applied:
            reason = "duplicate_result"
        elif event.state != ArrivalState.CONFIRMED or not event.final_confirmed:
            reason = "not_final_confirmed"
        if reason is not None:
            self.audit.append(
                {
                    "event": "arrival_result_rejected",
                    "reason": reason,
                    **event.identity.__dict__,
                    "state": event.state.value,
                    "final_confirmed": event.final_confirmed,
                    "node_switched": False,
                    "bpl_updated": False,
                    "stop_authorized": False,
                }
            )
            return False

        record = dict(
            on_final_confirmed_transaction(event.identity, event.evidence) or {}
        )
        key = self._key(event.identity)
        self.applied.add(key)
        self.audit.append(
            {
                "event": "arrival_result_applied",
                **event.identity.__dict__,
                "state": event.state.value,
                "final_confirmed": True,
                "node_switched": True,
                "bpl_updated": True,
                "stop_authorized": event.identity.final_goal,
                "transaction_record": record,
            }
        )
        return True


class ArrivalSequenceV7StateMachine:
    def __init__(
        self,
        *,
        deep_threshold_m: float = 0.5,
        candidate_threshold_m: float = 1.0,
        post_candidate_approaches: int = 3,
        minimum_rgb_change: float = 1.0,
        async_workers: int = 1,
    ) -> None:
        self.deep_threshold_m = float(deep_threshold_m)
        self.candidate_threshold_m = float(candidate_threshold_m)
        self.post_candidate_approaches = int(post_candidate_approaches)
        self.minimum_rgb_change = float(minimum_rgb_change)
        self.executor = ThreadPoolExecutor(max_workers=async_workers)
        self.gate = HardenedConfirmedEventGate()
        self.generation = 0
        self.identity: TargetIdentity | None = None
        self.state = ArrivalState.NAVIGATING
        self.candidate_index: int | None = None
        self.candidate_started: float | None = None
        self.changes: dict[int, float] = {}
        self.pending: dict[int, Future[FrameAssessment]] = {}
        self.retired_pending: dict[int, Future[FrameAssessment]] = {}
        self.next_token = 0
        self.audit: list[dict[str, Any]] = []
        self.hold_latencies_s: list[float] = []
        self.last_assessment: FrameAssessment | None = None

    def _transition(self, state: ArrivalState, reason: str, **extra: Any) -> None:
        previous = self.state
        self.state = state
        self.audit.append(
            {
                "event": "state_transition",
                "previous_state": previous.value,
                "state": state.value,
                "reason": reason,
                "generation": self.generation,
                "timestamp_monotonic_s": time.monotonic(),
                "node_switched": False,
                "bpl_updated": False,
                "stop_authorized": False,
                **extra,
            }
        )

    def begin_target(
        self,
        *,
        request_id: str,
        candidate_node_id: str,
        target_hash: str,
        final_goal: bool,
    ) -> TargetIdentity:
        self.retired_pending.update(self.pending)
        self.pending.clear()
        self.generation += 1
        self.identity = TargetIdentity(
            request_id=request_id,
            generation=self.generation,
            candidate_node_id=candidate_node_id,
            target_hash=target_hash,
            final_goal=bool(final_goal),
        )
        self.state = ArrivalState.NAVIGATING
        self.candidate_index = None
        self.candidate_started = None
        self.changes.clear()
        self.last_assessment = None
        self.gate.begin(self.identity)
        self.audit.append(
            {
                "event": "target_generation_started",
                **self.identity.__dict__,
                "pending_future_count": len(self.pending),
                "retired_stale_future_count": len(self.retired_pending),
                "old_filter_state_cleared": True,
                "node_switched": False,
                "bpl_updated": False,
                "stop_authorized": False,
            }
        )
        return self.identity

    def submit(
        self,
        evaluator: Callable[..., FrameAssessment],
        *args: Any,
        **kwargs: Any,
    ) -> int:
        if self.identity is None:
            raise RuntimeError("begin_target must be called before submit")
        token = self.next_token
        self.next_token += 1
        identity = self.identity
        self.pending[token] = self.executor.submit(evaluator, identity, *args, **kwargs)
        self.audit.append(
            {
                "event": "assessment_submitted",
                "token": token,
                **identity.__dict__,
                "node_switched": False,
                "bpl_updated": False,
                "stop_authorized": False,
            }
        )
        return token

    def resolve(
        self,
        token: int,
        *,
        timeout_s: float,
        on_final_confirmed_transaction: Callable[
            [TargetIdentity, Mapping[str, Any]], Mapping[str, Any] | None
        ],
    ) -> bool:
        if token in self.pending:
            future = self.pending.pop(token)
        else:
            future = self.retired_pending.pop(token)
        try:
            assessment = future.result(timeout=timeout_s)
        except TimeoutError:
            self.mark_timeout("ASSESSMENT_TIMEOUT")
            return False
        return self.apply_assessment(
            assessment,
            on_final_confirmed_transaction=on_final_confirmed_transaction,
        )

    def _confirm(
        self,
        assessment: FrameAssessment,
        *,
        mode: str,
        on_final_confirmed_transaction: Callable[
            [TargetIdentity, Mapping[str, Any]], Mapping[str, Any] | None
        ],
    ) -> bool:
        self._transition(
            ArrivalState.CONFIRMED,
            mode,
            view_index=assessment.view_index,
            final_confirmed=True,
        )
        event = ArrivalEvent(
            identity=assessment.identity,
            state=ArrivalState.CONFIRMED,
            final_confirmed=True,
            evidence={
                **dict(assessment.evidence),
                "decision_mode": mode,
                "view_index": assessment.view_index,
                "predicted_distance_m": assessment.predicted_distance_m,
            },
        )
        applied = self.gate.apply(
            event,
            on_final_confirmed_transaction=on_final_confirmed_transaction,
        )
        if applied and self.candidate_started is not None:
            self.hold_latencies_s.append(time.monotonic() - self.candidate_started)
        return applied

    def apply_assessment(
        self,
        assessment: FrameAssessment,
        *,
        on_final_confirmed_transaction: Callable[
            [TargetIdentity, Mapping[str, Any]], Mapping[str, Any] | None
        ],
    ) -> bool:
        if self.identity is None or assessment.identity != self.identity:
            self.audit.append(
                {
                    "event": "assessment_discarded",
                    "reason": "stale_or_target_identity_mismatch",
                    **assessment.identity.__dict__,
                    "view_index": assessment.view_index,
                    "node_switched": False,
                    "bpl_updated": False,
                    "stop_authorized": False,
                }
            )
            return False
        if self.state == ArrivalState.CONFIRMED:
            self.audit.append(
                {
                    "event": "assessment_discarded",
                    "reason": "already_confirmed",
                    **assessment.identity.__dict__,
                    "view_index": assessment.view_index,
                    "node_switched": False,
                    "bpl_updated": False,
                    "stop_authorized": False,
                }
            )
            return False
        if assessment.rgb_change_from_previous is not None:
            self.changes[assessment.view_index - 1] = float(
                assessment.rgb_change_from_previous
            )
        self.last_assessment = assessment
        self.audit.append(
            {
                "event": "assessment_applied",
                **assessment.identity.__dict__,
                "view_index": assessment.view_index,
                "predicted_distance_m": assessment.predicted_distance_m,
                "safe": assessment.safe,
                "safety_reasons": list(assessment.safety_reasons),
                "inference_latency_ms": assessment.inference_latency_ms,
                "node_switched": False,
                "bpl_updated": False,
                "stop_authorized": False,
            }
        )

        if assessment.safe and assessment.predicted_distance_m <= self.deep_threshold_m:
            if self.candidate_index is None:
                self.candidate_index = assessment.view_index
                self.candidate_started = time.monotonic()
                self._transition(
                    ArrivalState.CANDIDATE,
                    "DEEP_ARRIVAL_CANDIDATE",
                    view_index=assessment.view_index,
                )
                self._transition(ArrivalState.HOLD, "SAFE_HOLD_FOR_FINAL_COMMIT")
            return self._confirm(
                assessment,
                mode="DEEP_ARRIVAL_FRESH_FRAME",
                on_final_confirmed_transaction=on_final_confirmed_transaction,
            )
        if self.candidate_index is None:
            if (
                assessment.safe
                and assessment.predicted_distance_m <= self.candidate_threshold_m
            ):
                self.candidate_index = assessment.view_index
                self.candidate_started = time.monotonic()
                self._transition(
                    ArrivalState.CANDIDATE,
                    "BOUNDARY_CANDIDATE",
                    view_index=assessment.view_index,
                )
                self._transition(ArrivalState.HOLD, "SAFE_HOLD_FOR_MARGIN")
            return False

        approaches = assessment.view_index - self.candidate_index
        if approaches < self.post_candidate_approaches:
            self._transition(
                ArrivalState.REOBSERVE,
                "POST_CANDIDATE_APPROACH",
                approaches_since_candidate=approaches,
            )
            return False
        if approaches == self.post_candidate_approaches:
            transition_changes = [
                self.changes.get(index)
                for index in range(self.candidate_index, assessment.view_index)
            ]
            changes_pass = all(
                value is not None and value >= self.minimum_rgb_change
                for value in transition_changes
            )
            if (
                changes_pass
                and assessment.safe
                and assessment.predicted_distance_m <= self.candidate_threshold_m
            ):
                return self._confirm(
                    assessment,
                    mode="BOUNDARY_MARGIN_CONFIRMED",
                    on_final_confirmed_transaction=on_final_confirmed_transaction,
                )
            self._transition(
                ArrivalState.REJECTED,
                "BOUNDARY_FINAL_RECHECK_FAILED",
                approaches_since_candidate=approaches,
                transition_changes=transition_changes,
            )
            if self.candidate_started is not None:
                self.hold_latencies_s.append(time.monotonic() - self.candidate_started)
            self._transition(ArrivalState.RECOVERY, "REJECTED_RESUME_NAVIGATION")
        return False

    def mark_timeout(self, reason: str) -> None:
        if self.state == ArrivalState.CONFIRMED:
            return
        self._transition(ArrivalState.TIMEOUT, reason)
        if self.candidate_started is not None:
            self.hold_latencies_s.append(time.monotonic() - self.candidate_started)
        self._transition(ArrivalState.RECOVERY, "TIMEOUT_RESUME_NAVIGATION")

    def complete_recovery(self, reason: str = "RECOVERY_ACTION_OBSERVED") -> None:
        if self.state != ArrivalState.RECOVERY:
            raise RuntimeError("recovery can only complete from RECOVERY")
        self._transition(ArrivalState.NAVIGATING, reason)

    def close(self) -> None:
        self.executor.shutdown(wait=True, cancel_futures=False)


class V7SequenceDecisionEngine:
    def __init__(
        self,
        *,
        model_path: Path,
        expected_model_sha256: str,
        integration_root: Path,
        hard_safety_config: Mapping[str, Any],
    ) -> None:
        if self.sha256(model_path) != expected_model_sha256:
            raise RuntimeError("v7 live model hash mismatch")
        import joblib
        import sys

        sys.path.insert(0, str(integration_root))
        from modules.Panoramic_Place_Compass.localization import hard_safety, make_features
        from modules.Panoramic_Place_Compass.localization import rgb_change

        self.model = joblib.load(model_path)
        self.make_features = make_features
        self.hard_safety = hard_safety
        self.rgb_change = rgb_change
        self.hard_safety_config = dict(hard_safety_config)

    @staticmethod
    def sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def assess(
        self,
        identity: TargetIdentity,
        *,
        active_evidence: dict[str, Any],
        parallax_features: dict[str, Any],
        extension: list[dict[str, Any]],
    ) -> FrameAssessment:
        started = time.perf_counter()
        synthetic = dict(active_evidence)
        synthetic["views"] = [*active_evidence["views"], *extension]
        features = self.make_features(parallax_features, synthetic)[None]
        predicted_distance = float(self.model.predict(features)[0])
        safe, reasons = self.hard_safety(synthetic, self.hard_safety_config)
        change = None
        if len(extension) >= 2:
            change = float(self.rgb_change(extension[-2], extension[-1]))
        return FrameAssessment(
            identity=identity,
            view_index=len(extension) - 1,
            predicted_distance_m=predicted_distance,
            safe=bool(safe),
            safety_reasons=tuple(reasons),
            rgb_change_from_previous=change,
            inference_latency_ms=(time.perf_counter() - started) * 1000.0,
            evidence={"feature_count": int(features.shape[1])},
        )


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(dict(value), indent=2, sort_keys=True) + "\n")


# ---------------------------------------------------------------------------
# Return and re-observation coordinator
# ---------------------------------------------------------------------------

import hashlib
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from PIL import Image

from modules.Panoramic_Place_Compass.geometry import DynamicParallaxExtractor


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass
class VerifierDirective:
    mode: str
    linear_velocity_mps: float
    angular_velocity_rps: float
    count_toward_phase: bool
    requires_traversability: bool
    relative_target_bearing_degrees: float | None = None


class LiveV7ReturnCoordinator:
    """Dynamic RGB verifier for one active historical-node return at a time."""

    ROTATION_ANGULAR_RPS = math.radians(10.0) / (3 * 0.25)
    APPROACH_LINEAR_MPS = 0.16
    APPROACH_ACTIONS = 4
    POST_CANDIDATE_APPROACH_ACTIONS = 2
    MAX_APPROACH_VIEWS = 6
    RECOVERY_ACTIONS = 4

    def __init__(
        self,
        *,
        graph: Any,
        r361: Any,
        geometry: Any,
        dynamic_parallax: DynamicParallaxExtractor,
        decision_engine: V7SequenceDecisionEngine,
        run_dir: Path,
        scene_id: str,
        time_step_s: float,
        candidate_cooldown_steps: int = 12,
    ) -> None:
        self.graph = graph
        self.r361 = r361
        self.geometry = geometry
        self.dynamic_parallax = dynamic_parallax
        self.engine = decision_engine
        self.run_dir = run_dir
        self.scene_id = scene_id
        self.time_step_s = float(time_step_s)
        self.candidate_cooldown_steps = int(candidate_cooldown_steps)
        self.machine = ArrivalSequenceV7StateMachine(async_workers=1)
        self.active = False
        self.phase = "IDLE"
        self.phase_remaining = 0
        self.cooldown_until_step = 0
        self.started_step = 0
        self.target_node_id: int | None = None
        self.target_rgb: np.ndarray | None = None
        self.target_hash: str | None = None
        self.target_encoding: Mapping[str, torch.Tensor] | None = None
        self.rotation_views: list[dict[str, Any]] = []
        self.approach_views: list[dict[str, Any]] = []
        self.approach_commanded_distances: list[float] = []
        self.current_approach_commanded_m = 0.0
        self.events: list[dict[str, Any]] = []
        self.frame_latencies_ms: list[float] = []
        self.parallax_latencies_ms: list[float] = []
        self.decision_latencies_ms: list[float] = []
        self.confirmed_count = 0
        self.rejected_count = 0
        self.timeout_count = 0
        self.recovery_count = 0
        self.budget_exhausted_abstain_count = 0

    @staticmethod
    def _wrap_degrees(value: float) -> float:
        return float((float(value) + 180.0) % 360.0 - 180.0)

    def _gallery(self, query_encoding: Mapping[str, torch.Tensor]) -> tuple[int, float]:
        node_ids = sorted(self.graph.r361_node_descriptors)
        query = query_encoding["global_descriptor"].detach().cpu().float().numpy()[0]
        query /= max(float(np.linalg.norm(query)), 1e-12)
        scores = np.asarray(
            [float(self.graph.r361_node_descriptors[node] @ query) for node in node_ids],
            dtype=np.float64,
        )
        order = np.argsort(-scores)
        top1 = int(node_ids[int(order[0])])
        target_index = node_ids.index(int(self.target_node_id))
        other = [index for index in order if int(index) != target_index]
        margin = (
            float(scores[target_index] - scores[int(other[0])])
            if other
            else float(scores[target_index])
        )
        return top1, margin

    def _measure(self, rgb: np.ndarray, label: str, step: int) -> dict[str, Any]:
        if self.target_rgb is None or self.target_encoding is None:
            raise RuntimeError("target encoding is unavailable")
        started = time.perf_counter()
        from modules.Panoramic_Place_Compass.localization import pair_from_encoding

        query_encoding = self.r361.encode_panorama(rgb)
        pair = pair_from_encoding(
            self.r361.runtime, query_encoding, self.target_encoding
        )
        top1, margin = self._gallery(query_encoding)
        geometry = self.geometry.analyze(rgb, self.target_rgb, pair["yaw_degrees"])
        evidence = {
            "arrival_probability": pair["arrival_probability"],
            "vpr_similarity": pair["vpr_similarity"],
            "candidate_margin": margin,
            "e_inliers": geometry["e_inliers"],
            "e_inlier_ratio": geometry["e_inlier_ratio"],
            "grid_coverage": geometry["grid_coverage"],
            "hull_area": geometry["hull_area"],
            "horizontal_span": geometry["horizontal_span"],
            "vertical_span": geometry["vertical_span"],
            "reprojection_error": geometry["reprojection_error"],
            "positive_depth_ratio": geometry["positive_depth_ratio"],
            "supported_sectors": geometry["supported_sectors"],
            "h_dominance": geometry["h_dominance"],
        }
        image_path = self.run_dir / "v7_return_views" / (
            f"g{self.machine.generation:03d}_s{step:04d}_{label}.png"
        )
        image_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rgb).save(image_path)
        latency_ms = (time.perf_counter() - started) * 1000.0
        self.frame_latencies_ms.append(latency_ms)
        return {
            "view_label": label,
            # Keep the verifier self-contained when debug image writing is
            # disabled; downstream geometry must not depend on disk I/O.
            "image_rgb": np.asarray(rgb).copy(),
            "image_path": str(image_path),
            "image_sha256": sha256(image_path),
            "target_top1": top1 == int(self.target_node_id),
            "top_candidate_node_id": top1,
            "r361": pair,
            "evidence": evidence,
            "geometry": geometry,
            "frame_latency_ms": latency_ms,
            "runtime_gt_inputs": [],
        }

    def _candidate_trigger(self, rgb: np.ndarray, step: int) -> dict[str, Any] | None:
        state = self.graph.return_state
        if state is None or step < self.cooldown_until_step:
            return None
        target = self.graph.graph.regular_nodes[state.target_node]
        descriptor = self.graph.v31.rgb_descriptor(rgb)
        similarity = self.graph.v31.cosine(descriptor, target.descriptor)
        reference = np.asarray(Image.open(str(target.keyframe_rgb)).convert("RGB"))
        panorama = self.graph.v31.panorama_verification(rgb, reference)
        legacy_rgb_accepted = bool(similarity >= 0.992 or panorama.get("accepted", False))
        r361 = self.graph._r361_retrieve(rgb)
        target_top1 = bool(
            r361 is not None
            and int(r361["top1_node_id"]) == int(state.target_node)
        )
        accepted = bool(legacy_rgb_accepted and target_top1)
        row = {
            "step": int(step),
            "event": "v7_return_candidate_observation",
            "generation": int(state.generation),
            "candidate_node_id": int(state.target_node),
            "descriptor_similarity": float(similarity),
            "panorama": panorama,
            "legacy_rgb_accepted": legacy_rgb_accepted,
            "r361_retrieval": r361,
            "r361_target_top1": target_top1,
            "candidate_accepted": accepted,
            "final_confirmed": False,
            "node_switched": False,
            "bpl_updated": False,
            "runtime_gt_inputs": [],
        }
        self.graph.arrival_events.append(row)
        self.events.append(row)
        if not accepted:
            return row

        self.active = True
        self.started_step = int(step)
        self.target_node_id = int(state.target_node)
        self.target_rgb = reference
        target_path = Path(str(target.keyframe_rgb))
        self.target_hash = sha256(target_path)
        self.target_encoding = self.r361.encode_panorama(reference)
        identity = self.machine.begin_target(
            request_id=f"v7-return:{self.scene_id}:{state.generation}:{step}",
            candidate_node_id=str(self.target_node_id),
            target_hash=self.target_hash,
            final_goal=False,
        )
        self.rotation_views = [self._measure(rgb, "ORIGINAL", step)]
        self.approach_views = []
        self.approach_commanded_distances = []
        self.current_approach_commanded_m = 0.0
        self.phase = "ROTATE_LEFT_10"
        self.phase_remaining = 3
        started = {
            "step": int(step),
            "event": "v7_return_verification_started",
            **identity.__dict__,
            "final_confirmed": False,
            "node_switched": False,
            "bpl_updated": False,
            "runtime_gt_inputs": [],
        }
        self.graph.arrival_events.append(started)
        self.events.append(started)
        return started

    def _commit(self, identity: Any, evidence: Mapping[str, Any]) -> dict[str, Any]:
        state = self.graph.return_state
        if (
            state is None
            or int(identity.generation) != int(self.machine.generation)
            or int(identity.candidate_node_id) != int(state.target_node)
            or identity.target_hash != self.target_hash
        ):
            raise RuntimeError("v7 return transaction identity mismatch")
        source = state.route[state.route_index - 1]
        target_node = state.target_node
        transition = self.graph.graph.directed_transitions[(source, target_node)]
        transition.mark_success(
            confidence=1.0,
            step=int(evidence["step"]),
            record={
                "source": "dynamic_parallax_v7_final_confirmed",
                "generation": state.generation,
                "target_hash": identity.target_hash,
            },
        )
        self.graph.current_node = target_node
        self.graph._event_bpl_update(
            target_node, int(evidence["step"]), "historical_node_return_v7_final_confirmed"
        )
        state.route_index += 1
        route_complete = state.route_index >= len(state.route)
        next_target = None
        if route_complete:
            self.graph.return_state = None
        else:
            state.target_node = state.route[state.route_index]
            state.started_step = int(evidence["step"])
            state.consecutive_confirmations = 0
            next_target = int(state.target_node)
        self.confirmed_count += 1
        return {
            "transaction": "NODE_SWITCH_PLUS_EVENT_BPL",
            "source_node_id": int(source),
            "target_node_id": int(target_node),
            "route_complete": route_complete,
            "next_target_node": next_target,
            "node_switch_count": 1,
            "bpl_update_count": 1,
            "stop_count": 0,
        }

    def _assess(self, step: int) -> dict[str, Any]:
        if len(self.approach_views) < 2 or self.target_rgb is None:
            raise RuntimeError("dynamic parallax requires two completed approaches")
        triplet_source = (
            [self.rotation_views[-1], *self.approach_views]
            if len(self.approach_views) == 2
            else self.approach_views[-3:]
        )
        triplet = []
        for label, view in zip(
            ("RESTORE_10", "APPROACH_1", "APPROACH_2"), triplet_source
        ):
            image_rgb = view.get("image_rgb")
            if image_rgb is None:
                image_path = view.get("image_path")
                if not image_path:
                    raise RuntimeError(
                        "dynamic parallax view has neither in-memory RGB nor image_path"
                    )
                image_rgb = np.asarray(Image.open(image_path).convert("RGB"))
            triplet.append(
                {
                    "label": label,
                    "image_rgb": np.asarray(image_rgb),
                    "target_yaw_degrees": view["r361"]["yaw_degrees"],
                }
            )
        commanded = self.approach_commanded_distances[-2:]
        step_m = float(np.mean(commanded))
        if not np.isfinite(step_m) or step_m <= 1e-6:
            # A verifier approach is only valid when OmniGuard actually
            # accepted a positive forward command.  Treating a zero-distance
            # pair as a parallax measurement would either crash the runtime
            # or manufacture a distance estimate from no motion.  Preserve
            # the transaction gate and resume through bounded recovery.
            self.machine.mark_timeout("NO_EFFECTIVE_APPROACH_MOTION")
            self.timeout_count += 1
            self.phase = "RECOVERY_SCAN"
            self.phase_remaining = self.RECOVERY_ACTIONS
            row = {
                "step": int(step),
                "event": "v7_return_no_effective_approach_motion",
                **self.machine.identity.__dict__,
                "commanded_forward_distance_m": step_m,
                "final_confirmed": False,
                "node_switched": False,
                "bpl_updated": False,
                "runtime_gt_inputs": [],
            }
            self.graph.arrival_events.append(row)
            self.events.append(row)
            return row
        parallax = self.dynamic_parallax.extract(
            request_id=self.machine.identity.request_id,
            generation=self.machine.identity.generation,
            candidate_node_id=self.machine.identity.candidate_node_id,
            target_hash=self.machine.identity.target_hash,
            target_rgb=self.target_rgb,
            views=triplet,
            commanded_forward_distance_per_step_m=step_m,
        )
        self.parallax_latencies_ms.append(float(parallax["latency_ms"]))
        synthetic = {
            "trial_id": self.machine.identity.request_id,
            "views": [*self.rotation_views, *self.approach_views],
        }
        started = time.perf_counter()
        features = self.engine.make_features(parallax, synthetic)[None]
        predicted_distance = float(self.engine.model.predict(features)[0])
        safe, reasons = self.engine.hard_safety(
            synthetic, self.engine.hard_safety_config
        )
        change = None
        if len(self.approach_views) >= 2:
            change = float(
                self.engine.rgb_change(
                    self.approach_views[-2], self.approach_views[-1]
                )
            )
        decision_latency_ms = (time.perf_counter() - started) * 1000.0
        self.decision_latencies_ms.append(decision_latency_ms)
        assessment = FrameAssessment(
            identity=self.machine.identity,
            view_index=len(self.approach_views) - 1,
            predicted_distance_m=predicted_distance,
            safe=bool(safe),
            safety_reasons=tuple(reasons),
            rgb_change_from_previous=change,
            inference_latency_ms=decision_latency_ms,
            evidence={
                "step": int(step),
                "feature_count": int(features.shape[1]),
                "dynamic_parallax_latency_ms": parallax["latency_ms"],
                "commanded_forward_distance_m": step_m,
            },
        )
        applied = self.machine.apply_assessment(
            assessment, on_final_confirmed_transaction=self._commit
        )
        row = {
            "step": int(step),
            "event": "v7_return_assessment",
            **self.machine.identity.__dict__,
            "predicted_distance_m": predicted_distance,
            "safe": bool(safe),
            "safety_reasons": list(reasons),
            "rgb_change_from_previous": change,
            "dynamic_parallax_latency_ms": parallax["latency_ms"],
            "decision_latency_ms": decision_latency_ms,
            "state": self.machine.state.value,
            "final_confirmed": bool(applied),
            "node_switched": bool(applied),
            "bpl_updated": bool(applied),
            "runtime_gt_inputs": [],
        }
        self.graph.arrival_events.append(row)
        self.events.append(row)
        if applied:
            row["event"] = "return_arrival_final_confirmed_v7"
            self.active = False
            self.phase = "IDLE"
        elif self.machine.state in (ArrivalState.HOLD, ArrivalState.REOBSERVE):
            self.phase = "APPROACH_MARGIN"
            self.phase_remaining = self.POST_CANDIDATE_APPROACH_ACTIONS
            self.current_approach_commanded_m = 0.0
        elif (
            self.machine.state == ArrivalState.NAVIGATING
            and len(self.approach_views) < self.MAX_APPROACH_VIEWS
        ):
            self.phase = "APPROACH_MARGIN"
            self.phase_remaining = self.APPROACH_ACTIONS
            self.current_approach_commanded_m = 0.0
        else:
            if self.machine.state == ArrivalState.NAVIGATING:
                self.machine.mark_timeout("V7_NO_CANDIDATE_AFTER_DYNAMIC_PARALLAX")
                self.timeout_count += 1
            elif self.machine.state == ArrivalState.RECOVERY:
                self.rejected_count += 1
            self.phase = "RECOVERY_SCAN"
            self.phase_remaining = self.RECOVERY_ACTIONS
        return row

    def observe(self, rgb: np.ndarray, step: int) -> dict[str, Any] | None:
        if not self.active:
            return self._candidate_trigger(rgb, step)
        if self.phase_remaining > 0:
            return None

        if self.phase == "ROTATE_LEFT_10":
            self.rotation_views.append(self._measure(rgb, "LEFT_10", step))
            self.phase = "ROTATE_RIGHT_20"
            self.phase_remaining = 6
        elif self.phase == "ROTATE_RIGHT_20":
            self.rotation_views.append(self._measure(rgb, "RIGHT_20", step))
            self.phase = "ROTATE_RESTORE_10"
            self.phase_remaining = 3
        elif self.phase == "ROTATE_RESTORE_10":
            self.rotation_views.append(self._measure(rgb, "RESTORE_10", step))
            self.phase = "APPROACH_MARGIN"
            self.phase_remaining = self.APPROACH_ACTIONS
            self.current_approach_commanded_m = 0.0
        elif self.phase == "APPROACH_MARGIN":
            label = f"MARGIN_{len(self.approach_views) + 1}"
            self.approach_views.append(self._measure(rgb, label, step))
            self.approach_commanded_distances.append(
                self.current_approach_commanded_m
            )
            if len(self.approach_views) < 2:
                self.phase_remaining = self.APPROACH_ACTIONS
                self.current_approach_commanded_m = 0.0
            else:
                return self._assess(step)
        elif self.phase == "RECOVERY_SCAN":
            self.machine.complete_recovery("V7_RETURN_RECOVERY_SCAN_COMPLETE")
            self.recovery_count += 1
            self.cooldown_until_step = int(step) + self.candidate_cooldown_steps
            row = {
                "step": int(step),
                "event": "v7_return_recovery_complete",
                "generation": self.machine.generation,
                "candidate_node_id": self.target_node_id,
                "final_confirmed": False,
                "node_switched": False,
                "bpl_updated": False,
                "runtime_gt_inputs": [],
            }
            self.graph.arrival_events.append(row)
            self.events.append(row)
            self.active = False
            self.phase = "IDLE"
            return row
        return None

    def directive(self, relative_target_bearing_degrees: float) -> VerifierDirective | None:
        if not self.active:
            return None
        if self.phase == "ROTATE_LEFT_10":
            return VerifierDirective(
                self.phase, 0.0, -self.ROTATION_ANGULAR_RPS, True, False
            )
        if self.phase == "ROTATE_RIGHT_20":
            return VerifierDirective(
                self.phase, 0.0, self.ROTATION_ANGULAR_RPS, True, False
            )
        if self.phase == "ROTATE_RESTORE_10":
            return VerifierDirective(
                self.phase, 0.0, -self.ROTATION_ANGULAR_RPS, True, False
            )
        if self.phase == "RECOVERY_SCAN":
            return VerifierDirective(
                self.phase, 0.0, 0.15, True, False
            )
        if self.phase == "APPROACH_MARGIN":
            return VerifierDirective(
                self.phase,
                self.APPROACH_LINEAR_MPS,
                0.0,
                True,
                True,
                float(relative_target_bearing_degrees),
            )
        return None

    def action_executed(
        self,
        directive: VerifierDirective,
        *,
        linear_velocity_mps: float,
        phase_action_accepted: bool,
    ) -> None:
        if not self.active or directive.mode != self.phase or not phase_action_accepted:
            return
        if self.phase_remaining <= 0:
            raise RuntimeError("v7 verifier phase action underflow")
        self.phase_remaining -= 1
        if self.phase == "APPROACH_MARGIN":
            self.current_approach_commanded_m += (
                float(linear_velocity_mps) * self.time_step_s
            )

    def abort_for_route_timeout(self, step: int) -> dict[str, Any] | None:
        if not self.active:
            return None
        self.machine.mark_timeout("GLOBAL_RETURN_ROUTE_TIMEOUT")
        self.timeout_count += 1
        self.phase = "RECOVERY_SCAN"
        self.phase_remaining = self.RECOVERY_ACTIONS
        row = {
            "step": int(step),
            "event": "v7_return_route_timeout",
            "generation": self.machine.generation,
            "candidate_node_id": self.target_node_id,
            "final_confirmed": False,
            "node_switched": False,
            "bpl_updated": False,
            "runtime_gt_inputs": [],
        }
        self.graph.arrival_events.append(row)
        self.events.append(row)
        return row

    def abort_unsafe_approach(self, step: int, reason: str) -> dict[str, Any] | None:
        if not self.active or self.phase != "APPROACH_MARGIN":
            return None
        self.machine.mark_timeout(str(reason))
        self.timeout_count += 1
        self.phase = "RECOVERY_SCAN"
        self.phase_remaining = self.RECOVERY_ACTIONS
        row = {
            "step": int(step),
            "event": "v7_return_unsafe_approach_aborted",
            "reason": str(reason),
            "generation": self.machine.generation,
            "candidate_node_id": self.target_node_id,
            "final_confirmed": False,
            "node_switched": False,
            "bpl_updated": False,
            "runtime_gt_inputs": [],
        }
        self.graph.arrival_events.append(row)
        self.events.append(row)
        return row

    def finalize_action_budget(self, step: int) -> dict[str, Any] | None:
        if not self.active:
            return None
        self.machine.mark_timeout("ACTION_BUDGET_EXHAUSTED_ABSTAIN")
        self.budget_exhausted_abstain_count += 1
        row = {
            "step": int(step),
            "event": "v7_return_budget_exhausted_abstain",
            "generation": self.machine.generation,
            "candidate_node_id": self.target_node_id,
            "final_confirmed": False,
            "node_switched": False,
            "bpl_updated": False,
            "runtime_gt_inputs": [],
        }
        self.graph.arrival_events.append(row)
        self.events.append(row)
        self.active = False
        self.phase = "IDLE"
        return row

    def payload(self) -> dict[str, Any]:
        return {
            "schema": "integration_v2_v7_return_coordinator_v1",
            "events": self.events,
            "machine_audit": self.machine.audit,
            "gate_audit": self.machine.gate.audit,
            "confirmed_count": self.confirmed_count,
            "rejected_count": self.rejected_count,
            "timeout_count": self.timeout_count,
            "recovery_count": self.recovery_count,
            "budget_exhausted_abstain_count": self.budget_exhausted_abstain_count,
            "frame_latencies_ms": self.frame_latencies_ms,
            "dynamic_parallax_latencies_ms": self.parallax_latencies_ms,
            "decision_latencies_ms": self.decision_latencies_ms,
            "runtime_gt_inputs": [],
            "ordinary_frame_bpl_mutations": 0,
            "bpl_can_affirm_stop": False,
            "event_bpl_can_affirm_stop": False,
        }

    def close(self) -> None:
        self.machine.close()


# ---------------------------------------------------------------------------
# Goal-image arrival verifier
# ---------------------------------------------------------------------------

"""Conditional V3.3.12/V7 RGB arrival adapter for the candidate-frontier path.

The frozen verifier is deliberately kept behind a small bridge.  It can
observe and produce a final-confirmed event, but it cannot mutate the topology
graph or authorize Stop except through ``ArrivalGate``.
"""

import hashlib
import json
import os
import types
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from modules.Topological_Belief_Cascade.topology import (
    GOAL_FINAL_CONFIRMED,
    CANDIDATE_FRONTIER_FINAL_CONFIRMED,
)


ROOT = Path(__file__).resolve().parent
ARRIVAL_PROTOCOL = Path(os.environ.get(
    "PIVOTNAV_ARRIVAL_PROTOCOL",
    str(ROOT / "arrival_protocol.json"),
)).expanduser().resolve()


def resolve_protocol_resource(value: str | os.PathLike[str]) -> Path:
    """Resolve an external frozen resource relative to the selected protocol."""
    candidate = Path(value).expanduser()
    if candidate.is_absolute():
        return candidate
    return (ARRIVAL_PROTOCOL.parent / candidate).resolve()


def typed_commit_payload(target_kind: str, stop_authorized: bool) -> dict[str, Any]:
    """Normalize the frozen coordinator result to the V4 typed event contract."""
    if target_kind == "ghost":
        target_kind = "candidate_frontier"
    if target_kind == "goal":
        assert stop_authorized
        return {
            "transaction": "V4_GOAL_FINAL_CONFIRMED_STOP",
            "event_type": GOAL_FINAL_CONFIRMED,
            "stop_authorized": True,
            "node_switch_count": 0,
            "bpl_update_count": 0,
        }
    if target_kind == "candidate_frontier":
        assert not stop_authorized
        return {
            "transaction": "V4_CANDIDATE_FRONTIER_FINAL_CONFIRMED_TRANSACTION",
            "event_type": CANDIDATE_FRONTIER_FINAL_CONFIRMED,
            "stop_authorized": False,
            "node_switch_count": 1,
            "bpl_update_count": 1,
        }
    raise ValueError(f"unsupported arrival target kind: {target_kind}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class _GoalGraph:
    """Read-only gallery bridge used by the frozen measurement code."""

    def __init__(self, descriptor: np.ndarray, target_path: Path) -> None:
        self.r361_node_descriptors = {1: np.asarray(descriptor, dtype=np.float32)}
        self.graph = types.SimpleNamespace(
            regular_nodes={1: types.SimpleNamespace(keyframe_rgb=str(target_path), descriptor=descriptor)},
        )
        self.arrival_events: list[dict[str, Any]] = []
        self.current_node = 0
        self.return_state = None


class GoalImageArrivalVerifier:
    """V7 sequence gate specialized for an explicit Goal ERP target."""

    schema = "integration_v4_conditional_goal_arrival_verifier_v1"

    def __init__(
        self,
        *,
        policy: Any,
        r361: Any,
        goal_rgb: np.ndarray,
        goal_path: Path,
        run_dir: Path,
        scene_id: str,
        verifier_device: str = "cuda:0",
        vpr_candidate_threshold: float = 0.992,
        candidate_observe_cadence: int = 2,
        target_kind: str = "goal",
        candidate_frontier_id: str | None = None,
        shared_dynamic_parallax: Any | None = None,
        shared_decision_engine: Any | None = None,
    ) -> None:
        import sys

        frozen_root = Path(os.environ.get("PIVOTNAV_ARRIVAL_FROZEN_ROOT", str(ROOT))).expanduser()
        sys.path[:0] = [str(ROOT), str(frozen_root)]
        from modules.Panoramic_Place_Compass.geometry import DynamicParallaxExtractor
        from modules.Panoramic_Place_Compass.localization import pair_from_encoding

        protocol = json.loads(ARRIVAL_PROTOCOL.read_text())
        model = protocol["model"]
        self.protocol_id = protocol["protocol_id"]
        self.protocol_status = protocol["status"]
        self.production_approved = bool(protocol.get("production_approved", False))
        self.r361 = r361
        self.policy = policy
        self.goal_rgb = np.asarray(goal_rgb)[..., :3].astype(np.uint8)
        self.goal_path = Path(goal_path)
        self.goal_hash = sha256(self.goal_path)
        self.goal_encoding = r361.encode_panorama(self.goal_rgb)
        descriptor = self.goal_encoding["global_descriptor"].detach().cpu().float().numpy()[0]
        descriptor /= max(float(np.linalg.norm(descriptor)), 1e-12)
        self.graph = _GoalGraph(descriptor, self.goal_path)
        self.gate = ArrivalGate(policy)
        self.run_dir = Path(run_dir)
        self.scene_id = str(scene_id)
        self.vpr_candidate_threshold = float(vpr_candidate_threshold)
        self.candidate_observe_cadence = max(1, int(candidate_observe_cadence))
        self.candidate_observation_skipped_count = 0
        self.candidate_observation_count = 0
        self.candidate_accept_count = 0
        # Candidate evidence images are diagnostic-only. Keep them opt-in so
        # long ablation rollouts do not spend time on filesystem I/O.
        self.write_debug_return_views = os.environ.get(
            "PHASE6_WRITE_DEBUG_RETURN_VIEWS", "0"
        ).strip().lower() in {"1", "true", "yes", "on"}
        # Known regular-node navigation is deliberately handled by
        # ``KnownNodeLocalizer`` (VPR+BPL+multi-frame) and must never enter the
        # final Goal geometry verifier.  This verifier is only for the final
        # Goal or a candidate-frontier-to-regular arrival transaction.
        if target_kind == "ghost":
            target_kind = "candidate_frontier"
        if target_kind not in {"goal", "candidate_frontier"}:
            raise ValueError(f"unsupported arrival target kind: {target_kind}")
        if target_kind == "candidate_frontier" and not candidate_frontier_id:
            raise ValueError("candidate-frontier arrival requires candidate_frontier_id")
        self.arrival_target_kind = str(target_kind)
        self.arrival_candidate_frontier_id = str(candidate_frontier_id) if candidate_frontier_id is not None else None
        self.arrival_final_goal = self.arrival_target_kind == "goal"
        self.pair_from_encoding = pair_from_encoding
        self.dynamic_parallax = shared_dynamic_parallax or DynamicParallaxExtractor(
            device=str(verifier_device), commanded_forward_distance_per_step_m=0.16
        )
        configured_model = os.environ.get("PIVOTNAV_ARRIVAL_MODEL")
        model_path = (
            Path(configured_model).expanduser().resolve()
            if configured_model
            else resolve_protocol_resource(model["path"])
        )
        if not model_path.is_file():
            raise RuntimeError(
                "arrival model is not configured; set PIVOTNAV_ARRIVAL_MODEL "
                f"to a readable joblib file (looked for {model_path})"
            )
        self.engine = shared_decision_engine or V7SequenceDecisionEngine(
            model_path=model_path, expected_model_sha256=model["sha256"],
            integration_root=ROOT, hard_safety_config=protocol["hard_safety"]
        )
        super().__init__ if False else None
        # Compose the frozen coordinator methods without letting its graph
        # transaction run.  The subclass below overrides candidate and commit.
        self._coordinator = _GoalCoordinator(
            graph=self.graph,
            r361=r361,
            geometry=self.dynamic_parallax.geometry,
            dynamic_parallax=self.dynamic_parallax,
            decision_engine=self.engine,
            run_dir=self.run_dir,
            scene_id=self.scene_id,
            time_step_s=0.25,
            policy=self.policy,
            gate=self.gate,
            goal_rgb=self.goal_rgb,
            goal_path=self.goal_path,
            goal_hash=self.goal_hash,
            goal_encoding=self.goal_encoding,
            pair_from_encoding=self.pair_from_encoding,
            vpr_candidate_threshold=self.vpr_candidate_threshold,
            write_debug_return_views=self.write_debug_return_views,
            arrival_target_kind=self.arrival_target_kind,
            arrival_candidate_frontier_id=self.arrival_candidate_frontier_id,
            arrival_final_goal=self.arrival_final_goal,
        )

    def observe(self, rgb: np.ndarray, step: int) -> dict[str, Any] | None:
        if not self.active and (int(step) - 1) % self.candidate_observe_cadence:
            self.candidate_observation_skipped_count += 1
            event = {
                "step": int(step),
                "event": "v4_goal_candidate_observation_skipped_cadence",
                "candidate_observe_cadence": self.candidate_observe_cadence,
                "final_confirmed": False,
                "node_switched": False,
                "bpl_updated": False,
                "stop_authorized": False,
                "runtime_gt_inputs": [],
            }
            self._coordinator.events.append(event)
            return event
        return self._coordinator.observe(rgb, int(step))

    def directive(self, bearing_degrees: float):
        return self._coordinator.directive(float(bearing_degrees))

    def action_executed(self, directive: Any, *, linear_velocity_mps: float, phase_action_accepted: bool) -> None:
        self._coordinator.action_executed(
            directive, linear_velocity_mps=float(linear_velocity_mps), phase_action_accepted=bool(phase_action_accepted)
        )

    def abort_unsafe_approach(self, step: int, reason: str):
        return self._coordinator.abort_unsafe_approach(int(step), str(reason))

    def finalize_action_budget(self, step: int):
        return self._coordinator.finalize_action_budget(int(step))

    @property
    def active(self) -> bool:
        return bool(self._coordinator.active)

    @property
    def phase(self) -> str:
        return str(self._coordinator.phase)

    @property
    def stop_authorized(self) -> bool:
        return bool(self.gate.stop_authorized)

    def payload(self) -> dict[str, Any]:
        payload = self._coordinator.payload()
        payload.update({
            "schema": self.schema,
            "protocol_id": self.protocol_id,
            "protocol_status": self.protocol_status,
            "production_approved": self.production_approved,
            "goal_hash": self.goal_hash,
            "candidate_threshold": self.vpr_candidate_threshold,
            "candidate_observe_cadence": self.candidate_observe_cadence,
            "candidate_observation_skipped_count": self.candidate_observation_skipped_count,
            "candidate_observation_count": self._coordinator.candidate_observation_count,
            "candidate_accept_count": self._coordinator.candidate_accept_count,
            "query_encode_count": self._coordinator.query_encode_count,
            "query_cache_hit_count": self._coordinator.query_cache_hit_count,
            "current_target_order_contract": {
                "vpr": "CURRENT_QUERY_THEN_GOAL_TARGET",
                "geometry": "CURRENT_OBSERVATION_THEN_GOAL_TARGET",
            },
            "gate_audit": self.gate.audit,
            "stop_authorized": self.stop_authorized,
        })
        return payload

    def close(self) -> None:
        self._coordinator.close()


class _GoalCoordinator:
    """Small subclass-like copy of the frozen coordinator's live lifecycle."""

    def __init__(self, *, graph, r361, geometry, dynamic_parallax, decision_engine,
                 run_dir, scene_id, time_step_s, policy, gate, goal_rgb, goal_path,
                 goal_hash, goal_encoding, pair_from_encoding, vpr_candidate_threshold,
                 write_debug_return_views,
                 arrival_target_kind, arrival_candidate_frontier_id, arrival_final_goal):
        class Coordinator(LiveV7ReturnCoordinator):
            def _candidate_trigger(inner, rgb, step):
                if inner.active or step < inner.cooldown_until_step:
                    return None
                inner.candidate_observation_count += 1
                query_encoding = inner._encode_query_once(rgb)
                pair = pair_from_encoding(inner.r361.runtime, query_encoding, inner.target_encoding_goal)
                similarity = float(pair["vpr_similarity"])
                arrival_probability = float(pair["arrival_probability"])
                candidate_node_id = "GOAL" if inner.arrival_target_kind == "goal" else inner.arrival_candidate_frontier_id
                row = {
                    "step": int(step), "event": (
                        "v4_goal_arrival_candidate_observation"
                        if inner.arrival_target_kind == "goal"
                        else "v4_candidate_frontier_arrival_candidate_observation"
                    ),
                    "candidate_node_id": candidate_node_id, "vpr_similarity": similarity,
                    "arrival_probability": arrival_probability,
                    "candidate_accepted": bool(similarity >= inner.vpr_candidate_threshold),
                    "final_confirmed": False, "node_switched": False,
                    "bpl_updated": False, "runtime_gt_inputs": [],
                    "vpr_input_order": "CURRENT_QUERY_THEN_GOAL_TARGET",
                    "current_target_order_asserted": True,
                }
                inner.events.append(row)
                if similarity < inner.vpr_candidate_threshold:
                    return row
                inner.candidate_accept_count += 1
                inner.active = True
                inner.started_step = int(step)
                inner.target_node_id = 1
                inner.target_rgb = inner.goal_rgb
                inner.target_hash = inner.goal_hash
                inner.target_encoding = inner.target_encoding_goal
                inner.gate.begin(target_kind=inner.arrival_target_kind,
                                 candidate_frontier_id=inner.arrival_candidate_frontier_id,
                                 final_goal=inner.arrival_final_goal)
                identity = inner.machine.begin_target(
                    request_id=f"v4-{inner.arrival_target_kind}:{inner.scene_id}:{step}",
                    candidate_node_id=candidate_node_id, target_hash=inner.target_hash,
                    final_goal=inner.arrival_final_goal,
                )
                inner.rotation_views = [inner._measure(rgb, "ORIGINAL", step)]
                inner.approach_views = []
                inner.approach_commanded_distances = []
                inner.current_approach_commanded_m = 0.0
                inner.phase = "ROTATE_LEFT_10"
                inner.phase_remaining = 3
                started = {"step": int(step), "event": (
                               "v4_goal_verification_started"
                               if inner.arrival_target_kind == "goal"
                               else "v4_candidate_frontier_verification_started"
                           ),
                           **identity.__dict__, "final_confirmed": False,
                           "node_switched": False, "bpl_updated": False,
                           "runtime_gt_inputs": []}
                inner.events.append(started)
                return started

            def _gallery(inner, query_encoding):
                query = query_encoding["global_descriptor"].detach().cpu().float().numpy()[0]
                query /= max(float(np.linalg.norm(query)), 1e-12)
                target = inner.graph.r361_node_descriptors[1]
                return 1, float(np.dot(target, query))

            def _assess(inner, step):
                row = super(Coordinator, inner)._assess(step)
                # The explicit Goal target is not a topology node.  The
                # frozen coordinator uses generic node-event fields for its
                # result row, so normalize those fields at this boundary.
                if row and row.get("final_confirmed") and row.get("candidate_node_id") == "GOAL":
                    row["node_switched"] = False
                    row["bpl_updated"] = False
                return row

            def _measure(inner, rgb, label, step):
                if inner.target_rgb is None or inner.target_encoding is None:
                    raise RuntimeError("target encoding is unavailable")
                started = time.perf_counter()
                query_encoding = inner._encode_query_once(rgb)
                pair = pair_from_encoding(inner.r361.runtime, query_encoding, inner.target_encoding)
                top1, margin = inner._gallery(query_encoding)
                geometry = inner.geometry.analyze(rgb, inner.target_rgb, pair["yaw_degrees"])
                evidence = {
                    "arrival_probability": pair["arrival_probability"],
                    "vpr_similarity": pair["vpr_similarity"],
                    "candidate_margin": margin,
                    **{key: geometry[key] for key in (
                        "e_inliers", "e_inlier_ratio", "grid_coverage", "hull_area",
                        "horizontal_span", "vertical_span", "reprojection_error",
                        "positive_depth_ratio", "supported_sectors", "h_dominance",
                    )},
                }
                image_path = None
                image_sha256 = None
                if getattr(inner, "write_debug_return_views", False):
                    image_path = inner.run_dir / "v7_return_views" / (
                        f"g{inner.machine.generation:03d}_s{step:04d}_{label}.png"
                    )
                    image_path.parent.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(rgb).save(image_path)
                    image_sha256 = sha256(image_path)
                latency_ms = (time.perf_counter() - started) * 1000.0
                inner.frame_latencies_ms.append(latency_ms)
                return {
                    "view_label": label,
                    # Keep the verifier self-contained when debug image writing
                    # is disabled; downstream geometry must not depend on disk I/O.
                    "image_rgb": np.asarray(rgb).copy(),
                    "image_path": str(image_path) if image_path is not None else None,
                    "image_sha256": image_sha256,
                    "target_top1": top1 == int(inner.target_node_id),
                    "top_candidate_node_id": top1,
                    "r361": pair,
                    "evidence": evidence,
                    "geometry": geometry,
                    "frame_latency_ms": latency_ms,
                    "geometry_input_order": "CURRENT_OBSERVATION_THEN_GOAL_TARGET",
                    "current_target_order_asserted": True,
                    "runtime_gt_inputs": [],
                }

            def _encode_query_once(inner, rgb):
                array = np.ascontiguousarray(np.asarray(rgb)[..., :3].astype(np.uint8, copy=False))
                key = hashlib.sha256(array.tobytes()).hexdigest()
                if key == inner._last_query_key and inner._last_query_encoding is not None:
                    inner.query_cache_hit_count += 1
                    return inner._last_query_encoding
                encoding = inner.r361.encode_panorama(array)
                inner.query_encode_count += 1
                inner._last_query_key = key
                inner._last_query_encoding = encoding
                return encoding

            def _commit(inner, identity, evidence):
                applied = inner.gate.apply(
                    generation=int(identity.generation), final_confirmed=True,
                    step=int(evidence.get("step", inner.started_step)),
                    evidence={"decision": "V7_FINAL_CONFIRMED", **dict(evidence)},
                )
                if not applied:
                    raise RuntimeError("V4 goal final-confirmed transaction was rejected")
                inner.confirmed_count += 1
                return typed_commit_payload(
                    inner.arrival_target_kind,
                    bool(inner.gate.stop_authorized),
                )

            def payload(inner):
                payload = super(Coordinator, inner).payload()
                payload["gate_audit"] = inner.gate.audit
                payload["stop_authorized"] = inner.gate.stop_authorized
                return payload

        self._impl = Coordinator(
            graph=graph, r361=r361, geometry=geometry, dynamic_parallax=dynamic_parallax,
            decision_engine=decision_engine, run_dir=run_dir, scene_id=scene_id,
            time_step_s=time_step_s,
        )
        self._impl.candidate_observation_count = 0
        self._impl.candidate_accept_count = 0
        self._impl.query_encode_count = 0
        self._impl.query_cache_hit_count = 0
        self._impl._last_query_key = None
        self._impl._last_query_encoding = None
        self._impl.policy = policy
        self._impl.gate = gate
        self._impl.goal_rgb = goal_rgb
        self._impl.goal_path = goal_path
        self._impl.goal_hash = goal_hash
        self._impl.target_encoding_goal = goal_encoding
        self._impl.vpr_candidate_threshold = vpr_candidate_threshold
        self._impl.write_debug_return_views = bool(write_debug_return_views)
        self._impl.arrival_target_kind = arrival_target_kind
        self._impl.arrival_candidate_frontier_id = arrival_candidate_frontier_id
        self._impl.arrival_final_goal = arrival_final_goal
        self._impl.graph = graph
        self._impl.current_target_order_contract = {
            "vpr": "CURRENT_QUERY_THEN_GOAL_TARGET",
            "geometry": "CURRENT_OBSERVATION_THEN_GOAL_TARGET",
        }

    def __getattr__(self, name):
        return getattr(self._impl, name)

    def observe(self, *args, **kwargs):
        return self._impl.observe(*args, **kwargs)

    def directive(self, *args, **kwargs):
        return self._impl.directive(*args, **kwargs)

    def action_executed(self, *args, **kwargs):
        return self._impl.action_executed(*args, **kwargs)

    def abort_unsafe_approach(self, *args, **kwargs):
        return self._impl.abort_unsafe_approach(*args, **kwargs)

    def finalize_action_budget(self, *args, **kwargs):
        return self._impl.finalize_action_budget(*args, **kwargs)

    def payload(self):
        return self._impl.payload()

    def close(self):
        return self._impl.close()
