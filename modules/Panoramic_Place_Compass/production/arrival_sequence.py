from __future__ import annotations

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
        from modules.Panoramic_Place_Compass.production.evaluate_parallax import hard_safety, make_features
        from modules.Panoramic_Place_Compass.production.evaluate_safety_margin import rgb_change

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
