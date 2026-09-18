from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict


class AnchorEvidenceState(str, Enum):
    UNKNOWN = "UNKNOWN"
    PROVISIONAL = "PROVISIONAL"
    CONFIRMED = "CONFIRMED"


@dataclass(frozen=True)
class R35MultiFrameEvidenceConfig:
    provisional_presence_threshold: float = 0.80
    confirmation_presence_threshold: float = 0.85
    minimum_candidate_probability: float = 0.75
    minimum_observation_confidence: float = 0.60
    contradiction_unknown_probability: float = 0.70
    contradiction_other_candidate_probability: float = 0.80
    minimum_independent_observations: int = 2


@dataclass(frozen=True)
class AnchorObservation:
    observation_id: str
    candidate_id: str | None
    presence_probability: float
    candidate_probability: float
    confidence: float

    @property
    def unknown_probability(self) -> float:
        return 1.0 - self.presence_probability


@dataclass
class R35AnchorEvidenceTracker:
    cfg: R35MultiFrameEvidenceConfig = field(default_factory=R35MultiFrameEvidenceConfig)
    state: AnchorEvidenceState = AnchorEvidenceState.UNKNOWN
    candidate_id: str | None = None
    independent_observation_ids: set[str] = field(default_factory=set)
    history: list[dict[str, Any]] = field(default_factory=list)

    @property
    def architecture_record(self) -> Dict[str, Any]:
        return {
            "schema_version": "r35_multiframe_goal_anchor_evidence_v1",
            "config": asdict(self.cfg),
            "states": [state.value for state in AnchorEvidenceState],
            "confirmation_rule": "at least two distinct observation_id values supporting the same candidate",
            "contradiction_rule": "strong UNKNOWN or strong different candidate resets state to UNKNOWN",
        }

    def _supports(self, observation: AnchorObservation, *, confirmation: bool) -> bool:
        threshold = (
            self.cfg.confirmation_presence_threshold
            if confirmation
            else self.cfg.provisional_presence_threshold
        )
        return (
            observation.candidate_id is not None
            and observation.presence_probability >= threshold
            and observation.candidate_probability >= self.cfg.minimum_candidate_probability
            and observation.confidence >= self.cfg.minimum_observation_confidence
        )

    def _is_unknown_contradiction(self, observation: AnchorObservation) -> bool:
        return observation.unknown_probability >= self.cfg.contradiction_unknown_probability

    def _is_candidate_contradiction(self, observation: AnchorObservation) -> bool:
        return (
            observation.candidate_id is not None
            and self.candidate_id is not None
            and observation.candidate_id != self.candidate_id
            and observation.presence_probability >= self.cfg.provisional_presence_threshold
            and observation.candidate_probability
            >= self.cfg.contradiction_other_candidate_probability
        )

    def _reset_unknown(self) -> None:
        self.state = AnchorEvidenceState.UNKNOWN
        self.candidate_id = None
        self.independent_observation_ids.clear()

    def update(self, observation: AnchorObservation) -> Dict[str, Any]:
        if not observation.observation_id:
            raise ValueError("observation_id must be non-empty")
        for name, value in (
            ("presence_probability", observation.presence_probability),
            ("candidate_probability", observation.candidate_probability),
            ("confidence", observation.confidence),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0,1]")
        previous_state = self.state
        previous_candidate = self.candidate_id
        transition_reason = "insufficient_evidence"
        duplicate = observation.observation_id in self.independent_observation_ids

        if self.state is AnchorEvidenceState.UNKNOWN:
            if self._supports(observation, confirmation=False):
                self.state = AnchorEvidenceState.PROVISIONAL
                self.candidate_id = observation.candidate_id
                self.independent_observation_ids = {observation.observation_id}
                transition_reason = "first_independent_support"
        elif self._is_unknown_contradiction(observation) or self._is_candidate_contradiction(observation):
            transition_reason = (
                "strong_unknown_contradiction"
                if self._is_unknown_contradiction(observation)
                else "strong_different_candidate_contradiction"
            )
            self._reset_unknown()
        elif (
            observation.candidate_id == self.candidate_id
            and self._supports(observation, confirmation=True)
            and not duplicate
        ):
            self.independent_observation_ids.add(observation.observation_id)
            if len(self.independent_observation_ids) >= self.cfg.minimum_independent_observations:
                self.state = AnchorEvidenceState.CONFIRMED
                transition_reason = "minimum_independent_support_reached"
            else:
                transition_reason = "additional_independent_support"
        elif duplicate:
            transition_reason = "duplicate_observation_ignored"

        record = {
            "observation_id": observation.observation_id,
            "previous_state": previous_state.value,
            "new_state": self.state.value,
            "previous_candidate_id": previous_candidate,
            "candidate_id": self.candidate_id,
            "independent_observation_count": len(self.independent_observation_ids),
            "transition_reason": transition_reason,
        }
        self.history.append(record)
        return self.snapshot(record)

    def snapshot(self, latest_transition: dict[str, Any] | None = None) -> Dict[str, Any]:
        return {
            "state": self.state.value,
            "candidate_id": self.candidate_id,
            "independent_observation_count": len(self.independent_observation_ids),
            "independent_observation_ids": sorted(self.independent_observation_ids),
            "latest_transition": latest_transition,
        }
