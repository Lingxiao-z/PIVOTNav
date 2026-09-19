"""Final-confirmed transaction bridge between a frozen verifier and NTS graph."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from modules.Topological_Belief_Cascade.runtime.goal_policy import NTSGoalNavigator
from modules.Topological_Belief_Cascade.runtime.event_semantics import (
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


class NTSArrivalGateAdapter:
    """Apply typed arrival transactions with a single Goal Stop authority.

    ``goal`` is the only target that can authorize Stop. Candidate-frontier and historical
    node transactions mutate topology only after a final confirmation and are
    emitted with distinct event types for downstream audits.
    """

    schema = "integration_v4_nts_arrival_gate_adapter_v1"

    def __init__(self, policy: NTSGoalNavigator) -> None:
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
