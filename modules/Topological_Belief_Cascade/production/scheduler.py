"""Global regular-node/ghost FS scheduler for Integration V5."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from modules.Topological_Belief_Cascade.production.protocol import FSSchedule, assert_fs_event

TERMINAL_STATES = frozenset({"BLOCKED", "EXHAUSTED", "VISITED", "CONFIRMED", "REJECTED", "EXECUTING"})


@dataclass
class CandidateFrontierCandidate:
    candidate_frontier_id: str
    parent_node_id: int
    sector: int
    fg_score: float
    fs_score: float
    valid: bool = True
    lifecycle: str = "UNTRIED"
    visit_count: int = 0
    topology_return_cost: float = 0.0
    turn_cost: float = 0.0
    metadata: dict = field(default_factory=dict)

    def audit(self, selected: bool, rank: int | None) -> dict:
        return {
            "ghost_id": self.candidate_frontier_id, "parent_node_id": self.parent_node_id,
            "sector": self.sector, "fg_score": self.fg_score, "fs_score": self.fs_score,
            "valid_mask": self.valid, "candidate_lifecycle": self.lifecycle,
            "visit_count": self.visit_count, "topology_return_cost": self.topology_return_cost,
            "turn_cost": self.turn_cost, "global_rank": rank, "selected": selected,
        }


class EventDrivenGlobalScheduler:
    def __init__(self) -> None:
        self.fs_call_count = 0
        self.ordinary_frame_fs_call_count = 0
        self.last_fs_step: int | None = None
        self.records: list[dict] = []

    def ordinary_frame(self, step: int) -> None:
        # Deliberately no FS call or graph mutation.
        del step

    def schedule(self, *, reason: str, step: int, graph_version: int,
                 current_node: int, candidates: Iterable[CandidateFrontierCandidate]) -> FSSchedule:
        assert_fs_event(reason)
        self.fs_call_count += 1
        rows = list(candidates)
        eligible = [c for c in rows if c.valid and c.lifecycle not in TERMINAL_STATES]
        # Stable global ranking: FS, FG, visits, return cost, turn cost, ids.
        eligible.sort(key=lambda c: (-c.fs_score, -c.fg_score, c.visit_count,
                                     c.topology_return_cost, c.turn_cost,
                                     c.parent_node_id, c.candidate_frontier_id))
        selected = eligible[0] if eligible else None
        rank = {c.candidate_frontier_id: i + 1 for i, c in enumerate(eligible)}
        audited = tuple(c.audit(c is selected, rank.get(c.candidate_frontier_id)) for c in rows)
        schedule = FSSchedule(
            reason=reason, graph_version=int(graph_version), current_node=int(current_node),
            selected_candidate_frontier_id=selected.candidate_frontier_id if selected else None,
            selected_parent_node_id=selected.parent_node_id if selected else None,
            selected_sector=selected.sector if selected else None,
            selected_fg_score=selected.fg_score if selected else None,
            selected_fs_score=selected.fs_score if selected else None,
            candidates=audited,
        )
        self.records.append({
            **schedule.__dict__, "step": int(step),
            "actions_since_previous_fs_call": None if self.last_fs_step is None else int(step) - self.last_fs_step,
            "selected_global_rank": 1 if selected else None,
            "fs_call_on_ordinary_frame": False,
        })
        self.last_fs_step = int(step)
        return schedule


def build_parent_candidates(parent_node_id: int, fg_scores, fs_scores, *, valid_threshold: float = 0.5,
                            lifecycle_by_sector: dict[int, str] | None = None,
                            return_cost: float = 0.0) -> list[CandidateFrontierCandidate]:
    if len(fg_scores) != 12 or len(fs_scores) != 12:
        raise ValueError("Expanded v6 outputs must have 12 sectors")
    # ``valid_threshold`` is retained only for API compatibility. Expanded-v6
    # FG is auxiliary evidence; physical reachability/lifecycle must decide
    # validity, and FS remains the primary ordering key.
    del valid_threshold
    lifecycle_by_sector = lifecycle_by_sector or {}
    return [CandidateFrontierCandidate(
        candidate_frontier_id=f"g{parent_node_id}s{sector}", parent_node_id=int(parent_node_id), sector=sector,
        fg_score=float(fg_scores[sector]), fs_score=float(fs_scores[sector]),
        valid=bool(return_cost < 1e6 and lifecycle_by_sector.get(sector, "UNTRIED") not in TERMINAL_STATES),
        lifecycle=lifecycle_by_sector.get(sector, "UNTRIED"), topology_return_cost=float(return_cost),
    ) for sector in range(12)]
