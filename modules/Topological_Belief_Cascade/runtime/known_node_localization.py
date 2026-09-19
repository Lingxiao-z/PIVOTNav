"""Geometry-free R36.1 retrieval + BPL + multi-frame known-node localization."""
from __future__ import annotations

from typing import Any, Collection, Mapping, Sequence

import numpy as np

from modules.Topological_Belief_Cascade.runtime.event_semantics import KNOWN_NODE_LOCALIZED, assert_known_node_event


def _normalize(vector: Any) -> np.ndarray:
    array = np.asarray(vector, dtype=np.float32).reshape(-1)
    return array / max(float(np.linalg.norm(array)), 1e-12)


class KnownNodeLocalizer:
    """Known-gallery retrieval with graph constraints and temporal stability."""

    schema = "integration_v6_known_node_localization_v1"

    def __init__(
        self,
        gallery: Mapping[int, Any],
        *,
        bpl_prior: Mapping[int, float] | None = None,
        score_threshold: float = 0.0,
        stable_frames: int = 3,
        localizable_node_ids: Collection[int] | None = None,
        routing_edges: Sequence[tuple[int, int]] | None = None,
        bpl_self_weight: float = 0.70,
        bpl_log_weight: float = 0.005,
        bpl_prior_weight: float = 0.05,
        bpl_visual_margin_guard: float = 0.01,
    ) -> None:
        if not gallery:
            raise ValueError("known-node gallery cannot be empty")
        if int(stable_frames) < 1:
            raise ValueError("stable_frames must be positive")
        self.gallery = {int(node_id): _normalize(value) for node_id, value in gallery.items()}
        self.bpl_prior = {int(node_id): float(value) for node_id, value in (bpl_prior or {}).items()}
        self.score_threshold = float(score_threshold)
        self.stable_frames = int(stable_frames)
        self.localizable_node_ids = frozenset(
            self.gallery if localizable_node_ids is None else (int(value) for value in localizable_node_ids)
        )
        if not self.localizable_node_ids.issubset(self.gallery):
            raise ValueError("localizable nodes must exist in the known-node gallery")
        self.current_node: int | None = None
        self.routing_edges = {(int(source), int(target)) for source, target in (routing_edges or ())}
        self.bpl_self_weight = float(np.clip(bpl_self_weight, 0.0, 1.0))
        self.bpl_log_weight = float(max(0.0, bpl_log_weight))
        self.bpl_prior_weight = float(max(0.0, bpl_prior_weight))
        self.bpl_visual_margin_guard = float(max(0.0, bpl_visual_margin_guard))
        self.persistent_belief: dict[int, float] | None = None
        self._candidate: int | None = None
        self._streak = 0
        self.events: list[dict[str, Any]] = []
        self.bpl_update_count = 0
        self.hop_generation = 0

    def reset_for_new_hop(
        self,
        *,
        current_node: int,
        bpl_prior: Mapping[int, float] | None = None,
        localizable_node_ids: Collection[int] | None = None,
        routing_edges: Sequence[tuple[int, int]] | None = None,
    ) -> None:
        """Clear temporal state before a new known-node subgoal generation."""
        self.current_node = int(current_node)
        self._candidate = None
        self._streak = 0
        self.persistent_belief = None
        if bpl_prior is not None:
            self.bpl_prior = {int(node_id): float(value) for node_id, value in bpl_prior.items()}
        if localizable_node_ids is not None:
            self.localizable_node_ids = frozenset(int(value) for value in localizable_node_ids)
        if routing_edges is not None:
            self.routing_edges = {(int(source), int(target)) for source, target in routing_edges}
        self.hop_generation += 1

    def rank(self, query: Any, *, use_bpl: bool = True) -> list[tuple[int, float]]:
        vector = _normalize(query)
        visual_scores = {node_id: float(np.dot(vector, descriptor)) for node_id, descriptor in self.gallery.items()}
        if not use_bpl:
            ranked = list(visual_scores.items())
            ranked.sort(key=lambda pair: (-pair[1], pair[0]))
            return ranked
        prior = self._transient_bpl_prior()
        ranked = []
        for node_id, score in visual_scores.items():
            # Keep the BPL contribution bounded. It is a routing prior, never a
            # replacement for visual evidence and never a Stop signal.
            score += self.bpl_log_weight * float(np.log(max(prior.get(node_id, 1e-12), 1e-12)))
            # Caller-provided routing hints are deliberately bounded.  They
            # may resolve visually ambiguous candidates, but must not replace
            # a stronger R36.1 observation with the desired route target.
            score += self.bpl_prior_weight * float(self.bpl_prior.get(node_id, 0.0))
            ranked.append((node_id, score))
        ranked.sort(key=lambda pair: (-pair[1], pair[0]))
        return ranked

    def _transient_bpl_prior(self) -> dict[int, float]:
        ids = sorted(self.gallery)
        if self.persistent_belief:
            previous = {node_id: float(self.persistent_belief.get(node_id, 0.0)) for node_id in ids}
        elif self.current_node in self.gallery:
            previous = {node_id: (1.0 if node_id == self.current_node else 0.0) for node_id in ids}
        else:
            previous = {node_id: 1.0 / max(len(ids), 1) for node_id in ids}
        propagated = {
            node_id: self.bpl_self_weight * previous[node_id] for node_id in ids
        }
        for source in ids:
            neighbors = [target for left, target in self.routing_edges if left == source and target in propagated]
            if not neighbors:
                propagated[source] += (1.0 - self.bpl_self_weight) * previous[source]
            else:
                share = (
                    (1.0 - self.bpl_self_weight)
                    * previous[source]
                    / len(neighbors)
                )
                for target in neighbors:
                    propagated[target] += share
        total = sum(propagated.values())
        return {node_id: value / max(total, 1e-12) for node_id, value in propagated.items()}

    def observe(
        self,
        query_descriptor: Any,
        *,
        step: int,
        localization_admitted: bool = True,
    ) -> dict[str, Any]:
        """Process one RGB query without allowing a non-node event to mutate BPL.

        ``localization_admitted`` is an optional caller-side routing admission
        gate.  A return controller may require action-odometry progress along
        an already observed sequential edge before a visually repetitive
        corridor can commit a known-node switch.  The gate is not geometry,
        GT pose, or a Stop authority.  Resetting the streak while closed means
        that the first admissible node event still has independent temporal
        RGB evidence.
        """
        visual_rank = self.rank(query_descriptor, use_bpl=False)
        ranked = self.rank(query_descriptor, use_bpl=True)
        # A known-node return has a bounded routing hypothesis: the current
        # regular node and the Dijkstra next-hop.  Keep full-gallery rankings
        # for audit, but select and stabilize only inside that legal set.
        # Otherwise a visually similar remote node can suppress a valid
        # next-hop even though the robot is not allowed to switch to it.
        route_visual_rank = [
            pair for pair in visual_rank if pair[0] in self.localizable_node_ids
        ]
        route_rank = [
            pair for pair in ranked if pair[0] in self.localizable_node_ids
        ]
        if not route_visual_rank or not route_rank:
            raise AssertionError("known-node route has no localizable gallery candidate")
        visual_margin = (
            float(route_visual_rank[0][1] - route_visual_rank[1][1])
            if len(route_visual_rank) > 1
            else float("inf")
        )
        bpl_override_blocked = bool(
            route_rank[0][0] != route_visual_rank[0][0]
            and visual_margin > self.bpl_visual_margin_guard
        )
        if bpl_override_blocked:
            route_rank = route_visual_rank
        node_id, score = route_rank[0]
        localization_admitted = bool(localization_admitted)
        if not localization_admitted:
            self._candidate, self._streak = None, 0
        elif score < self.score_threshold:
            self._candidate, self._streak = None, 0
        elif self._candidate == node_id:
            self._streak += 1
        else:
            self._candidate, self._streak = node_id, 1
        allowed = node_id in self.localizable_node_ids
        is_new_node = self.current_node is None or self.current_node != node_id
        localized = bool(
            localization_admitted
            and
            allowed
            and is_new_node
            and self._candidate is not None
            and self._streak >= self.stable_frames
        )
        event_type = KNOWN_NODE_LOCALIZED if localized else (
            "KNOWN_NODE_CANDIDATE" if allowed else "KNOWN_NODE_REJECTED_IMPOSSIBLE_JUMP"
        )
        switched = localized
        if switched:
            prior = self._transient_bpl_prior()
            self.current_node = int(node_id)
            self.bpl_update_count += 1
            self.persistent_belief = {key: float(value) for key, value in prior.items()}
            self.persistent_belief[node_id] = max(self.persistent_belief.get(node_id, 0.0), 0.70)
            total = sum(self.persistent_belief.values())
            self.persistent_belief = {key: value / max(total, 1e-12) for key, value in self.persistent_belief.items()}
        event = {
            "event": event_type,
            "event_type": event_type,
            "step": int(step),
            "hop_generation": int(self.hop_generation),
            "node_id": int(node_id),
            "top1_score": float(score),
            "top3": [{"node_id": int(item), "score": float(value)} for item, value in route_rank[:3]],
            "route_filtered_vpr_only_top1": int(route_visual_rank[0][0]),
            "route_filtered_vpr_only_top1_score": float(route_visual_rank[0][1]),
            "route_filtered_vpr_bpl_top1": int(route_rank[0][0]),
            "route_filtered_vpr_bpl_top1_score": float(route_rank[0][1]),
            "raw_gallery_top1": int(visual_rank[0][0]),
            "raw_gallery_top1_score": float(visual_rank[0][1]),
            "raw_gallery_top1_outside_routing_hypothesis": bool(
                visual_rank[0][0] not in self.localizable_node_ids
            ),
            "stability_streak": int(self._streak),
            "stable_frames_required": int(self.stable_frames),
            "candidate_allowed_by_routing_graph": bool(allowed),
            "candidate_is_current_node": not is_new_node,
            "localization_admitted": localization_admitted,
            "node_switched": switched,
            "geometry_called": False,
            "geometry_call_count": 0,
            "GOAL_FINAL_CONFIRMED": 0,
            "stop_authorized": False,
            "ordinary_rgb_frame_bpl_mutation": False,
            "ordinary_rgb_frame_topology_mutation": False,
            "bpl_persistent_mutation": bool(switched),
            "bpl_transient_only": not bool(switched),
            "bpl_posterior_normalized": True,
            "vpr_only_top1": int(visual_rank[0][0]),
            "vpr_only_top1_score": float(visual_rank[0][1]),
            "vpr_only_margin_to_top2": visual_margin,
            "bpl_override_blocked": bpl_override_blocked,
        }
        self.events.append(event)
        assert_known_node_event(event)
        return event

    def audit(self) -> dict[str, Any]:
        localized_count = sum(
            event["event_type"] == KNOWN_NODE_LOCALIZED for event in self.events
        )
        result = {
            "schema": self.schema,
            "geometry_call_count": 0,
            "GOAL_FINAL_CONFIRMED": 0,
            "stop_authorized": False,
            "ordinary_rgb_frame_bpl_mutation_count": 0,
            "bpl_update_count": int(self.bpl_update_count),
            "known_node_localized_count": localized_count,
            "localized_events_match_bpl_updates": localized_count == self.bpl_update_count,
        }
        result["passed"] = (
            result["geometry_call_count"] == 0
            and result["GOAL_FINAL_CONFIRMED"] == 0
            and not result["stop_authorized"]
            and result["ordinary_rgb_frame_bpl_mutation_count"] == 0
            and result["localized_events_match_bpl_updates"]
        )
        return result
