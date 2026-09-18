"""Conditional V3.3.12/V7 RGB arrival adapter for the V4 NTS goal path.

The frozen verifier is deliberately kept behind a small bridge.  It can
observe and produce a final-confirmed event, but it cannot mutate the NTS
graph or authorize Stop except through ``NTSArrivalGateAdapter``.
"""
from __future__ import annotations

import hashlib
import json
import os
import types
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from modules.Panoramic_Place_Compass.production.arrival_gate import NTSArrivalGateAdapter
from modules.Topological_Belief_Cascade.production.event_semantics import GOAL_FINAL_CONFIRMED, GHOST_FINAL_CONFIRMED


ROOT = Path(__file__).resolve().parent
ARRIVAL_PROTOCOL = Path(os.environ.get(
    "PIVOTNAV_ARRIVAL_PROTOCOL",
    str(ROOT / "arrival_protocol.json"),
)).expanduser()


def typed_commit_payload(target_kind: str, stop_authorized: bool) -> dict[str, Any]:
    """Normalize the frozen coordinator result to the V4 typed event contract."""
    if target_kind == "goal":
        assert stop_authorized
        return {
            "transaction": "V4_GOAL_FINAL_CONFIRMED_STOP",
            "event_type": GOAL_FINAL_CONFIRMED,
            "stop_authorized": True,
            "node_switch_count": 0,
            "bpl_update_count": 0,
        }
    if target_kind == "ghost":
        assert not stop_authorized
        return {
            "transaction": "V4_GHOST_FINAL_CONFIRMED_TRANSACTION",
            "event_type": GHOST_FINAL_CONFIRMED,
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

        frozen_root = Path(os.environ.get("PIVOTNAV_ARRIVAL_FROZEN_ROOT", str(ROOT / "arrival_frozen"))).expanduser()
        sys.path[:0] = [str(ROOT), str(frozen_root)]
        from modules.Panoramic_Place_Compass.production.arrival_sequence import V7SequenceDecisionEngine
        from modules.Panoramic_Place_Compass.production.dynamic_parallax import DynamicParallaxExtractor
        from modules.Panoramic_Place_Compass.production.return_coordinator import LiveV7ReturnCoordinator
        from modules.Panoramic_Place_Compass.production.active_evidence import pair_from_encoding

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
        self.gate = NTSArrivalGateAdapter(policy)
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
        # Goal or a ghost-to-regular arrival transaction.
        if target_kind not in {"goal", "ghost"}:
            raise ValueError(f"unsupported arrival target kind: {target_kind}")
        if target_kind == "ghost" and not candidate_frontier_id:
            raise ValueError("ghost arrival requires ghost_id")
        self.arrival_target_kind = str(target_kind)
        self.arrival_ghost_id = str(candidate_frontier_id) if candidate_frontier_id is not None else None
        self.arrival_final_goal = self.arrival_target_kind == "goal"
        self.pair_from_encoding = pair_from_encoding
        self.dynamic_parallax = shared_dynamic_parallax or DynamicParallaxExtractor(
            device=str(verifier_device), commanded_forward_distance_per_step_m=0.16
        )
        self.engine = shared_decision_engine or V7SequenceDecisionEngine(
            model_path=Path(model["path"]), expected_model_sha256=model["sha256"],
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
            arrival_ghost_id=self.arrival_ghost_id,
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
                 arrival_target_kind, arrival_ghost_id, arrival_final_goal):
        from modules.Panoramic_Place_Compass.production.return_coordinator import LiveV7ReturnCoordinator

        class Coordinator(LiveV7ReturnCoordinator):
            def _candidate_trigger(inner, rgb, step):
                if inner.active or step < inner.cooldown_until_step:
                    return None
                inner.candidate_observation_count += 1
                query_encoding = inner._encode_query_once(rgb)
                pair = pair_from_encoding(inner.r361.runtime, query_encoding, inner.target_encoding_goal)
                similarity = float(pair["vpr_similarity"])
                arrival_probability = float(pair["arrival_probability"])
                candidate_node_id = "GOAL" if inner.arrival_target_kind == "goal" else inner.arrival_ghost_id
                row = {
                    "step": int(step), "event": (
                        "v4_goal_arrival_candidate_observation"
                        if inner.arrival_target_kind == "goal"
                        else "v4_ghost_arrival_candidate_observation"
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
                                 candidate_frontier_id=inner.arrival_ghost_id,
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
                               else "v4_ghost_verification_started"
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
        self._impl.arrival_ghost_id = arrival_ghost_id
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
