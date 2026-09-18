from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from PIL import Image

from modules.Panoramic_Place_Compass.production.arrival_sequence import (
    ArrivalSequenceV7StateMachine,
    ArrivalState,
    FrameAssessment,
    V7SequenceDecisionEngine,
)
from modules.Panoramic_Place_Compass.production.dynamic_parallax import DynamicParallaxExtractor


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
        from modules.Panoramic_Place_Compass.production.active_evidence import pair_from_encoding

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
