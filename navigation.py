#!/usr/bin/env python3
"""Event-driven PIVOTNav image-goal navigation runner."""
from __future__ import annotations

import argparse
import gzip
import heapq
import json
import math
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent
PPC = PROJECT_ROOT / "modules/Panoramic_Place_Compass"

from modules.Panoramic_Place_Compass.localization import (  # noqa: E402
    EgocentricBearingTracker,
    R361Adapter,
    R363BearingAdapter,
)
from modules.Topological_Belief_Cascade.localization import velocity_action, wrap_degrees  # noqa: E402
from modules.Topological_Belief_Cascade.localization import KnownNodeLocalizer  # noqa: E402
from modules.Navigable_Curiosity_Field.workers import (  # noqa: E402
    DT,
    MAX_V,
    MAX_W,
    ExpandedWorkerClient,
    OmniGuardClient,
    atomic_json,
    save_rgb,
    write_jsonl,
)
from modules.Topological_Belief_Cascade.topology import (  # noqa: E402
    CandidateFrontierCandidate,
    EventDrivenGlobalScheduler,
    fs_sector_to_robot_relative_bearing,
)
from modules.Panoramic_Place_Compass.arrival import GoalImageArrivalVerifier  # noqa: E402
from habitat_gs import env_config, _patch_habitat_opencv_compatibility  # noqa: E402

CHECKPOINT_SHA = "44aa451546691f35659ce1ecc0d616d67d706217ceb5a8f2ed43cba9b132760f"
R361_SHA = "4090261bcef45f70ba771533d1283a3ebe2b5e7d84c400d829cac25e01d79f4a"
SEGMENT_DISTANCE_M = 3.0
SEGMENT_MAX_STEPS = 480
BLOCKED_FRAME_LIMIT = 12
DIFFICULTY_BUDGETS = {
    "easy": 1200,
    "medium": 1500,
    "hard": 2000,
    "hard+": 3000,
    "hard++": 4000,
}
DIFFICULTY_ORDER = {"Easy": 0, "Medium": 1, "Hard": 2, "Hard+": 3, "Hard++": 4}


def build_dataset(tasks: list[dict], path: Path) -> None:
    episodes = []
    for task in tasks:
        with gzip.open(task["runtime_inputs"]["episode_path"], "rt") as stream:
            episode = json.load(stream)["episodes"][0]
        episode["episode_id"] = task["task_id"]
        episodes.append(episode)
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as stream:
        json.dump({"episodes": episodes}, stream)


def graph_path(edges: list[dict[str, Any]], source: int, target: int) -> list[int]:
    adjacency: dict[int, set[int]] = {}
    for edge in edges:
        left, right = int(edge["source"]), int(edge["target"])
        adjacency.setdefault(left, set()).add(right)
        if edge["edge_type"] == "SEQUENTIAL_EXECUTED":
            adjacency.setdefault(right, set()).add(left)
    queue = [(0, int(source))]
    distance = {int(source): 0}
    parent: dict[int, int] = {}
    while queue:
        cost, node = heapq.heappop(queue)
        if node == int(target):
            break
        if cost != distance[node]:
            continue
        for neighbor in sorted(adjacency.get(node, ())):
            if cost + 1 < distance.get(neighbor, 10**9):
                distance[neighbor] = cost + 1
                parent[neighbor] = node
                heapq.heappush(queue, (cost + 1, neighbor))
    if int(target) not in distance:
        return []
    path = [int(target)]
    while path[-1] != int(source):
        path.append(parent[path[-1]])
    return list(reversed(path))


def cheap_rgb_descriptor(rgb: np.ndarray) -> np.ndarray:
    image = Image.fromarray(np.asarray(rgb)[..., :3].astype(np.uint8)).resize((32, 16), Image.Resampling.BILINEAR)
    vector = np.asarray(image, dtype=np.float32).reshape(-1)
    vector -= float(vector.mean())
    return vector / max(float(np.linalg.norm(vector)), 1e-12)


def tensor_scalar(value: Any) -> float:
    if hasattr(value, "detach"):
        return float(value.detach().float().cpu().reshape(-1)[0])
    return float(np.asarray(value).reshape(-1)[0])


def r361_descriptor(adapter: R361Adapter, rgb: np.ndarray) -> np.ndarray:
    """Encode one RGB panorama for a known-node localization query."""
    value = adapter.encode_query(np.asarray(rgb)[..., :3].astype(np.uint8))["global_descriptor"]
    descriptor = value.detach().float().cpu().numpy().reshape(-1)
    return descriptor / max(float(np.linalg.norm(descriptor)), 1e-12)


class VisualBearingState:
    def __init__(self) -> None:
        self.generation = 0
        self.target_node: int | None = None
        self.target_encoding: dict[str, Any] | None = None
        self.last_valid: float | None = None
        self.last_valid_ttl = 0
        self.valid_history: list[float] = []

    def reset(self, target_node: int, target_encoding: dict[str, Any]) -> None:
        self.generation += 1
        self.target_node = int(target_node)
        self.target_encoding = target_encoding
        self.last_valid = None
        self.last_valid_ttl = 0
        self.valid_history.clear()

    @staticmethod
    def circular_mean(values: list[float]) -> float:
        radians = np.radians(np.asarray(values[-5:], dtype=np.float64))
        return wrap_degrees(math.degrees(math.atan2(float(np.sin(radians).mean()), float(np.cos(radians).mean()))))

    def predict(self, adapter: R363BearingAdapter, rgb: np.ndarray) -> dict[str, Any]:
        assert self.target_encoding is not None
        started = time.perf_counter()
        output = adapter.predict_bearing_to_encoding(rgb, self.target_encoding)
        bearing = tensor_scalar(output["bearing_degrees"])
        confidence = tensor_scalar(output["bearing_confidence"])
        # R36.3 does not expose a calibrated validity head. A finite bearing
        # is the runtime usability criterion; confidence remains telemetry.
        valid_probability = 1.0 if math.isfinite(bearing) else 0.0
        valid = bool(math.isfinite(bearing))
        fallback = None
        if valid:
            self.valid_history.append(bearing)
            control = self.circular_mean(self.valid_history)
            self.last_valid = control
            self.last_valid_ttl = 2
        elif self.last_valid is not None and self.last_valid_ttl > 0:
            control = self.last_valid
            self.last_valid_ttl -= 1
            fallback = "SAME_GENERATION_LAST_VALID_TTL"
        else:
            control = None
            fallback = "STOP_AND_SCAN"
        return {"generation": self.generation, "target_node_id": self.target_node,
                "raw_bearing_degrees": bearing, "control_bearing_degrees": control,
                "valid_probability": valid_probability, "confidence": confidence, "valid": valid,
                "fallback": fallback, "last_valid_ttl_remaining": self.last_valid_ttl,
                "latency_ms": (time.perf_counter() - started) * 1000.0}


def run_episode(env: Any, observation: Any, task: dict, phase_root: Path,
                fs_model: ExpandedWorkerClient, omni: OmniGuardClient,
                r361: R361Adapter, r363: R363BearingAdapter, bearing_mode: str,
                goal_runtime: dict[str, Any] | None = None,
                goal_candidate_threshold: float = 0.975) -> dict[str, Any]:
    task_id = task["task_id"]
    run_dir = phase_root / "runs" / task_id
    result_path = run_dir / "result.json"
    if result_path.is_file():
        return json.loads(result_path.read_text())
    run_dir.mkdir(parents=True, exist_ok=True)
    difficulty = str(task["difficulty"]).strip().lower()
    budget = int(task.get("budget_steps") or DIFFICULTY_BUDGETS[difficulty])
    if bearing_mode != "visual":
        raise ValueError("PIVOTNav requires the visual-bearing control mode")
    regular: dict[int, dict[str, Any]] = {}
    candidate_frontiers: dict[str, dict[str, Any]] = {}
    edges: list[dict[str, Any]] = []
    scheduler = EventDrivenGlobalScheduler()
    current_node, next_node, graph_version = 0, 1, 0
    selected_id: str | None = None
    fs_model_start = fs_model.inference_count
    fs_latency_ms = 0.0
    per_step: list[dict[str, Any]] = []
    fs_events: list[dict[str, Any]] = []
    node_events: list[dict[str, Any]] = []
    candidate_frontier_events: list[dict[str, Any]] = []
    known_events: list[dict[str, Any]] = []
    bpl_events: list[dict[str, Any]] = []
    action_history: list[dict[str, Any]] = []
    target_encoding_cache: dict[int, dict[str, Any]] = {}
    localizer: KnownNodeLocalizer | None = None
    route_target: int | None = None
    visual_bearing = VisualBearingState()
    candidate_frontier_tracker = EgocentricBearingTracker()
    segment_id: str | None = None
    segment_aligned = False
    segment_steps = 0
    segment_commanded_distance = 0.0
    segment_blocked_frames = 0
    segment_stasis_frames = 0
    previous_rgb_descriptor: np.ndarray | None = None
    commanded_path_length = 0.0
    executed_path_length = 0.0
    action_count = 0
    active_hold_count = 0
    non_action_event_count = 0
    non_action_events: list[dict[str, Any]] = []
    last_audit_position = np.asarray(env.sim.get_agent_state().position, dtype=np.float64)
    gt_upper_bound_ended = False
    goal_verifier = None
    goal_confirmed = False
    goal_verifier_events: list[dict[str, Any]] = []
    if goal_runtime is not None:
        goal_verifier = GoalImageArrivalVerifier(
            policy=object(), r361=r361,
            goal_rgb=np.asarray(Image.open(task["runtime_inputs"]["goal_erp_path"]).convert("RGB")),
            goal_path=Path(task["runtime_inputs"]["goal_erp_path"]), run_dir=run_dir,
            scene_id=task["scene_id"], verifier_device=goal_runtime["device"],
            shared_dynamic_parallax=goal_runtime["dynamic_parallax"],
            shared_decision_engine=goal_runtime["decision_engine"],
            target_kind="goal",
            vpr_candidate_threshold=float(goal_candidate_threshold),
        )
    collision_proxy_count = 0
    no_displacement_count = 0
    omni_inference_start = omni.inference_count
    selected_change_events: list[dict[str, Any]] = []
    estimated_heading_deg = 0.0
    estimated_heading_by_node: dict[int, float] = {0: 0.0}
    # This position is retained only for post-run success/SPL auditing.  It is
    # never passed to FS, VPR, OmniGuard, the scheduler, or the verifier.
    # Legacy base tasks and the Hard+/Hard++ extension use different
    # names for the offline-only audit block.  Both carry the same goal
    # position; normalize the schema at this boundary without exposing it to
    # the online policy.
    offline_audit = task.get("offline_preflight_gt_audit_only") or task.get("offline_gt_audit")
    if not isinstance(offline_audit, dict) or "goal_position" not in offline_audit:
        raise KeyError("task manifest lacks offline goal position audit")
    audit_goal_position = np.asarray(offline_audit["goal_position"], dtype=np.float64)
    min_goal_distance = float(
        np.linalg.norm((last_audit_position - audit_goal_position)[[0, 2]])
    )

    def add_regular(node_id: int, parent: int | None, step: int, obs: Any,
                    heading_estimate_deg: float) -> None:
        nonlocal graph_version
        path = run_dir / "node_rgb" / f"node_{node_id:04d}.png"
        save_rgb(obs, path)
        rgb = np.asarray(obs["rgb"])[..., :3].astype(np.uint8)
        encoding = r363.encode_node_package_once(node_id, rgb)
        descriptor = encoding["global_descriptor"].detach().float().cpu().numpy().reshape(-1)
        descriptor /= max(float(np.linalg.norm(descriptor)), 1e-12)
        regular[node_id] = {"node_id": node_id, "parent_id": parent, "created_step": step,
                            "rgb_path": str(path), "r361_descriptor": descriptor.tolist(),
                            "online_pose_inputs": [],
                            "heading_estimate_deg": float(heading_estimate_deg)}
        target_encoding_cache[node_id] = encoding
        for sector in range(12):
            candidate_frontier_id = f"g{node_id}s{sector}"
            candidate_frontiers[candidate_frontier_id] = {"candidate_frontier_id": candidate_frontier_id, "parent_node_id": node_id, "sector": sector,
                                "state": "UNTRIED", "fg_score": float("-inf"), "fs_score": float("-inf"),
                                "visit_count": 0, "created_step": step}
            candidate_frontier_events.append({"event_type": "CANDIDATE_FRONTIER_CREATED", "step": step, **candidate_frontiers[candidate_frontier_id]})
        graph_version += 1
        node_events.append({"event_type": "REGULAR_CREATED", "step": step, "graph_version": graph_version,
                            "node_id": node_id, "parent_id": parent, "ordinary_rgb_frame": False})
        belief = {str(key): (1.0 if key == node_id else 0.0) for key in regular}
        bpl_events.append({"event_type": "CANDIDATE_FRONTIER_FINAL_CONFIRMED" if parent is not None else "ROOT_REGULAR_CREATED",
                           "step": step, "node_id": node_id, "belief_after": belief,
                           "belief_sum": sum(belief.values()), "ordinary_rgb_frame": False,
                           "stop_authorized": False})

    def schedule(reason: str, step: int) -> str | None:
        nonlocal fs_latency_ms
        candidates = []
        for node_id, node in sorted(regular.items()):
            fg, fs, latency = fs_model.infer(Path(node["rgb_path"]), Path(task["runtime_inputs"]["goal_erp_path"]))
            fs_latency_ms += latency
            route = graph_path(edges, current_node, node_id)
            return_cost = 0.0 if node_id == current_node else (len(route) - 1 if route else 1e6)
            for sector in range(12):
                candidate_frontier = candidate_frontiers[f"g{node_id}s{sector}"]
                candidate_frontier["fg_score"], candidate_frontier["fs_score"] = float(fg[sector]), float(fs[sector])
                candidates.append(CandidateFrontierCandidate(
                    candidate_frontier["candidate_frontier_id"], node_id, sector, float(fg[sector]), float(fs[sector]),
                    # FG is auxiliary evidence only; OmniGuard owns runtime
                    # safety and FS remains the primary policy value.
                    valid=bool(return_cost < 1e6),
                    lifecycle=candidate_frontier["state"], visit_count=candidate_frontier["visit_count"],
                    topology_return_cost=float(return_cost),
                    turn_cost=abs(fs_sector_to_robot_relative_bearing(sector)),
                ))
        result = scheduler.schedule(reason=reason, step=step, graph_version=graph_version,
                                    current_node=current_node, candidates=candidates)
        fs_events.append({**result.__dict__, "step": step, "all_regular_nodes_rescored": True,
                          "inference_count_this_event": len(regular), "runtime_gt_inputs": [],
                          "ordinary_frame_fs_call": False})
        if result.selected_candidate_frontier_id is not None:
            previous_selected = selected_id
            candidate_frontiers[result.selected_candidate_frontier_id]["state"] = "SELECTED"
            candidate_frontiers[result.selected_candidate_frontier_id]["visit_count"] += 1
            if previous_selected != result.selected_candidate_frontier_id:
                selected_change_events.append({
                    "step": step,
                    "reason": reason,
                    "old_candidate_frontier_id": previous_selected,
                    "new_candidate_frontier_id": result.selected_candidate_frontier_id,
                    "old_sector": candidate_frontiers[previous_selected]["sector"] if previous_selected in candidate_frontiers else None,
                    "new_sector": candidate_frontiers[result.selected_candidate_frontier_id]["sector"],
                })
        return result.selected_candidate_frontier_id

    def start_segment(candidate_frontier_id: str) -> None:
        nonlocal segment_id, segment_aligned, segment_steps, segment_commanded_distance
        nonlocal segment_blocked_frames, segment_stasis_frames
        segment_id = candidate_frontier_id
        segment_aligned = False
        segment_steps = 0
        segment_commanded_distance = 0.0
        segment_blocked_frames = 0
        segment_stasis_frames = 0
        candidate_frontier_tracker.reset_for_new_hop()
        candidate_frontiers[candidate_frontier_id]["state"] = "EXECUTING"

    try:
        omni.reset()
        add_regular(0, None, 0, observation, estimated_heading_deg)
        selected_id = schedule("ROOT_REGULAR_CREATED", 0)
        # ``action_count`` is the only online budget counter.  Evaluations,
        # FS scheduling, topology mutations, and localization events do not
        # consume budget.  A velocity-control command (including a zero-speed
        # rotation) or an explicit HOLD command consumes exactly one action.
        while action_count < budget:
            step = action_count + 1
            scheduler.ordinary_frame(step)
            if goal_verifier is not None:
                verifier_event = goal_verifier.observe(np.asarray(observation["rgb"])[..., :3].astype(np.uint8), step)
                if verifier_event is not None:
                    goal_verifier_events.append(dict(verifier_event))
                    if bool(verifier_event.get("final_confirmed")) or verifier_event.get("event_type") == "GOAL_FINAL_CONFIRMED":
                        goal_confirmed = True
                        per_step.append({"step": step, "event_type": "GOAL_FINAL_CONFIRMED",
                                         "verifier_event": verifier_event, "stop_authorized": True,
                                         "GOAL_FINAL_CONFIRMED": 1, "runtime_gt_inputs": []})
                        break
            if selected_id is None:
                per_step.append({"step": step, "event_type": "NO_VALID_GLOBAL_CANDIDATE_FRONTIER"})
                break
            # Snapshot the target before this control frame. A schedule may run
            # after env.step; that new target belongs to the next frame's log.
            selected_candidate_frontier_before_step = selected_id
            candidate_frontier = candidate_frontiers[selected_candidate_frontier_before_step]
            parent = int(candidate_frontier["parent_node_id"])
            control_bearing: float | None
            control_source: str
            bearing_event: dict[str, Any] | None = None
            if current_node != parent:
                route = graph_path(edges, current_node, parent)
                if len(route) < 2:
                    candidate_frontier["state"] = "BLOCKED"
                    candidate_frontier_events.append({"event_type": "CANDIDATE_FRONTIER_BLOCKED", "step": step,
                                         "candidate_frontier_id": selected_id, "reason": "PARENT_UNREACHABLE"})
                    non_action_event_count += 1
                    non_action_events.append({"step": step, "event_type": "CANDIDATE_FRONTIER_BLOCKED",
                                              "reason": "PARENT_UNREACHABLE", "action_count": action_count})
                    selected_id = schedule("CANDIDATE_FRONTIER_BLOCKED", step)
                    continue
                next_return = int(route[1])
                if route_target != next_return:
                    route_target = next_return
                    gallery = {node_id: np.asarray(node["r361_descriptor"], dtype=np.float32)
                               for node_id, node in regular.items()}
                    prior = {node_id: -1_000_000.0 for node_id in gallery}
                    prior[current_node], prior[next_return] = 0.01, 0.04
                    localizer = KnownNodeLocalizer(gallery, bpl_prior=prior, stable_frames=3,
                                                   localizable_node_ids={current_node, next_return})
                    localizer.current_node = current_node
                    visual_bearing.reset(next_return, target_encoding_cache[next_return])
                rgb = np.asarray(observation["rgb"])[..., :3].astype(np.uint8)
                query = r361_descriptor(r361, rgb)
                assert localizer is not None
                known = localizer.observe(query, step=step)
                known.update({"target_node_id": next_return, "current_node_before": current_node,
                              "geometry_call_count": 0, "GOAL_FINAL_CONFIRMED": 0,
                              "stop_authorized": False, "ordinary_rgb_frame_bpl_mutation": False,
                              "ordinary_rgb_frame_topology_mutation": False,
                              "known_next_hop_switched": False, "bpl_updated_on_node_event": False})
                known_events.append(known)
                if known["event_type"] == "KNOWN_NODE_LOCALIZED" and known["node_id"] == next_return:
                    source_node = current_node
                    current_node = next_return
                    estimated_heading_deg = float(
                        regular[next_return].get("heading_estimate_deg", estimated_heading_deg)
                    )
                    known["known_next_hop_switched"] = True
                    known["known_next_hop_switch_authority"] = "KNOWN_NODE_LOCALIZED"
                    known["bpl_updated_on_node_event"] = True
                    if not any(edge["source"] == source_node and edge["target"] == current_node for edge in edges):
                        edges.append({"source": source_node, "target": current_node,
                                      "edge_type": "HISTORICAL_RETURN_EXECUTED", "confirmed_step": step})
                    bpl_events.append({"event_type": "KNOWN_NODE_LOCALIZED", "step": step,
                                       "node_id": current_node, "belief_after": {str(current_node): 1.0},
                                       "belief_sum": 1.0, "ordinary_rgb_frame": False,
                                       "stop_authorized": False})
                    route_target = None
                    if current_node == parent:
                        selected_id = schedule("HISTORICAL_NODE_FINAL_CONFIRMED", step)
                    non_action_event_count += 1
                    non_action_events.append({"step": step, "event_type": "HISTORICAL_NODE_FINAL_CONFIRMED",
                                              "node_id": current_node, "action_count": action_count})
                    continue
                bearing_event = visual_bearing.predict(r363, rgb)
                control_bearing = bearing_event["control_bearing_degrees"]
                control_source = "R361_VISUAL_BEARING_KNOWN_RETURN"
            else:
                if segment_id != selected_id:
                    start_segment(selected_id)
                # The trained FS package defines sectors relative to the
                # panorama's capture heading. Convert that bearing through the
                # stored parent heading and the current estimated heading.
                parent_heading = float(regular[parent].get("heading_estimate_deg", 0.0))
                sector_relative = fs_sector_to_robot_relative_bearing(int(candidate_frontier["sector"]))
                sector_world = wrap_degrees(parent_heading + sector_relative)
                control_bearing = wrap_degrees(sector_world - estimated_heading_deg)
                control_source = "FS_PARENT_HEADING_RELATIVE_SECTOR_TO_BODY_BEARING"
                align_only = not segment_aligned and abs(float(control_bearing)) > 5.0
                if not align_only:
                    segment_aligned = True
            if current_node != parent:
                align_only = False
            rgb = np.asarray(observation["rgb"])[..., :3].astype(np.uint8)
            control_started = time.perf_counter()
            verifier_active_before_action = bool(goal_verifier is not None and goal_verifier.active)
            verifier_directive = goal_verifier.directive(control_bearing or 0.0) if goal_verifier is not None and goal_verifier.active else None
            if verifier_directive is not None:
                if verifier_directive.requires_traversability:
                    verifier_bearing = float(verifier_directive.relative_target_bearing_degrees or 0.0)
                    response = omni.infer_rgb(rgb, goal_heading_rad=math.radians(verifier_bearing), goal_distance_m=2.0)
                    omni_latency = (time.perf_counter() - control_started) * 1000.0
                    linear = min(float(np.clip(response["linear_velocity_mps"], 0.0, MAX_V)),
                                 float(verifier_directive.linear_velocity_mps))
                    angular = float(np.clip(response["angular_velocity_rps"], -MAX_W, MAX_W))
                    raw = np.asarray(response["raw_distance_m"], dtype=np.float32)
                    raw_summary = {"minimum": float(raw.min()), "p50": float(np.percentile(raw, 50)),
                                   "maximum": float(raw.max())}
                    controller_mode = f"GOAL_VERIFIER_{verifier_directive.mode}_VIA_OMNIGUARD"
                else:
                    response = None
                    linear = float(np.clip(verifier_directive.linear_velocity_mps, 0.0, MAX_V))
                    angular = float(np.clip(verifier_directive.angular_velocity_rps, -MAX_W, MAX_W))
                    omni_latency = 0.0
                    controller_mode = f"GOAL_VERIFIER_{verifier_directive.mode}"
                    raw_summary = None
            elif control_bearing is None:
                response = None
                linear, angular = 0.0, (-0.15 if step % 8 < 4 else 0.15)
                omni_latency = 0.0
                controller_mode = "VISUAL_BEARING_STOP_AND_SCAN"
                raw_summary = None
            else:
                response = omni.infer_rgb(rgb, goal_heading_rad=math.radians(control_bearing), goal_distance_m=2.0)
                omni_latency = (time.perf_counter() - control_started) * 1000.0
                linear = float(np.clip(response["linear_velocity_mps"], 0.0, MAX_V))
                angular = float(np.clip(response["angular_velocity_rps"], -MAX_W, MAX_W))
                controller_mode = str(response.get("mode"))
                raw = np.asarray(response["raw_distance_m"], dtype=np.float32)
                raw_summary = {"minimum": float(raw.min()), "p50": float(np.percentile(raw, 50)),
                               "maximum": float(raw.max())}
            action_kind = "VELOCITY_CONTROL"
            if verifier_directive is not None and str(getattr(verifier_directive, "mode", "")) == "HOLD":
                linear, angular = 0.0, 0.0
                controller_mode = "GOAL_VERIFIER_HOLD"
                action_kind = "HOLD"
                raw_summary = None
            if current_node == parent and align_only and not verifier_active_before_action:
                linear = 0.0
                controller_mode = "ALIGN_PARENT_TO_FS_CANDIDATE_FRONTIER"
            # Sliding is part of the validated Habitat velocity-control bridge:
            # it lets OmniGuard's local policy deflect around an obstacle instead
            # of repeatedly issuing forward commands that cannot move the agent.
            observation = env.step(velocity_action(linear, angular, allow_sliding=True,
                                                    min_linear_velocity_mps=0.0,
                                                    max_linear_velocity_mps=MAX_V,
                                                    max_angular_velocity_rps=MAX_W, time_step_s=DT))
            if verifier_directive is not None:
                goal_verifier.action_executed(verifier_directive, linear_velocity_mps=linear, phase_action_accepted=True)
            action_count += 1
            if action_kind == "HOLD":
                active_hold_count += 1
            candidate_frontier_tracker.apply_executed_angular(angular, DT)
            estimated_heading_deg = wrap_degrees(
                estimated_heading_deg - math.degrees(float(angular) * DT)
            )
            commanded_step_distance = max(0.0, linear) * DT
            commanded_path_length += commanded_step_distance
            action_history.append({"step": action_count, "action_kind": action_kind,
                                   "linear_velocity_mps": linear,
                                   "angular_velocity_rps": angular, "time_step_s": DT,
                                   "official_habitat_action": "velocity_control"})
            post_position = np.asarray(env.sim.get_agent_state().position, dtype=np.float64)
            executed_path_length += float(np.linalg.norm((post_position - last_audit_position)[[0, 2]]))
            last_audit_position = post_position
            min_goal_distance = min(
                min_goal_distance,
                float(np.linalg.norm((post_position - audit_goal_position)[[0, 2]])),
            )
            current_descriptor = cheap_rgb_descriptor(rgb)
            novelty = None if previous_rgb_descriptor is None else max(0.0, 1.0 - float(np.dot(current_descriptor, previous_rgb_descriptor)))
            previous_rgb_descriptor = current_descriptor
            no_displacement_proxy = bool(linear > 0.01 and novelty is not None and novelty < 0.0025)
            if no_displacement_proxy:
                no_displacement_count += 1
                collision_proxy_count += 1
            # A goal-verifier rotation/approach action belongs exclusively to
            # final-goal confirmation.  It must not advance or complete the
            # currently selected FS candidate_frontier segment.
            if (not verifier_active_before_action) and current_node == parent and segment_id == selected_id:
                segment_steps += 1
                forward_ready = bool(segment_aligned and abs(float(control_bearing or 0.0)) <= 15.0)
                segment_blocked_frames = (
                    segment_blocked_frames + 1
                    if forward_ready and linear <= 0.01
                    else max(0, segment_blocked_frames - 1)
                )
                segment_stasis_frames = (
                    segment_stasis_frames + 1
                    if forward_ready and linear > 0.01 and no_displacement_proxy
                    else max(0, segment_stasis_frames - 1)
                )
                if segment_blocked_frames >= BLOCKED_FRAME_LIMIT or segment_stasis_frames >= BLOCKED_FRAME_LIMIT:
                    candidate_frontier["state"] = "BLOCKED"
                    candidate_frontier_events.append({"event_type": "CANDIDATE_FRONTIER_BLOCKED", "step": step, "candidate_frontier_id": selected_id,
                                         "reason": ("OMNIGUARD_NO_FORWARD_COMMAND"
                                                    if segment_blocked_frames >= BLOCKED_FRAME_LIMIT
                                                    else "RGB_STASIS_DURING_FORWARD_COMMAND"),
                                         "segment_steps": segment_steps,
                                         "segment_stasis_frames": segment_stasis_frames})
                    segment_id = None
                    selected_id = schedule("CANDIDATE_FRONTIER_BLOCKED", step)
                else:
                    # The task contract grants the agent the capability to
                    # execute a commanded 3m segment.  RGB stasis is tracked
                    # separately as a blocked-segment signal; it must not
                    # silently redefine the project's commanded-distance
                    # contract or consume a different arrival signal.
                    segment_commanded_distance += max(0.0, linear) * DT
                if segment_id == selected_id and segment_commanded_distance >= SEGMENT_DISTANCE_M:
                    source_node = parent
                    new_node = next_node
                    next_node += 1
                    edges.append({"source": source_node, "target": new_node,
                                  "edge_type": "SEQUENTIAL_EXECUTED", "confirmed_step": step})
                    candidate_frontier["state"] = "CONFIRMED"
                    estimated_heading_by_node[new_node] = float(estimated_heading_deg)
                    add_regular(new_node, source_node, step, observation, estimated_heading_deg)
                    current_node = new_node
                    candidate_frontier_events.append({"event_type": "CANDIDATE_FRONTIER_FINAL_CONFIRMED", "step": step,
                                         "candidate_frontier_id": selected_id, "new_regular_node_id": new_node,
                                         "confirmation_authority": "PROJECT_3M_PARENT_TO_ROBOT_EUCLIDEAN_CAPABILITY",
                                         "commanded_distance_m": segment_commanded_distance,
                                         "geometry_call_count": 0, "GOAL_FINAL_CONFIRMED": 0,
                                         "stop_authorized": False})
                    segment_id = None
                    selected_id = schedule("CANDIDATE_FRONTIER_FINAL_CONFIRMED", step)
                elif segment_steps >= SEGMENT_MAX_STEPS:
                    candidate_frontier["state"] = "EXHAUSTED"
                    candidate_frontier_events.append({"event_type": "CANDIDATE_FRONTIER_EXHAUSTED", "step": step,
                                         "candidate_frontier_id": selected_id, "reason": "LOCAL_SEGMENT_STEP_LIMIT"})
                    segment_id = None
                    selected_id = schedule("CANDIDATE_FRONTIER_EXHAUSTED", step)
            per_step.append({"step": action_count, "selected_candidate_frontier_id": selected_candidate_frontier_before_step,
                             "selected_parent_node_id": parent, "fs_selected_sector": candidate_frontier["sector"],
                             "control_bearing_degrees_right_positive": control_bearing,
                             "control_bearing_source": control_source,
                             "fs_parent_heading_estimate_deg": float(regular[parent].get("heading_estimate_deg", 0.0)),
                             "fs_sector_relative_bearing_deg": (None if current_node != parent else float(sector_relative)),
                             "fs_sector_world_bearing_deg": (None if current_node != parent else float(sector_world)),
                             "segment_aligned_before_action": bool(segment_aligned if current_node == parent else False),
                             "align_only_action": bool(align_only),
                             "bearing_event": bearing_event, "linear_velocity_mps": linear,
                             "angular_velocity_rps": angular, "omniguard_mode": controller_mode,
                             "omniguard_latency_ms": omni_latency, "omnitrav_raw_dist_summary": raw_summary,
                             "fs_selection_changed_after_frame": selected_id != selected_candidate_frontier_before_step,
                             "official_habitat_action": "velocity_control", "action_kind": action_kind,
                             "fs_called_this_frame": False,
                             "ordinary_rgb_frame_bpl_mutation": False,
                             "ordinary_rgb_frame_topology_mutation": False,
                             "rgb_novelty": novelty, "no_displacement_proxy_rgb_only": no_displacement_proxy,
                             "runtime_gt_inputs": []})
        final_position = np.asarray(env.sim.get_agent_state().position, dtype=np.float64)
        final_goal_distance = float(np.linalg.norm((final_position - audit_goal_position)[[0, 2]]))
        physical_success = bool(min_goal_distance <= 1.0)
        reference_path_value = offline_audit.get("reference_shortest_path_m")
        if reference_path_value is None:
            reference_path_value = offline_audit.get("skeleton_geodesic_m")
        if reference_path_value is None:
            raise KeyError("task manifest lacks offline reference shortest path audit")
        reference_path = float(reference_path_value)
        algorithm_success = bool(goal_confirmed and physical_success)
        physical_spl = (reference_path / max(reference_path, executed_path_length)) if algorithm_success else None
        goal_final_confirmed_event_count = sum(
            bool(event.get("final_confirmed"))
            or event.get("event_type") == "GOAL_FINAL_CONFIRMED"
            for event in goal_verifier_events
        )
        goal_candidate_event_count = sum(
            event.get("event") == "v4_goal_arrival_candidate_observation"
            for event in goal_verifier_events
        )
        payload = {"schema": "integration_v6_phase6_result_v2",
                   "classification": "PHASE6_EVENT_DRIVEN_FS_GOAL_NAV",
                   "task_id": task_id, "scene_id": task["scene_id"], "difficulty": task["difficulty"],
                   "bearing_mode": bearing_mode, "physical_success_1m": physical_success,
                   "physical_success": physical_success,
                   "algorithm_success": algorithm_success, "physical_spl": physical_spl,
                   "steps": action_count, "action_budget": budget,
                   "commanded_path_length_m": commanded_path_length,
                   "executed_path_length_m_gt_audit_only": executed_path_length,
                   "final_goal_distance_m_gt_audit_only": final_goal_distance,
                   "min_goal_distance_m_gt_audit_only": min_goal_distance,
                   "run_ended_by_gt_goal_upper_bound_evaluator": False,
                   "GOAL_FINAL_CONFIRMED": int(goal_confirmed), "stop_authorized": bool(goal_confirmed),
                   "goal_verifier_event_count": len(goal_verifier_events),
                   "goal_verifier_non_skip_event_count": sum(
                       event.get("event") != "v4_goal_candidate_observation_skipped_cadence"
                       for event in goal_verifier_events
                   ),
                   "goal_candidate_event_count": goal_candidate_event_count,
                   "goal_final_confirmed_event_count": goal_final_confirmed_event_count,
                   "goal_verifier_payload": goal_verifier.payload() if goal_verifier is not None else None,
                   "goal_candidate_observation_count": (goal_verifier.payload().get("candidate_observation_count", 0) if goal_verifier is not None else 0),
                   "goal_candidate_accept_count": (goal_verifier.payload().get("candidate_accept_count", 0) if goal_verifier is not None else 0),
                   "goal_query_encode_count": (goal_verifier.payload().get("query_encode_count", 0) if goal_verifier is not None else 0),
                   "goal_query_cache_hit_count": (goal_verifier.payload().get("query_cache_hit_count", 0) if goal_verifier is not None else 0),
                   "fs_schedule_event_count": scheduler.fs_call_count,
                   "fs_model_inference_count": fs_model.inference_count - fs_model_start,
                   "ordinary_frame_fs_call_count": scheduler.ordinary_frame_fs_call_count,
                   "env_step_count": len(action_history), "action_count": action_count,
                   "velocity_control_action_count": action_count - active_hold_count,
                   "active_hold_count": active_hold_count,
                   "non_action_event_count": non_action_event_count,
                   "non_action_events": non_action_events,
                   "fs_latency_ms_total": fs_latency_ms,
                   "omniguard_inference_count": omni.inference_count - omni_inference_start,
                   "collision_proxy_count": collision_proxy_count,
                   "official_collision_count": None,
                   "collision_proxy_definition": "RGB-only proxy: commanded linear velocity > 0.01 m/s and cheap RGB novelty < 0.0025; not Habitat collision telemetry",
                   "no_displacement_count": no_displacement_count,
                   "no_displacement_ratio": no_displacement_count / max(len(action_history), 1),
                   "selected_candidate_frontier_change_count": len(selected_change_events),
                   "regular_node_count": len(regular), "candidate_frontier_count": len(candidate_frontiers),
                   "directed_edge_count": len(edges),
                   "candidate_frontier_final_confirmed_count": sum(event["event_type"] == "CANDIDATE_FRONTIER_FINAL_CONFIRMED" for event in candidate_frontier_events),
                   "known_node_query_count": len(known_events),
                   "known_node_localized_count": sum(event["event_type"] == "KNOWN_NODE_LOCALIZED" for event in known_events),
                   "known_node_geometry_call_count": 0, "known_node_GOAL_FINAL_CONFIRMED": 0,
                   "known_node_stop_authorized": False, "ordinary_frame_bpl_mutation_count": 0,
                   "bpl_update_count": len(bpl_events), "checkpoint_sha256": CHECKPOINT_SHA,
                   "r361_checkpoint_sha256": R361_SHA,
                   "runtime_gt_inputs": [],
                   "gt_local_planner_used": False, "navmesh_used_for_control": False,
                   "gt_candidate_frontier_arrival_used": False, "per_frame_disk_rgb_roundtrip": False,
                   "status": ("ONLINE_GOAL_CONFIRMED_PHYSICAL_SUCCESS" if algorithm_success
                              else "ONLINE_GOAL_CONFIRMED_PHYSICAL_AUDIT_FAIL" if goal_confirmed
                              else "BUDGET_OR_FRONTIER_EXHAUSTED")}
        write_jsonl(run_dir / "per_step.jsonl", per_step)
        write_jsonl(run_dir / "fs_event_log.jsonl", fs_events)
        write_jsonl(run_dir / "node_lifecycle.jsonl", node_events)
        write_jsonl(run_dir / "candidate_frontier_lifecycle.jsonl", candidate_frontier_events)
        write_jsonl(run_dir / "known_node_events.jsonl", known_events)
        write_jsonl(run_dir / "bpl_update_log.jsonl", bpl_events)
        write_jsonl(run_dir / "fs_selection_changes.jsonl", selected_change_events)
        write_jsonl(run_dir / "goal_verifier_events.jsonl", goal_verifier_events)
        if goal_verifier is not None:
            goal_verifier.close()
        write_jsonl(run_dir / "action_history.jsonl", action_history)
        atomic_json(run_dir / "routing_graph.json", {"directed_edges": edges, "node_count": len(regular)})
        atomic_json(result_path, payload)
        return payload
    except Exception as error:
        payload = {"schema": "integration_v6_phase6_result_v2", "classification": "PHASE6_TASK_ERROR",
                   "action_budget": int(task.get("budget_steps") or DIFFICULTY_BUDGETS[difficulty]),
                   "action_count": len(action_history),
                   "velocity_control_action_count": sum(
                       item.get("action_kind") == "VELOCITY_CONTROL" for item in action_history
                   ),
                   "active_hold_count": sum(
                       item.get("action_kind") == "HOLD" for item in action_history
                   ),
                   "non_action_event_count": len(non_action_events),
                   "physical_success_1m": None, "algorithm_success": False, "physical_spl": None,
                   "runtime_gt_inputs": [], "task_id": task_id,
                   "scene_id": task["scene_id"], "difficulty": task["difficulty"],
                   "bearing_mode": bearing_mode, "status": "TASK_ERROR",
                   "error": f"{type(error).__name__}: {error}"}
        atomic_json(result_path, payload)
        return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--phase-root", type=Path, required=True)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--bearing-mode", choices=("gt", "visual"), required=True)
    parser.add_argument("--goal-verifier", action="store_true")
    parser.add_argument("--disable-goal-verifier", action="store_true")
    parser.add_argument("--goal-candidate-threshold", type=float, default=0.975)
    parser.add_argument("--task-id")
    parser.add_argument("--max-tasks", type=int)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    tasks = [task for task in manifest["tasks"] if task["scene_id"] == args.scene_id]
    if args.task_id is not None:
        tasks = [task for task in tasks if task["task_id"] == args.task_id]
    tasks.sort(key=lambda task: (DIFFICULTY_ORDER[task["difficulty"]], task["task_id"]))
    if args.max_tasks is not None:
        tasks = tasks[:args.max_tasks]
    dataset = args.phase_root / "worker_datasets" / f"{args.scene_id}.json.gz"
    build_dataset(tasks, dataset)
    # Habitat-Lab imports its visualization map module eagerly. Apply the
    # OpenCV/NumPy compatibility shim before that import, not only inside the
    # later environment-config builder.
    _patch_habitat_opencv_compatibility()
    import habitat
    from habitat.config.read_write import read_write
    habitat_root = os.environ.get("PIVOTNAV_HABITAT_ROOT")
    if not habitat_root:
        raise RuntimeError("PIVOTNAV_HABITAT_ROOT is required")
    config = env_config(dataset, habitat_root=habitat_root, gpu=args.gpu, dt=DT, max_v=MAX_V, max_w=MAX_W)
    with read_write(config):
        config.habitat.dataset.data_path = str(dataset)
        config.habitat.environment.max_episode_steps = max(
            [int(task.get("budget_steps") or DIFFICULTY_BUDGETS[str(task["difficulty"]).lower()])
             for task in tasks] or [max(DIFFICULTY_BUDGETS.values())]
        ) + 1
    # Establish the Gaussian-splat simulator context and render its first
    # frame before creating CUDA inference contexts.  Habitat-GS can otherwise
    # hit a renderer illegal-memory-access when GL and CUDA contexts are born
    # concurrently on the shared host.
    env = habitat.Env(config=config)
    warmup_observation = env.reset()
    warmup_episode_id = str(env.current_episode.episode_id)
    fs_model = ExpandedWorkerClient(f"cuda:{args.gpu}", args.phase_root / "logs" / f"{args.scene_id}.fs.stderr.log")
    omni = OmniGuardClient(args.gpu, args.phase_root / "worker_debug" / args.scene_id,
                           args.phase_root / "logs" / f"{args.scene_id}.omniguard.stderr.log")
    weights_root = Path(os.environ.get("PIVOTNAV_WEIGHTS_ROOT", "")).expanduser().resolve()
    if not weights_root.is_dir():
        raise RuntimeError("PIVOTNAV_WEIGHTS_ROOT must point to the external model directory")
    r361 = R361Adapter(
        PPC, device=f"cuda:{args.gpu}", precision="fp32",
        checkpoint_path=weights_root / "r361/r361_modular.pt",
    )
    r363 = R363BearingAdapter(
        PPC, PROJECT_ROOT / "modules/Panoramic_Place_Compass/model",
        adapter_root=PPC, device=f"cuda:{args.gpu}", r361_adapter=r361,
        checkpoint_path=weights_root / "r363/r363_bearing.pt",
    )
    goal_runtime = None
    if args.goal_verifier and not args.disable_goal_verifier:
        from modules.Panoramic_Place_Compass.geometry import DynamicParallaxExtractor
        from modules.Panoramic_Place_Compass.arrival import V7SequenceDecisionEngine
        protocol_path = Path(os.environ.get(
            "PIVOTNAV_ARRIVAL_PROTOCOL",
            PROJECT_ROOT / "modules/Panoramic_Place_Compass/arrival_protocol.json",
        )).expanduser().resolve()
        protocol = json.loads(protocol_path.read_text())
        model = protocol["model"]
        configured_model = os.environ.get("PIVOTNAV_ARRIVAL_MODEL")
        model_path = Path(configured_model).expanduser().resolve() if configured_model else Path(model["path"]).expanduser()
        if not model_path.is_absolute():
            model_path = (protocol_path.parent / model_path).resolve()
        if not model_path.is_file():
            raise RuntimeError(
                "arrival model is not configured; set PIVOTNAV_ARRIVAL_MODEL "
                f"to a readable joblib file (looked for {model_path})"
            )
        goal_runtime = {
            "device": f"cuda:{args.gpu}",
            "dynamic_parallax": DynamicParallaxExtractor(
                device=f"cuda:{args.gpu}", commanded_forward_distance_per_step_m=0.16
            ),
            "decision_engine": V7SequenceDecisionEngine(
                model_path=model_path, expected_model_sha256=model["sha256"],
                integration_root=PPC, hard_safety_config=protocol["hard_safety"],
            ),
        }
    summaries = []
    reset_count = 1
    try:
        pending = {task["task_id"]: task for task in tasks}
        use_warmup = warmup_episode_id in pending
        while pending:
            if use_warmup:
                observation = warmup_observation
                episode_id = warmup_episode_id
                use_warmup = False
            else:
                observation = env.reset()
                reset_count += 1
                episode_id = str(env.current_episode.episode_id)
            if episode_id not in pending:
                if reset_count > len(tasks) * 3:
                    raise RuntimeError("EPISODE_ITERATOR_DID_NOT_COVER_PENDING")
                continue
            task = pending.pop(episode_id)
            result = run_episode(env, observation, task, args.phase_root, fs_model, omni, r361, r363,
                                 args.bearing_mode, goal_runtime, args.goal_candidate_threshold)
            summaries.append(result)
            print(json.dumps({"task_id": task["task_id"], "status": result["status"]}), flush=True)
    finally:
        env.close()
        omni.close()
        fs_model.close()
        atomic_json(args.phase_root / "worker_audits" / f"{args.scene_id}.json", {
            "scene_id": args.scene_id, "gpu": args.gpu, "pid": os.getpid(),
            "bearing_mode": args.bearing_mode, "task_count": len(tasks), "reset_count": reset_count,
            "habitat_env_init_count": 1, "fs_model_load_count": 1,
            "omniguard_model_load_count": 1, "r361_model_load_count": 1,
            "goal_geometry_model_load_count": 1 if args.goal_verifier else 0,
            "goal_decision_model_load_count": 1 if args.goal_verifier else 0,
            "external_process_actions": [], "exclusive_gpu_required": False,
        })


if __name__ == "__main__":
    main()
