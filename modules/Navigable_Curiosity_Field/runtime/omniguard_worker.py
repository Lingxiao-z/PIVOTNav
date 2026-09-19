#!/usr/bin/env python3
"""Persistent OmniGuard worker for V2 layer-replacement diagnostics."""

from __future__ import annotations

import argparse
import base64
import io
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml
from PIL import Image


def jsonable(value):
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def reset_controller(controller) -> None:
    controller._prev_ema_distances_m = None
    controller._prev_best_theta_rad = 0.0
    controller._state = "polar_esdf"
    controller._recovery_active = False
    controller._recovery_turn_direction = 1.0
    controller._recovery_steps_since_switch = 0


def deep_update(target: dict, updates: dict) -> dict:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            deep_update(target[key], value)
        else:
            target[key] = value
    return target


def save_debug_npz(path: str | None, *, source: str, geometry, scoring, raw, effective, exist) -> None:
    if not path:
        return
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        source=np.asarray(source),
        raw_distance_m=np.asarray(raw, dtype=np.float32),
        effective_distance_m=np.asarray(effective, dtype=np.float32),
        exist_probability=np.asarray(exist, dtype=np.float32),
        clipped_distance_m=np.asarray(geometry.processed.clipped_distance_m, dtype=np.float32),
        smoothed_distance_m=np.asarray(geometry.processed.smoothed_distance_m, dtype=np.float32),
        observed_angle_mask=np.asarray(geometry.processed.observed_angle_mask, dtype=np.uint8),
        obstacle_hit_mask=np.asarray(geometry.processed.obstacle_hit_mask, dtype=np.uint8),
        occupancy_grid=np.asarray(geometry.occupancy.occupancy_grid, dtype=np.int8),
        occupied_mask=np.asarray(geometry.occupancy.occupied_mask, dtype=np.uint8),
        free_grid_mask=np.asarray(geometry.occupancy.free_grid_mask, dtype=np.uint8),
        esdf_grid_m=np.asarray(geometry.esdf_grid_m, dtype=np.float32),
        candidate_angles_rad=np.asarray(scoring.angles_rad, dtype=np.float32),
        candidate_scores=np.asarray(scoring.scores, dtype=np.float32),
        candidate_min_esdf_m=np.asarray(scoring.min_esdf_m, dtype=np.float32),
        candidate_mean_esdf_m=np.asarray(scoring.mean_esdf_m, dtype=np.float32),
        rollout_points_local_m=np.asarray(scoring.rollout_points_local_m, dtype=np.float32),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--profile", default="current")
    parser.add_argument("--overrides-json", default="{}")
    parser.add_argument("--distance-scale", type=float, default=1.0)
    parser.add_argument("--angle-roll-bins", type=int, default=0)
    parser.add_argument("--mirror-angles", action="store_true")
    args = parser.parse_args()

    repo = Path(args.repo).resolve()
    sys.path.insert(0, str(repo / "deployment/src"))
    from guide import RuleEsdfNavigator
    from omniguard_esdf.controllers.polar_esdf_controller import (
        GridSpec,
        LocalGeometryResult,
        OccupancyResult,
        ProcessedDistances,
    )
    from omniguard_esdf.controllers.radar import analyze_radar

    config_path = repo / "deployment/config/omniguard_algorithm.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))["omniguard_rule"]
    config["traversability_checkpoint"] = str(Path(args.checkpoint).resolve())
    config["output_dir"] = str(Path(args.output_dir).resolve())
    overrides = json.loads(args.overrides_json)
    config.setdefault("esdf", {})
    config.setdefault("navigation", {}).setdefault("polar_esdf", {})
    deep_update(config, overrides)

    started = time.perf_counter()
    navigator = RuleEsdfNavigator(args.device, config)
    controller = navigator.controller
    if torch.cuda.is_available() and str(args.device).startswith("cuda"):
        torch.cuda.synchronize()
    print(json.dumps({
        "ready": True,
        "profile": args.profile,
        "overrides": overrides,
        "distance_scale": args.distance_scale,
        "angle_roll_bins": args.angle_roll_bins,
        "mirror_angles": args.mirror_angles,
        "rule_config": jsonable(controller.rule_config),
        "init_seconds": time.perf_counter() - started,
    }), flush=True)

    def transformed(values):
        array = np.asarray(values, dtype=np.float32).reshape(-1)
        if args.mirror_angles:
            array = array[::-1]
            array = np.roll(array, 1)
        if args.angle_roll_bins:
            array = np.roll(array, int(args.angle_roll_bins))
        return array

    def preserve_origin_free(geometry):
        enabled = bool(config.get("esdf", {}).get("preserve_origin_free_after_geometry", False))
        if not enabled:
            return geometry, False
        origin_index = geometry.occupancy.grid_spec.local_to_index(0.0, 0.0)
        if origin_index is None:
            return geometry, False
        occupancy = geometry.occupancy
        occupancy.occupancy_grid[origin_index] = 0
        occupancy.occupied_mask[origin_index] = False
        occupancy.observed_grid_mask[origin_index] = True
        occupancy.free_grid_mask[origin_index] = True
        esdf = controller.compute_esdf(occupancy)
        corrected = LocalGeometryResult(
            processed=geometry.processed,
            occupancy=occupancy,
            esdf_grid_m=esdf,
            geometry_time_ms=geometry.geometry_time_ms,
        )
        return corrected, True

    def control_distances(*, raw, effective, exist, observed, goal_heading, goal_distance, point_goal, debug_path,
                          force_debug, source):
        raw = transformed(raw)
        effective = transformed(effective) * float(args.distance_scale)
        exist = transformed(exist)
        observed = transformed(observed).astype(bool)
        radar = analyze_radar(
            effective_distance_m=effective,
            raw_distance_m=raw,
            exist_probability=exist,
            config=navigator.runtime_config,
            goal_angle_rad=float(goal_heading),
        )
        geometry = controller.build_local_geometry(radar, observed_angle_mask=observed)
        geometry, origin_forced_free = preserve_origin_free(geometry)
        scoring = controller.score_candidate_directions(
            esdf_grid_m=geometry.esdf_grid_m,
            grid_spec=geometry.occupancy.grid_spec,
            theta_goal_rad=float(goal_heading),
            observed_angle_mask=geometry.processed.observed_angle_mask,
            processed=geometry.processed,
            goal_distance_m=float(goal_distance),
            point_goal_active=bool(point_goal),
        )
        linear, angular = controller.compute_control_from_scoring(
            scoring=scoring,
            goal_distance_m=float(goal_distance),
        )
        best = scoring.best_index
        path_min = None if best is None else float(scoring.min_esdf_m[best])
        path_mean = None if best is None else float(scoring.mean_esdf_m[best])
        if debug_path and (force_debug or path_min is None or path_min <= 1e-9):
            save_debug_npz(
                debug_path,
                source=source,
                geometry=geometry,
                scoring=scoring,
                raw=raw,
                effective=effective,
                exist=exist,
            )
        origin_index = geometry.occupancy.grid_spec.local_to_index(0.0, 0.0)
        origin_occupancy = None if origin_index is None else int(geometry.occupancy.occupancy_grid[origin_index])
        origin_esdf = None if origin_index is None else float(geometry.esdf_grid_m[origin_index])
        finite_scores = np.isfinite(scoring.scores)
        return {
            "linear_velocity_mps": float(linear),
            "angular_velocity_rps": float(angular),
            "mode": str(scoring.selected_mode),
            "rule_reason": str(scoring.selected_reason),
            "goal_heading_rad": float(goal_heading),
            "goal_distance_m": float(goal_distance),
            "chosen_heading_rad": None if best is None else float(scoring.angles_rad[best]),
            "path_min_esdf_m": path_min,
            "path_mean_esdf_m": path_mean,
            "origin_occupancy": origin_occupancy,
            "origin_esdf_m": origin_esdf,
            "origin_forced_free_after_geometry": origin_forced_free,
            "occupied_cell_count": int(np.count_nonzero(geometry.occupancy.occupancy_grid == 1)),
            "free_cell_count": int(np.count_nonzero(geometry.occupancy.occupancy_grid == 0)),
            "unknown_cell_count": int(np.count_nonzero(geometry.occupancy.occupancy_grid == -1)),
            "candidate_count": int(scoring.scores.size),
            "finite_candidate_count": int(np.count_nonzero(finite_scores)),
            "candidate_angles_deg": np.degrees(scoring.angles_rad).astype(np.float32),
            "candidate_scores": scoring.scores,
            "candidate_min_esdf_m": scoring.min_esdf_m,
            "candidate_mean_esdf_m": scoring.mean_esdf_m,
            "raw_distance_m": raw,
            "effective_distance_m": effective,
            "exist_probability": exist,
            "observed_angle_mask": observed,
            "debug_npz_path": debug_path if debug_path and (force_debug or path_min is None or path_min <= 1e-9) else None,
            "source": source,
        }

    for line in sys.stdin:
        try:
            request = json.loads(line)
            command = request.get("command")
            if command == "close":
                print(json.dumps({"closed": True}), flush=True)
                break
            if command == "reset":
                reset_controller(controller)
                print(json.dumps({"reset": True}), flush=True)
                continue
            if command == "control_distances":
                distance = request["distance_360_m"]
                count = len(distance)
                response = control_distances(
                    raw=distance,
                    effective=distance,
                    exist=request.get("exist_probability", [1.0] * count),
                    observed=request.get("observed_angle_mask", [True] * count),
                    goal_heading=request.get("goal_heading_rad", 0.0),
                    goal_distance=request.get("goal_distance_m", 2.0),
                    point_goal=request.get("point_goal_active", True),
                    debug_path=request.get("debug_npz_path"),
                    force_debug=bool(request.get("force_debug_npz", False)),
                    source=request.get("source", "external_360_distance"),
                )
            elif command == "control_occupancy":
                occupancy_grid = np.asarray(request["occupancy_grid"], dtype=np.int8)
                spec = request["grid_spec"]
                grid_spec = GridSpec(
                    x_min_m=float(spec["x_min_m"]),
                    x_max_m=float(spec["x_max_m"]),
                    y_min_m=float(spec["y_min_m"]),
                    y_max_m=float(spec["y_max_m"]),
                    resolution_m=float(spec["resolution_m"]),
                )
                occupied = occupancy_grid == 1
                free = occupancy_grid == 0
                occupancy = OccupancyResult(
                    occupancy_grid=occupancy_grid,
                    occupied_mask=occupied,
                    observed_grid_mask=occupancy_grid != -1,
                    free_grid_mask=free,
                    boundary_points_local_m=np.empty((0, 2), dtype=np.float32),
                    grid_spec=grid_spec,
                )
                esdf = controller.compute_esdf(occupancy)
                observed = np.ones((360,), dtype=bool)
                processed = ProcessedDistances(
                    clipped_distance_m=np.full((360,), 8.0, dtype=np.float32),
                    ema_distance_m=np.full((360,), 8.0, dtype=np.float32),
                    smoothed_distance_m=np.full((360,), 8.0, dtype=np.float32),
                    observed_angle_mask=observed,
                    obstacle_hit_mask=np.zeros((360,), dtype=bool),
                )
                geometry = LocalGeometryResult(processed=processed, occupancy=occupancy, esdf_grid_m=esdf, geometry_time_ms=0.0)
                goal_heading = float(request.get("goal_heading_rad", 0.0))
                goal_distance = float(request.get("goal_distance_m", 2.0))
                scoring = controller.score_candidate_directions(
                    esdf_grid_m=esdf,
                    grid_spec=grid_spec,
                    theta_goal_rad=goal_heading,
                    observed_angle_mask=observed,
                    processed=processed,
                    goal_distance_m=goal_distance,
                    point_goal_active=True,
                )
                linear, angular = controller.compute_control_from_scoring(scoring=scoring, goal_distance_m=goal_distance)
                best = scoring.best_index
                path_min = None if best is None else float(scoring.min_esdf_m[best])
                path_mean = None if best is None else float(scoring.mean_esdf_m[best])
                debug_path = request.get("debug_npz_path")
                if debug_path and (path_min is None or path_min <= 1e-9):
                    save_debug_npz(debug_path, source="gt_local_occupancy", geometry=geometry, scoring=scoring,
                                   raw=processed.clipped_distance_m, effective=processed.clipped_distance_m,
                                   exist=np.ones((360,), dtype=np.float32))
                origin_index = grid_spec.local_to_index(0.0, 0.0)
                response = {
                    "linear_velocity_mps": float(linear),
                    "angular_velocity_rps": float(angular),
                    "mode": str(scoring.selected_mode),
                    "rule_reason": str(scoring.selected_reason),
                    "goal_heading_rad": goal_heading,
                    "goal_distance_m": goal_distance,
                    "chosen_heading_rad": None if best is None else float(scoring.angles_rad[best]),
                    "path_min_esdf_m": path_min,
                    "path_mean_esdf_m": path_mean,
                    "origin_occupancy": None if origin_index is None else int(occupancy_grid[origin_index]),
                    "origin_esdf_m": None if origin_index is None else float(esdf[origin_index]),
                    "occupied_cell_count": int(np.count_nonzero(occupied)),
                    "free_cell_count": int(np.count_nonzero(free)),
                    "unknown_cell_count": int(np.count_nonzero(occupancy_grid == -1)),
                    "candidate_count": int(scoring.scores.size),
                    "finite_candidate_count": int(np.count_nonzero(np.isfinite(scoring.scores))),
                    "candidate_angles_deg": np.degrees(scoring.angles_rad).astype(np.float32),
                    "candidate_scores": scoring.scores,
                    "candidate_min_esdf_m": scoring.min_esdf_m,
                    "candidate_mean_esdf_m": scoring.mean_esdf_m,
                    "debug_npz_path": debug_path if debug_path and (path_min is None or path_min <= 1e-9) else None,
                    "source": "gt_local_occupancy",
                }
            elif command in {"infer_image", "infer_rgb_png_base64"}:
                if command == "infer_rgb_png_base64":
                    image = Image.open(io.BytesIO(base64.b64decode(request["png_base64"]))).convert("RGB")
                else:
                    image = Image.open(request["image_path"]).convert("RGB")
                width, height = image.size
                camera_info = {
                    "fx": float(width), "fy": float(width), "cx": width / 2.0, "cy": height / 2.0,
                    "width": width, "height": height, "frame_id": "habitat_gs_native_erp",
                    "distortion_model": "equirectangular", "d": [], "k": [], "r": [], "p": [],
                }
                navigator.set_runtime_camera_info(camera_info)
                frame_bgr = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
                model_output = navigator.inference.run(frame_bgr)
                response = control_distances(
                    raw=model_output.raw_distance_m,
                    effective=model_output.effective_distance_m,
                    exist=model_output.exist_probability,
                    observed=model_output.observed_angle_mask,
                    goal_heading=request.get("goal_heading_rad", 0.0),
                    goal_distance=request.get("goal_distance_m", 2.0),
                    point_goal=request.get("point_goal_active", True),
                    debug_path=request.get("debug_npz_path"),
                    force_debug=bool(request.get("force_debug_npz", False)),
                    source="omnitrav_image",
                )
            else:
                raise ValueError(f"unknown command: {command}")
            print(json.dumps(jsonable(response), allow_nan=False), flush=True)
        except Exception as exc:
            print(json.dumps({"error": repr(exc)}), flush=True)


if __name__ == "__main__":
    main()
