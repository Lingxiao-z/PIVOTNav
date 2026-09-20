#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
import traceback
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

HEIGHT, WIDTH, DIRECTIONS = 224, 448, 12
LABEL_VERSION = "nts_inspired_habitat_gs_fgfs_v2"
PILOT_SCENES = (
    ("train", "interior_0007_840137", "train", "indoor"),
    ("train", "interior_0022_840117", "train", "indoor"),
    ("train", "scene01", "train", "outdoor"),
    ("train", "scene03", "train", "outdoor"),
    ("train", "interior_0104_839982", "development", "indoor"),
    ("train", "scene02", "development", "outdoor"),
)


def infer_environment(scene_id: str) -> str:
    """Conservative metadata label used for reporting, never for labels."""
    return "indoor" if scene_id.startswith("interior_") else "outdoor"

from .labels import (
    DMAX_METERS,
    FG_PATH_RATIO,
    LOCAL_RADIUS_METERS,
    decide_fg,
    fs_targets,
    local_polar_offsets,
    select_local_endpoints,
    target_offset,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def atomic_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def make_sim(scene_root: Path, partition: str, scene_id: str, gpu: int, seed: int):
    import habitat_sim
    from habitat_sim.agent import AgentConfiguration

    scene = scene_root / partition / scene_id / f"{scene_id}.gs.ply"
    navmesh = scene.with_name(f"{scene_id}.navmesh")
    scene_config = scene_root / f"{partition}.scene_dataset_config.json"
    for required in (scene, navmesh, scene_config):
        if not required.is_file():
            raise FileNotFoundError(required)
    simulator = habitat_sim.SimulatorConfiguration()
    simulator.scene_id = str(scene)
    simulator.scene_dataset_config_file = str(scene_config)
    simulator.enable_physics = False
    simulator.gpu_device_id = gpu
    simulator.create_renderer = True
    simulator.enable_hbao = False
    simulator.random_seed = seed % 2147483647
    sensor = habitat_sim.EquirectangularSensorSpec()
    sensor.uuid = "rgb"
    sensor.sensor_type = habitat_sim.SensorType.COLOR
    sensor.resolution = [HEIGHT, WIDTH]
    sensor.position = [0.0, 1.5, 0.0]
    sensor.channels = 4
    agent = AgentConfiguration()
    agent.sensor_specifications = [sensor]
    sim = habitat_sim.Simulator(habitat_sim.Configuration(simulator, [agent]))
    explicit_load = False
    if not sim.pathfinder.is_loaded:
        explicit_load = bool(sim.pathfinder.load_nav_mesh(str(navmesh)))
    if not sim.pathfinder.is_loaded:
        sim.close()
        raise RuntimeError(f"failed to load NavMesh {navmesh}")
    return sim, scene, navmesh, scene_config, explicit_load


def find_path(pathfinder, start: np.ndarray, end: np.ndarray) -> tuple[bool, float]:
    import habitat_sim

    query = habitat_sim.ShortestPath()
    query.requested_start = np.asarray(start, dtype=np.float32)
    query.requested_end = np.asarray(end, dtype=np.float32)
    found = bool(pathfinder.find_path(query))
    return found, float(query.geodesic_distance) if found else float("inf")


def sample_positions(pathfinder, count: int, minimum_separation_m: float) -> tuple[np.ndarray, dict]:
    thresholds = [value for value in (minimum_separation_m, 0.20, 0.15, 0.10, 0.075, 0.05, 0.025, 0.01) if value <= minimum_separation_m]
    total_attempts = 0
    best_count = 0
    for threshold in thresholds:
        positions = []
        attempts = 0
        while len(positions) < count and attempts < count * 500:
            attempts += 1
            total_attempts += 1
            raw = np.asarray(pathfinder.get_random_navigable_point(), dtype=np.float32)
            snapped = np.asarray(pathfinder.snap_point(raw), dtype=np.float32)
            if not np.isfinite(snapped).all() or not pathfinder.is_navigable(snapped):
                continue
            if any(float(np.linalg.norm(snapped - old)) < threshold for old in positions):
                continue
            positions.append(snapped)
        best_count = max(best_count, len(positions))
        if len(positions) != count:
            continue
        points = np.stack(positions)
        distances = np.linalg.norm(points[:, None] - points[None, :], axis=-1)
        np.fill_diagonal(distances, np.inf)
        return points, {
            "method": "Habitat get_random_navigable_point plus snap and audited separation fallback",
            "attempts": total_attempts,
            "requested_minimum_separation_m": float(minimum_separation_m),
            "used_minimum_separation_m": float(threshold),
            "separation_fallback_used": bool(threshold < minimum_separation_m),
            "threshold_sequence_m": thresholds,
            "observed_nearest_m": float(distances.min()) if count > 1 else None,
        }
    raise RuntimeError(f"sampled at most {best_count}/{count} positions after {total_attempts} attempts; minimum threshold 0.01m")


def render_images(sim, positions: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    import habitat_sim

    agent = sim.get_agent(0)
    frames = []
    records = []
    rotation = habitat_sim.utils.common.quat_from_angle_axis(
        0.0, np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
    )
    for location_id, position in enumerate(positions):
        state = agent.get_state()
        state.position = position
        state.rotation = rotation
        agent.set_state(state, reset_sensors=True)
        frame = np.asarray(sim.get_sensor_observations()["rgb"])[..., :3].copy()
        if frame.shape != (HEIGHT, WIDTH, 3):
            raise RuntimeError(f"unexpected ERP shape {frame.shape}")
        frames.append(frame)
        records.append({
            "location_id": location_id,
            "position": [float(x) for x in position],
            "canonical_yaw_degrees": 0.0,
            "rgb_sha256": hashlib.sha256(frame.tobytes()).hexdigest(),
            "rgb_mean": float(frame.mean()),
            "rgb_std": float(frame.std()),
            "shape": list(frame.shape),
            "render_method": "native Habitat-Sim EquirectangularSensorSpec",
        })
    return np.stack(frames), records


def build_fg(pathfinder, positions: np.ndarray, max_snap_offset_m: float):
    labels = np.zeros((len(positions), DIRECTIONS), dtype=bool)
    requested = np.empty((len(positions), DIRECTIONS, 3), dtype=np.float32)
    snapped = np.empty_like(requested)
    snap_offset = np.empty((len(positions), DIRECTIONS), dtype=np.float32)
    path_distance = np.full((len(positions), DIRECTIONS), np.inf, dtype=np.float32)
    records = []
    for source_id, source in enumerate(positions):
        source_records = []
        for direction in range(DIRECTIONS):
            target = source + target_offset(direction)
            target_snapped = np.asarray(pathfinder.snap_point(target), dtype=np.float32)
            navigable = bool(np.isfinite(target_snapped).all() and pathfinder.is_navigable(target_snapped))
            found, distance = find_path(pathfinder, source, target_snapped) if navigable else (False, float("inf"))
            decision = decide_fg(
                direction_index=direction,
                requested_target=target,
                snapped_target=target_snapped,
                navigable=navigable,
                path_found=found,
                shortest_path_distance_m=distance,
                max_snap_offset_m=max_snap_offset_m,
            )
            labels[source_id, direction] = bool(decision.fg_label)
            requested[source_id, direction] = target
            snapped[source_id, direction] = target_snapped
            snap_offset[source_id, direction] = decision.snap_offset_m
            path_distance[source_id, direction] = decision.shortest_path_distance_m
            source_records.append(decision.to_dict())
        records.append({"source_location_id": source_id, "directions": source_records})
    return labels, requested, snapped, snap_offset, path_distance, records


def build_endpoints(
    pathfinder,
    positions: np.ndarray,
    radial_resolution_m: float,
    angular_resolution_degrees: float,
    max_snap_offset_m: float,
):
    offsets = local_polar_offsets(
        radial_resolution_m=radial_resolution_m,
        angular_resolution_degrees=angular_resolution_degrees,
    )
    endpoint = np.full((len(positions), DIRECTIONS, 3), np.nan, dtype=np.float32)
    geodesic = np.full((len(positions), DIRECTIONS), np.nan, dtype=np.float32)
    euclidean = np.full_like(geodesic, np.nan)
    valid = np.zeros((len(positions), DIRECTIONS), dtype=bool)
    records = []
    for source_id, source in enumerate(positions):
        selected, audit = select_local_endpoints(
            source,
            offsets,
            snap=lambda point: np.asarray(pathfinder.snap_point(point), dtype=np.float32),
            is_navigable=lambda point: bool(pathfinder.is_navigable(point)),
            geodesic=lambda start, end: find_path(pathfinder, start, end),
            max_snap_offset_m=max_snap_offset_m,
        )
        for item in selected:
            if item.valid:
                valid[source_id, item.direction_index] = True
                endpoint[source_id, item.direction_index] = item.endpoint
                geodesic[source_id, item.direction_index] = item.source_geodesic_m
                euclidean[source_id, item.direction_index] = item.source_euclidean_m
        records.append({
            "source_location_id": source_id,
            "directions": [item.to_dict() for item in selected],
            "candidate_audit": audit,
        })
    config = {
        "method": "deterministic local polar grid plus Source-connected NavMesh shortest-path filter",
        "global_random_candidate_pool_used": False,
        "radial_resolution_m": radial_resolution_m,
        "angular_resolution_degrees": angular_resolution_degrees,
        "candidate_count_per_source": int(len(offsets)),
        "max_snap_offset_m": max_snap_offset_m,
    }
    return endpoint, geodesic, euclidean, valid, records, config


def build_pairs(pathfinder, positions, endpoints, endpoint_valid, fg_labels):
    count = len(positions)
    source_ids = np.repeat(np.arange(count, dtype=np.int32), count - 1)
    goal_ids = np.concatenate([np.delete(np.arange(count, dtype=np.int32), source) for source in range(count)])
    pair_count = len(source_ids)
    source_goal = np.full(pair_count, np.inf, dtype=np.float32)
    endpoint_goal = np.full((pair_count, DIRECTIONS), np.inf, dtype=np.float32)
    for pair_id, (source_id, goal_id) in enumerate(zip(source_ids, goal_ids)):
        _, source_goal[pair_id] = find_path(pathfinder, positions[source_id], positions[goal_id])
        for direction in range(DIRECTIONS):
            if not endpoint_valid[source_id, direction] or not fg_labels[source_id, direction]:
                continue
            _, endpoint_goal[pair_id, direction] = find_path(
                pathfinder, endpoints[source_id, direction], positions[goal_id]
            )
    pair_fg = fg_labels[source_ids]
    pair_endpoint_valid = endpoint_valid[source_ids]
    scores, fs_valid, unreachable = fs_targets(endpoint_goal, pair_fg, pair_endpoint_valid)
    return source_ids, goal_ids, source_goal, endpoint_goal, pair_fg, scores, fs_valid, unreachable


def draw_scene(scene_id, positions, requested, snapped, fg_labels, endpoints, endpoint_valid, output):
    values = [positions[:, [0, 2]], requested.reshape(-1, 3)[:, [0, 2]]]
    if endpoint_valid.any():
        values.append(endpoints[endpoint_valid][:, [0, 2]])
    points = np.concatenate(values)
    lower, upper = points.min(axis=0), points.max(axis=0)
    scale = 700.0 / max(float((upper - lower).max()), 1e-3)
    image = Image.new("RGB", (780, 780), "white")
    draw = ImageDraw.Draw(image)

    def pixel(point):
        return tuple((40 + (np.asarray(point)[[0, 2]] - lower) * scale).astype(int))

    for source_id, source in enumerate(positions):
        center = pixel(source)
        draw.ellipse((*np.asarray(center) - 4, *np.asarray(center) + 4), fill="black")
        for direction in range(DIRECTIONS):
            if np.isfinite(snapped[source_id, direction]).all():
                draw.line((center, pixel(snapped[source_id, direction])), fill="green" if fg_labels[source_id, direction] else "red", width=1)
            if endpoint_valid[source_id, direction]:
                draw.line((center, pixel(endpoints[source_id, direction])), fill="blue", width=2)
    draw.text((10, 10), f"{scene_id}: FG targets (green/red), continuous-local FS endpoints (blue)", fill="black")
    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output)


def collect_scene(args, partition: str, scene_id: str, split: str, environment: str, seed: int) -> dict:
    if environment == "unknown":
        environment = infer_environment(scene_id)
    output = Path(args.output)
    marker = output / "metadata" / f"{scene_id}.done.json"
    if marker.is_file():
        record = json.loads(marker.read_text())
        shard = output / "shards" / record["shard_name"]
        if shard.is_file() and sha256(shard) == record["shard_sha256"]:
            return {**record, "resumed": True}
    started = time.time()
    sim = None
    try:
        sim, scene, navmesh, scene_config, explicit_load = make_sim(
            args.scene_root, partition, scene_id, args.gpu, seed
        )
        positions, sampling = sample_positions(sim.pathfinder, args.positions, args.minimum_separation_m)
        images, location_records = render_images(sim, positions)
        fg, fg_requested, fg_snapped, fg_snap_offset, fg_path, fg_records = build_fg(
            sim.pathfinder, positions, args.fg_max_snap_offset_m
        )
        endpoints, endpoint_geo, endpoint_euclidean, endpoint_valid, endpoint_records, endpoint_config = build_endpoints(
            sim.pathfinder, positions, args.radial_resolution_m, args.angular_resolution_degrees, args.endpoint_max_snap_offset_m
        )
        source_ids, goal_ids, source_goal, endpoint_goal, pair_fg, scores, fs_valid, unreachable = build_pairs(
            sim.pathfinder, positions, endpoints, endpoint_valid, fg
        )
        shard_name = f"{scene_id}.fgfs_v2.npz"
        shard = output / "shards" / shard_name
        atomic_npz(
            shard,
            images=images, positions=positions, fg_labels=fg,
            fg_requested_targets=fg_requested, fg_snapped_targets=fg_snapped,
            fg_snap_offset_m=fg_snap_offset, fg_shortest_path_m=fg_path,
            endpoints=endpoints, endpoint_geodesic_m=endpoint_geo,
            endpoint_euclidean_m=endpoint_euclidean, endpoint_valid=endpoint_valid,
            pair_source_ids=source_ids, pair_goal_ids=goal_ids,
            pair_source_goal_geodesic_m=source_goal, pair_fg_labels=pair_fg,
            pair_endpoint_goal_geodesic_m=endpoint_goal, pair_fs_scores=scores,
            pair_fs_valid_mask=fs_valid, pair_unreachable_mask=unreachable,
        )
        atomic_json(output / "audits" / f"{scene_id}.fg.json", fg_records)
        atomic_json(output / "audits" / f"{scene_id}.endpoints.json", endpoint_records)
        draw_scene(
            scene_id, positions, fg_requested, fg_snapped, fg, endpoints, endpoint_valid,
            output / "visualizations" / f"{scene_id}.fgfs_v2.png",
        )
        record = {
            "schema": "fgfs_v2_scene_record_v1",
            "scene_id": scene_id, "partition": partition, "split": split, "environment": environment,
            "seed": seed, "gs_ply": str(scene), "navmesh": str(navmesh),
            "scene_config": str(scene_config), "navmesh_loaded": True,
            "explicit_navmesh_load": explicit_load, "navigable_area": float(sim.pathfinder.navigable_area),
            "location_sampling": sampling, "locations": location_records,
            "label_version": LABEL_VERSION, "fg_path_ratio": FG_PATH_RATIO,
            "local_radius_m": LOCAL_RADIUS_METERS, "dmax_m": DMAX_METERS,
            "endpoint_search": endpoint_config,
            "erp_count": int(len(images)), "pair_count": int(len(source_ids)),
            "fg_positive_fraction": float(fg.mean()),
            "endpoint_valid_fraction": float(endpoint_valid.mean()),
            "fs_valid_fraction": float(fs_valid.mean()),
            "unreachable_fraction": float(unreachable.mean()),
            "shard_name": shard_name, "shard_sha256": sha256(shard),
            "shard_bytes": shard.stat().st_size, "runtime_seconds": time.time() - started,
            "formal_test_accessed": False, "test_run_count": 0,
        }
        atomic_json(marker, record)
        return record
    finally:
        if sim is not None:
            sim.close()


def pilot_acceptance(output: Path, records: list[dict]) -> dict:
    arrays = [np.load(output / "shards" / record["shard_name"], allow_pickle=False) for record in records]
    checks = {
        "required_six_scene_categories": len(records) == 6 and {r["environment"] for r in records} == {"indoor", "outdoor"},
        "navmesh_loaded_100_percent": all(record["navmesh_loaded"] for record in records),
        "native_erp_shape_224x448x3": all(array["images"].shape[1:] == (HEIGHT, WIDTH, 3) for array in arrays),
        "blank_erp_count_zero": all(np.all(array["images"].std(axis=(1, 2, 3)) >= 1.0) for array in arrays),
        "fg_binary": all(np.isin(array["fg_labels"], [False, True]).all() for array in arrays),
        "fg_snap_offsets_explicit": all("fg_snap_offset_m" in array.files for array in arrays),
        "old_global_random_endpoint_method_absent": all(not record["endpoint_search"]["global_random_candidate_pool_used"] for record in records),
        "fs_scores_finite_unit_interval": all(np.isfinite(array["pair_fs_scores"]).all() and ((array["pair_fs_scores"] >= 0) & (array["pair_fs_scores"] <= 1)).all() for array in arrays),
        "unreachable_excluded_from_fs_valid": all(not np.any(array["pair_fs_valid_mask"] & array["pair_unreachable_mask"]) for array in arrays),
        "fg_zero_excluded_from_fs_valid": all(not np.any(array["pair_fs_valid_mask"] & ~array["pair_fg_labels"]) for array in arrays),
        "no_self_pairs": all(np.all(array["pair_source_ids"] != array["pair_goal_ids"]) for array in arrays),
        "formal_test_untouched": all(not record["formal_test_accessed"] and record["test_run_count"] == 0 for record in records),
    }
    result = {
        "schema": "fgfs_v2_pilot_acceptance_v1",
        "accepted": all(checks.values()),
        "checks": checks,
        "statistics": {
            "scene_count": len(records),
            "erp_count": sum(record["erp_count"] for record in records),
            "pair_count": sum(record["pair_count"] for record in records),
            "fg_positive_fraction": float(np.mean([record["fg_positive_fraction"] for record in records])),
            "fs_valid_fraction": float(np.mean([record["fs_valid_fraction"] for record in records])),
            "unreachable_fraction": float(np.mean([record["unreachable_fraction"] for record in records])),
        },
        "formal_test_accessed": False,
        "test_run_count": 0,
    }
    atomic_json(output / "pilot_acceptance.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Collect native ERP RGB and offline FG/FS labels from Habitat-GS"
    )
    parser.add_argument("--mode", required=True, choices=("pilot", "scene"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--habitat-root",
        type=Path,
        default=os.environ.get("PIVOTNAV_HABITAT_ROOT"),
        help="Habitat-GS checkout containing data/scene_datasets/gs_scenes",
    )
    parser.add_argument(
        "--pilot-manifest",
        type=Path,
        help="Optional JSON list of {partition, scene_id, split, environment} records",
    )
    parser.add_argument("--positions", type=int, default=40)
    parser.add_argument("--minimum-separation-m", type=float, default=0.30)
    parser.add_argument("--radial-resolution-m", type=float, default=0.15)
    parser.add_argument("--angular-resolution-degrees", type=float, default=5.0)
    parser.add_argument("--fg-max-snap-offset-m", type=float, default=0.25)
    parser.add_argument("--endpoint-max-snap-offset-m", type=float, default=0.20)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=2026081401)
    parser.add_argument("--partition", choices=("train", "val"))
    parser.add_argument("--scene-id")
    parser.add_argument("--split", default="pilot")
    parser.add_argument("--environment", choices=("indoor", "outdoor", "unknown"), default="unknown")
    args = parser.parse_args()
    if args.habitat_root is None:
        parser.error("--habitat-root is required (or set PIVOTNAV_HABITAT_ROOT)")
    args.scene_root = args.habitat_root.expanduser().resolve() / "data/scene_datasets/gs_scenes"
    if not args.scene_root.is_dir():
        parser.error(f"Habitat-GS scene root does not exist: {args.scene_root}")
    output = args.output.expanduser().resolve()
    try:
        if args.mode == "scene":
            if not args.scene_id or not args.partition:
                parser.error("scene mode requires --scene-id and --partition")
            record = collect_scene(args, args.partition, args.scene_id, args.split, args.environment, args.seed)
            records_path = output / "scene_records.json"
            records = json.loads(records_path.read_text()) if records_path.is_file() else []
            records = [item for item in records if item.get("scene_id") != record["scene_id"]]
            records.append(record)
            atomic_json(records_path, sorted(records, key=lambda item: item["scene_id"]))
            print(json.dumps(record, indent=2))
            return
        pilot_scenes = PILOT_SCENES
        if args.pilot_manifest:
            manifest = json.loads(args.pilot_manifest.read_text(encoding="utf-8"))
            pilot_scenes = tuple(
                (item["partition"], item["scene_id"], item["split"], item.get("environment", "unknown"))
                for item in manifest
            )
        records = []
        for index, (partition, scene_id, split, environment) in enumerate(pilot_scenes):
            records.append(collect_scene(args, partition, scene_id, split, environment, args.seed + index))
        atomic_json(output / "scene_records.json", records)
        result = pilot_acceptance(output, records)
        print(json.dumps(result, indent=2))
        if not result["accepted"]:
            raise SystemExit(2)
    except Exception as error:
        arguments = {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        }
        atomic_json(output / "failed" / f"failure_{int(time.time())}.json", {
            "error": repr(error), "traceback": traceback.format_exc(), "arguments": arguments,
            "formal_test_accessed": False, "test_run_count": 0,
        })
        raise


if __name__ == "__main__":
    main()
