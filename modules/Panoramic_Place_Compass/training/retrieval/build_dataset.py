#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import random
import sys
import tarfile
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(os.environ.get("PIVOTNAV_HABITAT_ROOT", "")).expanduser().resolve()
ROOT = Path(os.environ.get("PIVOTNAV_PPC_TRAINING_ROOT", Path.cwd())).expanduser().resolve()
sys.path.insert(0, str(ROOT / "src"))

from modules.Panoramic_Place_Compass.training.spatial_protocol import (
    PROTOCOL_VERSION,
    SpatialProtocol,
    build_spatial_label,
)
SCENE_DATASET_ROOT = PROJECT_ROOT / "data/scene_datasets/gs_scenes"
SEED = 20260730 + 220
SENSOR_HW = (224, 448)
JPEG_QUALITY = 90
NEAR_DISTS = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
NEAR_PATHS_PER_PLACE = 4
SYNTHETIC_ROLL_YAWS = [0, 15, 30, 45, 60, 90, 135, 180, 225, 270, 315]
MIN_BASE_SEP_M = 2.0
RELIABLE_NEGATIVE_M = 3.0
MAX_SAMPLES_PER_SHARD = 1024
MAX_PLACES_PER_SHARD = 512


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha_path(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_key(text: str) -> str:
    return hashlib.sha256(f"{SEED}:{text}".encode()).hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def write_tar_entry(tar: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name=name)
    info.size = len(data)
    info.mtime = 0
    tar.addfile(info, io.BytesIO(data))


def build_sample_index(sample_shards: Sequence[Dict[str, Any]], split_tag: str) -> Dict[str, Any]:
    rows: Dict[str, Dict[str, Any]] = {}
    for shard_record in sample_shards:
        shard = Path(shard_record["path"])
        pending: Dict[str, Dict[str, str]] = {}
        with tarfile.open(shard, "r") as tar:
            for member in tar:
                if not member.isfile():
                    continue
                if member.name.endswith(".jpg"):
                    key = member.name[:-4]
                    pending.setdefault(key, {})["jpg_member"] = member.name
                elif member.name.endswith(".json"):
                    key = member.name[:-5]
                    f = tar.extractfile(member)
                    if f is None:
                        continue
                    meta = json.loads(f.read().decode("utf-8"))
                    pending.setdefault(key, {})["sample_id"] = meta["sample_id"]
                group = pending.get(key, {})
                if "sample_id" in group and "jpg_member" in group:
                    rows[group["sample_id"]] = {
                        "shard": str(shard),
                        "jpg_member": group["jpg_member"],
                    }
                    pending.pop(key, None)
    return {
        "schema_version": "3.0.0",
        "created_at": now(),
        "split_tag": split_tag,
        "sample_count": len(rows),
        "sample_shard_count": len(sample_shards),
        "rule_zh": "预生成 sample_id -> shard/member 索引；dataloader worker 不再每个 epoch 扫描全部 tar 元数据。",
        "samples": rows,
    }


class ShardWriter:
    def __init__(self, out_dir: Path, split_tag: str, kind: str, max_items: int):
        self.out_dir = out_dir
        self.split_tag = split_tag
        self.kind = kind
        self.max_items = max_items
        self.index = 0
        self.current = 0
        self.total = 0
        self.tar: tarfile.TarFile | None = None
        self.path: Path | None = None
        self.records: List[Dict[str, Any]] = []
        out_dir.mkdir(parents=True, exist_ok=True)

    def _open(self) -> None:
        if self.tar is not None and self.current < self.max_items:
            return
        if self.tar is not None:
            self.close_current()
        self.path = self.out_dir / f"{self.split_tag}-{self.kind}-{self.index:06d}.tar"
        self.tar = tarfile.open(self.path, "w")
        self.records.append({"split_tag": self.split_tag, "kind": self.kind, "path": str(self.path), "item_count": 0, "sha256": None})
        self.index += 1
        self.current = 0

    def close_current(self) -> None:
        if self.tar is None:
            return
        self.tar.close()
        assert self.path is not None
        self.records[-1]["sha256"] = sha_path(self.path)
        self.tar = None

    def add_sample(self, sample_id: str, jpg: bytes, meta: Dict[str, Any]) -> None:
        self._open()
        assert self.tar is not None
        write_tar_entry(self.tar, f"{sample_id}.jpg", jpg)
        write_tar_entry(self.tar, f"{sample_id}.json", json.dumps(meta, ensure_ascii=False, sort_keys=True).encode("utf-8"))
        self.current += 1
        self.total += 1
        self.records[-1]["item_count"] += 1

    def add_place(self, place_id: str, place: Dict[str, Any]) -> None:
        self._open()
        assert self.tar is not None
        safe = place_id.replace("/", "_")
        write_tar_entry(self.tar, f"{safe}.json", json.dumps(place, ensure_ascii=False, sort_keys=True).encode("utf-8"))
        self.current += 1
        self.total += 1
        self.records[-1]["item_count"] += 1

    def close(self) -> List[Dict[str, Any]]:
        self.close_current()
        return self.records


def scene_partition(scene_id: str) -> str:
    for partition in ("train", "val"):
        if (SCENE_DATASET_ROOT / partition / scene_id).is_dir():
            return partition
    raise FileNotFoundError(f"scene is absent from train/val: {scene_id}")


def scene_asset(scene_id: str, suffix: str) -> Path:
    partition = scene_partition(scene_id)
    return SCENE_DATASET_ROOT / partition / scene_id / f"{scene_id}.{suffix}"


def make_sim(scene_id: str, gpu: int, seed: int):
    import habitat_sim
    from habitat_sim.agent import ActionSpec, ActuationSpec, AgentConfiguration

    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = str(scene_asset(scene_id, "gs.ply"))
    partition = scene_partition(scene_id)
    sim_cfg.scene_dataset_config_file = str(SCENE_DATASET_ROOT / f"{partition}.scene_dataset_config.json")
    sim_cfg.enable_physics = False
    sim_cfg.gpu_device_id = int(gpu)
    sim_cfg.create_renderer = True
    sim_cfg.enable_hbao = False
    # habitat_sim binding expects a signed C int; keep large experiment seeds deterministic.
    sim_cfg.random_seed = int(seed) % 2147483647
    sensor = habitat_sim.EquirectangularSensorSpec()
    sensor.uuid = "rgb"
    sensor.sensor_type = habitat_sim.SensorType.COLOR
    sensor.resolution = list(SENSOR_HW)
    sensor.position = [0.0, 1.5, 0.0]
    sensor.channels = 4
    agent_cfg = AgentConfiguration()
    agent_cfg.sensor_specifications = [sensor]
    agent_cfg.action_space = {
        "move_forward": ActionSpec("move_forward", ActuationSpec(amount=0.25)),
        "turn_left": ActionSpec("turn_left", ActuationSpec(amount=30.0)),
        "turn_right": ActionSpec("turn_right", ActuationSpec(amount=30.0)),
    }
    sim = habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg]))
    navmesh = scene_asset(scene_id, "navmesh")
    if not sim.pathfinder.is_loaded and navmesh.is_file():
        sim.pathfinder.load_nav_mesh(str(navmesh))
    return sim


def quat_from_render_yaw(yaw: float):
    # Habitat quaternion yaw uses the opposite sign from V2 render/action labels.
    import habitat_sim

    quat_yaw = -float(yaw)
    return habitat_sim.utils.common.quat_from_angle_axis(math.radians(quat_yaw), np.asarray([0.0, 1.0, 0.0], dtype=np.float32))


def set_agent(agent, position: Sequence[float], render_yaw: float) -> None:
    state = agent.get_state()
    state.position = np.asarray(position, dtype=np.float32)
    state.rotation = quat_from_render_yaw(render_yaw)
    agent.set_state(state, reset_sensors=True)


def geodesic(pathfinder, a: Sequence[float], b: Sequence[float]) -> float:
    import habitat_sim

    path = habitat_sim.ShortestPath()
    path.requested_start = np.asarray(a, dtype=np.float32)
    path.requested_end = np.asarray(b, dtype=np.float32)
    if not pathfinder.find_path(path):
        return float("inf")
    return float(path.geodesic_distance)


def point_along_path(pathfinder, start: Sequence[float], end: Sequence[float], dist: float) -> List[float] | None:
    import habitat_sim

    path = habitat_sim.ShortestPath()
    path.requested_start = np.asarray(start, dtype=np.float32)
    path.requested_end = np.asarray(end, dtype=np.float32)
    if not pathfinder.find_path(path) or len(path.points) < 2:
        return None
    remaining = float(dist)
    pts = [np.asarray(p, dtype=np.float32) for p in path.points]
    for a, b in zip(pts, pts[1:]):
        length = float(np.linalg.norm(b - a))
        if length >= remaining and length > 1e-6:
            raw = a + (remaining / length) * (b - a)
            snapped = np.asarray(pathfinder.snap_point(raw), dtype=np.float32)
            if np.isfinite(snapped).all():
                return [float(x) for x in snapped]
            return [float(x) for x in raw]
        remaining -= length
    return None


def directional_targets(pathfinder, base: Sequence[float], candidates: Sequence[Sequence[float]], count: int) -> List[Sequence[float]]:
    eligible = []
    base_array = np.asarray(base, dtype=np.float32)
    for candidate in candidates:
        candidate_array = np.asarray(candidate, dtype=np.float32)
        path_distance = geodesic(pathfinder, base, candidate)
        if not math.isfinite(path_distance) or path_distance < max(NEAR_DISTS) + 0.25:
            continue
        delta = candidate_array - base_array
        bearing = math.atan2(float(delta[2]), float(delta[0]))
        eligible.append((candidate, bearing, path_distance))
    if not eligible:
        return []
    selected = [max(eligible, key=lambda item: item[2])]
    while len(selected) < min(int(count), len(eligible)):
        used = {id(item) for item in selected}
        def separation(item):
            if id(item) in used:
                return -1.0
            return min(abs(math.atan2(math.sin(item[1] - old[1]), math.cos(item[1] - old[1]))) for old in selected)
        choice = max(eligible, key=separation)
        if choice in selected:
            break
        selected.append(choice)
    return [item[0] for item in selected]


def observe_jpeg(sim) -> Tuple[bytes, Dict[str, Any]]:
    frame = np.asarray(sim.get_sensor_observations()["rgb"])[..., :3]
    img = Image.fromarray(frame.astype(np.uint8), mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=JPEG_QUALITY, optimize=True)
    data = buf.getvalue()
    return data, {
        "shape": list(frame.shape),
        "rgb_mean": float(frame.mean()),
        "rgb_std": float(frame.std()),
        "nonblank": bool(frame.std() >= 1.0 and frame.max() > frame.min()),
        "jpeg_sha256": sha_bytes(data),
        "jpeg_bytes": len(data),
        "sensor_hw": list(SENSOR_HW),
        "jpeg_quality": int(JPEG_QUALITY),
    }


def sample_positions(pathfinder, scene_id: str, target_bases: int, absent_count: int, rng: random.Random) -> Tuple[List[List[float]], List[List[float]], Dict[str, Any]]:
    candidates: List[List[float]] = []
    attempts = 0
    target_candidates = max(target_bases * 8, target_bases + absent_count + 50)
    while len(candidates) < target_candidates and attempts < target_candidates * 40:
        attempts += 1
        p = np.asarray(pathfinder.get_random_navigable_point(), dtype=np.float32)
        if not np.isfinite(p).all():
            continue
        point = [float(x) for x in p]
        if any(float(np.linalg.norm(p - np.asarray(old))) < 0.35 for old in candidates):
            continue
        candidates.append(point)
    ranked = sorted(candidates, key=lambda p: stable_key(f"{scene_id}:{json.dumps(p)}"))
    bases: List[List[float]] = []
    for p in ranked:
        if len(bases) >= target_bases:
            break
        if all(geodesic(pathfinder, p, old) >= MIN_BASE_SEP_M for old in bases):
            bases.append(p)
    absents: List[List[float]] = []
    for p in ranked:
        if len(absents) >= absent_count:
            break
        if any(np.allclose(p, b) for b in bases):
            continue
        # R3 UNKNOWN means outside the complete gallery, not merely a prefix.
        if bases and all(geodesic(pathfinder, p, b) >= RELIABLE_NEGATIVE_M for b in bases):
            absents.append(p)
    all_pts = bases + absents
    dists = [geodesic(pathfinder, a, b) for i, a in enumerate(all_pts) for b in all_pts[i + 1 :]]
    audit = {
        "scene_id": scene_id,
        "candidate_count": len(candidates),
        "attempts": attempts,
        "target_base_count": target_bases,
        "base_count": len(bases),
        "absent_count": len(absents),
        "base_shortfall": max(0, target_bases - len(bases)),
        "pairwise_min_geodesic_m": float(np.min(dists)) if dists else None,
        "pairwise_median_geodesic_m": float(np.median(dists)) if dists else None,
    }
    return bases, absents, audit


def label_for_distance(dist: float) -> str:
    if dist <= 0.25:
        return "strict_owner_0_0.25m"
    if dist <= 1.0:
        return "spatial_multi_positive_0.25_1m"
    if dist <= 2.0:
        return "adjacent_boundary_1_2m"
    return "unknown_distance"


def render_scene(split_tag: str, scene_id: str, scene_idx: int, gpu: int, bases_per_scene: int, absent_per_scene: int, near_paths_per_place: int, sample_writer: ShardWriter, place_writer: ShardWriter) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    sim = make_sim(scene_id, gpu, SEED + scene_idx)
    samples: List[Dict[str, Any]] = []
    places: List[Dict[str, Any]] = []
    try:
        sim.reset()
        agent = sim.get_agent(0)
        bases, absents, audit = sample_positions(sim.pathfinder, scene_id, bases_per_scene, absent_per_scene, random.Random(SEED + scene_idx))
        protocol = SpatialProtocol(node_separation_m=MIN_BASE_SEP_M)
        protocol.validate()
        gallery_positions = {f"{scene_id}__v3place{i:04d}": base for i, base in enumerate(bases)}
        for i, base in enumerate(bases):
            place_id = f"{scene_id}__v3place{i:04d}"
            views = []
            specs = [("anchor_0m", base, 0.0, 0.0)]
            targets = directional_targets(sim.pathfinder, base, [value for index, value in enumerate(bases) if index != i], near_paths_per_place)
            for direction_index, target in enumerate(targets):
                for dist in NEAR_DISTS:
                    near = point_along_path(sim.pathfinder, base, target, dist)
                    if near is not None and all(float(np.linalg.norm(np.asarray(near) - np.asarray(old[1]))) >= 0.1 for old in specs):
                        actual_distance = geodesic(sim.pathfinder, base, near)
                        specs.append((f"near_dir{direction_index:02d}_{dist:.2f}m", near, actual_distance, 0.0))
            for view_idx, (variant, pos, dist, yaw) in enumerate(specs):
                set_agent(agent, pos, yaw)
                jpg, stats = observe_jpeg(sim)
                sample_id = f"{scene_id}__v3place{i:04d}__{variant}__view{view_idx:02d}"
                spatial_label = build_spatial_label(
                    pos, place_id, gallery_positions,
                    lambda a, b: geodesic(sim.pathfinder, a, b), protocol,
                )
                meta = {
                    "sample_id": sample_id,
                    "split_tag": split_tag,
                    "scene_id": scene_id,
                    "place_id": place_id,
                    "view_id": variant,
                    "role": "place_view",
                    "variant": variant,
                    "position": [float(x) for x in pos],
                    "yaw_degrees": float(yaw),
                    "base_distance_m": float(dist),
                    "label_grade": label_for_distance(float(dist)),
                    "precise_yaw_label": False,
                    "rgb_only_model_input": True,
                    "synthetic_roll_source": False,
                    **spatial_label,
                    **stats,
                }
                sample_writer.add_sample(sample_id, jpg, meta)
                samples.append(meta)
                views.append({
                    "sample_id": sample_id,
                    "view_id": variant,
                    "base_distance_m": float(dist),
                    "label_grade": meta["label_grade"],
                    "yaw_degrees": float(yaw),
                    "precise_yaw_label": False,
                    "synthetic_roll_yaws_available": SYNTHETIC_ROLL_YAWS if dist == 0.0 else [],
                    **spatial_label,
                })
            place = {
                "place_id": place_id,
                "split_tag": split_tag,
                "scene_id": scene_id,
                "base_position": [float(x) for x in base],
                "role": "gallery_place",
                "views": views,
                "synthetic_roll_yaws_available": SYNTHETIC_ROLL_YAWS,
                "spatial_protocol_version": PROTOCOL_VERSION,
                "positive_rule_zh": "0-0.25m 且唯一覆盖时为严格 owner-ID；1m 内所有 gallery 节点共同构成空间多正样本集合；1-2m 仅作边界诊断。",
            }
            place_writer.add_place(place_id, place)
            places.append(place)
        gallery_prefix_size = max(1, len(bases) // 2)
        for j, pos in enumerate(absents):
            place_id = f"{scene_id}__v3absent{j:04d}"
            set_agent(agent, pos, 0.0)
            jpg, stats = observe_jpeg(sim)
            sample_id = f"{scene_id}__v3absent{j:04d}__anchor_0m__view00"
            min_prefix_dist = min((geodesic(sim.pathfinder, pos, b) for b in bases[:gallery_prefix_size]), default=float("inf"))
            min_full_dist = min((geodesic(sim.pathfinder, pos, b) for b in bases), default=float("inf"))
            meta = {
                "sample_id": sample_id,
                "split_tag": split_tag,
                "scene_id": scene_id,
                "place_id": place_id,
                "view_id": "absent_anchor",
                "role": "absent_outside_full_gallery",
                "variant": "absent_anchor",
                "position": [float(x) for x in pos],
                "yaw_degrees": 0.0,
                "base_distance_m": None,
                "label_grade": "absent_not_in_full_gallery",
                "spatial_protocol_version": PROTOCOL_VERSION,
                "spatial_positive_place_ids": [],
                "nearest_gallery_distance_m": float(min_full_dist),
                "min_geodesic_to_full_gallery_m": float(min_full_dist),
                "gallery_prefix_size": gallery_prefix_size,
                "min_geodesic_to_gallery_prefix_m": float(min_prefix_dist),
                "precise_yaw_label": False,
                "rgb_only_model_input": True,
                "synthetic_roll_source": False,
                **stats,
            }
            sample_writer.add_sample(sample_id, jpg, meta)
            samples.append(meta)
            place = {
                "place_id": place_id,
                "split_tag": split_tag,
                "scene_id": scene_id,
                "role": "absent_outside_full_gallery",
                "base_position": [float(x) for x in pos],
                "gallery_prefix_size": gallery_prefix_size,
                "min_geodesic_to_gallery_prefix_m": float(min_prefix_dist),
                "min_geodesic_to_full_gallery_m": float(min_full_dist),
                "spatial_protocol_version": PROTOCOL_VERSION,
                "views": [{
                    "sample_id": sample_id,
                    "view_id": "absent_anchor",
                    "base_distance_m": None,
                    "label_grade": "absent_not_in_full_gallery",
                    "spatial_protocol_version": PROTOCOL_VERSION,
                    "spatial_positive_place_ids": [],
                    "yaw_degrees": 0.0,
                    "precise_yaw_label": False,
                }],
            }
            place_writer.add_place(place_id, place)
            places.append(place)
        base_pair_distances = [
            geodesic(sim.pathfinder, a, b)
            for index, a in enumerate(bases)
            for b in bases[index + 1 :]
        ]
        absent_clearances = [
            min((geodesic(sim.pathfinder, position, base) for base in bases), default=float("inf"))
            for position in absents
        ]
        audit.update({
            "sample_count": len(samples),
            "place_count": len(places),
            "all_nonblank": all(bool(s.get("nonblank")) for s in samples),
            "gallery_min_pairwise_geodesic_m": min(base_pair_distances) if base_pair_distances else None,
            "absent_min_full_gallery_geodesic_m": min(absent_clearances) if absent_clearances else None,
        })
        return samples, places, audit
    finally:
        sim.close()


def load_split_scenes(split: str) -> List[str]:
    manifest = json.loads((ROOT / "data/manifests/v3_scene_split_manifest.json").read_text(encoding="utf-8"))
    return list(manifest["splits"][split])


def clear_existing(split_tag: str) -> None:
    for kind in ("samples", "places"):
        for path in (ROOT / f"data/shards/{kind}").glob(f"{split_tag}-{kind}-*.tar"):
            path.unlink()
    for path in (ROOT / "data/manifests").glob(f"{split_tag}_*.json*"):
        path.unlink()


def main() -> None:
    global SENSOR_HW, JPEG_QUALITY, NEAR_DISTS, NEAR_PATHS_PER_PLACE, MIN_BASE_SEP_M, MAX_SAMPLES_PER_SHARD, MAX_PLACES_PER_SHARD, SEED
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train", choices=["train", "development", "test", "fresh_confirmation"])
    ap.add_argument("--split-tag", default="")
    ap.add_argument("--max-scenes", type=int, default=1)
    ap.add_argument("--bases-per-scene", type=int, default=8)
    ap.add_argument("--absent-per-scene", type=int, default=4)
    ap.add_argument("--gpu", type=int, default=1)
    ap.add_argument("--sensor-height", type=int, default=SENSOR_HW[0])
    ap.add_argument("--sensor-width", type=int, default=SENSOR_HW[1])
    ap.add_argument("--jpeg-quality", type=int, default=JPEG_QUALITY)
    ap.add_argument("--near-dists", default=chr(44).join(str(x) for x in NEAR_DISTS))
    ap.add_argument("--near-paths-per-place", type=int, default=NEAR_PATHS_PER_PLACE)
    ap.add_argument("--min-base-sep-m", type=float, default=MIN_BASE_SEP_M)
    ap.add_argument("--max-samples-per-shard", type=int, default=MAX_SAMPLES_PER_SHARD)
    ap.add_argument("--max-places-per-shard", type=int, default=MAX_PLACES_PER_SHARD)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--allow-confirmation", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if args.split == "fresh_confirmation" and not args.allow_confirmation:
        raise SystemExit("拒绝访问 fresh_confirmation：必须在模型/阈值/代码/manifest 冻结后显式 --allow-confirmation。")
    SENSOR_HW = (int(args.sensor_height), int(args.sensor_width))
    JPEG_QUALITY = int(args.jpeg_quality)
    NEAR_DISTS = [float(x) for x in args.near_dists.split(chr(44)) if x.strip()]
    NEAR_PATHS_PER_PLACE = max(1, int(args.near_paths_per_place))
    MIN_BASE_SEP_M = float(args.min_base_sep_m)
    SpatialProtocol(node_separation_m=MIN_BASE_SEP_M).validate()
    MAX_SAMPLES_PER_SHARD = int(args.max_samples_per_shard)
    MAX_PLACES_PER_SHARD = int(args.max_places_per_shard)
    SEED = int(args.seed)
    split_tag = args.split_tag or f"{args.split}_pilot"
    if args.force:
        clear_existing(split_tag)
    scenes = load_split_scenes(args.split)[: max(0, args.max_scenes)]
    if not scenes:
        raise RuntimeError(f"no scenes selected for {args.split}")
    sample_writer = ShardWriter(ROOT / "data/shards/samples", split_tag, "samples", MAX_SAMPLES_PER_SHARD)
    place_writer = ShardWriter(ROOT / "data/shards/places", split_tag, "places", MAX_PLACES_PER_SHARD)
    all_samples: List[Dict[str, Any]] = []
    all_places: List[Dict[str, Any]] = []
    audits: List[Dict[str, Any]] = []
    started = time.perf_counter()
    failures = []
    for local_idx, scene_id in enumerate(scenes):
        try:
            samples, places, audit = render_scene(split_tag, scene_id, local_idx, args.gpu, args.bases_per_scene, args.absent_per_scene, NEAR_PATHS_PER_PLACE, sample_writer, place_writer)
            all_samples.extend(samples)
            all_places.extend(places)
            audits.append(audit)
            print(json.dumps({"split_tag": split_tag, "scene_id": scene_id, "samples": len(samples), "places": len(places), "elapsed_s": time.perf_counter() - started}, ensure_ascii=False), flush=True)
        except Exception as exc:
            failures.append({"scene_id": scene_id, "error": repr(exc)})
            print(json.dumps({"failed_scene": scene_id, "error": repr(exc)}, ensure_ascii=False), flush=True)
    sample_shards = sample_writer.close()
    place_shards = place_writer.close()
    manifest_dir = ROOT / "data/manifests"
    sample_index = build_sample_index(sample_shards, split_tag)
    atomic_jsonl(manifest_dir / f"{split_tag}_sample_manifest.jsonl", all_samples)
    atomic_jsonl(manifest_dir / f"{split_tag}_place_manifest.jsonl", all_places)
    atomic_json(manifest_dir / f"{split_tag}_sample_index.json", sample_index)
    shard_manifest = {
        "schema_version": "3.0.0",
        "spatial_protocol_version": PROTOCOL_VERSION,
        "created_at": now(),
        "split": args.split,
        "split_tag": split_tag,
        "scene_count": len(audits),
        "sample_count": len(all_samples),
        "place_count": len(all_places),
        "gallery_place_count": sum(1 for p in all_places if p.get("role") == "gallery_place"),
        "absent_place_count": sum(1 for p in all_places if p.get("role") == "absent_outside_full_gallery"),
        "sample_shards": sample_shards,
        "sample_index": {
            "path": str(manifest_dir / f"{split_tag}_sample_index.json"),
            "sample_count": sample_index["sample_count"],
            "sha256": sha_path(manifest_dir / f"{split_tag}_sample_index.json") if (manifest_dir / f"{split_tag}_sample_index.json").is_file() else None,
        },
        "place_shards": place_shards,
        "scene_audits": audits,
        "failures": failures,
        "elapsed_seconds": time.perf_counter() - started,
        "storage_rule_zh": "每张 panorama JPEG 只在 sample shard 中存储一次；place shard 只保存 sample_id 和标签/分组元数据，不复制图像。",
        "confirmation_accessed": args.split == "fresh_confirmation",
        "generation_config": {
            "sensor_hw": list(SENSOR_HW),
            "jpeg_quality": JPEG_QUALITY,
            "near_dists": NEAR_DISTS,
            "near_paths_per_place": NEAR_PATHS_PER_PLACE,
            "min_base_sep_m": MIN_BASE_SEP_M,
            "bases_per_scene": args.bases_per_scene,
            "absent_per_scene": args.absent_per_scene,
            "max_samples_per_shard": MAX_SAMPLES_PER_SHARD,
            "max_places_per_shard": MAX_PLACES_PER_SHARD,
            "seed": SEED,
            "spatial_positive_radius_m": 1.0,
            "strict_owner_radius_m": 0.25,
            "absent_clearance_m": 3.0,
        },
    }
    atomic_json(manifest_dir / f"{split_tag}_shard_manifest.json", shard_manifest)
    (ROOT / "runtime").mkdir(parents=True, exist_ok=True)
    with (ROOT / "runtime/command_records_v3.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps({"created_at": now(), "step": "build_v3_sample_place_shards", "split": args.split, "split_tag": split_tag, "outputs": [str(manifest_dir / f"{split_tag}_shard_manifest.json"), str(manifest_dir / f"{split_tag}_sample_manifest.jsonl"), str(manifest_dir / f"{split_tag}_place_manifest.jsonl"), str(manifest_dir / f"{split_tag}_sample_index.json")], "returncode": 0 if not failures else 1}, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps({"split_tag": split_tag, "sample_count": len(all_samples), "place_count": len(all_places), "scene_count": len(audits), "failures": failures, "manifest": str(manifest_dir / f"{split_tag}_shard_manifest.json")}, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit("some scenes failed")


if __name__ == "__main__":
    main()
