#!/usr/bin/env python3
from __future__ import annotations

import argparse
import collections
import datetime as dt
import hashlib
import heapq
import json
import math
import os
import statistics
import sys
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = "r35_bearing_pairs_v1"
PRIMARY_BUCKETS = (
    ("0.5-1m", 0.5, 1.0, False),
    ("1-2m", 1.0, 2.0, False),
    ("2-3m", 2.0, 3.0, False),
    ("3-4m", 3.0, 4.0001, False),
)
DIAGNOSTIC_BUCKETS = (("4-6m_diagnostic", 4.0001, 6.0001, True),)


@dataclass(frozen=True)
class Sample:
    sample_id: str
    scene_id: str
    place_id: str
    position: tuple[float, float, float]
    yaw_degrees: float
    jpeg_sha256: str
    variant: str
    base_distance_m: float
    shard_path: str
    jpeg_member: str


@dataclass(frozen=True)
class Candidate:
    source_index: int
    target_index: int
    distance_m: float
    bucket: str
    source_bearing: float
    target_bearing: float
    source_bin: int
    target_bin: int
    physical_pair_id: str
    tie_breaker: int


def now_iso() -> str:
    return dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).isoformat(timespec="seconds")


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(text: str, seed: int, length: int = 16) -> str:
    return hashlib.sha256(f"{seed}|{text}".encode("utf-8")).hexdigest()[:length]


def stable_int(text: str, seed: int) -> int:
    return int(stable_hash(text, seed, 16), 16)


def splitmix64(value: int) -> int:
    mask = (1 << 64) - 1
    value = (value + 0x9E3779B97F4A7C15) & mask
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & mask
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & mask
    return (value ^ (value >> 31)) & mask


def canonical_position(position: Iterable[float]) -> tuple[int, int, int]:
    return tuple(int(round(float(value) * 1000.0)) for value in position)  # type: ignore[return-value]


def physical_pair_id(scene_id: str, a: Sample, b: Sample) -> str:
    keys = sorted((canonical_position(a.position), canonical_position(b.position)))
    payload = f"{scene_id}|{keys[0]}|{keys[1]}"
    return f"r35phys_{hashlib.sha256(payload.encode('utf-8')).hexdigest()[:24]}"


def distance_bucket(distance_m: float) -> str | None:
    for name, lower, upper, _ in PRIMARY_BUCKETS + DIAGNOSTIC_BUCKETS:
        if lower <= distance_m < upper:
            return name
    return None


def resolve_shard(repo: Path, source_path: str) -> Path:
    source = Path(source_path)
    candidates = (
        source,
        repo / "data" / "shards" / "samples" / source.name,
        repo / source.name,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Cannot resolve sample shard {source_path}")


def sample_from_metadata(metadata: dict[str, Any], shard: Path, member_name: str) -> Sample | None:
    if metadata.get("role") != "place_view" or not metadata.get("nonblank", False):
        return None
    position = metadata.get("position")
    if not isinstance(position, list) or len(position) != 3:
        return None
    sample_id = str(metadata["sample_id"])
    return Sample(
        sample_id=sample_id,
        scene_id=str(metadata["scene_id"]),
        place_id=str(metadata["place_id"]),
        position=(float(position[0]), float(position[1]), float(position[2])),
        yaw_degrees=float(metadata.get("yaw_degrees", 0.0)),
        jpeg_sha256=str(metadata["jpeg_sha256"]),
        variant=str(metadata.get("variant", metadata.get("view_id", "unknown"))),
        base_distance_m=float(metadata.get("base_distance_m", -1.0)),
        shard_path=str(shard.relative_to(shard.parents[3])) if len(shard.parents) >= 4 else str(shard),
        jpeg_member=f"{sample_id}.jpg",
    )


def relative_bearing(
    geometry_module: Any,
    source: Sample,
    target: Sample,
) -> tuple[float, bool, float]:
    return geometry_module.relative_translation_bearing_degrees(
        source.position,
        target.position,
        source.yaw_degrees,
        minimum_translation_m=0.05,
    )


def generate_candidates(
    samples: list[Sample],
    geometry_module: Any,
    seed: int,
    max_vertical_delta_m: float,
    pool_per_source_bin: int,
) -> tuple[dict[str, list[Candidate]], int, dict[str, int]]:
    unique_by_position_and_yaw: dict[tuple[tuple[int, int, int], int], Sample] = {}
    duplicate_count = 0
    for sample in samples:
        key = (canonical_position(sample.position), int(round(sample.yaw_degrees * 1000.0)))
        if key in unique_by_position_and_yaw:
            duplicate_count += 1
            continue
        unique_by_position_and_yaw[key] = sample
    rows = sorted(unique_by_position_and_yaw.values(), key=lambda item: item.sample_id)
    candidate_counts: collections.Counter[str] = collections.Counter()
    pools: dict[str, dict[int, list[tuple[int, int, int, Candidate]]]] = collections.defaultdict(
        lambda: collections.defaultdict(list)
    )
    place_seed = stable_int(rows[0].place_id, seed) if rows else seed
    for i, source in enumerate(rows):
        for j in range(i + 1, len(rows)):
            target = rows[j]
            if abs(source.position[1] - target.position[1]) > max_vertical_delta_m:
                continue
            dx = target.position[0] - source.position[0]
            dz = target.position[2] - source.position[2]
            distance_m = math.hypot(dx, dz)
            bucket = distance_bucket(distance_m)
            if bucket is None:
                continue
            source_bearing, source_valid, source_distance = relative_bearing(
                geometry_module, source, target
            )
            target_bearing, target_valid, target_distance = relative_bearing(
                geometry_module, target, source
            )
            if not source_valid or not target_valid:
                continue
            if abs(source_distance - distance_m) > 1e-6 or abs(target_distance - distance_m) > 1e-6:
                raise RuntimeError("Bearing geometry distance mismatch")
            candidate_counts[bucket] += 1
            source_bin = geometry_module.bearing_to_bin(source_bearing, 72)
            target_bin = geometry_module.bearing_to_bin(target_bearing, 72)
            tie_breaker = splitmix64(
                place_seed
                ^ ((i + 1) * 0xD6E8FEB86659FD93)
                ^ ((j + 1) * 0xA5A3564E27F8862D)
            )
            heap = pools[bucket][source_bin]
            if len(heap) >= pool_per_source_bin and tie_breaker >= -heap[0][0]:
                continue
            candidate = Candidate(
                source_index=i,
                target_index=j,
                distance_m=distance_m,
                bucket=bucket,
                source_bearing=source_bearing,
                target_bearing=target_bearing,
                source_bin=source_bin,
                target_bin=target_bin,
                physical_pair_id=physical_pair_id(source.scene_id, source, target),
                tie_breaker=tie_breaker,
            )
            entry = (-tie_breaker, i, j, candidate)
            if len(heap) < pool_per_source_bin:
                heapq.heappush(heap, entry)
            else:
                heapq.heapreplace(heap, entry)
    candidates = {
        bucket: [entry[3] for bin_heap in bin_heaps.values() for entry in bin_heap]
        for bucket, bin_heaps in pools.items()
    }
    return candidates, duplicate_count, dict(candidate_counts)


def choose_candidates(
    candidates: list[Candidate],
    sample_count: int,
    limit: int,
    global_bin_counts: list[int],
    globally_selected_physical_ids: set[str],
) -> list[Candidate]:
    available = [
        candidate
        for candidate in candidates
        if candidate.physical_pair_id not in globally_selected_physical_ids
    ]
    selected: list[Candidate] = []
    local_bin_counts = [0] * 72
    sample_reuse = [0] * sample_count
    while available and len(selected) < limit:
        def score(candidate: Candidate) -> tuple[int, int, int, int, int]:
            global_sum = global_bin_counts[candidate.source_bin] + global_bin_counts[candidate.target_bin]
            global_peak = max(
                global_bin_counts[candidate.source_bin],
                global_bin_counts[candidate.target_bin],
            )
            local_sum = local_bin_counts[candidate.source_bin] + local_bin_counts[candidate.target_bin]
            reuse = sample_reuse[candidate.source_index] + sample_reuse[candidate.target_index]
            return global_sum, global_peak, local_sum, reuse, candidate.tie_breaker

        best = min(available, key=score)
        available.remove(best)
        selected.append(best)
        globally_selected_physical_ids.add(best.physical_pair_id)
        global_bin_counts[best.source_bin] += 1
        global_bin_counts[best.target_bin] += 1
        local_bin_counts[best.source_bin] += 1
        local_bin_counts[best.target_bin] += 1
        sample_reuse[best.source_index] += 1
        sample_reuse[best.target_index] += 1
    return selected


def sample_payload(sample: Sample) -> dict[str, Any]:
    return {
        "sample_id": sample.sample_id,
        "position_xyz": list(sample.position),
        "render_yaw_degrees": sample.yaw_degrees,
        "jpeg_sha256": sample.jpeg_sha256,
        "variant": sample.variant,
        "base_distance_m": sample.base_distance_m,
        "sample_shard": sample.shard_path,
        "jpeg_member": sample.jpeg_member,
    }


def directed_pair_payload(
    split_name: str,
    source: Sample,
    target: Sample,
    candidate: Candidate,
    bearing: float,
    bearing_bin: int,
    reverse: bool,
) -> dict[str, Any]:
    direction_key = f"{source.sample_id}->{target.sample_id}"
    reciprocal_key = f"{target.sample_id}->{source.sample_id}"
    return {
        "schema_version": SCHEMA_VERSION,
        "split": split_name,
        "directed_pair_id": f"r35dir_{hashlib.sha256(direction_key.encode('utf-8')).hexdigest()[:24]}",
        "reciprocal_pair_id": f"r35dir_{hashlib.sha256(reciprocal_key.encode('utf-8')).hexdigest()[:24]}",
        "physical_pair_id": candidate.physical_pair_id,
        "scene_id": source.scene_id,
        "owner_place_id": source.place_id,
        "source": sample_payload(source),
        "target": sample_payload(target),
        "distance_m": candidate.distance_m,
        "distance_bucket": candidate.bucket,
        "bearing_degrees": bearing,
        "bearing_bin_72": bearing_bin,
        "bearing_valid": True,
        "reciprocal_direction": bool(reverse),
        "coordinate_convention": {
            "range": "[-180, 180)",
            "zero": "source_camera_forward",
            "positive": "source_camera_right_omniguard",
            "habitat_yaw_zero_forward": "-Z",
        },
        "label_provenance": "Habitat GT position and source render yaw; offline training only",
        "overlap_proxy": {
            "kind": "shared_owner_anchor_geometric_candidate",
            "shared_owner": True,
            "source_owner_distance_m": source.base_distance_m,
            "target_owner_distance_m": target.base_distance_m,
            "visual_overlap_verified": False,
            "requires_teacher_or_image_audit": True,
        },
        "test_r32_confirmation_accessed": False,
    }


def iter_metadata_groups(
    repo: Path,
    manifest: dict[str, Any],
    max_shards: int,
    stats: dict[str, Any],
) -> Iterable[list[Sample]]:
    shard_entries = manifest["sample_shards"]
    selected_entries = shard_entries[:max_shards] if max_shards else shard_entries
    truncated = len(selected_entries) < len(shard_entries)
    current_place: str | None = None
    current_samples: list[Sample] = []
    completed_places: set[str] = set()
    for shard_number, shard_entry in enumerate(selected_entries, start=1):
        shard = resolve_shard(repo, str(shard_entry["path"]))
        stats["sample_shards_scanned"] += 1
        stats["last_shard_scanned"] = str(shard)
        with tarfile.open(shard, "r:") as archive:
            for member in archive:
                if not member.isfile() or not member.name.endswith(".json"):
                    continue
                extracted = archive.extractfile(member)
                if extracted is None:
                    raise RuntimeError(f"Cannot read {shard}:{member.name}")
                metadata = json.load(extracted)
                stats["metadata_rows_scanned"] += 1
                sample = sample_from_metadata(metadata, shard, member.name)
                if sample is None:
                    stats["metadata_rows_excluded"] += 1
                    continue
                if current_place is None:
                    current_place = sample.place_id
                if sample.place_id != current_place:
                    if sample.place_id in completed_places:
                        raise RuntimeError(f"Non-contiguous place metadata: {sample.place_id}")
                    completed_places.add(current_place)
                    yield current_samples
                    current_place = sample.place_id
                    current_samples = []
                current_samples.append(sample)
        stats["scan_progress"] = {
            "completed_shards": shard_number,
            "selected_shards": len(selected_entries),
            "total_shards": len(shard_entries),
        }
    if current_samples and not truncated:
        yield current_samples
    elif current_samples:
        stats["truncated_place_dropped"] = current_place
        stats["truncated_place_rows_dropped"] = len(current_samples)


def build_split(
    repo: Path,
    output_dir: Path,
    split_name: str,
    source_manifest_path: Path,
    geometry_module: Any,
    pairs_per_bucket: int,
    diagnostic_pairs_per_bucket: int,
    max_vertical_delta_m: float,
    candidate_pool_per_source_bin: int,
    max_shards: int,
    seed: int,
) -> tuple[dict[str, Any], set[str], set[str]]:
    with source_manifest_path.open("r", encoding="utf-8") as handle:
        source_manifest = json.load(handle)
    jsonl_path = output_dir / f"{split_name}_bearing_pairs.jsonl"
    temp_path = jsonl_path.with_suffix(jsonl_path.suffix + ".tmp")
    if jsonl_path.exists() or temp_path.exists():
        raise FileExistsError(f"Refusing to overwrite {jsonl_path} or {temp_path}")

    stats: dict[str, Any] = {
        "metadata_rows_scanned": 0,
        "metadata_rows_excluded": 0,
        "sample_shards_scanned": 0,
        "places_scanned": 0,
        "eligible_samples": 0,
        "duplicate_position_yaw_rows": 0,
        "candidate_unordered_pairs_by_bucket": collections.Counter(),
        "selected_unordered_pairs_by_bucket": collections.Counter(),
        "selected_directed_pairs_by_bucket": collections.Counter(),
        "selected_scenes": collections.Counter(),
        "bearing_bin_72_counts": [0] * 72,
        "distance_values": collections.defaultdict(list),
    }
    globally_selected_physical_ids: set[str] = set()
    selected_scene_ids: set[str] = set()
    digest = hashlib.sha256()
    directed_count = 0
    with temp_path.open("wb") as output:
        for samples in iter_metadata_groups(repo, source_manifest, max_shards, stats):
            if not samples:
                continue
            stats["places_scanned"] += 1
            stats["eligible_samples"] += len(samples)
            candidates_by_bucket, duplicate_count, candidate_counts = generate_candidates(
                samples,
                geometry_module,
                seed,
                max_vertical_delta_m,
                candidate_pool_per_source_bin,
            )
            stats["duplicate_position_yaw_rows"] += duplicate_count
            ordered_samples = sorted(
                {
                    (canonical_position(sample.position), int(round(sample.yaw_degrees * 1000.0))): sample
                    for sample in samples
                }.values(),
                key=lambda item: item.sample_id,
            )
            for bucket_name, _, _, diagnostic in PRIMARY_BUCKETS + DIAGNOSTIC_BUCKETS:
                bucket_candidates = candidates_by_bucket.get(bucket_name, [])
                stats["candidate_unordered_pairs_by_bucket"][bucket_name] += candidate_counts.get(
                    bucket_name, 0
                )
                limit = diagnostic_pairs_per_bucket if diagnostic else pairs_per_bucket
                selected = choose_candidates(
                    bucket_candidates,
                    len(ordered_samples),
                    limit,
                    stats["bearing_bin_72_counts"],
                    globally_selected_physical_ids,
                )
                stats["selected_unordered_pairs_by_bucket"][bucket_name] += len(selected)
                stats["selected_directed_pairs_by_bucket"][bucket_name] += 2 * len(selected)
                for candidate in selected:
                    a = ordered_samples[candidate.source_index]
                    b = ordered_samples[candidate.target_index]
                    selected_scene_ids.add(a.scene_id)
                    stats["selected_scenes"][a.scene_id] += 2
                    stats["distance_values"][bucket_name].append(candidate.distance_m)
                    payloads = (
                        directed_pair_payload(
                            split_name,
                            a,
                            b,
                            candidate,
                            candidate.source_bearing,
                            candidate.source_bin,
                            False,
                        ),
                        directed_pair_payload(
                            split_name,
                            b,
                            a,
                            candidate,
                            candidate.target_bearing,
                            candidate.target_bin,
                            True,
                        ),
                    )
                    for payload in payloads:
                        line = (json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
                        output.write(line)
                        digest.update(line)
                        directed_count += 1
    os.replace(temp_path, jsonl_path)

    distance_stats: dict[str, Any] = {}
    for bucket, values in stats.pop("distance_values").items():
        if values:
            distance_stats[bucket] = {
                "count_unordered": len(values),
                "min_m": min(values),
                "median_m": statistics.median(values),
                "max_m": max(values),
            }
    for key in (
        "candidate_unordered_pairs_by_bucket",
        "selected_unordered_pairs_by_bucket",
        "selected_directed_pairs_by_bucket",
        "selected_scenes",
    ):
        stats[key] = dict(sorted(stats[key].items()))
    bin_counts = stats["bearing_bin_72_counts"]
    nonzero_bins = [value for value in bin_counts if value > 0]
    stats["bearing_balance"] = {
        "nonzero_bins": len(nonzero_bins),
        "min_count": min(nonzero_bins) if nonzero_bins else 0,
        "max_count": max(nonzero_bins) if nonzero_bins else 0,
        "max_to_min_ratio": (max(nonzero_bins) / min(nonzero_bins)) if nonzero_bins else None,
    }
    stats["distance_stats"] = distance_stats
    stats["selected_directed_pair_count"] = directed_count
    stats["selected_unordered_pair_count"] = len(globally_selected_physical_ids)
    manifest_record = {
        "schema_version": f"{SCHEMA_VERSION}_manifest",
        "created_at": now_iso(),
        "split": split_name,
        "source_manifest": str(source_manifest_path.relative_to(repo)),
        "source_manifest_sha256": sha256_file(source_manifest_path),
        "source_sample_count": source_manifest.get("sample_count"),
        "source_place_count": source_manifest.get("place_count"),
        "source_sample_shard_count": len(source_manifest.get("sample_shards", [])),
        "selection_rule": {
            "seed": seed,
            "primary_pairs_per_bucket_per_place": pairs_per_bucket,
            "diagnostic_pairs_per_bucket_per_place": diagnostic_pairs_per_bucket,
            "bearing_bins": 72,
            "distance_buckets": [item[0] for item in PRIMARY_BUCKETS + DIAGNOSTIC_BUCKETS],
            "balance_order": [
                "lowest global directed-bin count",
                "lowest global peak-bin count",
                "lowest local directed-bin count",
                "lowest sample reuse",
                "stable SplitMix64 tie breaker seeded by SHA256(place_id)",
            ],
            "reciprocal_pairs_are_atomic": True,
            "max_vertical_delta_m": max_vertical_delta_m,
            "candidate_reservoir_per_source_bin": candidate_pool_per_source_bin,
            "same_owner_place_required": True,
            "geometric_overlap_is_only_a_proxy": True,
            "visual_overlap_requires_followup_teacher_or_image_audit": True,
        },
        "pair_jsonl": str(jsonl_path.relative_to(output_dir)),
        "pair_jsonl_sha256": digest.hexdigest(),
        "pair_jsonl_bytes": jsonl_path.stat().st_size,
        "stats": stats,
        "test_r32_confirmation_accessed": False,
    }
    manifest_path = output_dir / f"{split_name}_bearing_pair_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest_record, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest_record, globally_selected_physical_ids, selected_scene_ids


def write_visualization(output_dir: Path, manifests: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {"created": False}
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        figure, axes = plt.subplots(1, len(manifests), figsize=(8 * len(manifests), 5), subplot_kw={"projection": "polar"})
        if len(manifests) == 1:
            axes = [axes]
        theta = np.deg2rad(np.arange(72) * 5.0 - 177.5)
        for axis, manifest in zip(axes, manifests):
            counts = np.asarray(manifest["stats"]["bearing_bin_72_counts"], dtype=float)
            axis.bar(theta, counts, width=np.deg2rad(5.0), alpha=0.85)
            axis.set_theta_zero_location("N")
            axis.set_theta_direction(-1)
            axis.set_title(f"{manifest['split']} bearing labels (0=forward, +=right)")
        figure.tight_layout()
        path = output_dir / "bearing_label_direction_distribution.png"
        figure.savefig(path, dpi=160)
        plt.close(figure)
        result = {
            "created": True,
            "path": path.name,
            "sha256": sha256_file(path),
            "note_zh": "极坐标图零度为正前方，顺时针为正角（机器人右侧）。",
        }
    except Exception as exc:  # visual artifact is audited explicitly
        result = {"created": False, "error": repr(exc)}
    return result


def write_combined_reports(
    output_dir: Path,
    manifests: list[dict[str, Any]],
    physical_sets: list[set[str]],
    scene_sets: list[set[str]],
    minimum_train_pairs: int,
    minimum_development_pairs: int,
) -> None:
    by_split = {manifest["split"]: manifest for manifest in manifests}
    train_count = by_split.get("train", {}).get("stats", {}).get("selected_directed_pair_count", 0)
    development_count = by_split.get("development", {}).get("stats", {}).get("selected_directed_pair_count", 0)
    physical_overlap = set.intersection(*physical_sets) if len(physical_sets) > 1 else set()
    scene_overlap = set.intersection(*scene_sets) if len(scene_sets) > 1 else set()
    reciprocal_even = all(
        manifest["stats"]["selected_directed_pair_count"]
        == 2 * manifest["stats"]["selected_unordered_pair_count"]
        for manifest in manifests
    )
    isolation = {
        "schema_version": "r35_scene_pair_isolation_audit_v1",
        "created_at": now_iso(),
        "passed": not scene_overlap and not physical_overlap and reciprocal_even,
        "train_development_scene_intersection_count": len(scene_overlap),
        "train_development_physical_pair_intersection_count": len(physical_overlap),
        "reciprocal_pairs_atomic_within_split": reciprocal_even,
        "alternate_yaw_position_leakage_impossible_by_scene_disjointness": not scene_overlap,
        "test_r32_confirmation_accessed": False,
        "note_zh": "本审计只比较Train与Internal Development；未读取Test、R32或任何Confirmation内容。",
    }
    (output_dir / "scene_pair_isolation_audit.json").write_text(
        json.dumps(isolation, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    visual = write_visualization(output_dir, manifests)
    combined = {
        "schema_version": "r35_bearing_data_manifest_v1",
        "created_at": now_iso(),
        "splits": {
            manifest["split"]: {
                "manifest": f"{manifest['split']}_bearing_pair_manifest.json",
                "manifest_sha256": sha256_file(output_dir / f"{manifest['split']}_bearing_pair_manifest.json"),
                "directed_pairs": manifest["stats"]["selected_directed_pair_count"],
                "unordered_pairs": manifest["stats"]["selected_unordered_pair_count"],
                "scene_count": len(manifest["stats"]["selected_scenes"]),
            }
            for manifest in manifests
        },
        "minimum_requirements": {
            "train_directed_pairs": minimum_train_pairs,
            "development_directed_pairs": minimum_development_pairs,
        },
        "minimum_pair_counts_passed": (
            train_count >= minimum_train_pairs and development_count >= minimum_development_pairs
        ),
        "isolation_audit": "scene_pair_isolation_audit.json",
        "isolation_passed": isolation["passed"],
        "label_visualization": visual,
        "semantic_scene_type_status": "not_available_in_r3_sample_metadata",
        "visual_overlap_status": "pending_teacher_descriptor_and_sample_image_audit",
        "test_r32_confirmation_accessed": False,
    }
    (output_dir / "bearing_data_manifest.json").write_text(
        json.dumps(combined, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report = f"""# R35 Bearing数据充分性与缺口报告

生成时间：{now_iso()}

## 当前结果

- Train有向pair：{train_count:,}（最低要求：{minimum_train_pairs:,}）
- Internal Development有向pair：{development_count:,}（最低要求：{minimum_development_pairs:,}）
- Train/Development场景交集：{len(scene_overlap)}
- Train/Development物理pair交集：{len(physical_overlap)}
- reciprocal pair是否保持在同一split：{'是' if reciprocal_even else '否'}
- 72方向bin统计：见各split manifest与 `bearing_label_direction_distribution.png`

## 已确认边界

- bearing标签由Habitat GT位置和source render yaw离线生成，不使用relative_yaw替代。
- 每个无序物理pair以A到B和B到A两个方向原子写入同一split。
- Train和Internal Development按scene-disjoint隔离；未读取Test、R32或Confirmation内容。
- 当前几何筛选只证明两张图共享owner anchor且位于指定距离范围，不能单独证明无遮挡或显著视觉重叠。

## 尚待完成

- 使用冻结R3 teacher描述符和抽样图像完成视觉重叠、严重遮挡、多峰歧义审计，再冻结正式训练pair。
- R3 sample metadata没有室内/室外、走廊、房间、岔路等可靠语义标签；必须从已有非禁用元数据或人工抽样建立语义覆盖审计，不能根据scene名称臆测。
- 4到6米仅为诊断范围；若现有pair不足，不触发首轮补采。
"""
    (output_dir / "data_sufficiency_gap_report.md").write_text(report, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build scene-disjoint R35 bearing pair manifests")
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", choices=("train", "development"), default=("train", "development"))
    parser.add_argument("--pairs-per-bucket", type=int, default=8)
    parser.add_argument("--diagnostic-pairs-per-bucket", type=int, default=2)
    parser.add_argument("--max-vertical-delta-m", type=float, default=0.25)
    parser.add_argument("--candidate-pool-per-source-bin", type=int, default=8)
    parser.add_argument("--max-shards", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260811)
    parser.add_argument("--minimum-train-pairs", type=int, default=500_000)
    parser.add_argument("--minimum-development-pairs", type=int, default=50_000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo = args.repo.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    from modules.Panoramic_Place_Compass.training import bearing_geometry as geometry

    manifest_paths = {
        "train": repo / "data" / "manifests" / "train_r3_full_shard_manifest.json",
        "development": repo / "data" / "manifests" / "development_r3_full_shard_manifest.json",
    }
    manifests: list[dict[str, Any]] = []
    physical_sets: list[set[str]] = []
    scene_sets: list[set[str]] = []
    for split_name in args.splits:
        manifest, physical_ids, scene_ids = build_split(
            repo=repo,
            output_dir=output_dir,
            split_name=split_name,
            source_manifest_path=manifest_paths[split_name],
            geometry_module=geometry,
            pairs_per_bucket=args.pairs_per_bucket,
            diagnostic_pairs_per_bucket=args.diagnostic_pairs_per_bucket,
            max_vertical_delta_m=args.max_vertical_delta_m,
            candidate_pool_per_source_bin=args.candidate_pool_per_source_bin,
            max_shards=args.max_shards,
            seed=args.seed,
        )
        manifests.append(manifest)
        physical_sets.append(physical_ids)
        scene_sets.append(scene_ids)
    if set(args.splits) == {"train", "development"}:
        write_combined_reports(
            output_dir,
            manifests,
            physical_sets,
            scene_sets,
            args.minimum_train_pairs,
            args.minimum_development_pairs,
        )
    command_record = {
        "created_at": now_iso(),
        "argv": sys.argv,
        "script_sha256": sha256_file(Path(__file__)),
        "cwd": str(Path.cwd()),
        "test_r32_confirmation_accessed": False,
    }
    (output_dir / "build_command_record.json").write_text(
        json.dumps(command_record, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "ok", "output_dir": str(output_dir)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
