from __future__ import annotations

import hashlib
import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info

from .data import DecodedImageStore, MEAN, STD, roll_tensor_by_degrees


MAX_CANDIDATES = 16
LOGICAL_CATEGORIES = (
    "ordinary_positive",
    "hard_positive",
    "near_but_wrong",
    "same_scene_repeated_texture",
    "clear_absent",
    "yaw_roll_absent",
    "cross_scene_high_score",
)
SOURCE_CATEGORIES = (
    "present",
    "near_but_wrong",
    "same_scene_repeated_texture",
    "clear_absent",
    "yaw_roll_absent",
    "cross_scene_high_score",
)
FORBIDDEN_PATH_TOKENS = (
    "/r32/",
    "/test/",
    "confirmation",
    "final_test",
)


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"JSONL记录不是对象: {path}")
                yield value


def _assert_input_boundary(paths: Sequence[Path]) -> None:
    violations = []
    for path in paths:
        lowered = f"/{str(path.resolve()).lower().strip('/')}"
        if any(token in lowered for token in FORBIDDEN_PATH_TOKENS):
            violations.append(str(path))
    if violations:
        raise RuntimeError(f"Stage 3 raw-image输入触碰禁止数据边界: {violations}")


def _anchor_view(row: dict[str, Any]) -> dict[str, Any] | None:
    for view in row.get("views", []):
        distance = view.get("base_distance_m")
        if view.get("view_id") == "anchor_0m" or (
            distance is not None and float(distance) == 0.0
        ):
            return view
    views = row.get("views", [])
    return views[0] if views else None


def load_place_anchors(path: Path) -> dict[str, str]:
    anchors: dict[str, str] = {}
    for row in iter_jsonl(path):
        if row.get("role") != "gallery_place":
            continue
        view = _anchor_view(row)
        if view is not None:
            anchors[str(row["place_id"])] = str(view["sample_id"])
    if not anchors:
        raise RuntimeError("Stage 3 place manifest没有gallery anchor")
    return anchors


def _load_hard_positive_indices(path: Path) -> tuple[set[int], dict[str, Any]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version") != "r35_track_c_recall_pool_manifest_v1"
        or manifest.get("test_or_confirmation_accessed") is not False
    ):
        raise RuntimeError("Stage 3 hard-positive manifest不合法")
    pool = Path(manifest["hard_positive_pool"])
    if sha256_path(pool) != manifest.get("hard_positive_pool_sha256"):
        raise RuntimeError("Stage 3 hard-positive pool SHA256不匹配")
    indices = {
        int(row["r341_train_record_index"])
        for row in iter_jsonl(pool)
        if row.get("source_partition") == "R34.1 Train only"
        and row.get("development_sample_copied") is False
        and row.get("test_r32_confirmation_accessed") is False
    }
    if len(indices) != int(manifest.get("hard_positive_count", -1)):
        raise RuntimeError("Stage 3 hard-positive数量与manifest不一致")
    return indices, {
        "manifest": str(path),
        "manifest_sha256": sha256_path(path),
        "pool": str(pool),
        "pool_sha256": sha256_path(pool),
        "count": len(indices),
        "train_only": True,
    }


def _load_internal_hard_positive_indices(
    path: Path,
) -> tuple[set[int], dict[str, Any]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version")
        != "r35_internal_hard_positive_eval_manifest_v1"
        or manifest.get("evaluation_only") is not True
        or manifest.get("copied_into_training") is not False
        or manifest.get("test_r32_confirmation_accessed") is not False
    ):
        raise RuntimeError("Stage 3 Internal hard-positive manifest不合法")
    pool = Path(manifest["pool"])
    if sha256_path(pool) != manifest.get("pool_sha256"):
        raise RuntimeError("Stage 3 Internal hard-positive pool SHA256不匹配")
    indices = {
        int(row["r341_internal_record_index"])
        for row in iter_jsonl(pool)
        if row.get("evaluation_only") is True
        and row.get("copied_into_training") is False
        and row.get("test_r32_confirmation_accessed") is False
    }
    if len(indices) != int(manifest.get("hard_positive_count", -1)):
        raise RuntimeError("Stage 3 Internal hard-positive数量与manifest不一致")
    return indices, {
        "manifest": str(path),
        "manifest_sha256": sha256_path(path),
        "pool": str(pool),
        "pool_sha256": sha256_path(pool),
        "count": len(indices),
        "evaluation_only": True,
        "copied_into_training": False,
    }


class R35RawImageReader:
    def __init__(
        self,
        cache_manifests: Sequence[Path],
        cache_indexes: Sequence[dict[str, int]],
    ) -> None:
        if len(cache_manifests) != len(cache_indexes) or not cache_manifests:
            raise ValueError("Stage 3 decoded cache配置不完整")
        self.stores = [
            DecodedImageStore(path, sample_to_row=index)
            for path, index in zip(cache_manifests, cache_indexes)
        ]

    def read_uint8(self, sample_id: str) -> torch.Tensor:
        for store in self.stores:
            try:
                value = store.read_tensor(sample_id)
                if value.dtype != torch.uint8 or tuple(value.shape) != (3, 224, 448):
                    raise RuntimeError(f"Stage 3 cache image shape/dtype不合法: {sample_id}")
                return value
            except KeyError:
                continue
        raise KeyError(f"Stage 3所有冻结cache均缺少sample: {sample_id}")

    def close(self) -> None:
        for store in self.stores:
            store.close()


class R35TrackCRawBatchDataset(IterableDataset):
    """Yield score-independent raw-image batches containing all seven Track C categories."""

    def __init__(
        self,
        *,
        raw_data_audit: str | Path,
        hard_positive_manifest: str | Path,
        sets_per_category: int,
        seed: int,
        partition: str = "train",
        deterministic: bool = False,
    ) -> None:
        super().__init__()
        if partition != "train":
            raise ValueError("Stage 3正式训练raw batch dataset只允许Train partition")
        if int(sets_per_category) not in (1, 2, 4, 6, 8, 12, 16, 24, 32):
            raise ValueError("sets_per_category必须是预注册的1、2、4、6、8、12、16、24或32")
        self.audit_path = Path(raw_data_audit).resolve()
        self.hard_positive_manifest_path = Path(hard_positive_manifest).resolve()
        self.sets_per_category = int(sets_per_category)
        self.seed = int(seed)
        self.deterministic = bool(deterministic)
        audit = json.loads(self.audit_path.read_text(encoding="utf-8"))
        if (
            audit.get("schema_version")
            != "r35_stage3_raw_data_feasibility_audit_v1"
            or audit.get("passed") is not True
            or not all(audit.get("gates", {}).values())
            or audit.get("test_r32_confirmation_accessed") is not False
        ):
            raise RuntimeError("Stage 3 raw data feasibility audit未通过")
        partition_record = audit.get("partitions", {}).get(partition)
        if not partition_record or partition_record.get("passed") is not True:
            raise RuntimeError("Stage 3 Train raw partition audit未通过")
        metadata_records = {
            str(record["category"]): record
            for record in partition_record.get("metadata", [])
        }
        if not set(SOURCE_CATEGORIES).issubset(metadata_records):
            raise RuntimeError("Stage 3 Track C Train metadata类别不完整")
        place_manifest = Path(audit["place_manifest"])
        if sha256_path(place_manifest) != audit.get("place_manifest_sha256"):
            raise RuntimeError("Stage 3 place manifest SHA256不匹配")
        cache_records = audit.get("decoded_caches", [])
        if len(cache_records) < 2:
            raise RuntimeError("Stage 3需要主cache和absent cache")
        self.cache_manifests = [Path(record["manifest"]) for record in cache_records]
        self.cache_indexes_paths = [Path(record["index"]) for record in cache_records]
        paths = [
            self.audit_path,
            self.hard_positive_manifest_path,
            place_manifest,
            *self.cache_manifests,
            *self.cache_indexes_paths,
            *(Path(record["path"]) for record in metadata_records.values()),
        ]
        _assert_input_boundary(paths)
        for record, manifest_path, index_path in zip(
            cache_records, self.cache_manifests, self.cache_indexes_paths
        ):
            if sha256_path(manifest_path) != record.get("manifest_sha256"):
                raise RuntimeError("Stage 3 decoded cache manifest SHA256不匹配")
            if sha256_path(index_path) != record.get("index_sha256"):
                raise RuntimeError("Stage 3 decoded cache index SHA256不匹配")
        self.cache_indexes = [
            {str(key): int(value) for key, value in json.loads(path.read_text()).items()}
            for path in self.cache_indexes_paths
        ]
        self.anchors = load_place_anchors(place_manifest)
        hard_indices, self.hard_positive_provenance = _load_hard_positive_indices(
            self.hard_positive_manifest_path
        )
        source_rows: dict[str, list[dict[str, Any]]] = {}
        for category, record in metadata_records.items():
            path = Path(record["path"])
            if sha256_path(path) != record.get("sha256"):
                raise RuntimeError(f"Stage 3 metadata SHA256不匹配: {category}")
            rows = list(iter_jsonl(path))
            if len(rows) != int(record.get("record_count", -1)):
                raise RuntimeError(f"Stage 3 metadata数量不匹配: {category}")
            source_rows[category] = rows
        present = source_rows.pop("present")
        hard = [row for row in present if int(row["record_index"]) in hard_indices]
        ordinary = [row for row in present if int(row["record_index"]) not in hard_indices]
        if not hard or not ordinary:
            raise RuntimeError("Stage 3 present池缺少ordinary或hard-positive")
        self.pools = {
            "ordinary_positive": ordinary,
            "hard_positive": hard,
            **source_rows,
        }
        if set(self.pools) != set(LOGICAL_CATEGORIES):
            raise RuntimeError("Stage 3七类逻辑池构建失败")
        self.provenance = {
            "schema_version": "r35_stage3_raw_batch_data_v1",
            "raw_data_audit": str(self.audit_path),
            "raw_data_audit_sha256": sha256_path(self.audit_path),
            "hard_positive": self.hard_positive_provenance,
            "sets_per_category": self.sets_per_category,
            "sets_per_rank_microbatch": self.sets_per_category * len(LOGICAL_CATEGORIES),
            "deterministic_sampling": self.deterministic,
            "category_pool_counts": {
                category: len(rows) for category, rows in self.pools.items()
            },
            "selection_uses_model_scores": False,
            "test_r32_confirmation_accessed": False,
        }

    def _reader(self) -> R35RawImageReader:
        return R35RawImageReader(self.cache_manifests, self.cache_indexes)

    def _materialize(
        self,
        row: dict[str, Any],
        logical_category: str,
        reader: R35RawImageReader,
    ) -> dict[str, Any]:
        candidate_ids = [str(value) for value in row["candidate_place_ids"]]
        valid = min(
            int(row.get("valid_candidate_count", len(candidate_ids))),
            len(candidate_ids),
            MAX_CANDIDATES,
        )
        if valid < 1:
            raise RuntimeError("Stage 3 Track C候选集合为空")
        candidate_ids = candidate_ids[:valid]
        missing = [value for value in candidate_ids if value not in self.anchors]
        if missing:
            raise KeyError(f"Stage 3 candidate place缺少anchor: {missing[:3]}")
        query = reader.read_uint8(str(row["query_sample_id"]))
        roll = float(row.get("roll_degrees", 0.0))
        if roll:
            query, _ = roll_tensor_by_degrees(query, roll)
        candidates = [reader.read_uint8(self.anchors[value]) for value in candidate_ids]
        candidates.extend([candidates[0]] * (MAX_CANDIDATES - valid))
        mask = torch.zeros(MAX_CANDIDATES, dtype=torch.bool)
        mask[:valid] = True
        same = torch.zeros(MAX_CANDIDATES, dtype=torch.bool)
        for value in row.get("positive_candidate_indices", []):
            index = int(value)
            if 0 <= index < valid:
                same[index] = True
        source_present = logical_category in ("ordinary_positive", "hard_positive")
        if source_present != bool(same.any()):
            raise RuntimeError("Stage 3 present/absent标签与positive candidate矛盾")
        reciprocal = torch.zeros(MAX_CANDIDATES, dtype=torch.float32)
        reciprocal[:valid] = 1.0 / (torch.arange(valid, dtype=torch.float32) + 1.0)
        return {
            "query": query,
            "candidates": torch.stack(candidates),
            "candidate_mask": mask,
            "same_targets": same,
            "hard_negative_targets": mask & ~same,
            "hard_positive_set": logical_category == "hard_positive",
            "reciprocal_rank_score": reciprocal,
            "logical_category": logical_category,
            "source_record_index": int(row["record_index"]),
            "scene_id": str(row["scene_id"]),
            "query_sample_id": str(row["query_sample_id"]),
            "source_place_id": str(row.get("source_place_id", "")),
            "candidate_place_ids": candidate_ids
            + [candidate_ids[0]] * (MAX_CANDIDATES - valid),
        }

    def __iter__(self) -> Iterator[dict[str, Any]]:
        worker = get_worker_info()
        worker_id = worker.id if worker else 0
        rank = int(os.environ.get("RANK", "0"))
        generator = torch.Generator().manual_seed(
            self.seed
            + rank * 100_003
            + worker_id * 10_007
            + (0 if self.deterministic else int(time.time_ns() % 1_000_003))
        )
        reader = self._reader()
        try:
            while True:
                rows: list[dict[str, Any]] = []
                for category in LOGICAL_CATEGORIES:
                    pool = self.pools[category]
                    indices = torch.randint(
                        len(pool),
                        (self.sets_per_category,),
                        generator=generator,
                    )
                    rows.extend(
                        self._materialize(pool[int(index)], category, reader)
                        for index in indices
                    )
                permutation = torch.randperm(len(rows), generator=generator).tolist()
                rows = [rows[index] for index in permutation]
                yield {
                    "query_images_uint8": torch.stack([row["query"] for row in rows]),
                    "candidate_images_uint8": torch.stack(
                        [row["candidates"] for row in rows]
                    ),
                    "candidate_mask": torch.stack([row["candidate_mask"] for row in rows]),
                    "same_targets": torch.stack([row["same_targets"] for row in rows]),
                    "hard_negative_targets": torch.stack(
                        [row["hard_negative_targets"] for row in rows]
                    ),
                    "hard_positive_set": torch.tensor(
                        [row["hard_positive_set"] for row in rows], dtype=torch.bool
                    ),
                    "reciprocal_rank_score": torch.stack(
                        [row["reciprocal_rank_score"] for row in rows]
                    ),
                    "categories": [row["logical_category"] for row in rows],
                    "source_record_indices": [row["source_record_index"] for row in rows],
                    "scene_ids": [row["scene_id"] for row in rows],
                    "query_sample_ids": [row["query_sample_id"] for row in rows],
                    "category_counts": {
                        category: self.sets_per_category
                        for category in LOGICAL_CATEGORIES
                    },
                }
        finally:
            reader.close()


class R35TrackCRawEvaluationDataset(Dataset):
    """Deterministically traverse the frozen scene-disjoint Internal partition."""

    def __init__(
        self,
        *,
        raw_data_audit: str | Path,
        internal_hard_positive_manifest: str | Path,
    ) -> None:
        super().__init__()
        self.audit_path = Path(raw_data_audit).resolve()
        self.hard_positive_manifest_path = Path(
            internal_hard_positive_manifest
        ).resolve()
        audit = json.loads(self.audit_path.read_text(encoding="utf-8"))
        if (
            audit.get("schema_version")
            != "r35_stage3_raw_data_feasibility_audit_v1"
            or audit.get("passed") is not True
            or not all(audit.get("gates", {}).values())
            or audit.get("test_r32_confirmation_accessed") is not False
        ):
            raise RuntimeError("Stage 3 raw data feasibility audit未通过")
        partition = audit.get("partitions", {}).get("internal_validation")
        if not partition or partition.get("passed") is not True:
            raise RuntimeError("Stage 3 Internal raw partition audit未通过")
        metadata_records = {
            str(record["category"]): record
            for record in partition.get("metadata", [])
        }
        if not set(SOURCE_CATEGORIES).issubset(metadata_records):
            raise RuntimeError("Stage 3 Internal metadata类别不完整")
        place_manifest = Path(audit["place_manifest"])
        if sha256_path(place_manifest) != audit.get("place_manifest_sha256"):
            raise RuntimeError("Stage 3 Internal place manifest SHA256不匹配")
        cache_records = audit.get("decoded_caches", [])
        if len(cache_records) < 2:
            raise RuntimeError("Stage 3 Internal需要主cache和absent cache")
        self.cache_manifests = [Path(record["manifest"]) for record in cache_records]
        self.cache_indexes_paths = [Path(record["index"]) for record in cache_records]
        paths = [
            self.audit_path,
            self.hard_positive_manifest_path,
            place_manifest,
            *self.cache_manifests,
            *self.cache_indexes_paths,
            *(Path(record["path"]) for record in metadata_records.values()),
        ]
        _assert_input_boundary(paths)
        for record, manifest_path, index_path in zip(
            cache_records, self.cache_manifests, self.cache_indexes_paths
        ):
            if sha256_path(manifest_path) != record.get("manifest_sha256"):
                raise RuntimeError("Stage 3 Internal cache manifest SHA256不匹配")
            if sha256_path(index_path) != record.get("index_sha256"):
                raise RuntimeError("Stage 3 Internal cache index SHA256不匹配")
        self.cache_indexes = [
            {str(key): int(value) for key, value in json.loads(path.read_text()).items()}
            for path in self.cache_indexes_paths
        ]
        self.anchors = load_place_anchors(place_manifest)
        hard_indices, hard_provenance = _load_internal_hard_positive_indices(
            self.hard_positive_manifest_path
        )
        self.rows: list[tuple[str, dict[str, Any], bool]] = []
        for category in SOURCE_CATEGORIES:
            record = metadata_records[category]
            path = Path(record["path"])
            if sha256_path(path) != record.get("sha256"):
                raise RuntimeError(f"Stage 3 Internal metadata SHA256不匹配: {category}")
            rows = list(iter_jsonl(path))
            if len(rows) != int(record.get("record_count", -1)):
                raise RuntimeError(f"Stage 3 Internal metadata数量不匹配: {category}")
            self.rows.extend(
                (
                    category,
                    row,
                    category == "present"
                    and int(row["record_index"]) in hard_indices,
                )
                for row in rows
            )
        observed_hard = sum(hard for _, _, hard in self.rows)
        if observed_hard != len(hard_indices):
            raise RuntimeError("Stage 3 Internal hard-positive索引未完整映射")
        if len(self.rows) != int(partition.get("record_count", -1)):
            raise RuntimeError("Stage 3 Internal完整遍历数量与audit不一致")
        self._reader_instance: R35RawImageReader | None = None
        self.provenance = {
            "schema_version": "r35_stage3_raw_internal_evaluation_v1",
            "raw_data_audit": str(self.audit_path),
            "raw_data_audit_sha256": sha256_path(self.audit_path),
            "partition": "internal_validation",
            "record_count": len(self.rows),
            "scene_count": int(partition["scene_count"]),
            "category_counts": dict(partition["category_counts"]),
            "hard_positive": hard_provenance,
            "deterministic_complete_traversal": True,
            "evaluation_only": True,
            "test_r32_confirmation_accessed": False,
        }

    def __len__(self) -> int:
        return len(self.rows)

    def _reader(self) -> R35RawImageReader:
        if self._reader_instance is None:
            self._reader_instance = R35RawImageReader(
                self.cache_manifests,
                self.cache_indexes,
            )
        return self._reader_instance

    def __getitem__(self, index: int) -> dict[str, Any]:
        category, row, hard_positive = self.rows[int(index)]
        candidate_ids = [str(value) for value in row["candidate_place_ids"]]
        valid = min(
            int(row.get("valid_candidate_count", len(candidate_ids))),
            len(candidate_ids),
            MAX_CANDIDATES,
        )
        if valid < 1:
            raise RuntimeError("Stage 3 Internal候选集合为空")
        candidate_ids = candidate_ids[:valid]
        missing = [value for value in candidate_ids if value not in self.anchors]
        if missing:
            raise KeyError(f"Stage 3 Internal candidate缺少anchor: {missing[:3]}")
        reader = self._reader()
        query = reader.read_uint8(str(row["query_sample_id"]))
        roll = float(row.get("roll_degrees", 0.0))
        if roll:
            query, _ = roll_tensor_by_degrees(query, roll)
        candidates = [reader.read_uint8(self.anchors[value]) for value in candidate_ids]
        candidates.extend([candidates[0]] * (MAX_CANDIDATES - valid))
        mask = torch.zeros(MAX_CANDIDATES, dtype=torch.bool)
        mask[:valid] = True
        same = torch.zeros(MAX_CANDIDATES, dtype=torch.bool)
        for value in row.get("positive_candidate_indices", []):
            positive_index = int(value)
            if 0 <= positive_index < valid:
                same[positive_index] = True
        if (category == "present") != bool(same.any()):
            raise RuntimeError("Stage 3 Internal present/absent标签矛盾")
        reciprocal = torch.zeros(MAX_CANDIDATES, dtype=torch.float32)
        reciprocal[:valid] = 1.0 / (
            torch.arange(valid, dtype=torch.float32) + 1.0
        )
        return {
            "query_images_uint8": query,
            "candidate_images_uint8": torch.stack(candidates),
            "candidate_mask": mask,
            "same_targets": same,
            "hard_negative_targets": mask & ~same,
            "hard_positive_set": torch.tensor(hard_positive, dtype=torch.bool),
            "reciprocal_rank_score": reciprocal,
            "category": category,
            "source_record_index": int(row["record_index"]),
            "scene_id": str(row["scene_id"]),
            "query_sample_id": str(row["query_sample_id"]),
            "source_place_id": str(row.get("source_place_id", "")),
            "candidate_place_ids": candidate_ids
            + [candidate_ids[0]] * (MAX_CANDIDATES - valid),
        }

    def __del__(self) -> None:
        reader = getattr(self, "_reader_instance", None)
        if reader is not None:
            reader.close()


def collate_r35_track_c_raw(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("Stage 3 raw collate不接受空batch")
    return {
        "query_images_uint8": torch.stack(
            [row["query_images_uint8"] for row in rows]
        ),
        "candidate_images_uint8": torch.stack(
            [row["candidate_images_uint8"] for row in rows]
        ),
        "candidate_mask": torch.stack([row["candidate_mask"] for row in rows]),
        "same_targets": torch.stack([row["same_targets"] for row in rows]),
        "hard_negative_targets": torch.stack(
            [row["hard_negative_targets"] for row in rows]
        ),
        "hard_positive_set": torch.stack(
            [row["hard_positive_set"] for row in rows]
        ),
        "reciprocal_rank_score": torch.stack(
            [row["reciprocal_rank_score"] for row in rows]
        ),
        "categories": [str(row["category"]) for row in rows],
        "source_record_indices": [int(row["source_record_index"]) for row in rows],
        "scene_ids": [str(row["scene_id"]) for row in rows],
        "query_sample_ids": [str(row["query_sample_id"]) for row in rows],
        "source_place_ids": [str(row["source_place_id"]) for row in rows],
        "candidate_place_ids": [list(row["candidate_place_ids"]) for row in rows],
    }


def raw_batch_sampling_acceptance(batch: dict[str, Any]) -> dict[str, Any]:
    counts = Counter(str(value) for value in batch.get("categories", []))
    set_count = int(batch["query_images_uint8"].shape[0])
    gates = {
        "all_seven_categories_present": set(counts) == set(LOGICAL_CATEGORIES),
        "equal_positive_preserving_category_counts": (
            len(set(counts.values())) == 1 and min(counts.values(), default=0) > 0
        ),
        "set_count_matches": sum(counts.values()) == set_count,
        "uint8_images": (
            batch["query_images_uint8"].dtype == torch.uint8
            and batch["candidate_images_uint8"].dtype == torch.uint8
        ),
        "candidate_shape_k16": tuple(batch["candidate_images_uint8"].shape[1:3])
        == (MAX_CANDIDATES, 3),
        "every_set_has_candidate": bool(batch["candidate_mask"].any(dim=1).all()),
        "hard_positive_marked": int(batch["hard_positive_set"].sum())
        == counts.get("hard_positive", 0),
    }
    return {
        "schema_version": "r35_stage3_raw_batch_sampling_acceptance_v1",
        "counts": dict(sorted(counts.items())),
        "gates": gates,
        "passed": all(gates.values()),
        "test_r32_confirmation_accessed": False,
    }


def prepare_raw_candidate_count(
    batch: dict[str, Any],
    candidate_count: int,
    generator: torch.Generator,
) -> dict[str, Any]:
    if candidate_count not in (4, 8, 16):
        raise ValueError("Stage 3 candidate_count必须是4、8或16")
    set_count = int(batch["query_images_uint8"].shape[0])
    permutation = torch.stack(
        [torch.randperm(candidate_count, generator=generator) for _ in range(set_count)]
    )
    result = dict(batch)
    for name in (
        "candidate_images_uint8",
        "candidate_mask",
        "same_targets",
        "hard_negative_targets",
        "reciprocal_rank_score",
    ):
        value = batch[name][:, :candidate_count]
        index = permutation
        for _ in range(value.ndim - 2):
            index = index.unsqueeze(-1)
        index = index.expand(-1, -1, *value.shape[2:])
        result[name] = value.gather(1, index)
    same = result["same_targets"] & result["candidate_mask"]
    target_present = same.any(dim=1)
    target_index = torch.full((set_count,), -1, dtype=torch.long)
    target_index[target_present] = same[target_present].float().argmax(dim=1)
    result.update(
        {
            "hard_positive_candidate_mask": (
                same & batch["hard_positive_set"].unsqueeze(1)
            ),
            "target_present": target_present,
            "target_candidate_index": target_index,
            "ordinary_positive_set": torch.tensor(
                [value == "ordinary_positive" for value in batch["categories"]],
                dtype=torch.bool,
            ),
            "near_wrong_set": torch.tensor(
                [value == "near_but_wrong" for value in batch["categories"]],
                dtype=torch.bool,
            ),
            "scene_group_id": torch.tensor(
                [
                    sorted(set(batch["scene_ids"])).index(scene_id)
                    for scene_id in batch["scene_ids"]
                ],
                dtype=torch.long,
            ),
            "scene_group_is_model_input": False,
            "candidate_count": candidate_count,
        }
    )
    return result


def normalize_uint8_images(
    images: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    if images.dtype != torch.uint8 or images.shape[-3:] != (3, 224, 448):
        raise ValueError("Stage 3 GPU归一化输入必须是uint8 ERP [*,3,224,448]")
    value = images.to(device=device, dtype=torch.float32, non_blocking=True).div_(255.0)
    leading = (1,) * (value.ndim - 3)
    mean = MEAN.to(device=device, dtype=value.dtype).reshape(*leading, 3, 1, 1)
    std = STD.to(device=device, dtype=value.dtype).reshape(*leading, 3, 1, 1)
    return (value - mean) / std
