from __future__ import annotations

import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, DistributedSampler

from .r35_bearing_head import circular_error_degrees_tensor
from .r35_multitask_system import R35MultitaskSystem
from .r35_stage2_metrics import compute_r35_stage2_metrics
from .r35_stage3_raw_data import (
    R35TrackCRawEvaluationDataset,
    collate_r35_track_c_raw,
    normalize_uint8_images,
    prepare_raw_candidate_count,
)


DISTANCE_NAMES = {
    0: "0.5-1m",
    1: "1-2m",
    2: "2-3m",
    3: "3-4m",
    4: "4-6m",
}


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _error_summary(errors: np.ndarray) -> dict[str, Any]:
    if errors.size == 0:
        return {
            "count": 0,
            "mae_degrees": None,
            "median_degrees": None,
            "p90_degrees": None,
            "accuracy_le_5deg": None,
            "accuracy_le_10deg": None,
            "accuracy_le_15deg": None,
            "accuracy_le_22_5deg": None,
            "catastrophic_gt_45deg_rate": None,
        }
    return {
        "count": int(errors.size),
        "mae_degrees": float(errors.mean()),
        "median_degrees": float(np.percentile(errors, 50)),
        "p90_degrees": float(np.percentile(errors, 90)),
        "accuracy_le_5deg": float((errors <= 5.0).mean()),
        "accuracy_le_10deg": float((errors <= 10.0).mean()),
        "accuracy_le_15deg": float((errors <= 15.0).mean()),
        "accuracy_le_22_5deg": float((errors <= 22.5).mean()),
        "catastrophic_gt_45deg_rate": float((errors > 45.0).mean()),
    }


def _shutdown_loader(loader: DataLoader) -> None:
    iterator = getattr(loader, "_iterator", None)
    shutdown = getattr(iterator, "_shutdown_workers", None)
    if callable(shutdown):
        shutdown()


def evaluate_bearing_and_track_y(
    *,
    student: R35MultitaskSystem,
    track_y_teacher: R35MultitaskSystem,
    bearing_development_manifest: str | Path,
    batch_per_rank: int,
    num_workers: int,
    prefetch_factor: int,
    rank: int,
    world: int,
    device: torch.device,
    global_step: int,
    seed: int,
) -> dict[str, Any]:
    from r35_bearing_data import R35BearingCompactDataset, collate_r35_bearing

    started = time.perf_counter()
    dataset = R35BearingCompactDataset(
        bearing_development_manifest,
        invalid_probability=0.0,
        seed=seed,
    )
    sampler = DistributedSampler(
        dataset,
        num_replicas=world,
        rank=rank,
        shuffle=False,
        drop_last=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_per_rank,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        collate_fn=collate_r35_bearing,
        drop_last=False,
    )
    local: dict[str, list[Any]] = defaultdict(list)
    student.eval()
    track_y_teacher.eval()
    try:
        with torch.inference_mode():
            for batch in loader:
                source = batch["source"].to(device, non_blocking=True)
                target = batch["target"].to(device, non_blocking=True)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    student_output = student.forward_pairs(source, target)
                    teacher_output = track_y_teacher.forward_pairs(source, target)
                bearing_error = circular_error_degrees_tensor(
                    student_output["bearing"]["bearing_angle_degrees"],
                    batch["bearing_degrees"].to(device, non_blocking=True),
                )
                yaw_target = torch.remainder(
                    batch["source_yaw_degrees"].to(device, non_blocking=True)
                    - batch["target_yaw_degrees"].to(device, non_blocking=True),
                    360.0,
                )
                student_yaw_error = circular_error_degrees_tensor(
                    student_output["yaw"]["predicted_yaw_degrees"], yaw_target
                )
                teacher_yaw_error = circular_error_degrees_tensor(
                    teacher_output["yaw"]["predicted_yaw_degrees"], yaw_target
                )
                local["bearing_error"].extend(
                    bearing_error.detach().float().cpu().tolist()
                )
                local["bearing_confidence"].extend(
                    student_output["bearing"]["bearing_confidence"]
                    .detach()
                    .float()
                    .cpu()
                    .tolist()
                )
                local["bearing_valid_probability"].extend(
                    student_output["bearing"]["bearing_valid_probability"]
                    .detach()
                    .float()
                    .cpu()
                    .tolist()
                )
                local["student_yaw_error"].extend(
                    student_yaw_error.detach().float().cpu().tolist()
                )
                local["teacher_yaw_error"].extend(
                    teacher_yaw_error.detach().float().cpu().tolist()
                )
                local["distance_bucket"].extend(batch["distance_bucket"].tolist())
                local["scene_index"].extend(batch["scene_index"].tolist())
    finally:
        _shutdown_loader(loader)
    gathered: list[dict[str, list[Any]] | None] = [None for _ in range(world)]
    torch.distributed.all_gather_object(gathered, dict(local))
    result = None
    if rank == 0:
        merged: dict[str, list[Any]] = defaultdict(list)
        for record in gathered:
            if record:
                for name, values in record.items():
                    merged[name].extend(values)
        bearing_error = np.asarray(merged["bearing_error"], dtype=np.float64)
        confidence = np.asarray(merged["bearing_confidence"], dtype=np.float64)
        valid_probability = np.asarray(
            merged["bearing_valid_probability"], dtype=np.float64
        )
        distance = np.asarray(merged["distance_bucket"], dtype=np.int64)
        student_yaw = np.asarray(merged["student_yaw_error"], dtype=np.float64)
        teacher_yaw = np.asarray(merged["teacher_yaw_error"], dtype=np.float64)
        by_distance = {
            DISTANCE_NAMES[index]: _error_summary(bearing_error[distance == index])
            for index in DISTANCE_NAMES
        }
        one_to_three = bearing_error[np.isin(distance, (1, 2))]
        high_confidence = confidence >= 0.8
        bearing_gates = {
            "distance_0_5_1m_accuracy_le_15_ge_95pct": by_distance["0.5-1m"][
                "accuracy_le_15deg"
            ]
            >= 0.95,
            "distance_1_2m_accuracy_le_15_ge_92pct": by_distance["1-2m"][
                "accuracy_le_15deg"
            ]
            >= 0.92,
            "distance_2_3m_accuracy_le_22_5_ge_90pct": by_distance["2-3m"][
                "accuracy_le_22_5deg"
            ]
            >= 0.90,
            "distance_3_4m_accuracy_le_22_5_ge_85pct": by_distance["3-4m"][
                "accuracy_le_22_5deg"
            ]
            >= 0.85,
            "distance_1_3m_p90_le_22_5deg": float(np.percentile(one_to_three, 90))
            <= 22.5,
            "high_confidence_catastrophic_gt_45_le_2pct": int(
                high_confidence.sum()
            )
            >= 100
            and float((bearing_error[high_confidence] > 45.0).mean()) <= 0.02,
            "bearing_valid_coverage_ge_80pct": float(
                (valid_probability >= 0.5).mean()
            )
            >= 0.80,
        }
        bearing_primary = (
            by_distance["0.5-1m"]["accuracy_le_15deg"]
            + by_distance["1-2m"]["accuracy_le_15deg"]
            + by_distance["2-3m"]["accuracy_le_22_5deg"]
            + by_distance["3-4m"]["accuracy_le_22_5deg"]
            - float(np.percentile(one_to_three, 90)) / 180.0
        )
        student_yaw_summary = _error_summary(student_yaw)
        teacher_yaw_summary = _error_summary(teacher_yaw)
        yaw_gates = {
            "mean_error_increase_le_0_5deg": student_yaw_summary["mae_degrees"]
            - teacher_yaw_summary["mae_degrees"]
            <= 0.5,
            "p90_error_increase_le_0_5deg": student_yaw_summary["p90_degrees"]
            - teacher_yaw_summary["p90_degrees"]
            <= 0.5,
            "accuracy_within_5deg_drop_le_0_5pct": teacher_yaw_summary[
                "accuracy_le_5deg"
            ]
            - student_yaw_summary["accuracy_le_5deg"]
            <= 0.005,
            "all_outputs_finite": bool(np.isfinite(student_yaw).all())
            and bool(np.isfinite(teacher_yaw).all()),
        }
        result = {
            "schema_version": "r35_stage3_bearing_track_y_development_v1",
            "created_at": _now(),
            "global_step": global_step,
            "sample_count": int(bearing_error.size),
            "bearing": {
                "overall": _error_summary(bearing_error),
                "by_distance": by_distance,
                "distance_1_3m": _error_summary(one_to_three),
                "high_confidence_threshold": 0.8,
                "high_confidence_count": int(high_confidence.sum()),
                "high_confidence_catastrophic_gt_45deg_rate": (
                    float((bearing_error[high_confidence] > 45.0).mean())
                    if high_confidence.any()
                    else None
                ),
                "bearing_valid_coverage_at_0_5": float(
                    (valid_probability >= 0.5).mean()
                ),
                "mean_confidence": float(confidence.mean()),
                "gates": bearing_gates,
                "all_bearing_gates_passed": all(bearing_gates.values()),
                "primary_checkpoint_selection_score": float(bearing_primary),
            },
            "track_y_protection": {
                "student": student_yaw_summary,
                "frozen_track_y_teacher": teacher_yaw_summary,
                "mean_error_increase_degrees": student_yaw_summary["mae_degrees"]
                - teacher_yaw_summary["mae_degrees"],
                "p90_error_increase_degrees": student_yaw_summary["p90_degrees"]
                - teacher_yaw_summary["p90_degrees"],
                "accuracy_within_5deg_drop": teacher_yaw_summary[
                    "accuracy_le_5deg"
                ]
                - student_yaw_summary["accuracy_le_5deg"],
                "gates": yaw_gates,
                "all_track_y_protection_gates_passed": all(yaw_gates.values()),
            },
            "full_scene_disjoint_development": True,
            "elapsed_seconds": time.perf_counter() - started,
            "test_r32_confirmation_accessed": False,
        }
    container = [result]
    torch.distributed.broadcast_object_list(container, src=0)
    student.train()
    return container[0]


def _retrieval_metrics(
    similarity: torch.Tensor,
    same_targets: torch.Tensor,
    candidate_mask: torch.Tensor,
    categories: list[str],
) -> dict[str, Any]:
    source_present = torch.tensor(
        [value == "present" for value in categories], dtype=torch.bool
    )
    eligible = source_present & (same_targets & candidate_mask).any(dim=1)
    masked = similarity.float().masked_fill(~candidate_mask, -math.inf)
    ranking = masked.argsort(dim=1, descending=True)
    target_at_rank = same_targets.gather(1, ranking)
    recall_at_1 = float(target_at_rank[eligible, :1].any(dim=1).float().mean())
    recall_at_5 = float(target_at_rank[eligible, :5].any(dim=1).float().mean())
    return {
        "eligible_present_count": int(eligible.sum()),
        "recall_at_1": recall_at_1,
        "recall_at_5": recall_at_5,
    }


def evaluate_goal_anchor_and_vpr(
    *,
    student: R35MultitaskSystem,
    r3_teacher: torch.nn.Module,
    raw_data_audit: str | Path,
    internal_hard_positive_manifest: str | Path,
    batch_sets_per_rank: int,
    num_workers: int,
    prefetch_factor: int,
    rank: int,
    world: int,
    device: torch.device,
    global_step: int,
    seed: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    dataset = R35TrackCRawEvaluationDataset(
        raw_data_audit=raw_data_audit,
        internal_hard_positive_manifest=internal_hard_positive_manifest,
    )
    sampler = DistributedSampler(
        dataset,
        num_replicas=world,
        rank=rank,
        shuffle=False,
        drop_last=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_sets_per_rank,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        collate_fn=collate_r35_track_c_raw,
        drop_last=False,
    )
    local: dict[int, dict[str, Any]] = {
        candidate_count: {
            "presence": [],
            "same_probability": [],
            "same_targets": [],
            "mask": [],
            "confidence": [],
            "hard_positive": [],
            "categories": [],
            "scene_ids": [],
            "student_similarity": [],
            "teacher_similarity": [],
        }
        for candidate_count in (4, 8, 16)
    }
    student.eval()
    r3_teacher.eval()
    try:
        with torch.inference_mode():
            for batch_index, raw in enumerate(loader):
                for candidate_count in (4, 8, 16):
                    generator = torch.Generator().manual_seed(
                        seed
                        + global_step * 31
                        + candidate_count * 100_003
                        + rank * 10_007
                        + batch_index
                    )
                    selected = prepare_raw_candidate_count(
                        raw, candidate_count, generator
                    )
                    query = normalize_uint8_images(
                        selected["query_images_uint8"], device
                    )
                    candidates = normalize_uint8_images(
                        selected["candidate_images_uint8"], device
                    )
                    mask = selected["candidate_mask"].to(
                        device, non_blocking=True
                    )
                    reciprocal = selected["reciprocal_rank_score"].to(
                        device, non_blocking=True
                    )
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        output = student.match_candidate_set(
                            query,
                            candidates,
                            candidate_mask=mask,
                            reciprocal_rank_score=reciprocal,
                        )
                        flat = candidates.flatten(0, 1)
                        teacher_tokens = r3_teacher.backbone.forward_tokens(
                            torch.cat((query, flat), dim=0)
                        )
                        teacher_descriptor = (
                            r3_teacher.descriptor_head.forward_tokens(
                                teacher_tokens.float()
                            )["global"]
                        )
                    set_count = query.shape[0]
                    teacher_query = teacher_descriptor[:set_count]
                    teacher_candidate = teacher_descriptor[set_count:].reshape(
                        set_count, candidate_count, -1
                    )
                    teacher_similarity = F.cosine_similarity(
                        teacher_query[:, None].expand_as(teacher_candidate),
                        teacher_candidate,
                        dim=-1,
                    )
                    student_similarity = F.cosine_similarity(
                        output["query_encoding"]["global"][:, None].expand_as(
                            output["candidate_encoding"]["global"]
                        ),
                        output["candidate_encoding"]["global"],
                        dim=-1,
                    )
                    record = local[candidate_count]
                    record["presence"].append(
                        output["set_presence"]["presence_probability"]
                        .float()
                        .cpu()
                    )
                    record["same_probability"].append(
                        output["candidate_match"]["same_place_probability"]
                        .float()
                        .cpu()
                    )
                    record["same_targets"].append(
                        selected["same_targets"].cpu()
                    )
                    record["mask"].append(selected["candidate_mask"].cpu())
                    record["confidence"].append(
                        output["set_presence"]["confidence"].float().cpu()
                    )
                    record["hard_positive"].append(
                        selected["hard_positive_set"].cpu()
                    )
                    record["student_similarity"].append(
                        student_similarity.float().cpu()
                    )
                    record["teacher_similarity"].append(
                        teacher_similarity.float().cpu()
                    )
                    record["categories"].extend(selected["categories"])
                    record["scene_ids"].extend(selected["scene_ids"])
    finally:
        _shutdown_loader(loader)
    compact: dict[int, dict[str, Any]] = {}
    for candidate_count, record in local.items():
        compact[candidate_count] = {
            name: torch.cat(values, dim=0)
            for name, values in record.items()
            if name not in ("categories", "scene_ids")
        }
        compact[candidate_count]["categories"] = record["categories"]
        compact[candidate_count]["scene_ids"] = record["scene_ids"]
    gathered: list[dict[int, dict[str, Any]] | None] = [None for _ in range(world)]
    torch.distributed.all_gather_object(gathered, compact)
    result = None
    if rank == 0:
        by_candidate_count: dict[str, Any] = {}
        vpr_by_candidate_count: dict[str, Any] = {}
        for candidate_count in (4, 8, 16):
            merged = {
                name: torch.cat(
                    [record[candidate_count][name] for record in gathered if record],
                    dim=0,
                )
                for name in (
                    "presence",
                    "same_probability",
                    "same_targets",
                    "mask",
                    "confidence",
                    "hard_positive",
                    "student_similarity",
                    "teacher_similarity",
                )
            }
            categories = [
                value
                for record in gathered
                if record
                for value in record[candidate_count]["categories"]
            ]
            scene_ids = [
                value
                for record in gathered
                if record
                for value in record[candidate_count]["scene_ids"]
            ]
            metrics = compute_r35_stage2_metrics(
                presence_probability=merged["presence"],
                same_place_probability=merged["same_probability"],
                same_targets=merged["same_targets"],
                candidate_mask=merged["mask"],
                confidence=merged["confidence"],
                categories=categories,
                scene_ids=scene_ids,
                hard_positive_set=merged["hard_positive"],
                candidate_count=candidate_count,
                fixed_presence_threshold=0.5,
            )
            metrics["global_step"] = global_step
            by_candidate_count[str(candidate_count)] = metrics
            student_retrieval = _retrieval_metrics(
                merged["student_similarity"],
                merged["same_targets"],
                merged["mask"],
                categories,
            )
            teacher_retrieval = _retrieval_metrics(
                merged["teacher_similarity"],
                merged["same_targets"],
                merged["mask"],
                categories,
            )
            vpr_by_candidate_count[str(candidate_count)] = {
                "student": student_retrieval,
                "frozen_r3_teacher": teacher_retrieval,
                "recall_at_1_drop": teacher_retrieval["recall_at_1"]
                - student_retrieval["recall_at_1"],
                "recall_at_5_drop": teacher_retrieval["recall_at_5"]
                - student_retrieval["recall_at_5"],
            }
        vpr_k16 = vpr_by_candidate_count["16"]
        vpr_gates = {
            "recall_at_1_drop_le_0_5pct": vpr_k16["recall_at_1_drop"] <= 0.005,
            "recall_at_5_drop_le_0_5pct": vpr_k16["recall_at_5_drop"] <= 0.005,
            "all_outputs_finite": all(
                math.isfinite(float(value))
                for side in ("student", "frozen_r3_teacher")
                for name, value in vpr_k16[side].items()
                if name.startswith("recall_at_")
            ),
        }
        result = {
            "schema_version": "r35_stage3_goal_anchor_vpr_development_v1",
            "created_at": _now(),
            "global_step": global_step,
            "goal_anchor_by_candidate_count": by_candidate_count,
            "checkpoint_selection_candidate_count": 16,
            "all_goal_anchor_gates_passed_k16": by_candidate_count["16"][
                "all_goal_anchor_gates_passed"
            ],
            "goal_anchor_primary_checkpoint_selection_score": by_candidate_count[
                "16"
            ]["primary_checkpoint_selection_score"],
            "vpr_protection_by_candidate_count": vpr_by_candidate_count,
            "vpr_protection_gates": vpr_gates,
            "all_vpr_protection_gates_passed": all(vpr_gates.values()),
            "full_internal_development": True,
            "fixed_presence_threshold": 0.5,
            "diagnostic_thresholds_used_for_selection": False,
            "raw_evaluation_provenance": dataset.provenance,
            "elapsed_seconds": time.perf_counter() - started,
            "test_r32_confirmation_accessed": False,
        }
    container = [result]
    torch.distributed.broadcast_object_list(container, src=0)
    student.train()
    return container[0]


def combine_stage3_development(
    bearing_track_y: dict[str, Any],
    goal_anchor_vpr: dict[str, Any],
) -> dict[str, Any]:
    protection_passed = (
        bearing_track_y["track_y_protection"][
            "all_track_y_protection_gates_passed"
        ]
        and goal_anchor_vpr["all_vpr_protection_gates_passed"]
    )
    task_passed = (
        bearing_track_y["bearing"]["all_bearing_gates_passed"]
        and goal_anchor_vpr["all_goal_anchor_gates_passed_k16"]
    )
    eligible = protection_passed and task_passed
    score = (
        float(
            bearing_track_y["bearing"][
                "primary_checkpoint_selection_score"
            ]
        )
        + float(
            goal_anchor_vpr[
                "goal_anchor_primary_checkpoint_selection_score"
            ]
        )
        if protection_passed
        else None
    )
    return {
        "schema_version": "r35_stage3_full_development_v1",
        "created_at": _now(),
        "global_step": int(bearing_track_y["global_step"]),
        "bearing_and_track_y": bearing_track_y,
        "goal_anchor_and_vpr": goal_anchor_vpr,
        "bearing_gates_passed": bearing_track_y["bearing"][
            "all_bearing_gates_passed"
        ],
        "goal_anchor_gates_passed": goal_anchor_vpr[
            "all_goal_anchor_gates_passed_k16"
        ],
        "track_y_protection_passed": bearing_track_y["track_y_protection"][
            "all_track_y_protection_gates_passed"
        ],
        "vpr_protection_passed": goal_anchor_vpr[
            "all_vpr_protection_gates_passed"
        ],
        "all_protection_gates_passed": protection_passed,
        "all_task_gates_passed": task_passed,
        "eligible_for_stage3_checkpoint_selection": eligible,
        "primary_checkpoint_selection_score": score,
        "selection_rule_zh": (
            "先要求VPR和Track Y保护门槛全部通过，再按Bearing与Goal Anchor冻结主分数之和选择；"
            "最终授权还要求两个任务门槛均通过。"
        ),
        "test_r32_confirmation_accessed": False,
    }
