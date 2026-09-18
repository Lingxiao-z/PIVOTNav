from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Iterable, Sequence

import torch

NEGATIVE_FAR_LIMITS = {
    "clear_absent": 0.0333,
    "yaw_roll_absent": 0.0333,
    "cross_scene_high_score": 0.0333,
    "near_but_wrong": 0.0833,
    "same_scene_repeated_texture": 0.0833,
}
DIAGNOSTIC_THRESHOLDS = tuple(index / 100.0 for index in range(101))


def _ratio(numerator: int, denominator: int) -> float:
    return float(numerator) / max(int(denominator), 1)


def _f1(precision: float, recall: float) -> float:
    return 2.0 * precision * recall / max(precision + recall, 1e-12)


def _binary_calibration(
    probability: torch.Tensor,
    target: torch.Tensor,
    bins: int = 10,
) -> dict[str, float]:
    probability = probability.float().clamp(0.0, 1.0)
    target = target.float()
    brier = float(((probability - target) ** 2).mean())
    ece = 0.0
    for index in range(bins):
        lower = index / bins
        upper = (index + 1) / bins
        mask = (
            (probability >= lower)
            & (
                probability <= upper
                if index == bins - 1
                else probability < upper
            )
        )
        if bool(mask.any()):
            weight = float(mask.float().mean())
            ece += weight * abs(
                float(probability[mask].mean())
                - float(target[mask].mean())
            )
    return {
        "brier_score": brier,
        "expected_calibration_error_10bin": ece,
    }


def _decision_counts(
    *,
    presence_probability: torch.Tensor,
    selected_correct: torch.Tensor,
    source_present: torch.Tensor,
    categories: Sequence[str],
    threshold: float,
) -> dict[str, Any]:
    accepted = presence_probability >= float(threshold)
    true_positive = accepted & source_present & selected_correct
    predicted_positive = accepted
    tp = int(true_positive.sum())
    predicted = int(predicted_positive.sum())
    present_count = int(source_present.sum())
    precision = _ratio(tp, predicted)
    recall = _ratio(tp, present_count)
    f1 = _f1(precision, recall)
    far: dict[str, float] = {}
    category_counts: Counter[str] = Counter(categories)
    for category in NEGATIVE_FAR_LIMITS:
        mask = torch.tensor(
            [value == category for value in categories],
            dtype=torch.bool,
        )
        far[category] = _ratio(
            int((accepted & mask).sum()),
            category_counts[category],
        )
    return {
        "threshold": float(threshold),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positive_count": tp,
        "predicted_present_count": predicted,
        "source_present_count": present_count,
        "present_false_negative_count": present_count - tp,
        "far_by_category": far,
    }


def compute_r35_stage2_metrics(
    *,
    presence_probability: torch.Tensor,
    same_place_probability: torch.Tensor,
    same_targets: torch.Tensor,
    candidate_mask: torch.Tensor,
    confidence: torch.Tensor,
    categories: Sequence[str],
    scene_ids: Sequence[str],
    hard_positive_set: torch.Tensor,
    candidate_count: int,
    fixed_presence_threshold: float = 0.5,
    diagnostic_thresholds: Iterable[float] = DIAGNOSTIC_THRESHOLDS,
) -> dict[str, Any]:
    count = int(presence_probability.numel())
    expected_set_shape = (count, int(candidate_count))
    if same_place_probability.shape != expected_set_shape:
        raise ValueError("same_place_probability shape不匹配")
    if same_targets.shape != expected_set_shape:
        raise ValueError("same_targets shape不匹配")
    if candidate_mask.shape != expected_set_shape:
        raise ValueError("candidate_mask shape不匹配")
    if confidence.shape != (count,) or hard_positive_set.shape != (count,):
        raise ValueError("confidence或hard_positive_set shape不匹配")
    if len(categories) != count or len(scene_ids) != count:
        raise ValueError("category/scene metadata数量不匹配")
    if count < 1 or not bool(candidate_mask.any(dim=1).all()):
        raise ValueError("每个集合必须至少有一个候选")
    tensors = (
        presence_probability,
        same_place_probability,
        confidence,
    )
    if not all(bool(torch.isfinite(value).all()) for value in tensors):
        raise FloatingPointError("Stage 2 Development输出含非有限值")
    presence_probability = presence_probability.float().cpu()
    same_place_probability = same_place_probability.float().cpu()
    same_targets = same_targets.bool().cpu()
    candidate_mask = candidate_mask.bool().cpu()
    confidence = confidence.float().cpu()
    hard_positive_set = hard_positive_set.bool().cpu()
    source_present = torch.tensor(
        [value == "present" for value in categories],
        dtype=torch.bool,
    )
    target_present_in_set = (same_targets & candidate_mask).any(dim=1)
    masked_probability = same_place_probability.masked_fill(
        ~candidate_mask,
        -1.0,
    )
    selected_index = masked_probability.argmax(dim=1)
    selected_correct = same_targets.gather(
        1,
        selected_index.unsqueeze(1),
    ).squeeze(1)
    selected_probability = masked_probability.gather(
        1,
        selected_index.unsqueeze(1),
    ).squeeze(1)
    fixed = _decision_counts(
        presence_probability=presence_probability,
        selected_correct=selected_correct,
        source_present=source_present,
        categories=categories,
        threshold=fixed_presence_threshold,
    )
    accepted = presence_probability >= float(fixed_presence_threshold)
    present_count = int(source_present.sum())
    present_top1 = _ratio(
        int((source_present & selected_correct).sum()),
        present_count,
    )
    present_retrieval_coverage = _ratio(
        int((source_present & target_present_in_set).sum()),
        present_count,
    )
    presence_only_recall = _ratio(
        int((source_present & accepted).sum()),
        present_count,
    )
    hard_mask = source_present & hard_positive_set
    hard_true_positive = (
        hard_mask & accepted & selected_correct
    )
    hard_positive_recall = _ratio(
        int(hard_true_positive.sum()),
        int(hard_mask.sum()),
    )
    decision_correct = (
        (accepted & source_present & selected_correct)
        | (~accepted & ~source_present)
    )
    calibration = {
        "presence_probability": _binary_calibration(
            presence_probability,
            source_present,
        ),
        "same_place_selected_probability": _binary_calibration(
            selected_probability.clamp(0.0, 1.0),
            source_present & selected_correct,
        ),
        "decision_confidence": _binary_calibration(
            confidence.clamp(0.0, 1.0),
            decision_correct,
        ),
    }
    curves = [
        _decision_counts(
            presence_probability=presence_probability,
            selected_correct=selected_correct,
            source_present=source_present,
            categories=categories,
            threshold=float(threshold),
        )
        for threshold in diagnostic_thresholds
    ]
    scene_indices: dict[str, list[int]] = defaultdict(list)
    for index, scene in enumerate(scene_ids):
        scene_indices[str(scene)].append(index)
    per_scene: dict[str, Any] = {}
    scene_stability_pass = True
    for scene, indices in sorted(scene_indices.items()):
        index = torch.tensor(indices, dtype=torch.long)
        scene_categories = [categories[value] for value in indices]
        scene_metrics = _decision_counts(
            presence_probability=presence_probability[index],
            selected_correct=selected_correct[index],
            source_present=source_present[index],
            categories=scene_categories,
            threshold=fixed_presence_threshold,
        )
        scene_present_count = int(source_present[index].sum())
        scene_negative_count = len(indices) - scene_present_count
        stable_eligible = (
            scene_present_count >= 50
            and scene_negative_count >= 50
        )
        stable = (
            not stable_eligible
            or (
                scene_metrics["precision"] >= 0.90
                and scene_metrics["recall"] >= 0.80
            )
        )
        scene_stability_pass &= stable
        per_scene[scene] = {
            **scene_metrics,
            "sample_count": len(indices),
            "present_count": scene_present_count,
            "negative_count": scene_negative_count,
            "stability_eligible": stable_eligible,
            "stability_pass": stable,
        }
    gates = {
        "precision_ge_95pct": fixed["precision"] >= 0.95,
        "recall_ge_90pct": fixed["recall"] >= 0.90,
        "f1_ge_92pct": fixed["f1"] >= 0.92,
        "present_top1_ge_95pct": present_top1 >= 0.95,
        "clear_absent_far_le_3_33pct": (
            fixed["far_by_category"]["clear_absent"]
            <= NEGATIVE_FAR_LIMITS["clear_absent"]
        ),
        "yaw_roll_absent_far_le_3_33pct": (
            fixed["far_by_category"]["yaw_roll_absent"]
            <= NEGATIVE_FAR_LIMITS["yaw_roll_absent"]
        ),
        "cross_scene_high_score_far_le_3_33pct": (
            fixed["far_by_category"]["cross_scene_high_score"]
            <= NEGATIVE_FAR_LIMITS["cross_scene_high_score"]
        ),
        "near_but_wrong_far_le_8_33pct": (
            fixed["far_by_category"]["near_but_wrong"]
            <= NEGATIVE_FAR_LIMITS["near_but_wrong"]
        ),
        "repeated_texture_far_le_8_33pct": (
            fixed["far_by_category"]["same_scene_repeated_texture"]
            <= NEGATIVE_FAR_LIMITS["same_scene_repeated_texture"]
        ),
        "scene_stability": scene_stability_pass,
        "all_outputs_finite": True,
    }
    far_excess = sum(
        max(
            0.0,
            fixed["far_by_category"][name] - limit,
        )
        for name, limit in NEGATIVE_FAR_LIMITS.items()
    )
    primary_score = (
        fixed["f1"]
        + 0.50 * fixed["recall"]
        + 0.25 * fixed["precision"]
        + 0.25 * present_top1
        - far_excess
    )
    return {
        "schema_version": "r35_stage2_development_metrics_v1",
        "candidate_count": int(candidate_count),
        "fixed_presence_threshold": float(fixed_presence_threshold),
        "fixed_threshold_metrics": fixed,
        "present_top1_accuracy": present_top1,
        "present_retrieval_coverage": present_retrieval_coverage,
        "presence_only_recall": presence_only_recall,
        "hard_positive_count": int(hard_mask.sum()),
        "hard_positive_recall": hard_positive_recall,
        "calibration": calibration,
        "diagnostic_curves_not_used_for_checkpoint_selection": curves,
        "diagnostic_threshold_grid": {
            "start": 0.0,
            "end": 1.0,
            "step": 0.01,
            "count": len(curves),
            "used_for_checkpoint_or_threshold_selection": False,
        },
        "per_scene": per_scene,
        "scene_count": len(per_scene),
        "gates": gates,
        "all_goal_anchor_gates_passed": all(gates.values()),
        "primary_checkpoint_selection_score": primary_score,
        "checkpoint_selection_rule_zh": (
            "固定presence threshold=0.5；按F1、Recall、Precision、present Top-1"
            "与超限FAR惩罚计算。诊断曲线不参与checkpoint选择。"
        ),
        "test_r32_confirmation_accessed": False,
    }
