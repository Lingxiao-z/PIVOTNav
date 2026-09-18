from __future__ import annotations

from collections import Counter
from typing import Any

import torch


FAR_LIMITS = {
    "boundary_1_1_25": 0.05,
    "near_1_25_1_5": 0.05,
    "early_1_5_2": 0.05,
    "wall_separated": 0.05,
    "repeated_texture": 0.05,
    "clear_absent": 0.0333,
    "yaw_roll_absent": 0.0333,
    "cross_scene_or_far": 0.0333,
}


def ratio(numerator: int, denominator: int) -> float:
    return float(numerator) / max(int(denominator), 1)


def compute_r36_goal_anchor_metrics(
    *,
    presence_probability: torch.Tensor,
    same_place_probability: torch.Tensor,
    same_targets: torch.Tensor,
    candidate_mask: torch.Tensor,
    confidence: torch.Tensor,
    categories: list[str],
    scene_ids: list[str],
    hard_positive: torch.Tensor,
    candidate_count: int,
    source_present_at_k16: torch.Tensor | None = None,
) -> dict[str, Any]:
    probability = presence_probability.float().cpu()
    same = same_place_probability.float().cpu()
    targets = same_targets.bool().cpu()
    mask = candidate_mask.bool().cpu()
    confidence = confidence.float().cpu()
    hard_positive = hard_positive.bool().cpu()
    if not all(bool(value.isfinite().all()) for value in (probability, same, confidence)):
        raise FloatingPointError("R36 Goal Anchor Development has non-finite output")
    present = (
        source_present_at_k16.bool().cpu()
        if source_present_at_k16 is not None
        else targets.any(dim=1)
    )
    selected = same.masked_fill(~mask, -1).argmax(dim=1)
    selected_correct = targets.gather(1, selected[:, None]).squeeze(1)
    accepted = probability >= 0.5
    true_positive = accepted & present & selected_correct
    tp = int(true_positive.sum()); predicted = int(accepted.sum()); positive = int(present.sum())
    precision = ratio(tp, predicted); recall = ratio(tp, positive)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    top1 = ratio(int((present & selected_correct).sum()), positive)
    hard_recall = ratio(int((accepted & selected_correct & hard_positive).sum()), int(hard_positive.sum()))
    counts = Counter(categories)
    far = {}
    for category in FAR_LIMITS:
        category_mask = torch.tensor([name == category for name in categories])
        far[category] = ratio(int((accepted & category_mask).sum()), counts[category])
    near_mask = torch.tensor([name in ("boundary_1_1_25", "near_1_25_1_5", "early_1_5_2", "wall_separated") for name in categories])
    near_far = ratio(int((accepted & near_mask).sum()), int(near_mask.sum()))

    per_scene = {}
    eligible_count = 0
    passed_count = 0
    for scene in sorted(set(scene_ids)):
        indices = torch.tensor([name == scene for name in scene_ids])
        scene_present = present & indices
        scene_accepted = accepted & indices
        scene_tp = true_positive & indices
        scene_precision = ratio(int(scene_tp.sum()), int(scene_accepted.sum()))
        scene_recall = ratio(int(scene_tp.sum()), int(scene_present.sum()))
        scene_f1 = 2 * scene_precision * scene_recall / max(scene_precision + scene_recall, 1e-12)
        scene_near = near_mask & indices
        scene_near_far = ratio(int((accepted & scene_near).sum()), int(scene_near.sum()))
        eligible = int(scene_present.sum()) >= 50 and int((indices & ~present).sum()) >= 50
        passed = eligible and scene_precision >= .95 and scene_recall >= .92 and scene_f1 >= .93 and scene_near_far <= .05
        eligible_count += int(eligible); passed_count += int(passed)
        per_scene[scene] = {
            "sample_count": int(indices.sum()), "present_count": int(scene_present.sum()),
            "precision": scene_precision, "recall": scene_recall, "f1": scene_f1,
            "near_but_wrong_far": scene_near_far, "eligible": eligible, "passed": passed,
        }
    scene_pass_fraction = ratio(passed_count, eligible_count)
    brier = float(((probability - present.float()) ** 2).mean())
    ece = 0.0
    for index in range(10):
        lower, upper = index / 10, (index + 1) / 10
        selected_bin = (probability >= lower) & ((probability <= upper) if index == 9 else (probability < upper))
        if bool(selected_bin.any()):
            ece += float(selected_bin.float().mean()) * abs(float(probability[selected_bin].mean()) - float(present[selected_bin].float().mean()))
    gates = {
        "present_top1_ge_99pct": top1 >= .99,
        "precision_ge_95pct": precision >= .95,
        "recall_ge_92pct": recall >= .92,
        "f1_ge_93pct": f1 >= .93,
        "near_but_wrong_far_le_5pct": near_far <= .05,
        "repeated_texture_far_le_5pct": far["repeated_texture"] <= .05,
        "clear_absent_far_le_3_33pct": far["clear_absent"] <= .0333,
        "yaw_roll_absent_far_le_3_33pct": far["yaw_roll_absent"] <= .0333,
        "cross_scene_far_le_3_33pct": far["cross_scene_or_far"] <= .0333,
        "at_least_80pct_scenes_pass_primary_gates": scene_pass_fraction >= .80,
        "all_outputs_finite": True,
    }
    excess = sum(max(0.0, far[name] - limit) for name, limit in FAR_LIMITS.items())
    score = f1 + .5 * recall + .25 * precision + .25 * top1 - 2.0 * excess - near_far
    return {
        "schema_version": "r36_goal_anchor_development_metrics_v1",
        "candidate_count": candidate_count, "fixed_presence_threshold": .5,
        "sample_count": len(categories), "precision": precision, "recall": recall, "f1": f1,
        "present_top1_accuracy": top1, "hard_positive_recall": hard_recall,
        "near_but_wrong_far": near_far, "far_by_category": far,
        "sample_count_by_category": dict(counts),
        "presence_calibration": {"brier_score": brier, "expected_calibration_error_10bin": ece},
        "per_scene": per_scene, "eligible_scene_count": eligible_count,
        "scene_pass_fraction": scene_pass_fraction, "gates": gates,
        "all_goal_anchor_gates_passed": all(gates.values()),
        "primary_checkpoint_selection_score": score,
        "checkpoint_selection_rule_zh": "固定阈值0.5和K16；按F1、Recall、Precision、Top-1及各类FAR惩罚选择，不使用阈值扫描。",
        "test_r32_confirmation_accessed": False,
    }
