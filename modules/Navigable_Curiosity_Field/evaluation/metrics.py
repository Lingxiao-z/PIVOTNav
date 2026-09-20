from __future__ import annotations

from collections import defaultdict
import numpy as np


def _binary_metrics(prediction: np.ndarray, target: np.ndarray) -> dict:
    prediction = prediction.astype(bool)
    target = target.astype(bool)
    tp = int(np.logical_and(prediction, target).sum())
    tn = int(np.logical_and(~prediction, ~target).sum())
    fp = int(np.logical_and(prediction, ~target).sum())
    fn = int(np.logical_and(~prediction, target).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    negative_precision = tn / max(tn + fn, 1)
    negative_recall = tn / max(tn + fp, 1)
    negative_f1 = 2 * negative_precision * negative_recall / max(negative_precision + negative_recall, 1e-12)
    return {
        "accuracy": (tp + tn) / max(tp + tn + fp + fn, 1),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "macro_f1": (f1 + negative_f1) / 2,
        "confusion_matrix": [[tn, fp], [fn, tp]],
    }


def fg_metrics(logits, target, scene_ids) -> dict:
    logits = np.asarray(logits, dtype=np.float64)
    target = np.asarray(target, dtype=bool)
    if logits.shape != target.shape or logits.ndim != 2 or logits.shape[1] != 12:
        raise ValueError("FG logits/target must have matching [N,12] shapes")
    scene_ids = np.asarray(scene_ids)
    if len(scene_ids) != len(logits):
        raise ValueError("scene_ids length mismatch")
    probability = 1.0 / (1.0 + np.exp(-np.clip(logits, -80.0, 80.0)))
    prediction = probability >= 0.5
    result = _binary_metrics(prediction, target)
    result["per_direction_f1"] = [_binary_metrics(prediction[:, i], target[:, i])["f1"] for i in range(12)]
    result["per_scene_f1"] = {
        str(scene): _binary_metrics(prediction[scene_ids == scene], target[scene_ids == scene])["f1"]
        for scene in np.unique(scene_ids)
    }
    result["probability_range"] = [float(probability.min()), float(probability.max())]
    return result


def _rankdata(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def _spearman_1d(prediction, target) -> float | None:
    p = np.asarray(prediction)
    t = np.asarray(target)
    if len(p) < 2:
        return None
    p_rank, t_rank = _rankdata(p), _rankdata(t)
    if p_rank.std() == 0.0 or t_rank.std() == 0.0:
        return None
    corr = np.corrcoef(p_rank, t_rank)[0, 1]
    return float(corr) if np.isfinite(corr) else None


def _mean_row_spearman(prediction, target, valid) -> float:
    correlations = []
    for prediction_row, target_row, valid_row in zip(prediction, target, valid):
        correlation = _spearman_1d(prediction_row[valid_row], target_row[valid_row])
        if correlation is not None:
            correlations.append(correlation)
    return float(np.mean(correlations)) if correlations else 0.0


def _fs_summary(prediction, target, valid) -> dict:
    valid = valid.astype(bool)
    errors = prediction[valid] - target[valid]
    top1, regrets, top3, adjacent = [], [], [], []
    for p, t, m in zip(prediction, target, valid):
        indices = np.flatnonzero(m)
        if not len(indices):
            continue
        chosen_i = int(indices[np.argmax(p[indices])])
        best_score = float(np.max(t[indices]))
        best_indices = indices[np.isclose(t[indices], best_score, rtol=0.0, atol=1e-7)]
        top1.append(bool(np.any(best_indices == chosen_i)))
        regrets.append(float(best_score - t[chosen_i]))
        top3_count = min(3, len(indices))
        top3_indices = indices[np.argsort(p[indices], kind="stable")[-top3_count:]]
        top3.append(bool(np.intersect1d(best_indices, top3_indices, assume_unique=True).size))
        delta = np.minimum((chosen_i - best_indices) % 12, (best_indices - chosen_i) % 12)
        adjacent.append(bool(np.min(delta) <= 1))
    return {
        "masked_mse": float(np.square(errors).mean()) if len(errors) else 0.0,
        "mae": float(np.abs(errors).mean()) if len(errors) else 0.0,
        "top1": float(np.mean(top1)) if top1 else 0.0,
        "top3_recall": float(np.mean(top3)) if top3 else 0.0,
        "adjacent_accuracy": float(np.mean(adjacent)) if adjacent else 0.0,
        "mean_regret": float(np.mean(regrets)) if regrets else 0.0,
        "p90_regret": float(np.percentile(regrets, 90)) if regrets else 0.0,
        "spearman": _mean_row_spearman(prediction, target, valid),
        "evaluated_rows": int(valid.any(axis=1).sum()),
        "no_valid_direction_rows": int((~valid.any(axis=1)).sum()),
    }


def _fs_direction_summary(prediction, target, valid) -> dict:
    prediction = np.asarray(prediction)[valid]
    target = np.asarray(target)[valid]
    errors = prediction - target
    correlation = _spearman_1d(prediction, target)
    return {
        "valid_count": int(len(errors)),
        "masked_mse": float(np.square(errors).mean()) if len(errors) else None,
        "mae": float(np.abs(errors).mean()) if len(errors) else None,
        "mean_error": float(errors.mean()) if len(errors) else None,
        "spearman_across_samples": float(correlation) if correlation is not None else 0.0,
    }


def fs_metrics(prediction, target, valid, scene_ids) -> dict:
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    if prediction.shape != target.shape or valid.shape != target.shape or prediction.ndim != 2 or prediction.shape[1] != 12:
        raise ValueError("FS prediction/target/mask must have matching [N,12] shapes")
    scene_ids = np.asarray(scene_ids)
    result = _fs_summary(prediction, target, valid)
    result["per_direction"] = [
        _fs_direction_summary(prediction[:, i], target[:, i], valid[:, i]) for i in range(12)
    ]
    result["per_scene"] = {
        str(scene): _fs_summary(prediction[scene_ids == scene], target[scene_ids == scene], valid[scene_ids == scene])
        for scene in np.unique(scene_ids)
    }
    result["score_range"] = [float(prediction.min()), float(prediction.max())]
    return result


def joint_output_audit(fg_logits, fs_scores, loss_components, gradient_norms) -> dict:
    fg_logits = np.asarray(fg_logits)
    fs_scores = np.asarray(fs_scores)
    finite = bool(np.isfinite(fg_logits).all() and np.isfinite(fs_scores).all())
    fg_probability = 1.0 / (1.0 + np.exp(-np.clip(fg_logits, -80.0, 80.0)))
    classes = fg_probability >= 0.5
    if fg_logits.ndim != 2 or fs_scores.ndim != 2 or fg_logits.shape[0] != fs_scores.shape[0]:
        raise ValueError("FG logits and FS scores must have matching [N,D] batch dimensions")
    if len(fg_logits) > 1:
        fg_between_sample_std = float(fg_probability.std(axis=0).mean())
        fs_between_sample_std = float(fs_scores.std(axis=0).mean())
        fg_sample_invariant = fg_between_sample_std < 1e-5
        fs_sample_invariant = fs_between_sample_std < 1e-5
    else:
        fg_between_sample_std = None
        fs_between_sample_std = None
        fg_sample_invariant = False
        fs_sample_invariant = False
    collapsed = bool(
        classes.all() or (~classes).all() or fs_scores.std() < 1e-5
        or fg_sample_invariant or fs_sample_invariant
    )
    return {
        "finite_outputs": finite,
        "nan_count": int(np.isnan(fg_logits).sum() + np.isnan(fs_scores).sum()),
        "inf_count": int(np.isinf(fg_logits).sum() + np.isinf(fs_scores).sum()),
        "fg_probability_range": [float(fg_probability.min()), float(fg_probability.max())],
        "fs_score_range": [float(fs_scores.min()), float(fs_scores.max())],
        "fg_between_sample_std_mean": fg_between_sample_std,
        "fs_between_sample_std_mean": fs_between_sample_std,
        "fg_sample_invariant": fg_sample_invariant,
        "fs_sample_invariant": fs_sample_invariant,
        "prediction_collapsed": collapsed,
        "loss_components": {str(k): float(v) for k, v in loss_components.items()},
        "gradient_norms": {str(k): float(v) for k, v in gradient_norms.items()},
        "all_gradient_norms_finite": all(np.isfinite(float(v)) for v in gradient_norms.values()),
        "all_required_gradient_norms_nonzero": all(float(v) > 0 for v in gradient_norms.values()),
    }
