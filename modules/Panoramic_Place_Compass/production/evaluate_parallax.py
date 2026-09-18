from __future__ import annotations

import hashlib
import json
import math
import os
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor


ROOT = Path(os.environ.get("PIVOTNAV_ARRIVAL_PARALLAX_ROOT", Path(__file__).resolve().parent))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def wrap_degrees(value: float) -> float:
    return (float(value) + 180.0) % 360.0 - 180.0


def classification(labels: np.ndarray, predictions: np.ndarray) -> dict:
    labels = labels.astype(bool)
    predictions = predictions.astype(bool)
    tp = int(np.sum(labels & predictions))
    fp = int(np.sum(~labels & predictions))
    fn = int(np.sum(labels & ~predictions))
    tn = int(np.sum(~labels & ~predictions))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": precision, "recall": recall, "f1": f1}


def feature_names() -> list[str]:
    names = []
    for view in ("restore", "approach1", "approach2"):
        for field in (
            "raw_matches", "essential_inliers", "essential_ratio",
            "homography_inlier_ratio", "homography_scale",
            "flow_mean", "flow_median", "flow_p25", "flow_p75", "flow_p90",
            "flow_x_median", "flow_y_median",
        ):
            names.append(f"{view}_{field}")
    names.extend([
        "flow_decrease_01", "flow_decrease_12",
        "distance_estimate_1", "distance_estimate_2", "distance_estimate_median",
        "monotonic_flow_decrease_count",
        "current_arrival_probability", "current_vpr_similarity",
        "current_e_inliers", "current_e_inlier_ratio", "current_grid_coverage",
        "current_hull_area", "current_reprojection_error",
        "current_supported_sectors", "current_h_dominance",
        "target_top1_fraction", "target_top1_current",
    ])
    return names


def make_features(parallax: dict, row: dict) -> np.ndarray:
    common_sector = int(parallax["causal"]["common_sector_id"])
    values = []
    for view in parallax["views"]:
        sector = view["sectors"][common_sector]
        values.extend([
            sector["raw_matches"], sector["essential_inliers"], sector["essential_ratio"],
            sector["homography_inlier_ratio"], sector["homography_scale"],
            sector["flow"]["mean"], sector["flow"]["median"],
            sector["flow"]["p25"], sector["flow"]["p75"], sector["flow"]["p90"],
            sector["flow_x"]["median"], sector["flow_y"]["median"],
        ])
    causal = parallax["causal"]
    estimates = [min(float(value), 10.0) for value in causal["remaining_distance_estimate_sequence_m"]]
    evidence = row["views"][-1]["evidence"]
    top1 = [bool(view["target_top1"]) for view in row["views"]]
    values.extend([
        *causal["flow_decrease_sequence"], *estimates,
        min(float(causal["remaining_distance_estimate_median_m"]), 10.0),
        causal["monotonic_flow_decrease_count"],
        evidence["arrival_probability"], evidence["vpr_similarity"],
        evidence["e_inliers"], evidence["e_inlier_ratio"], evidence["grid_coverage"],
        evidence["hull_area"], evidence["reprojection_error"],
        evidence["supported_sectors"], evidence["h_dominance"],
        np.mean(top1), float(top1[-1]),
    ])
    result = np.asarray(values, dtype=np.float32)
    if len(result) != len(feature_names()) or not np.isfinite(result).all():
        raise ValueError(f"invalid parallax feature vector for {row['trial_id']}")
    return result


def hard_safety(row: dict, config: dict) -> tuple[bool, list[str]]:
    by_label = {view["view_label"]: view for view in row["views"]}
    required = ["ORIGINAL", "LEFT_10", "RIGHT_20", "RESTORE_10"]
    reasons = []
    if [view["view_label"] for view in row["views"]][:4] != required:
        reasons.append("missing_rotation_cycle")
    else:
        original = float(by_label["ORIGINAL"]["r361"]["yaw_degrees"])
        errors = [
            abs(wrap_degrees(float(by_label["LEFT_10"]["r361"]["yaw_degrees"]) - original + 10.0)),
            abs(wrap_degrees(float(by_label["RIGHT_20"]["r361"]["yaw_degrees"]) - original - 10.0)),
            abs(wrap_degrees(float(by_label["RESTORE_10"]["r361"]["yaw_degrees"]) - original)),
        ]
        if max(errors) > config["maximum_yaw_cycle_error_degrees"]:
            reasons.append("yaw_cycle_inconsistent")
    evidence = row["views"][-1]["evidence"]
    if int(evidence["e_inliers"]) < config["minimum_current_e_inliers"]:
        reasons.append("insufficient_current_inliers")
    if int(evidence["supported_sectors"]) < config["minimum_current_supported_sectors"]:
        reasons.append("insufficient_current_sector_support")
    return not reasons, reasons


def build_model(config: dict, seed: int) -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=config["n_estimators"],
        max_depth=config["max_depth"],
        min_samples_leaf=config["min_samples_leaf"],
        max_features=config["max_features"],
        bootstrap=config["bootstrap"],
        n_jobs=config["n_jobs"],
        random_state=seed,
    )


def main() -> None:
    protocol_path = ROOT / "model_protocol.json"
    protocol = json.loads(protocol_path.read_text())
    metrics_path = ROOT / "development_parallax_oof_metrics.json"
    predictions_path = ROOT / "development_parallax_oof_predictions.jsonl"
    model_dir = ROOT / "models"
    if metrics_path.exists() or predictions_path.exists() or model_dir.exists():
        raise RuntimeError("refuse to overwrite parallax OOF")
    for key in ("parallax_features", "active_evidence", "source_manifest", "extraction_manifest", "evaluator"):
        record = protocol[key]
        if sha256(Path(record["path"])) != record["sha256"]:
            raise RuntimeError(f"frozen input changed: {key}")
    parallax_rows = read_jsonl(Path(protocol["parallax_features"]["path"]))
    evidence_rows = read_jsonl(Path(protocol["active_evidence"]["path"]))
    evidence_by_trial = {row["trial_id"]: row for row in evidence_rows}
    source_rows = read_jsonl(Path(protocol["source_manifest"]["path"]))
    scene_by_trial = {row["trial_id"]: row["scene_id_audit_only"] for row in source_rows}
    rows = [evidence_by_trial[row["trial_id"]] for row in parallax_rows]
    trial_ids = [row["trial_id"] for row in rows]
    if len(trial_ids) != len(set(trial_ids)):
        raise RuntimeError("duplicate parallax trial ids")
    features = np.stack([make_features(parallax, row) for parallax, row in zip(parallax_rows, rows)])
    distances = np.asarray([row["final_horizontal_distance_m_gt_audit_only"] for row in rows])
    labels = distances <= protocol["model"]["arrival_distance_threshold_m"]
    scenes = sorted({scene_by_trial[trial_id] for trial_id in trial_ids})
    folds = [scenes[index:index + 2] for index in range(0, len(scenes), 2)]
    if len(folds) != protocol["model"]["fold_count"] or any(
        len(fold) != protocol["model"]["test_scenes_per_fold"] for fold in folds
    ):
        raise RuntimeError("scene folds differ from frozen protocol")
    predicted_distance_mean = np.zeros(len(rows))
    predicted_distance_p75 = np.zeros(len(rows))
    predictions = np.zeros(len(rows), dtype=bool)
    safety_reasons = [[] for _ in rows]
    fold_records = []
    model_dir.mkdir()
    for fold_index, test_scenes in enumerate(folds):
        test_indices = [i for i, trial_id in enumerate(trial_ids) if scene_by_trial[trial_id] in test_scenes]
        train_indices = [i for i in range(len(rows)) if i not in test_indices]
        model = build_model(protocol["model"], protocol["model"]["random_state"] + fold_index)
        model.fit(features[train_indices], distances[train_indices])
        tree_predictions = np.stack([
            estimator.predict(features[test_indices]) for estimator in model.estimators_
        ])
        predicted_distance_mean[test_indices] = tree_predictions.mean(0)
        predicted_distance_p75[test_indices] = np.quantile(
            tree_predictions, protocol["model"]["distance_prediction_quantile"], axis=0
        )
        for data_index in test_indices:
            safe, reasons = hard_safety(rows[data_index], protocol["hard_safety"])
            safety_reasons[data_index] = reasons
            predictions[data_index] = bool(
                predicted_distance_p75[data_index] <= protocol["model"]["arrival_distance_threshold_m"]
                and safe
            )
        model_path = model_dir / f"fold_{fold_index}.joblib"
        joblib.dump(model, model_path)
        fold_records.append({
            "fold": fold_index,
            "test_scenes": test_scenes,
            "test_count": len(test_indices),
            "metrics": classification(labels[test_indices], predictions[test_indices]),
            "distance_mae_m": float(np.mean(np.abs(predicted_distance_p75[test_indices] - distances[test_indices]))),
            "model": str(model_path),
            "model_sha256": sha256(model_path),
        })
    overall = classification(labels, predictions)
    by_category = {}
    for category in sorted({row["category_pre_registered"] for row in rows}):
        indices = [i for i, row in enumerate(rows) if row["category_pre_registered"] == category]
        by_category[category] = classification(labels[indices], predictions[indices])
    gates_spec = protocol["acceptance_gates"]
    false_indices = [i for i in range(len(rows)) if predictions[i] and not labels[i]]
    gates = {
        "precision": overall["precision"] >= gates_spec["precision_min"],
        "recall": overall["recall"] >= gates_spec["recall_min"],
        "f1": overall["f1"] >= gates_spec["f1_min"],
        "false_final_stop": len(false_indices) <= gates_spec["false_final_stop_max"],
        "repeated_texture": by_category["repeated_texture"]["fp"] == 0,
        "through_wall": by_category["through_wall_or_detour"]["fp"] == 0,
    }
    full_model = build_model(protocol["model"], protocol["model"]["random_state"] + 100)
    full_model.fit(features, distances)
    full_model_path = model_dir / "development_full.joblib"
    joblib.dump(full_model, full_model_path)
    with predictions_path.open("x") as stream:
        for index, row in enumerate(rows):
            stream.write(json.dumps({
                "trial_id": row["trial_id"],
                "scene_group_gt_audit_only": scene_by_trial[row["trial_id"]],
                "category_gt_audit_only": row["category_pre_registered"],
                "distance_m_gt_audit_only": float(distances[index]),
                "label_gt_audit_only": bool(labels[index]),
                "predicted_distance_mean_m": float(predicted_distance_mean[index]),
                "predicted_distance_p75_m": float(predicted_distance_p75[index]),
                "hard_safety_reasons": safety_reasons[index],
                "final_confirmed": bool(predictions[index]),
                "runtime_gt_inputs": [],
            }, sort_keys=True) + "\n")
    metrics = {
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": sha256(protocol_path),
        "status": "PASS" if all(gates.values()) else "FAIL",
        "status_scope": "DEVELOPMENT_PARALLAX_FEASIBILITY_ONLY",
        "independent_validation_passed": False,
        "production_approved": False,
        "scene_count": len(scenes),
        "opportunity_count": len(rows),
        "feature_count": features.shape[1],
        "feature_names": feature_names(),
        "overall": overall,
        "distance_mae_p75_m": float(np.mean(np.abs(predicted_distance_p75 - distances))),
        "by_category": by_category,
        "false_final_stop_count": len(false_indices),
        "false_final_stop_trials": [rows[i]["trial_id"] for i in false_indices],
        "folds": fold_records,
        "gates": gates,
        "full_development_model": str(full_model_path),
        "full_development_model_sha256": sha256(full_model_path),
        "threshold_scan_performed": False,
        "runtime_gt_inputs": [],
    }
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
