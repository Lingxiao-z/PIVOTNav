from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import joblib
import numpy as np
from PIL import Image


ROOT = Path(os.environ.get("PIVOTNAV_ARRIVAL_SAFETY_ROOT", Path(__file__).resolve().parent))
PANO = Path(os.environ.get("PIVOTNAV_ARRIVAL_RUNTIME_ROOT", Path(__file__).resolve().parent))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


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


def align_by_predicted_yaw(rgb: np.ndarray, yaw_degrees: float) -> np.ndarray:
    shift = int(round(-float(yaw_degrees) / 360.0 * rgb.shape[1]))
    return np.roll(rgb, shift, axis=1)


def _view_rgb(view: dict) -> np.ndarray:
    """Read verifier evidence without requiring debug-image disk output."""
    image_rgb = view.get("image_rgb")
    if image_rgb is not None:
        array = np.asarray(image_rgb)
    else:
        image_path = view.get("image_path")
        if not image_path or not Path(image_path).is_file():
            raise RuntimeError("MISSING_RGB_EVIDENCE")
        array = np.asarray(Image.open(image_path).convert("RGB"))
    if array.ndim != 3 or array.shape[-1] < 3:
        raise RuntimeError("INVALID_RGB_EVIDENCE_SHAPE")
    return np.asarray(array[..., :3], dtype=np.float32)


def rgb_change(first: dict, second: dict) -> float:
    first_rgb = _view_rgb(first)
    second_rgb = _view_rgb(second)
    first_aligned = align_by_predicted_yaw(first_rgb, first["r361"]["yaw_degrees"])
    second_aligned = align_by_predicted_yaw(second_rgb, second["r361"]["yaw_degrees"])
    return float(np.mean(np.abs(first_aligned - second_aligned)))


def main() -> None:
    protocol_path = ROOT / "evaluation_protocol.json"
    protocol = json.loads(protocol_path.read_text())
    metrics_path = ROOT / "development_policy_metrics.json"
    predictions_path = ROOT / "development_policy_predictions.jsonl"
    if metrics_path.exists() or predictions_path.exists():
        raise RuntimeError("refuse to overwrite safety-margin evaluation")
    for key, record in protocol.items():
        if isinstance(record, dict) and set(record) == {"path", "sha256"}:
            if sha256(Path(record["path"])) != record["sha256"]:
                raise RuntimeError(f"frozen input changed: {key}")

    import sys
    sys.path.insert(0, str(PANO / "integration_v2"))
    from evaluate_arrival_parallax_v3 import hard_safety, make_features

    margin_rows = read_jsonl(Path(protocol["margin_evidence"]["path"]))
    active_rows = {row["trial_id"]: row for row in read_jsonl(Path(protocol["source_active_evidence"]["path"]))}
    parallax_rows = {row["trial_id"]: row for row in read_jsonl(Path(protocol["parallax_features"]["path"]))}
    model = joblib.load(Path(protocol["parallax_full_model"]["path"]))
    policy = protocol["policy"]
    evaluated = []
    for margin in margin_rows:
        trial_id = margin["trial_id"]
        if trial_id not in parallax_rows:
            continue
        active = active_rows[trial_id]
        extension = margin["views"]
        distance_predictions = []
        safety = []
        for view_index, view in enumerate(extension):
            synthetic = dict(active)
            synthetic["views"] = [*active["views"], *extension[:view_index + 1]]
            features = make_features(parallax_rows[trial_id], synthetic)[None]
            tree_predictions = np.asarray([tree.predict(features)[0] for tree in model.estimators_])
            p75 = float(np.quantile(tree_predictions, 0.75))
            safe, reasons = hard_safety(synthetic, protocol["hard_safety"])
            distance_predictions.append(p75)
            safety.append({"safe": safe, "reasons": reasons})
        candidate_index = None
        decision_mode = None
        confirmed_index = None
        for index, (distance, check) in enumerate(zip(distance_predictions, safety)):
            if distance <= policy["deep_arrival_distance_threshold_m"] and check["safe"]:
                candidate_index = index
                confirmed_index = index
                decision_mode = "DEEP_ARRIVAL_FRESH_FRAME"
                break
        if confirmed_index is None:
            for index, (distance, check) in enumerate(zip(distance_predictions, safety)):
                if distance <= policy["candidate_distance_threshold_m"] and check["safe"]:
                    candidate_index = index
                    decision_mode = "BOUNDARY_REQUIRES_SAFETY_MARGIN"
                    break
        changes = []
        if candidate_index is not None and confirmed_index is None:
            final_index = candidate_index + policy["post_candidate_required_approaches"]
            if final_index < len(extension):
                changes = [
                    rgb_change(extension[index], extension[index + 1])
                    for index in range(candidate_index, final_index)
                ]
                if (
                    all(value >= policy["minimum_aligned_rgb_mean_absolute_change"] for value in changes)
                    and distance_predictions[final_index] <= policy["candidate_distance_threshold_m"]
                    and safety[final_index]["safe"]
                ):
                    confirmed_index = final_index
                    decision_mode = "BOUNDARY_MARGIN_CONFIRMED"
        if confirmed_index is None:
            label = any(view["formal_arrival_label_gt_audit_only"] for view in extension)
            decision_distance = margin["final_horizontal_distance_m_gt_audit_only"]
        else:
            label = extension[confirmed_index]["formal_arrival_label_gt_audit_only"]
            decision_distance = extension[confirmed_index]["horizontal_distance_m_gt_audit_only"]
        evaluated.append({
            "trial_id": trial_id,
            "category_gt_audit_only": margin["category_pre_registered"],
            "candidate_index": candidate_index,
            "confirmed_index": confirmed_index,
            "decision_mode": decision_mode,
            "distance_predictions_m": distance_predictions,
            "rgb_change_after_candidate": changes,
            "final_confirmed": confirmed_index is not None,
            "label_gt_audit_only": bool(label),
            "decision_distance_m_gt_audit_only": float(decision_distance),
            "runtime_gt_inputs": [],
        })
    labels = np.asarray([row["label_gt_audit_only"] for row in evaluated])
    predictions = np.asarray([row["final_confirmed"] for row in evaluated])
    overall = classification(labels, predictions)
    by_category = {}
    for category in sorted({row["category_gt_audit_only"] for row in evaluated}):
        indices = [i for i, row in enumerate(evaluated) if row["category_gt_audit_only"] == category]
        by_category[category] = classification(labels[indices], predictions[indices])
    false_indices = [i for i in range(len(evaluated)) if predictions[i] and not labels[i]]
    gates_spec = protocol["acceptance_gates"]
    gates = {
        "precision": overall["precision"] >= gates_spec["precision_min"],
        "recall": overall["recall"] >= gates_spec["recall_min"],
        "f1": overall["f1"] >= gates_spec["f1_min"],
        "false_final_stop": len(false_indices) <= gates_spec["false_final_stop_max"],
        "repeated_texture": by_category["repeated_texture"]["fp"] == 0,
        "through_wall": by_category["through_wall_or_detour"]["fp"] == 0,
    }
    with predictions_path.open("x") as stream:
        for row in evaluated:
            stream.write(json.dumps(row, sort_keys=True) + "\n")
    metrics = {
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": sha256(protocol_path),
        "status": "PASS" if all(gates.values()) else "FAIL",
        "status_scope": "DEVELOPMENT_POLICY_DIAGNOSTIC_ONLY",
        "independent_validation_passed": False,
        "production_approved": False,
        "opportunity_count": len(evaluated),
        "overall": overall,
        "by_category": by_category,
        "false_final_stop_count": len(false_indices),
        "false_final_stop_trials": [evaluated[i]["trial_id"] for i in false_indices],
        "candidate_count": sum(row["candidate_index"] is not None for row in evaluated),
        "confirmed_count": int(predictions.sum()),
        "gates": gates,
        "threshold_scan_performed": False,
        "runtime_gt_inputs": [],
    }
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
