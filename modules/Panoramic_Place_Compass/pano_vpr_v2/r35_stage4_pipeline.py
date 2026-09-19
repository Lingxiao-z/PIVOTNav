from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable, Sequence

from .r35_multiframe_evidence import (
    AnchorEvidenceState,
    AnchorObservation,
    R35AnchorEvidenceTracker,
    R35MultiFrameEvidenceConfig,
)


PREDICTION_MANIFEST_SCHEMA = "r35_stage3_internal_observation_predictions_v1"
SEQUENCE_MANIFEST_SCHEMA = "r35_stage4_sequence_manifest_v1"
SEQUENCE_SCHEMA = "r35_stage4_sequence_v1"
FORMAL_ACCEPTANCE_SCHEMA = "r35_stage3_transition_acceptance_v1"
REQUIRED_NEGATIVE_CATEGORIES = (
    "clear_absent",
    "yaw_roll_absent",
    "near_but_wrong",
    "same_scene_repeated_texture",
    "cross_scene_high_score",
)
FAR_LIMITS = {
    "clear_absent": 0.0333,
    "yaw_roll_absent": 0.0333,
    "cross_scene_high_score": 0.0333,
    "near_but_wrong": 0.0833,
    "same_scene_repeated_texture": 0.0833,
}
FORBIDDEN_FORMAL_PATH_TOKENS = (
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


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_jsonl_exclusive(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
        handle.flush()
        os.fsync(handle.fileno())
    return count


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"JSONL第{line_number}行无法解析: {path}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"JSONL第{line_number}行不是对象: {path}")
            yield value


def _resolve(reference: str | Path, parent: Path) -> Path:
    path = Path(reference)
    return path if path.is_absolute() else (parent / path).resolve()


def _assert_formal_path_boundary(paths: Sequence[Path]) -> None:
    violations = []
    for path in paths:
        lowered = f"/{str(path).lower().strip('/')}"
        if any(token in lowered for token in FORBIDDEN_FORMAL_PATH_TOKENS):
            violations.append(str(path))
    if violations:
        raise RuntimeError(f"Stage 4正式输入触碰禁止数据边界: {violations}")


def _finite_probability(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError(f"{name}必须是[0,1]内有限数")
    return result


def _stable_key(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest()


def _required_stage3_gates(acceptance: dict[str, Any]) -> dict[str, bool]:
    return {
        "formal_stage4_authorized": acceptance.get("formal_stage4_authorized") is True,
        "bearing_all_gates_passed": acceptance.get("bearing_all_gates_passed") is True,
        "goal_anchor_all_gates_passed": acceptance.get("goal_anchor_all_gates_passed") is True,
        "vpr_protection_passed": acceptance.get("vpr_protection_passed") is True,
        "track_y_protection_passed": acceptance.get("track_y_protection_passed") is True,
        "development_only": acceptance.get("test_r32_confirmation_accessed") is False,
    }


def load_prediction_contract(
    manifest_path: Path,
    *,
    allow_synthetic: bool = False,
) -> tuple[dict[str, Any], Path, list[dict[str, Any]], dict[str, Any]]:
    manifest_path = manifest_path.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != PREDICTION_MANIFEST_SCHEMA:
        raise RuntimeError("Stage 3 observation prediction manifest schema不匹配")
    synthetic = manifest.get("synthetic") is True
    if synthetic and not allow_synthetic:
        raise RuntimeError("正式Stage 4拒绝synthetic prediction manifest")
    if not synthetic and allow_synthetic:
        raise RuntimeError("合成验收入口拒绝伪装为正式数据")
    if manifest.get("partition") != "internal_validation":
        raise RuntimeError("Stage 4只允许Internal Development预测")
    if manifest.get("test_r32_confirmation_accessed") is not False:
        raise RuntimeError("Stage 3 prediction manifest数据边界不合法")
    if int(manifest.get("candidate_count", -1)) != 16:
        raise RuntimeError("Stage 4冻结使用K=16候选集合")

    parent = manifest_path.parent
    prediction_path = _resolve(manifest["prediction_file"], parent)
    acceptance_path = _resolve(manifest["stage3_acceptance_file"], parent)
    if not synthetic:
        _assert_formal_path_boundary((manifest_path, prediction_path, acceptance_path))
    if sha256_path(prediction_path) != manifest.get("prediction_file_sha256"):
        raise RuntimeError("Stage 3 observation prediction SHA256不匹配")
    if sha256_path(acceptance_path) != manifest.get("stage3_acceptance_file_sha256"):
        raise RuntimeError("Stage 3 transition acceptance SHA256不匹配")
    acceptance = json.loads(acceptance_path.read_text(encoding="utf-8"))
    if not synthetic and acceptance.get("schema_version") != FORMAL_ACCEPTANCE_SCHEMA:
        raise RuntimeError("Stage 3 transition acceptance schema不匹配")
    gates = _required_stage3_gates(acceptance)
    if not all(gates.values()):
        raise RuntimeError(f"Stage 3尚未授权Stage 4: {gates}")
    checkpoint_sha = str(manifest.get("checkpoint_sha256", ""))
    if not checkpoint_sha or checkpoint_sha != str(
        acceptance.get("selected_checkpoint_sha256", "")
    ):
        raise RuntimeError("Stage 3 checkpoint与transition acceptance不一致")

    records = list(iter_jsonl(prediction_path))
    if len(records) != int(manifest.get("record_count", -1)):
        raise RuntimeError("Stage 3 observation prediction数量与manifest不一致")
    normalized: list[dict[str, Any]] = []
    observation_ids: set[str] = set()
    category_counts: Counter[str] = Counter()
    for index, record in enumerate(records):
        normalized_record = normalize_prediction_record(
            record,
            checkpoint_sha=checkpoint_sha,
            index=index,
        )
        observation_id = normalized_record["observation_id"]
        if observation_id in observation_ids:
            raise RuntimeError(f"Stage 3 observation_id重复: {observation_id}")
        observation_ids.add(observation_id)
        category_counts[normalized_record["category"]] += 1
        normalized.append(normalized_record)
    required = {"present", *REQUIRED_NEGATIVE_CATEGORIES}
    if not required.issubset(category_counts):
        raise RuntimeError(
            f"Stage 4 prediction类别不完整: {sorted(required - set(category_counts))}"
        )
    return manifest, prediction_path, normalized, acceptance


def normalize_prediction_record(
    record: dict[str, Any],
    *,
    checkpoint_sha: str,
    index: int,
) -> dict[str, Any]:
    required_strings = (
        "observation_id",
        "scene_id",
        "query_sample_id",
        "source_place_id",
        "category",
    )
    values = {name: str(record.get(name, "")).strip() for name in required_strings}
    if not all(values.values()):
        raise ValueError(f"prediction record {index}缺少字符串字段")
    category = values["category"]
    if category not in {"present", *REQUIRED_NEGATIVE_CATEGORIES}:
        raise ValueError(f"prediction record {index}类别未知: {category}")
    target_present = record.get("target_present") is True
    if target_present != (category == "present"):
        raise ValueError(f"prediction record {index}的target_present与category矛盾")
    target_candidate_id = record.get("target_candidate_id")
    if target_present and not target_candidate_id:
        raise ValueError(f"prediction record {index}的present样本缺少target_candidate_id")
    if not target_present and target_candidate_id is not None:
        raise ValueError(f"prediction record {index}的absent样本不得含target_candidate_id")
    selected_candidate_id = record.get("selected_candidate_id")
    if selected_candidate_id is not None:
        selected_candidate_id = str(selected_candidate_id)
    if str(record.get("model_checkpoint_sha256", "")) != checkpoint_sha:
        raise RuntimeError(f"prediction record {index} checkpoint SHA不一致")
    if record.get("test_r32_confirmation_accessed") is not False:
        raise RuntimeError(f"prediction record {index}数据边界不合法")
    if int(record.get("candidate_count", -1)) != 16:
        raise RuntimeError(f"prediction record {index}不是K=16")
    return {
        **values,
        "record_index": int(record.get("record_index", index)),
        "target_present": target_present,
        "target_candidate_id": (
            str(target_candidate_id) if target_candidate_id is not None else None
        ),
        "selected_candidate_id": selected_candidate_id,
        "presence_probability": _finite_probability(
            record.get("presence_probability"), "presence_probability"
        ),
        "selected_candidate_probability": _finite_probability(
            record.get("selected_candidate_probability"),
            "selected_candidate_probability",
        ),
        "confidence": _finite_probability(record.get("confidence"), "confidence"),
        "hard_positive": record.get("hard_positive") is True,
        "candidate_count": 16,
        "model_checkpoint_sha256": checkpoint_sha,
        "test_r32_confirmation_accessed": False,
    }


def _ordered(records: Iterable[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    return sorted(
        records,
        key=lambda row: _stable_key(seed, str(row["observation_id"])),
    )


def _sequence(
    *,
    sequence_id: str,
    scenario_type: str,
    category: str,
    scene_id: str,
    target_present: bool,
    target_candidate_id: str | None,
    expected_final_state: str,
    expected_reset_reason: str | None,
    observations: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": SEQUENCE_SCHEMA,
        "sequence_id": sequence_id,
        "scenario_type": scenario_type,
        "category": category,
        "scene_id": scene_id,
        "target_present": target_present,
        "target_candidate_id": target_candidate_id,
        "expected_final_state": expected_final_state,
        "expected_reset_reason": expected_reset_reason,
        "observation_count": len(observations),
        "observations": list(observations),
        "selection_uses_model_scores": False,
        "test_r32_confirmation_accessed": False,
    }


def construct_stage4_sequences(
    records: Sequence[dict[str, Any]],
    *,
    seed: int,
    max_sequences_per_scenario: int,
) -> list[dict[str, Any]]:
    if max_sequences_per_scenario < 1:
        raise ValueError("max_sequences_per_scenario必须为正数")
    positive_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    negatives_by_scene_category: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if record["target_present"]:
            positive_groups[
                (record["scene_id"], record["target_candidate_id"])
            ].append(record)
        else:
            negatives_by_scene_category[
                (record["scene_id"], record["category"])
            ].append(record)
    positive_groups = {
        key: _ordered(value, seed)
        for key, value in positive_groups.items()
        if len(value) >= 2
    }
    if not positive_groups:
        raise RuntimeError("Stage 4缺少同一真实候选的两条独立present观测")

    scenario_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    sequence_counter = 0

    def add(
        scenario: str,
        category: str,
        scene: str,
        target_present: bool,
        target: str | None,
        expected: str,
        reset_reason: str | None,
        observations: Sequence[dict[str, Any]],
    ) -> None:
        nonlocal sequence_counter
        if len(scenario_rows[scenario]) >= max_sequences_per_scenario:
            return
        sequence_counter += 1
        scenario_rows[scenario].append(
            _sequence(
                sequence_id=f"r35-s4-{sequence_counter:06d}",
                scenario_type=scenario,
                category=category,
                scene_id=scene,
                target_present=target_present,
                target_candidate_id=target,
                expected_final_state=expected,
                expected_reset_reason=reset_reason,
                observations=observations,
            )
        )

    for (scene, target), group in sorted(positive_groups.items()):
        ordinary = [row for row in group if not row["hard_positive"]]
        if len(ordinary) >= 2:
            add(
                "present_two_independent",
                "present",
                scene,
                True,
                target,
                AnchorEvidenceState.CONFIRMED.value,
                None,
                ordinary[:2],
            )
        if any(row["hard_positive"] for row in group):
            hard = next(row for row in group if row["hard_positive"])
            partner = next(row for row in group if row["observation_id"] != hard["observation_id"])
            add(
                "hard_positive_two_independent",
                "hard_positive",
                scene,
                True,
                target,
                AnchorEvidenceState.CONFIRMED.value,
                None,
                (hard, partner),
            )
        first = group[0]
        add(
            "duplicate_observation_guard",
            "present",
            scene,
            True,
            target,
            AnchorEvidenceState.PROVISIONAL.value,
            None,
            (first, first),
        )
        clear_absent = negatives_by_scene_category.get((scene, "clear_absent"), [])
        if clear_absent:
            contradiction = _ordered(clear_absent, seed)[0]
            add(
                "unknown_contradiction_reset",
                "present",
                scene,
                True,
                target,
                AnchorEvidenceState.UNKNOWN.value,
                "strong_unknown_contradiction",
                (group[0], group[1], contradiction),
            )
            if len(group) >= 3:
                add(
                    "recovery_after_unknown",
                    "present",
                    scene,
                    True,
                    target,
                    AnchorEvidenceState.CONFIRMED.value,
                    "strong_unknown_contradiction",
                    (group[0], contradiction, group[1], group[2]),
                )

    groups_by_scene: dict[str, list[tuple[str, list[dict[str, Any]]]]] = defaultdict(list)
    for (scene, target), group in positive_groups.items():
        groups_by_scene[scene].append((target, group))
    for scene, groups in sorted(groups_by_scene.items()):
        groups.sort(key=lambda item: item[0])
        for (target_a, group_a), (target_b, group_b) in zip(groups, groups[1:]):
            if target_a == target_b:
                continue
            add(
                "different_candidate_contradiction_reset",
                "present",
                scene,
                True,
                target_a,
                AnchorEvidenceState.UNKNOWN.value,
                "strong_different_candidate_contradiction",
                (group_a[0], group_b[0]),
            )

    for category in REQUIRED_NEGATIVE_CATEGORIES:
        category_records = [row for row in records if row["category"] == category]
        by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in _ordered(category_records, seed):
            by_scene[row["scene_id"]].append(row)
        for scene, group in sorted(by_scene.items()):
            for start in range(0, len(group) - 1, 2):
                add(
                    "absent_stays_unknown",
                    category,
                    scene,
                    False,
                    None,
                    AnchorEvidenceState.UNKNOWN.value,
                    None,
                    group[start : start + 2],
                )

    required_scenarios = {
        "present_two_independent",
        "hard_positive_two_independent",
        "duplicate_observation_guard",
        "unknown_contradiction_reset",
        "different_candidate_contradiction_reset",
        "recovery_after_unknown",
        "absent_stays_unknown",
    }
    missing_scenarios = required_scenarios - set(scenario_rows)
    negative_coverage = {
        row["category"]
        for row in scenario_rows.get("absent_stays_unknown", [])
    }
    missing_negative = set(REQUIRED_NEGATIVE_CATEGORIES) - negative_coverage
    if missing_scenarios or missing_negative:
        raise RuntimeError(
            "Stage 4冻结序列覆盖不足: "
            f"missing_scenarios={sorted(missing_scenarios)} "
            f"missing_negative={sorted(missing_negative)}"
        )
    rows = [row for scenario in sorted(scenario_rows) for row in scenario_rows[scenario]]
    return sorted(rows, key=lambda row: row["sequence_id"])


def build_stage4_sequence_artifacts(
    *,
    prediction_manifest_path: Path,
    output_dir: Path,
    protocol_path: Path,
    seed: int = 20260811,
    max_sequences_per_scenario: int = 512,
    allow_synthetic: bool = False,
) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"拒绝覆盖非空Stage 4输出目录: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("schema_version") != "r35_stage4_protocol_v1":
        raise RuntimeError("Stage 4 protocol schema不匹配")
    manifest, prediction_path, records, acceptance = load_prediction_contract(
        prediction_manifest_path,
        allow_synthetic=allow_synthetic,
    )
    sequences = construct_stage4_sequences(
        records,
        seed=seed,
        max_sequences_per_scenario=max_sequences_per_scenario,
    )
    sequence_path = output_dir / "stage4_sequences.jsonl"
    write_jsonl_exclusive(sequence_path, sequences)
    scenario_counts = Counter(row["scenario_type"] for row in sequences)
    category_counts = Counter(row["category"] for row in sequences)
    scene_counts = Counter(row["scene_id"] for row in sequences)
    sequence_manifest = {
        "schema_version": SEQUENCE_MANIFEST_SCHEMA,
        "synthetic": allow_synthetic,
        "partition": "internal_validation",
        "created_from_prediction_manifest": str(prediction_manifest_path.resolve()),
        "created_from_prediction_manifest_sha256": sha256_path(
            prediction_manifest_path.resolve()
        ),
        "prediction_file": str(prediction_path),
        "prediction_file_sha256": sha256_path(prediction_path),
        "stage3_acceptance_file": str(
            _resolve(manifest["stage3_acceptance_file"], prediction_manifest_path.parent)
        ),
        "stage3_selected_checkpoint_sha256": manifest["checkpoint_sha256"],
        "stage3_authorization_gates": _required_stage3_gates(acceptance),
        "protocol_file": str(protocol_path.resolve()),
        "protocol_file_sha256": sha256_path(protocol_path.resolve()),
        "sequence_file": str(sequence_path),
        "sequence_file_sha256": sha256_path(sequence_path),
        "sequence_count": len(sequences),
        "scenario_counts": dict(sorted(scenario_counts.items())),
        "category_counts": dict(sorted(category_counts.items())),
        "scene_count": len(scene_counts),
        "seed": seed,
        "max_sequences_per_scenario": max_sequences_per_scenario,
        "selection_rule_zh": (
            "仅按冻结标签、scene、真实候选和稳定哈希构造序列；"
            "不读取presence、candidate或confidence分数来挑选样本。"
        ),
        "selection_uses_model_scores": False,
        "internal_hard_positive_evaluation_only": True,
        "test_r32_confirmation_accessed": False,
    }
    manifest_output = output_dir / "stage4_sequence_manifest.json"
    atomic_json(manifest_output, sequence_manifest)
    return sequence_manifest


def _tracker_observation(record: dict[str, Any]) -> AnchorObservation:
    return AnchorObservation(
        observation_id=str(record["observation_id"]),
        candidate_id=record.get("selected_candidate_id"),
        presence_probability=float(record["presence_probability"]),
        candidate_probability=float(record["selected_candidate_probability"]),
        confidence=float(record["confidence"]),
    )


def evaluate_one_sequence(
    sequence: dict[str, Any],
    cfg: R35MultiFrameEvidenceConfig,
) -> dict[str, Any]:
    tracker = R35AnchorEvidenceTracker(cfg=cfg)
    transitions = []
    first_confirmed_at = None
    for index, record in enumerate(sequence["observations"], 1):
        snapshot = tracker.update(_tracker_observation(record))
        transitions.append(snapshot["latest_transition"])
        if snapshot["state"] == AnchorEvidenceState.CONFIRMED.value and first_confirmed_at is None:
            first_confirmed_at = index
    final = tracker.snapshot()
    expected = str(sequence["expected_final_state"])
    reset_reason = sequence.get("expected_reset_reason")
    observed_reasons = [row["transition_reason"] for row in transitions]
    reset_correct = reset_reason is None or reset_reason in observed_reasons
    target_candidate = sequence.get("target_candidate_id")
    candidate_correct = (
        not sequence["target_present"]
        or final["candidate_id"] == target_candidate
    )
    ever_confirmed = any(
        row["new_state"] == AnchorEvidenceState.CONFIRMED.value
        for row in transitions
    )
    return {
        "schema_version": "r35_stage4_sequence_result_v1",
        "sequence_id": sequence["sequence_id"],
        "scenario_type": sequence["scenario_type"],
        "category": sequence["category"],
        "scene_id": sequence["scene_id"],
        "target_present": sequence["target_present"],
        "target_candidate_id": target_candidate,
        "expected_final_state": expected,
        "actual_final_state": final["state"],
        "actual_final_candidate_id": final["candidate_id"],
        "final_state_correct": final["state"] == expected,
        "final_candidate_correct": candidate_correct,
        "ever_confirmed": ever_confirmed,
        "first_confirmed_observation_index": first_confirmed_at,
        "expected_reset_reason": reset_reason,
        "reset_correct": reset_correct,
        "transitions": transitions,
        "test_r32_confirmation_accessed": False,
    }


def _ratio(numerator: int, denominator: int) -> float:
    return float(numerator) / max(int(denominator), 1)


def _aggregate_rows(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    expected_confirm = [
        row for row in rows
        if row["expected_final_state"] == AnchorEvidenceState.CONFIRMED.value
    ]
    confirmed_correct = [
        row for row in expected_confirm
        if row["actual_final_state"] == AnchorEvidenceState.CONFIRMED.value
        and row["final_candidate_correct"]
    ]
    absent = [row for row in rows if not row["target_present"]]
    false_confirmed = [row for row in absent if row["ever_confirmed"]]
    reset = [row for row in rows if row["expected_reset_reason"] is not None]
    latencies = [
        int(row["first_confirmed_observation_index"])
        for row in confirmed_correct
        if row["first_confirmed_observation_index"] is not None
    ]
    return {
        "sequence_count": len(rows),
        "expected_confirmation_count": len(expected_confirm),
        "correct_confirmation_count": len(confirmed_correct),
        "confirmation_recall": _ratio(len(confirmed_correct), len(expected_confirm)),
        "missed_confirmation_rate": _ratio(
            len(expected_confirm) - len(confirmed_correct), len(expected_confirm)
        ),
        "absent_sequence_count": len(absent),
        "false_confirmation_count": len(false_confirmed),
        "false_confirmation_rate": _ratio(len(false_confirmed), len(absent)),
        "reset_sequence_count": len(reset),
        "reset_accuracy": _ratio(sum(row["reset_correct"] for row in reset), len(reset)),
        "final_state_accuracy": _ratio(
            sum(row["final_state_correct"] for row in rows), len(rows)
        ),
        "confirmation_delay_observations": {
            "count": len(latencies),
            "mean": mean(latencies) if latencies else None,
            "median": median(latencies) if latencies else None,
            "max": max(latencies) if latencies else None,
        },
    }


def compute_stage4_metrics(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not results:
        raise ValueError("Stage 4没有序列结果")
    by_scenario: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in results:
        by_scenario[row["scenario_type"]].append(row)
        by_category[row["category"]].append(row)
        by_scene[row["scene_id"]].append(row)
    overall = _aggregate_rows(results)
    scenario_metrics = {
        key: _aggregate_rows(value) for key, value in sorted(by_scenario.items())
    }
    category_metrics = {
        key: _aggregate_rows(value) for key, value in sorted(by_category.items())
    }
    per_scene = {
        key: _aggregate_rows(value) for key, value in sorted(by_scene.items())
    }
    hard_recall = category_metrics.get("hard_positive", {}).get(
        "confirmation_recall", 0.0
    )
    duplicate_accuracy = scenario_metrics.get("duplicate_observation_guard", {}).get(
        "final_state_accuracy", 0.0
    )
    reset_rows = [
        row for row in results if row["expected_reset_reason"] is not None
    ]
    reset_accuracy = _ratio(
        sum(row["reset_correct"] for row in reset_rows), len(reset_rows)
    )
    recovery_recall = scenario_metrics.get("recovery_after_unknown", {}).get(
        "confirmation_recall", 0.0
    )
    far_by_category = {
        category: category_metrics.get(category, {}).get(
            "false_confirmation_rate", 1.0
        )
        for category in REQUIRED_NEGATIVE_CATEGORIES
    }
    gates = {
        "confirmation_recall_ge_90pct": overall["confirmation_recall"] >= 0.90,
        "hard_positive_confirmation_recall_ge_90pct": hard_recall >= 0.90,
        "duplicate_observation_guard_100pct": duplicate_accuracy == 1.0,
        "contradiction_reset_accuracy_ge_95pct": reset_accuracy >= 0.95,
        "recovery_after_contradiction_ge_90pct": recovery_recall >= 0.90,
        **{
            f"{category}_false_confirmation_rate_within_limit": (
                far_by_category[category] <= FAR_LIMITS[category]
            )
            for category in REQUIRED_NEGATIVE_CATEGORIES
        },
        "all_outputs_finite": True,
    }
    return {
        "schema_version": "r35_stage4_metrics_v1",
        "overall": overall,
        "by_scenario": scenario_metrics,
        "by_category": category_metrics,
        "per_scene": per_scene,
        "hard_positive_confirmation_recall": hard_recall,
        "duplicate_guard_accuracy": duplicate_accuracy,
        "contradiction_reset_accuracy": reset_accuracy,
        "recovery_after_contradiction_recall": recovery_recall,
        "false_confirmation_rate_by_category": far_by_category,
        "gates": gates,
        "all_stage4_gates_passed": all(gates.values()),
        "test_r32_confirmation_accessed": False,
    }


def evaluate_stage4_artifacts(
    *,
    sequence_manifest_path: Path,
    output_dir: Path,
    allow_synthetic: bool = False,
) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"拒绝覆盖非空Stage 4评测目录: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(sequence_manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SEQUENCE_MANIFEST_SCHEMA:
        raise RuntimeError("Stage 4 sequence manifest schema不匹配")
    if (manifest.get("synthetic") is True) != allow_synthetic:
        raise RuntimeError("Stage 4 synthetic/formal入口不匹配")
    if manifest.get("selection_uses_model_scores") is not False:
        raise RuntimeError("Stage 4序列不得按模型分数选择")
    if manifest.get("test_r32_confirmation_accessed") is not False:
        raise RuntimeError("Stage 4 sequence manifest数据边界不合法")
    sequence_path = Path(manifest["sequence_file"])
    protocol_path = Path(manifest["protocol_file"])
    if sha256_path(sequence_path) != manifest["sequence_file_sha256"]:
        raise RuntimeError("Stage 4 sequence file SHA256不匹配")
    if sha256_path(protocol_path) != manifest["protocol_file_sha256"]:
        raise RuntimeError("Stage 4 protocol SHA256不匹配")
    if not allow_synthetic:
        _assert_formal_path_boundary((sequence_manifest_path, sequence_path, protocol_path))
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    cfg = R35MultiFrameEvidenceConfig(**protocol["state_machine_config"])
    sequences = list(iter_jsonl(sequence_path))
    if len(sequences) != int(manifest["sequence_count"]):
        raise RuntimeError("Stage 4 sequence数量与manifest不一致")
    results = [evaluate_one_sequence(sequence, cfg) for sequence in sequences]
    result_path = output_dir / "stage4_sequence_results.jsonl"
    write_jsonl_exclusive(result_path, results)
    metrics = compute_stage4_metrics(results)
    metrics.update(
        {
            "synthetic": allow_synthetic,
            "sequence_manifest": str(sequence_manifest_path.resolve()),
            "sequence_manifest_sha256": sha256_path(sequence_manifest_path.resolve()),
            "sequence_results": str(result_path),
            "sequence_results_sha256": sha256_path(result_path),
            "stage3_selected_checkpoint_sha256": manifest[
                "stage3_selected_checkpoint_sha256"
            ],
            "state_machine_config": asdict(cfg),
            "thresholds_frozen_before_formal_predictions": True,
        }
    )
    metrics_path = output_dir / "stage4_metrics.json"
    atomic_json(metrics_path, metrics)
    per_scene_path = output_dir / "stage4_per_scene_metrics.json"
    atomic_json(
        per_scene_path,
        {
            "schema_version": "r35_stage4_per_scene_metrics_v1",
            "per_scene": metrics["per_scene"],
            "test_r32_confirmation_accessed": False,
        },
    )
    acceptance = {
        "schema_version": "r35_stage4_acceptance_v1",
        "synthetic": allow_synthetic,
        "passed": metrics["all_stage4_gates_passed"],
        "formal_stage4_complete": (
            not allow_synthetic and metrics["all_stage4_gates_passed"]
        ),
        "metrics_file": str(metrics_path),
        "metrics_file_sha256": sha256_path(metrics_path),
        "sequence_results_file": str(result_path),
        "sequence_results_file_sha256": sha256_path(result_path),
        "per_scene_metrics_file": str(per_scene_path),
        "per_scene_metrics_file_sha256": sha256_path(per_scene_path),
        "gates": metrics["gates"],
        "test_r32_confirmation_accessed": False,
    }
    acceptance_path = output_dir / "stage4_acceptance.json"
    atomic_json(acceptance_path, acceptance)
    report_path = output_dir / "stage4_chinese_report.md"
    report_path.write_text(
        "# R35 Stage 4 多帧证据模拟报告\n\n"
        f"- 运行类型：{'合成代码验收（非正式Development结果）' if allow_synthetic else '正式Internal Development离线模拟'}\n"
        f"- 序列数：{metrics['overall']['sequence_count']}\n"
        f"- 确认Recall：{metrics['overall']['confirmation_recall']:.4%}\n"
        f"- 误确认率：{metrics['overall']['false_confirmation_rate']:.4%}\n"
        f"- 矛盾重置准确率：{metrics['contradiction_reset_accuracy']:.4%}\n"
        f"- 矛盾后恢复Recall：{metrics['recovery_after_contradiction_recall']:.4%}\n"
        f"- 重复观测保护准确率：{metrics['duplicate_guard_accuracy']:.4%}\n"
        f"- 全部门槛通过：{'是' if metrics['all_stage4_gates_passed'] else '否'}\n\n"
        "序列只按冻结标签、场景与稳定哈希构造，没有按模型分数筛样本。"
        "本次未访问Test、R32、历史Confirmation或Fresh Confirmation。\n",
        encoding="utf-8",
    )
    return acceptance
