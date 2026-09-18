from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from .r35_model_loading import (
    EXPECTED_R3_SHA256,
    EXPECTED_TRACK_Y_R3_SHA256,
    load_r35_frozen_base,
    sha256_file,
)
from .r35_multitask_system import R35MultitaskSystem, R35TrainingStage
from .r35_stage2_revision2 import R35SceneRobustPresenceHead
from .r35_stage2_revision3 import R35DualExpertPresenceHead
from .r35_stage2_revision4 import (
    R35RawPairCandidateVerifier,
    R35RawPairVerifierPresenceHead,
)
from .system import PanoramicVPRV2System


FORBIDDEN_PATH_TOKENS = (
    "/r32/",
    "/test/",
    "confirmation",
    "final_test",
)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON根节点不是对象: {path}")
    return value


def _assert_input_boundary(paths: tuple[Path, ...]) -> None:
    violations = []
    for path in paths:
        lowered = f"/{str(path.resolve()).lower().strip('/')}"
        if any(token in lowered for token in FORBIDDEN_PATH_TOKENS):
            violations.append(str(path))
    if violations:
        raise RuntimeError(f"Stage 3初始化触碰禁止数据边界: {violations}")


def _resolve(value: str | Path, parent: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    candidates = [Path.cwd() / path]
    candidates.extend(ancestor / path for ancestor in (parent, *parent.parents))
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return (parent / path).resolve()


def validate_stage3_initialization_contract(
    *,
    stage1_freeze_path: str | Path,
    stage2_freeze_path: str | Path,
    protocol_path: str | Path,
    raw_data_audit_path: str | Path,
) -> dict[str, Any]:
    stage1_freeze_path = Path(stage1_freeze_path).resolve()
    stage2_freeze_path = Path(stage2_freeze_path).resolve()
    protocol_path = Path(protocol_path).resolve()
    raw_data_audit_path = Path(raw_data_audit_path).resolve()
    _assert_input_boundary(
        (
            stage1_freeze_path,
            stage2_freeze_path,
            protocol_path,
            raw_data_audit_path,
        )
    )
    stage1 = _load_json(stage1_freeze_path)
    stage2 = _load_json(stage2_freeze_path)
    protocol = _load_json(protocol_path)
    raw_audit = _load_json(raw_data_audit_path)
    if (
        protocol.get("schema_version") != "r35_stage3_training_protocol_v1"
        or protocol.get("test_r32_confirmation_accessed") is not False
        or protocol.get("trainable_scope", {}).get("unfrozen_backbone_blocks") != [10, 11]
        or protocol.get("trainable_scope", {}).get("maximum_unfrozen_backbone_blocks") != 2
        or protocol.get("trainable_scope", {}).get("descriptor_head_trainable") is not False
        or protocol.get("trainable_scope", {}).get("track_y_head_trainable") is not False
        or protocol.get("runtime", {}).get("physical_gpus") != [0, 1]
        or int(protocol.get("runtime", {}).get("world_size", -1)) != 2
        or int(protocol.get("runtime", {}).get("max_steps", -1)) != 5000
    ):
        raise RuntimeError("Stage 3训练协议关键语义不合法")
    if (
        raw_audit.get("schema_version")
        != "r35_stage3_raw_data_feasibility_audit_v1"
        or raw_audit.get("passed") is not True
        or not all(raw_audit.get("gates", {}).values())
        or raw_audit.get("test_r32_confirmation_accessed") is not False
    ):
        raise RuntimeError("Stage 3 raw data feasibility audit未通过")
    activation = protocol.get("activation_artifacts", {})
    if (
        _resolve(activation.get("raw_data_feasibility_audit", ""), protocol_path.parent)
        != raw_data_audit_path
        or activation.get("raw_data_feasibility_audit_sha256")
        != sha256_file(raw_data_audit_path)
    ):
        raise RuntimeError("Stage 3 protocol与raw data audit绑定不一致")
    if (
        stage1.get("schema_version") != "r35_stage1_freeze_v1"
        or stage1.get("authorized_for_stage2") is not True
        or stage1.get("all_basic_transition_gates_passed") is not True
        or stage1.get("training_clean_no_nan_no_skip") is not True
        or int(stage1.get("checkpoint_count", -1)) != int(
            stage1.get("expected_checkpoint_count", -2)
        )
        or stage1.get("test_r32_confirmation_accessed") is not False
    ):
        raise RuntimeError("Stage 1 freeze未授权或不完整")
    if (
        stage2.get("schema_version") != "r35_stage2_freeze_v1"
        or stage2.get("authorized_for_stage3") is not True
        or stage2.get("all_goal_anchor_gates_passed") is not True
        or stage2.get("training_clean_no_nan_no_skip") is not True
        or stage2.get("all_seven_categories_sampled") is not True
        or float(stage2.get("fixed_presence_threshold", float("nan"))) != 0.5
        or int(stage2.get("checkpoint_count", -1)) != int(
            stage2.get("expected_checkpoint_count", -2)
        )
        or stage2.get("test_r32_confirmation_accessed") is not False
    ):
        raise RuntimeError("Stage 2 freeze未授权或不完整")

    stage1_checkpoint = _resolve(stage1["checkpoint"], stage1_freeze_path.parent)
    stage2_checkpoint = _resolve(stage2["checkpoint"], stage2_freeze_path.parent)
    stage1_development = _resolve(
        stage1["development_metrics"], stage1_freeze_path.parent
    )
    stage2_development = _resolve(
        stage2["development_metrics"], stage2_freeze_path.parent
    )
    _assert_input_boundary(
        (
            stage1_checkpoint,
            stage2_checkpoint,
            stage1_development,
            stage2_development,
        )
    )
    for path, expected, label in (
        (stage1_checkpoint, stage1["checkpoint_sha256"], "Stage 1 checkpoint"),
        (stage2_checkpoint, stage2["checkpoint_sha256"], "Stage 2 checkpoint"),
        (
            stage1_development,
            stage1["development_metrics_sha256"],
            "Stage 1 Development",
        ),
        (
            stage2_development,
            stage2["development_metrics_sha256"],
            "Stage 2 Development",
        ),
    ):
        if sha256_file(path) != expected:
            raise RuntimeError(f"{label} SHA256不匹配")
    stage1_payload = torch.load(
        stage1_checkpoint, map_location="cpu", weights_only=False
    )
    stage2_payload = torch.load(
        stage2_checkpoint, map_location="cpu", weights_only=False
    )
    if (
        stage1_payload.get("schema_version")
        not in {
            "r35_stage1_bearing_checkpoint_v1",
            "r35_bearing_close_range_r2_checkpoint_v1",
        }
        or int(stage1_payload.get("global_step", -1))
        != int(stage1.get("checkpoint_global_step", -2))
        or not isinstance(stage1_payload.get("bearing_head"), dict)
        or not stage1_payload["bearing_head"]
        or stage1_payload.get("test_r32_confirmation_accessed") is not False
    ):
        raise RuntimeError("Stage 1 checkpoint payload不合法")
    if (
        stage2_payload.get("schema_version")
        not in {
            "r35_stage2_checkpoint_v1",
            "r35_stage2_checkpoint_v2",
            "r35_stage2_checkpoint_v3",
            "r35_stage2_checkpoint_v4",
            "r35_stage2_checkpoint_v5",
        }
        or int(stage2_payload.get("global_step", -1))
        != int(stage2.get("checkpoint_global_step", -2))
        or not isinstance(stage2_payload.get("candidate_match_head"), dict)
        or not stage2_payload["candidate_match_head"]
        or not isinstance(stage2_payload.get("set_presence_head"), dict)
        or not stage2_payload["set_presence_head"]
        or (
            stage2_payload.get("schema_version") == "r35_stage2_checkpoint_v5"
            and (
                not isinstance(
                    stage2_payload.get("raw_pair_candidate_verifier"), dict
                )
                or not stage2_payload["raw_pair_candidate_verifier"]
            )
        )
        or stage2_payload.get("test_r32_confirmation_accessed") is not False
    ):
        raise RuntimeError("Stage 2 checkpoint payload不合法")
    stage1_freeze_sha = sha256_file(stage1_freeze_path)
    selection = stage2_payload.get("loader_selection", {})
    if selection.get("stage1_freeze_sha256") != stage1_freeze_sha:
        raise RuntimeError("Stage 2 checkpoint未绑定当前Stage 1 freeze")
    if stage2.get("frozen_artifact_hashes") != stage2_payload.get(
        "frozen_artifact_hashes"
    ):
        raise RuntimeError("Stage 2 freeze与checkpoint冻结artifact hashes不一致")
    confidence_mode = "gated_evidence"
    confidence_config_path = None
    if stage1.get("confidence_mode_config") is not None:
        confidence_config_path = _resolve(
            stage1["confidence_mode_config"], stage1_freeze_path.parent
        )
        if sha256_file(confidence_config_path) != stage1.get(
            "confidence_mode_config_sha256"
        ):
            raise RuntimeError("Stage 1 confidence mode配置SHA256不匹配")
        confidence = _load_json(confidence_config_path)
        if (
            confidence.get("schema_version") != "r35_bearing_confidence_mode_v1"
            or confidence.get("authorized") is not True
            or confidence.get("test_r32_confirmation_accessed") is not False
        ):
            raise RuntimeError("Stage 1 confidence mode配置未授权")
        confidence_mode = str(confidence["confidence_output_mode"])
    return {
        "schema_version": "r35_stage3_initialization_contract_v1",
        "stage1_freeze": str(stage1_freeze_path),
        "stage1_freeze_sha256": stage1_freeze_sha,
        "stage1_checkpoint": str(stage1_checkpoint),
        "stage1_checkpoint_sha256": stage1["checkpoint_sha256"],
        "stage1_checkpoint_global_step": int(stage1["checkpoint_global_step"]),
        "stage1_bearing_head_state": stage1_payload["bearing_head"],
        "bearing_confidence_output_mode": confidence_mode,
        "bearing_confidence_config": (
            str(confidence_config_path) if confidence_config_path else None
        ),
        "stage2_freeze": str(stage2_freeze_path),
        "stage2_freeze_sha256": sha256_file(stage2_freeze_path),
        "stage2_checkpoint": str(stage2_checkpoint),
        "stage2_checkpoint_sha256": stage2["checkpoint_sha256"],
        "stage2_checkpoint_global_step": int(stage2["checkpoint_global_step"]),
        "stage2_checkpoint_schema": stage2_payload["schema_version"],
        "stage2_revision": stage2.get("stage2_revision", "stage2_r1"),
        "stage2_candidate_match_state": stage2_payload["candidate_match_head"],
        "stage2_set_presence_state": stage2_payload["set_presence_head"],
        "stage2_raw_pair_candidate_verifier_state": stage2_payload.get(
            "raw_pair_candidate_verifier"
        ),
        "stage2_scene_robust_presence": (
            stage2_payload["schema_version"] == "r35_stage2_checkpoint_v3"
        ),
        "stage2_dual_expert_presence": (
            stage2_payload["schema_version"] == "r35_stage2_checkpoint_v4"
        ),
        "stage2_raw_pair_verifier_presence": (
            stage2_payload["schema_version"] == "r35_stage2_checkpoint_v5"
        ),
        "protocol": str(protocol_path),
        "protocol_sha256": sha256_file(protocol_path),
        "raw_data_audit": str(raw_data_audit_path),
        "raw_data_audit_sha256": sha256_file(raw_data_audit_path),
        "formal_initialization_authorized": True,
        "test_r32_confirmation_accessed": False,
    }


def audit_stage3_trainable_scope(model: R35MultitaskSystem) -> dict[str, Any]:
    trainable = [name for name, value in model.named_parameters() if value.requires_grad]
    forbidden = [
        name
        for name in trainable
        if (
            name.startswith("track_y.")
            or name.startswith("encoder_system.descriptor_head.")
            or name.startswith("encoder_system.matcher.")
        )
    ]
    encoder_trainable = [
        name for name in trainable if name.startswith("encoder_system.")
    ]
    allowed_encoder_prefixes = (
        "encoder_system.backbone.model.blocks.10.",
        "encoder_system.backbone.model.blocks.11.",
        "encoder_system.backbone.model.norm.",
    )
    encoder_outside_scope = [
        name for name in encoder_trainable if not name.startswith(allowed_encoder_prefixes)
    ]
    head_prefixes = (
        "bearing_head.",
        "candidate_match_head.",
        "set_presence_head.",
        "raw_pair_candidate_verifier.",
    )
    head_counts = {
        prefix.rstrip("."): sum(name.startswith(prefix) for name in trainable)
        for prefix in head_prefixes
    }
    gates = {
        "no_track_y_descriptor_or_matcher_trainable": not forbidden,
        "encoder_only_blocks_10_11_and_final_norm": not encoder_outside_scope,
        "block_10_trainable": any(
            name.startswith("encoder_system.backbone.model.blocks.10.")
            for name in trainable
        ),
        "block_11_trainable": any(
            name.startswith("encoder_system.backbone.model.blocks.11.")
            for name in trainable
        ),
        "bearing_head_trainable": head_counts["bearing_head"] > 0,
        "candidate_match_head_trainable": head_counts["candidate_match_head"] > 0,
        "set_presence_head_trainable": head_counts["set_presence_head"] > 0,
        "raw_pair_candidate_verifier_scope_valid": (
            not hasattr(model, "raw_pair_candidate_verifier")
            or model.raw_pair_candidate_verifier is None
            or head_counts["raw_pair_candidate_verifier"] > 0
        ),
    }
    return {
        "schema_version": "r35_stage3_trainable_scope_audit_v1",
        "trainable_tensor_count": len(trainable),
        "encoder_trainable_tensor_count": len(encoder_trainable),
        "head_trainable_tensor_counts": head_counts,
        "forbidden_trainable_names": forbidden,
        "encoder_outside_scope": encoder_outside_scope,
        "gates": gates,
        "passed": all(gates.values()),
        "test_r32_confirmation_accessed": False,
    }


def load_stage3_models(
    *,
    r3_checkpoint: str | Path,
    track_y_checkpoint: str | Path,
    stage1_freeze_path: str | Path,
    stage2_freeze_path: str | Path,
    protocol_path: str | Path,
    raw_data_audit_path: str | Path,
    student_device: str | torch.device,
    teacher_device: str | torch.device | None = None,
) -> tuple[
    R35MultitaskSystem,
    PanoramicVPRV2System,
    R35MultitaskSystem,
    dict[str, Any],
]:
    r3_checkpoint = Path(r3_checkpoint).resolve()
    track_y_checkpoint = Path(track_y_checkpoint).resolve()
    _assert_input_boundary((r3_checkpoint, track_y_checkpoint))
    if sha256_file(r3_checkpoint) != EXPECTED_R3_SHA256:
        raise RuntimeError("Stage 3 R3 teacher checkpoint SHA256不匹配")
    if sha256_file(track_y_checkpoint) != EXPECTED_TRACK_Y_R3_SHA256:
        raise RuntimeError("Stage 3 Track Y teacher checkpoint SHA256不匹配")
    contract = validate_stage3_initialization_contract(
        stage1_freeze_path=stage1_freeze_path,
        stage2_freeze_path=stage2_freeze_path,
        protocol_path=protocol_path,
        raw_data_audit_path=raw_data_audit_path,
    )
    student, base_provenance = load_r35_frozen_base(
        r3_checkpoint, track_y_checkpoint, student_device
    )
    student.bearing_head.load_state_dict(
        contract.pop("stage1_bearing_head_state"), strict=True
    )
    student.bearing_head.set_confidence_output_mode(
        contract["bearing_confidence_output_mode"]
    )
    student.candidate_match_head.load_state_dict(
        contract.pop("stage2_candidate_match_state"), strict=True
    )
    raw_pair_presence = contract.pop("stage2_raw_pair_verifier_presence")
    if raw_pair_presence:
        student.set_presence_head = R35RawPairVerifierPresenceHead().to(
            student_device
        )
        student.raw_pair_candidate_verifier = R35RawPairCandidateVerifier().to(
            student_device
        )
        student.raw_pair_candidate_verifier.load_state_dict(
            contract.pop("stage2_raw_pair_candidate_verifier_state"),
            strict=True,
        )
    elif contract.pop("stage2_dual_expert_presence"):
        student.set_presence_head = R35DualExpertPresenceHead().to(
            student_device
        )
    elif contract.pop("stage2_scene_robust_presence"):
        student.set_presence_head = R35SceneRobustPresenceHead().to(
            student_device
        )
    student.set_presence_head.load_state_dict(
        contract.pop("stage2_set_presence_state"), strict=True
    )
    if not raw_pair_presence:
        contract.pop("stage2_raw_pair_candidate_verifier_state")
    student.configure_training_stage(
        R35TrainingStage.LIMITED_JOINT,
        stage3_unfreeze_blocks=2,
    )
    if isinstance(
        student.set_presence_head,
        (R35SceneRobustPresenceHead, R35DualExpertPresenceHead),
    ):
        for obsolete_head in (
            student.set_presence_head.evidence_encoder.presence_head,
            student.set_presence_head.evidence_encoder.near_wrong_head,
            student.set_presence_head.evidence_encoder.confidence_head,
        ):
            for parameter in obsolete_head.parameters():
                parameter.requires_grad_(False)
    scope = audit_stage3_trainable_scope(student)
    if not scope["passed"]:
        raise RuntimeError(f"Stage 3 trainable scope非法: {scope}")

    teacher_device = teacher_device or student_device
    r3_payload = torch.load(r3_checkpoint, map_location="cpu", weights_only=False)
    r3_teacher = PanoramicVPRV2System().to(teacher_device)
    r3_teacher.load_state_dict(r3_payload["system"], strict=True)
    del r3_payload
    r3_teacher.eval()
    for parameter in r3_teacher.parameters():
        parameter.requires_grad_(False)
    track_y_teacher, track_y_provenance = load_r35_frozen_base(
        r3_checkpoint, track_y_checkpoint, teacher_device
    )
    track_y_teacher.eval()
    for parameter in track_y_teacher.parameters():
        parameter.requires_grad_(False)
    provenance = {
        **contract,
        "student_base": base_provenance,
        "r3_teacher_checkpoint": str(r3_checkpoint),
        "r3_teacher_checkpoint_sha256": EXPECTED_R3_SHA256,
        "track_y_teacher_checkpoint": str(track_y_checkpoint),
        "track_y_teacher_checkpoint_sha256": EXPECTED_TRACK_Y_R3_SHA256,
        "track_y_teacher_base": track_y_provenance,
        "trainable_scope": scope,
        "r3_teacher_all_parameters_frozen": True,
        "track_y_teacher_all_parameters_frozen": True,
        "test_r32_confirmation_accessed": False,
    }
    return student, r3_teacher, track_y_teacher, provenance
