from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import torch

from .r34_yaw_head import R34CrossPositionYawHead
from .r34_yaw_head_r3 import R34CrossPositionYawHeadR3
from .r35_multitask_system import R35MultitaskSystem
from .system import PanoramicVPRV2System


INTEGRATED_CHECKPOINT_SCHEMA = "r35_integrated_checkpoint_v1"


def sha256_path(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_r35_integrated_checkpoint(
    checkpoint: str | Path,
    device: str | torch.device = "cpu",
) -> tuple[R35MultitaskSystem, dict[str, Any]]:
    """Load the self-contained frozen R35 checkpoint without external weights."""

    checkpoint_path = Path(checkpoint).resolve()
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if payload.get("schema_version") != INTEGRATED_CHECKPOINT_SCHEMA:
        raise RuntimeError("R35集成checkpoint schema不匹配")
    state = payload.get("student_system")
    if not isinstance(state, dict) or not state:
        raise RuntimeError("R35集成checkpoint缺少student_system")
    provenance = payload.get("provenance")
    if not isinstance(provenance, dict):
        raise RuntimeError("R35集成checkpoint缺少provenance")
    if provenance.get("development_all_passed_before_freeze") is not True:
        raise RuntimeError("R35集成checkpoint未证明Development冻结门槛")
    if provenance.get("test_r32_confirmation_accessed") is not False:
        raise RuntimeError("R35集成checkpoint数据边界不合法")
    model = R35MultitaskSystem(
        PanoramicVPRV2System(),
        R34CrossPositionYawHeadR3(R34CrossPositionYawHead()),
    )
    model.load_state_dict(state, strict=True)
    thresholds = payload.get("thresholds")
    if not isinstance(thresholds, dict):
        raise RuntimeError("R35集成checkpoint缺少冻结阈值")
    confidence_mode = thresholds.get("bearing_confidence_mode")
    if not isinstance(confidence_mode, str):
        raise RuntimeError("R35集成checkpoint缺少bearing confidence模式")
    model.bearing_head.set_confidence_output_mode(confidence_mode)
    if not all(torch.isfinite(value).all() for value in model.state_dict().values()):
        raise RuntimeError("R35集成checkpoint包含非有限权重")
    model.requires_grad_(False)
    model.eval().to(device)
    metadata = {
        "schema_version": payload["schema_version"],
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_path(checkpoint_path),
        "architecture": payload.get("architecture"),
        "thresholds": thresholds,
        "provenance": provenance,
    }
    return model, metadata
