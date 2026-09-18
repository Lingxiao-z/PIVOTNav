from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict

import torch

from .r34_yaw_head import R34CrossPositionYawHead
from .r34_yaw_head_r3 import R34CrossPositionYawHeadR3
from .r35_multitask_system import R35MultitaskSystem
from .system import PanoramicVPRV2System


EXPECTED_R3_SHA256 = "40307d508bcb4e7164ad8ff146d032bc3cadd7f203647eac6c79bbf863302075"
EXPECTED_TRACK_Y_R3_SHA256 = "3da199c2731fd80390fc97e100f728766a3d4a463af5bd7ea991d6f0008d2be2"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _apply_checked_partial_state(
    module: torch.nn.Module,
    partial_state: Dict[str, torch.Tensor],
    *,
    allowed_prefixes: tuple[str, ...],
    label: str,
) -> list[str]:
    if not partial_state:
        raise RuntimeError(f"{label} state is empty")
    current = module.state_dict()
    applied: list[str] = []
    for key, value in partial_state.items():
        if not key.startswith(allowed_prefixes):
            raise RuntimeError(f"{label} tensor is outside frozen scope: {key}")
        if key not in current or current[key].shape != value.shape:
            raise RuntimeError(f"{label} tensor mismatch: {key}")
        current[key].copy_(value)
        applied.append(key)
    return sorted(applied)


def load_r35_frozen_base(
    r3_checkpoint: str | Path,
    track_y_checkpoint: str | Path,
    device: str | torch.device,
) -> tuple[R35MultitaskSystem, Dict[str, Any]]:
    r3_path = Path(r3_checkpoint).resolve()
    track_y_path = Path(track_y_checkpoint).resolve()
    r3_sha256 = sha256_file(r3_path)
    track_y_sha256 = sha256_file(track_y_path)
    if r3_sha256 != EXPECTED_R3_SHA256:
        raise RuntimeError(f"R3 checkpoint SHA mismatch: {r3_sha256}")
    if track_y_sha256 != EXPECTED_TRACK_Y_R3_SHA256:
        raise RuntimeError(f"Track Y-r3 checkpoint SHA mismatch: {track_y_sha256}")
    target = torch.device(device)
    r3_payload = torch.load(r3_path, map_location="cpu", weights_only=False)
    if "system" not in r3_payload:
        raise RuntimeError("R3 checkpoint has no system state")
    encoder = PanoramicVPRV2System().to(target)
    encoder.load_state_dict(r3_payload["system"], strict=True)
    del r3_payload

    track_y_payload = torch.load(track_y_path, map_location="cpu", weights_only=False)
    if track_y_payload.get("schema_version") != "r34_track_y_checkpoint_v3":
        raise RuntimeError("Track Y checkpoint schema mismatch")
    if track_y_payload.get("revision") != "Y-r3":
        raise RuntimeError("Track Y checkpoint is not frozen Y-r3")
    if track_y_payload.get("source_checkpoint_sha256") != r3_sha256:
        raise RuntimeError("Track Y checkpoint does not descend from the frozen R3 checkpoint")
    adapted_blocks = tuple(int(value) for value in track_y_payload.get("backbone_trainable_blocks", []))
    if not adapted_blocks or any(value < 0 or value >= 12 for value in adapted_blocks):
        raise RuntimeError("Track Y adapted backbone block scope is invalid")
    backbone_keys = _apply_checked_partial_state(
        encoder.backbone,
        track_y_payload["backbone_state"],
        allowed_prefixes=tuple(f"model.blocks.{index}." for index in adapted_blocks),
        label="Track Y backbone",
    )
    descriptor_layer = int(track_y_payload.get("descriptor_refinement_trainable_layer", -1))
    descriptor_keys = _apply_checked_partial_state(
        encoder.descriptor_head,
        track_y_payload["descriptor_refinement_state"],
        allowed_prefixes=(f"refine.layers.{descriptor_layer}.",),
        label="Track Y descriptor refinement",
    )
    track_y = R34CrossPositionYawHeadR3(R34CrossPositionYawHead()).to(target)
    track_y.load_state_dict(track_y_payload["yaw_head_r3"], strict=True)
    del track_y_payload
    model = R35MultitaskSystem(encoder, track_y).to(target)
    for parameter in model.encoder_system.parameters():
        parameter.requires_grad_(False)
    for parameter in model.track_y.parameters():
        parameter.requires_grad_(False)
    if not all(torch.isfinite(value).all() for value in model.state_dict().values()):
        raise RuntimeError("R35 frozen base contains non-finite tensors")
    provenance = {
        "schema_version": "r35_frozen_base_loading_v1",
        "r3_checkpoint": str(r3_path),
        "r3_checkpoint_sha256": r3_sha256,
        "track_y_checkpoint": str(track_y_path),
        "track_y_checkpoint_sha256": track_y_sha256,
        "track_y_revision": "Y-r3",
        "adapted_backbone_blocks": list(adapted_blocks),
        "adapted_backbone_tensor_count": len(backbone_keys),
        "descriptor_refinement_layer": descriptor_layer,
        "descriptor_refinement_tensor_count": len(descriptor_keys),
        "shared_backbone_instance_count": 1,
        "test_r32_confirmation_accessed": False,
    }
    return model, provenance
