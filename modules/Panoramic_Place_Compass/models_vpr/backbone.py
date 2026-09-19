from __future__ import annotations

import hashlib
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import torch
from torch import nn


DINOV2_REPO = "facebookresearch/dinov2"
DINOV2_MODEL = "dinov2_vits14"
DINOV2_COMMIT = "7764ea0f912e53c92e82eb78a2a1631e92725fc8"
DINOV2_HUBCONF_SHA256 = "c1f5090e78ff940b72c076d2bf9c0310d1707c946b3d10e2d6f2b0bdf56a6f64"
DINOV2_LICENSE = "Apache-2.0"
PROJECT_ROOT = Path(os.environ.get(
    "PIVOTNAV_REPO_ROOT", str(Path(__file__).resolve().parents[4])
)).resolve()
MODULE_ROOT = Path(__file__).resolve().parents[2]
DINOV2_LOCAL_CHECKOUT = Path(os.environ.get(
    "PANORAMIC_VPR_DINOV2_CHECKOUT",
    str(MODULE_ROOT / "third_party" / "dinov2"),
))
DINOV2_CACHED_WEIGHT = Path(os.environ.get(
    "PANORAMIC_VPR_DINOV2_WEIGHT",
    str(PROJECT_ROOT / "cache" / "torch" / "hub" / "checkpoints" / "dinov2_vits14_pretrain.pth"),
))
DINOV2_CACHED_WEIGHT_SHA256 = "b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9"


@dataclass(frozen=True)
class DINOv2Provenance:
    repo: str = DINOV2_REPO
    model: str = DINOV2_MODEL
    commit: str = DINOV2_COMMIT
    license: str = DINOV2_LICENSE
    local_checkout: str = str(DINOV2_LOCAL_CHECKOUT)
    cached_weight: str = str(DINOV2_CACHED_WEIGHT)
    cached_weight_sha256: str = DINOV2_CACHED_WEIGHT_SHA256
    patch_size: int = 14
    input_height: int = 224
    input_width: int = 448
    token_grid_h: int = 16
    token_grid_w: int = 32


def sha256_path(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_tree(root: Path) -> str:
    h = hashlib.sha256()
    excluded_dirs = {".git", "__pycache__", ".pytest_cache"}
    for path in sorted(root.rglob("*")):
        if any(part in excluded_dirs for part in path.relative_to(root).parts):
            continue
        if not path.is_file():
            continue
        rel = path.relative_to(root).as_posix().encode("utf-8")
        h.update(rel)
        h.update(b"\0")
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        h.update(b"\0")
    return h.hexdigest()


def _git_head(path: Path) -> str | None:
    if not (path / ".git").exists():
        return None
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def verify_cached_dinov2_weight() -> Dict[str, object]:
    exists = DINOV2_CACHED_WEIGHT.is_file()
    actual = sha256_path(DINOV2_CACHED_WEIGHT) if exists else None
    return {
        "path": str(DINOV2_CACHED_WEIGHT),
        "exists": exists,
        "expected_sha256": DINOV2_CACHED_WEIGHT_SHA256,
        "actual_sha256": actual,
        "matches": actual == DINOV2_CACHED_WEIGHT_SHA256,
    }


def verify_dinov2_source_tree(include_tree_sha: bool = False) -> Dict[str, object]:
    exists = DINOV2_LOCAL_CHECKOUT.is_dir()
    head = _git_head(DINOV2_LOCAL_CHECKOUT) if exists else None
    hubconf = DINOV2_LOCAL_CHECKOUT / "hubconf.py"
    license_file = DINOV2_LOCAL_CHECKOUT / "LICENSE"
    hubconf_sha = sha256_path(hubconf) if hubconf.is_file() else None
    status: Dict[str, object] = {
        "path": str(DINOV2_LOCAL_CHECKOUT),
        "exists": exists,
        "expected_commit": DINOV2_COMMIT,
        "actual_commit": head,
        "commit_matches": head == DINOV2_COMMIT or hubconf_sha == DINOV2_HUBCONF_SHA256,
        "hubconf_sha256": hubconf_sha,
        "license_sha256": sha256_path(license_file) if license_file.is_file() else None,
    }
    if include_tree_sha and exists:
        status["source_tree_sha256"] = sha256_tree(DINOV2_LOCAL_CHECKOUT)
    return status


class DINOv2S14Backbone(nn.Module):
    """Official DINOv2-S/14 patch-token extractor for 224x448 ERP panoramas."""

    def __init__(self, freeze: bool = True):
        super().__init__()
        weight_status = verify_cached_dinov2_weight()
        if not weight_status["matches"]:
            raise RuntimeError(f"DINOv2 cached weight mismatch: {weight_status}")
        source_status = verify_dinov2_source_tree(include_tree_sha=False)
        if not source_status["commit_matches"]:
            raise RuntimeError(f"DINOv2 local checkout mismatch: {source_status}")
        # Bind runtime code to the fixed local checkout. This avoids floating
        # torch.hub default-branch loads while still reusing the verified cached
        # official weight.
        os.environ.setdefault("TORCH_HOME", str(PROJECT_ROOT / "cache" / "torch"))
        self.model = torch.hub.load(
            str(DINOV2_LOCAL_CHECKOUT),
            DINOV2_MODEL,
            source="local",
            pretrained=True,
            weights=str(DINOV2_CACHED_WEIGHT),
        )
        self._trainable_block_count = 0
        if freeze:
            self.freeze_all()

    def freeze_all(self) -> None:
        for p in self.model.parameters():
            p.requires_grad_(False)
        self._trainable_block_count = 0
        self.model.eval()

    def unfreeze_last_blocks(self, count: int = 4) -> None:
        self.freeze_all()
        blocks = getattr(self.model, "blocks", None)
        if blocks is None:
            raise RuntimeError("DINOv2 model does not expose blocks")
        count = int(count)
        if count <= 0 or count > len(blocks):
            raise ValueError(f"count must be in [1,{len(blocks)}], got {count}")
        for block in list(blocks)[-count:]:
            for p in block.parameters():
                p.requires_grad_(True)
        final_norm = getattr(self.model, "norm", None)
        if final_norm is not None:
            for p in final_norm.parameters():
                p.requires_grad_(True)
        self._trainable_block_count = count
        self._restore_training_modes(self.training)

    def _restore_training_modes(self, mode: bool) -> None:
        # Frozen DINO layers must stay in eval mode. Only the explicitly
        # unfrozen tail blocks and final norm may use training behavior.
        self.model.eval()
        if not mode or self._trainable_block_count <= 0:
            return
        blocks = list(getattr(self.model, "blocks"))
        for block in blocks[-self._trainable_block_count :]:
            block.train(True)
        final_norm = getattr(self.model, "norm", None)
        if final_norm is not None:
            final_norm.train(True)

    def train(self, mode: bool = True) -> "DINOv2S14Backbone":
        super().train(mode)
        self._restore_training_modes(mode)
        return self

    def trainable_parameter_names(self) -> List[str]:
        return [name for name, parameter in self.model.named_parameters() if parameter.requires_grad]

    @property
    def trainable_block_count(self) -> int:
        return self._trainable_block_count

    def forward_tokens(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or images.shape[-2:] != (224, 448):
            raise ValueError(f"expected normalized RGB [B,3,224,448], got {tuple(images.shape)}")
        out = self.model.forward_features(images)
        tokens = out["x_norm_patchtokens"]
        if tokens.shape[1] != 16 * 32:
            raise RuntimeError(f"expected 512 patch tokens for 224x448 input, got {tokens.shape[1]}")
        return tokens.reshape(tokens.shape[0], 16, 32, tokens.shape[-1]).contiguous()


class PanoSaladRingV2(nn.Module):
    def __init__(self, head: nn.Module, freeze_backbone: bool = True):
        super().__init__()
        self.backbone = DINOv2S14Backbone(freeze=freeze_backbone)
        self.head = head

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        tokens = self.backbone.forward_tokens(images)
        return self.head.forward_tokens(tokens)
