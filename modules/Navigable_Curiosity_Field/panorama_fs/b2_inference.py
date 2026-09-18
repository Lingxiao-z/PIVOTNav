from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from torch import nn

from .dino_fgfs_revision3 import RevisedDINOv2PanoramaFGFSV3
from .dino_fgfs_revision3_b2 import RevisedDINOv2PanoramaFGFSV3LastBlockB2
from .inference import normalize_erp, validate_online_inputs


B2_CHECKPOINT_SHA256 = "275150e5e93ae971d00689bfd8d45fe07774280ef1689d93ede60dbc9c50488d"
DINO_WEIGHT_SHA256 = "b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9"
DEFAULT_ENSEMBLE_SECTOR_SHIFTS = (0, 3, 6, 9)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_checkpoint(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _validated_result(fg_logits: torch.Tensor, fs_scores: torch.Tensor, threshold: float) -> dict[str, torch.Tensor]:
    if fg_logits.shape != fs_scores.shape or fg_logits.ndim != 2 or fg_logits.shape[1] != 12:
        raise RuntimeError("B2 FG/FS outputs must have matching [B,12] shapes")
    if not bool(torch.isfinite(fg_logits).all() and torch.isfinite(fs_scores).all()):
        raise RuntimeError("B2 FG/FS produced non-finite output")
    fg_probabilities = torch.sigmoid(fg_logits)
    fs_valid_mask = fg_probabilities >= threshold
    masked_fs_scores = torch.where(fs_valid_mask, fs_scores, torch.full_like(fs_scores, -torch.inf))
    return {
        "fg_logits": fg_logits,
        "fg_probabilities": fg_probabilities,
        "fs_scores": fs_scores,
        "fs_valid_mask": fs_valid_mask,
        "masked_fs_scores": masked_fs_scores,
    }


class B2FGFSInference(nn.Module):
    """Exact pure-RGB inference adapter for the frozen last-block B2 model."""

    def __init__(
        self,
        package_root: str | Path,
        *,
        device: str | torch.device = "cuda",
        fg_threshold: float = 0.5,
    ) -> None:
        super().__init__()
        self.package_root = Path(package_root).resolve()
        self.device = torch.device(device)
        self.fg_threshold = float(fg_threshold)
        if not 0.0 <= self.fg_threshold <= 1.0:
            raise ValueError("fg_threshold must be in [0,1]")

        checkpoint_path = self.package_root / "b2/checkpoints/best.pt"
        weight_path = self.package_root / "dinov2/dinov2_vits14_pretrain.pth"
        repository_path = self.package_root / "dinov2/repository"
        if _sha256(checkpoint_path) != B2_CHECKPOINT_SHA256:
            raise RuntimeError("frozen B2 checkpoint SHA256 mismatch")
        if _sha256(weight_path) != DINO_WEIGHT_SHA256:
            raise RuntimeError("official DINOv2 weight SHA256 mismatch")
        if not repository_path.is_dir():
            raise RuntimeError("frozen official DINOv2 repository is missing")

        backbone = torch.hub.load(
            str(repository_path),
            "dinov2_vits14",
            pretrained=True,
            weights=str(weight_path),
            source="local",
        ).to(self.device)
        decoder = RevisedDINOv2PanoramaFGFSV3().to(self.device)
        self.model = RevisedDINOv2PanoramaFGFSV3LastBlockB2(
            backbone.blocks[-1], backbone.norm, decoder
        ).to(self.device)
        checkpoint = _load_checkpoint(checkpoint_path)
        if checkpoint.get("resume_schema") != "revised_dino_fgfs_v3_last_block_b2_exact_resume_v1":
            raise RuntimeError("unexpected B2 checkpoint schema")
        if checkpoint.get("formal_test_accessed") is not False or checkpoint.get("test_run_count") != 0:
            raise RuntimeError("B2 checkpoint violates frozen test-isolation contract")
        self.model.load_state_dict(checkpoint["model"], strict=True)
        self.backbone = backbone
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    def _encode_prefix(self, image: torch.Tensor) -> torch.Tensor:
        mean = image.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
        std = image.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
        image = (image - mean) / std
        if self.device.type == "cuda":
            context = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        else:
            context = torch.autocast(device_type="cpu", enabled=False)
        with context:
            tokens = self.backbone.prepare_tokens_with_masks(image)
            for block in self.backbone.blocks[:-1]:
                tokens = block(tokens)
        if tokens.shape[1:] != (513, 384):
            raise RuntimeError(f"unexpected B2 prefix shape: {tuple(tokens.shape)}")
        return tokens.float()

    def _forward_images(self, current: torch.Tensor, goal: torch.Tensor) -> dict[str, torch.Tensor]:
        # Encode both images in one batch, matching the frozen cache protocol.
        prefix = self._encode_prefix(torch.cat((current, goal), dim=0))
        current_tokens, goal_tokens = prefix.chunk(2, dim=0)
        if self.device.type == "cuda":
            context = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        else:
            context = torch.autocast(device_type="cpu", enabled=False)
        with context:
            output = self.model(current_tokens, goal_tokens)
        return _validated_result(output.fg_logits.float(), output.fs_scores.float(), self.fg_threshold)

    @torch.inference_mode()
    def single(self, inputs: Mapping[str, object]) -> dict[str, torch.Tensor]:
        validate_online_inputs(inputs)
        if "current_features" in inputs:
            raise ValueError("B2 online inference requires native ERP RGB, not cached features")
        current = normalize_erp(inputs["current_erp_rgb"], device=self.device)
        goal = normalize_erp(inputs["goal_erp_rgb"], device=self.device)
        if current.shape != goal.shape:
            raise ValueError("Current and Goal ERP RGB batch shapes must match")
        return self._forward_images(current, goal)

    @torch.inference_mode()
    def yaw_ensemble(
        self,
        inputs: Mapping[str, object],
        *,
        sector_shifts: Sequence[int] = DEFAULT_ENSEMBLE_SECTOR_SHIFTS,
    ) -> dict[str, torch.Tensor]:
        validate_online_inputs(inputs)
        if "current_features" in inputs:
            raise ValueError("B2 yaw ensemble requires native ERP RGB")
        shifts = tuple(int(shift) % 12 for shift in sector_shifts)
        if not shifts or any(shift % 3 for shift in shifts):
            raise ValueError("B2 exact-pixel ensemble supports integer 90-degree (3-sector) shifts")
        current = normalize_erp(inputs["current_erp_rgb"], device=self.device)
        goal = normalize_erp(inputs["goal_erp_rgb"], device=self.device)
        if current.shape != goal.shape:
            raise ValueError("Current and Goal ERP RGB batch shapes must match")

        aligned_fg = []
        aligned_fs = []
        for shift in shifts:
            pixel_shift = shift * 448 // 12
            rolled = self._forward_images(
                torch.roll(current, shifts=pixel_shift, dims=3),
                torch.roll(goal, shifts=pixel_shift, dims=3),
            )
            aligned_fg.append(torch.roll(rolled["fg_probabilities"], shifts=-shift, dims=1))
            aligned_fs.append(torch.roll(rolled["fs_scores"], shifts=-shift, dims=1))

        fg_probabilities = torch.stack(aligned_fg, dim=0).median(dim=0).values
        fs_scores = torch.stack(aligned_fs, dim=0).median(dim=0).values
        eps = torch.finfo(fg_probabilities.dtype).eps
        fg_logits = torch.logit(fg_probabilities.clamp(eps, 1.0 - eps))
        result = _validated_result(fg_logits, fs_scores, self.fg_threshold)
        result["ensemble_sector_shifts"] = torch.tensor(shifts, device=self.device)
        result["aggregation"] = "componentwise_median_after_inverse_roll"
        return result

    def forward(self, inputs: Mapping[str, object]) -> dict[str, torch.Tensor]:
        return self.single(inputs)
