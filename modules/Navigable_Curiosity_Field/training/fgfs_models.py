from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_resnet18_fgfs_encoder(weight_path: str | Path | None = None) -> nn.Module:
    """Build the single canonical 128-D encoder used by training and inference."""
    from torchvision.models import resnet18

    trunk = resnet18(weights=None)
    if weight_path is not None:
        trunk.load_state_dict(torch.load(weight_path, map_location="cpu", weights_only=True))
    return nn.Sequential(
        nn.Sequential(*list(trunk.children())[:-2]),
        nn.Conv2d(512, 32, kernel_size=1),
        nn.AdaptiveAvgPool2d((4, 4)),
        nn.Flatten(),
        nn.Linear(32 * 4 * 4, 256),
        nn.ReLU(inplace=True),
        nn.Linear(256, 128),
    )


def normalized_nts_views(erp: torch.Tensor) -> torch.Tensor:
    """Resize native ERP, extract wrapped NTS patches, and normalize them."""
    from ..runtime.fs.geometry import nts_wrapped_crops

    if erp.ndim != 4 or erp.shape[1] != 3:
        raise ValueError("ERP must have shape [B, 3, H, W]")
    resized = F.interpolate(erp, size=(128, 512), mode="bilinear", align_corners=False)
    views = nts_wrapped_crops(resized)
    mean = views.new_tensor(IMAGENET_MEAN)[None, None, :, None, None]
    std = views.new_tensor(IMAGENET_STD)[None, None, :, None, None]
    return (views - mean) / std


class FGFSJointModel(nn.Module):
    """Auditable shared-encoder FG/FS joint model."""

    def __init__(self, input_dim: int = 16, hidden_dim: int = 48, directions: int = 12) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.directions = directions
        self.shared_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
        )
        self.fg_head = nn.Linear(hidden_dim, 1)
        self.fs_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(), nn.Dropout(0.0), nn.Linear(hidden_dim, 1)
        )

    def encode(self, views: torch.Tensor) -> torch.Tensor:
        if views.ndim != 3 or views.shape[1] != self.directions or views.shape[2] != self.input_dim:
            raise ValueError(f"expected [B,{self.directions},{self.input_dim}] views")
        b = views.shape[0]
        return self.shared_encoder(views.reshape(b * self.directions, self.input_dim)).reshape(b, self.directions, self.hidden_dim)

    def forward(self, source: torch.Tensor, goal: torch.Tensor) -> dict[str, torch.Tensor]:
        source_encoded = self.encode(source)
        goal_encoded = self.encode(goal)
        fg_logits = self.fg_head(source_encoded).squeeze(-1)
        fs_scores = torch.sigmoid(self.fs_head(torch.cat((source_encoded, goal_encoded), dim=-1)).squeeze(-1))
        return {"fg_logits": fg_logits, "fs_scores": fs_scores}


def joint_loss(outputs, fg_target, fs_target, fs_valid, *, fg_weight=1.0, fs_weight=10.0):
    if fg_target.shape != outputs["fg_logits"].shape:
        raise ValueError("FG target shape mismatch")
    if fs_target.shape != outputs["fs_scores"].shape or fs_valid.shape != fs_target.shape:
        raise ValueError("FS target/mask shape mismatch")
    loss_fg = torch.nn.functional.binary_cross_entropy_with_logits(outputs["fg_logits"], fg_target)
    mask = fs_valid.to(dtype=outputs["fs_scores"].dtype)
    loss_fs = ((outputs["fs_scores"] - fs_target).square() * mask).sum() / mask.sum().clamp_min(1.0)
    total = fg_weight * loss_fg + fs_weight * loss_fs
    return total, {"loss_fg": loss_fg, "loss_fs": loss_fs, "loss_total": total}


class NTSInspiredResNet18FGFS(nn.Module):
    """Shared NTS-style RGB encoder with distinct FG and FS heads.

    The module consumes the paper's 12 wrapped patches. FG sees only all Source
    patch features. FS sees all Source and Goal patch features.
    """

    model_name = "nts_inspired_resnet18_fgfs_habitat_gs_v2"

    def __init__(self, encoder: nn.Module, feature_dim: int = 128, dropout: float = 0.5) -> None:
        super().__init__()
        self.encoder = encoder
        self.feature_dim = feature_dim
        self.fg_head = nn.Sequential(
            nn.Linear(12 * feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 12),
        )
        self.fs_head = nn.Sequential(
            nn.Linear(24 * feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 12),
        )
        self._reset_head_parameters()

    def _reset_head_parameters(self) -> None:
        for head in (self.fg_head, self.fs_head):
            nn.init.kaiming_normal_(head[0].weight, mode="fan_in", nonlinearity="relu")
            nn.init.constant_(head[0].bias, 0.1)
            nn.init.xavier_uniform_(head[-1].weight, gain=0.1)
            nn.init.zeros_(head[-1].bias)

    def _encode(self, views: torch.Tensor) -> torch.Tensor:
        if views.ndim != 5 or views.shape[1] != 12 or views.shape[2] != 3:
            raise ValueError("views must have shape [B, 12, 3, H, W]")
        batch = views.shape[0]
        return self.encoder(views.flatten(0, 1)).reshape(batch, 12, self.feature_dim)

    def forward(self, source_views: torch.Tensor, goal_views: torch.Tensor) -> dict[str, torch.Tensor]:
        if source_views.shape != goal_views.shape:
            raise ValueError("Source and Goal view shapes must match")
        source = self._encode(source_views)
        goal = self._encode(goal_views)
        return self.forward_encoded(source, goal)

    def forward_encoded(self, source: torch.Tensor, goal: torch.Tensor) -> dict[str, torch.Tensor]:
        if source.shape != goal.shape or source.ndim != 3 or source.shape[1:] != (12, self.feature_dim):
            raise ValueError(f"encoded features must have matching [B,12,{self.feature_dim}] shapes")
        source_flat = source.flatten(1)
        fg_logits = self.fg_head(source_flat)
        fs_scores = torch.sigmoid(self.fs_head(torch.cat((source, goal), dim=1).flatten(1)))
        return {"fg_logits": fg_logits, "fs_scores": fs_scores}

    def encode_erp(self, erp: torch.Tensor) -> torch.Tensor:
        return self._encode(normalized_nts_views(erp))

    def forward_erp(self, source_erp: torch.Tensor, goal_erp: torch.Tensor) -> dict[str, torch.Tensor]:
        if source_erp.ndim != 4 or source_erp.shape[1] != 3 or source_erp.shape != goal_erp.shape:
            raise ValueError("Source and Goal ERP must have matching [B, 3, H, W] shapes")
        return self.forward_encoded(self.encode_erp(source_erp), self.encode_erp(goal_erp))
