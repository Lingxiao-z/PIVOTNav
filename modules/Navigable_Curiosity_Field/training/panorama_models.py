from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from ..runtime.fs.geometry import circular_relative_indices


class NTSResNet18Encoder(nn.Module):
    """Supplement Figure 2 RGB encoder ending in a 128-D patch representation."""

    def __init__(self, imagenet_pretrained: bool = True) -> None:
        super().__init__()
        from torchvision.models import ResNet18_Weights, resnet18

        weights = ResNet18_Weights.IMAGENET1K_V1 if imagenet_pretrained else None
        trunk = resnet18(weights=weights)
        self.features = nn.Sequential(*list(trunk.children())[:-2])
        self.conv = nn.Conv2d(512, 32, kernel_size=1)
        self.fc1 = nn.Linear(32 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 128)
        self.dropout = nn.Dropout(0.5)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        feature = self.conv(self.features(image))
        feature = F.adaptive_avg_pool2d(feature, (4, 4)).flatten(1)
        feature = self.dropout(F.relu(self.fc1(feature)))
        return self.fc2(self.dropout(feature))


class NTSInterNodeFS(nn.Module):
    """Supplement-faithful 24x128 concatenation and 3072->256->12 score head."""

    def __init__(self, encoder: nn.Module, feature_dim: int = 128) -> None:
        super().__init__()
        self.encoder = encoder
        self.feature_dim = feature_dim
        self.score_head = nn.Sequential(
            nn.Linear(24 * feature_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 12),
        )

    def initialize_fs_output(self, prior_probability: float = 0.5) -> None:
        """Start the sigmoid head in its responsive region for F_S regression."""
        if not 0.0 < prior_probability < 1.0:
            raise ValueError("prior_probability must be strictly between zero and one")
        final = self.score_head[-1]
        nn.init.zeros_(final.weight)
        nn.init.constant_(final.bias, math.log(prior_probability / (1.0 - prior_probability)))

    def forward(self, source_views: torch.Tensor, goal_views: torch.Tensor) -> torch.Tensor:
        if source_views.shape[:2] != goal_views.shape[:2] or source_views.shape[1] != 12:
            raise ValueError("source and goal must each contain 12 views")
        b = source_views.shape[0]
        source = self.encoder(source_views.flatten(0, 1)).reshape(b, 12, self.feature_dim)
        goal = self.encoder(goal_views.flatten(0, 1)).reshape(b, 12, self.feature_dim)
        return torch.sigmoid(self.score_head(torch.cat((source, goal), dim=1).flatten(1)))


class CircularDirectionEncoder(nn.Module):
    def __init__(self, dim: int, layers: int = 3) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            nn.Sequential(
                nn.Conv1d(dim, dim, kernel_size=3, padding=1, padding_mode="circular"),
                nn.GELU(),
                nn.Conv1d(dim, dim, kernel_size=1),
            )
            for _ in range(layers)
        )
        self.norms = nn.ModuleList(nn.LayerNorm(dim) for _ in range(layers))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        for block, norm in zip(self.blocks, self.norms):
            tokens = norm(tokens + block(tokens.transpose(1, 2)).transpose(1, 2))
        return tokens


class DINOv2PanoramaFS(nn.Module):
    def __init__(self, backbone: nn.Module, backbone_dim: int = 384, dim: int = 384, heads: int = 6) -> None:
        super().__init__()
        self.backbone = backbone
        self.project = nn.Identity() if backbone_dim == dim else nn.Linear(backbone_dim, dim)
        self.circular = CircularDirectionEncoder(dim)
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)
        relative = torch.as_tensor(circular_relative_indices(12), dtype=torch.float32)
        cosine = torch.cos(relative * 2.0 * math.pi / 12.0)
        self.register_buffer("direction_cosine", cosine, persistent=True)
        self.direction_bias_scale = nn.Parameter(torch.tensor(1.0))
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 1))

    def _encode(self, views: torch.Tensor) -> torch.Tensor:
        b = views.shape[0]
        features = self.backbone(views.flatten(0, 1))
        if isinstance(features, dict):
            features = features["x_norm_clstoken"]
        return self.circular(self.project(features).reshape(b, 12, -1))

    def forward(self, source_views: torch.Tensor, goal_views: torch.Tensor) -> torch.Tensor:
        source, goal = self._encode(source_views), self._encode(goal_views)
        return self.forward_features(source, goal, already_encoded=True)

    def forward_features(self, source: torch.Tensor, goal: torch.Tensor, already_encoded: bool = False) -> torch.Tensor:
        if not already_encoded:
            source, goal = self.circular(self.project(source)), self.circular(self.project(goal))
        q = self.q(source)
        attended_alignments = []
        for shift in range(12):
            shifted_goal = torch.roll(goal, shifts=shift, dims=1)
            k, v = self.k(shifted_goal), self.v(shifted_goal)
            logits = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(q.shape[-1])
            logits = logits + self.direction_bias_scale * self.direction_cosine.to(logits.dtype)
            attended_alignments.append(torch.matmul(logits.softmax(dim=-1), v))
        attended = self.out(torch.stack(attended_alignments, dim=0).mean(dim=0))
        return torch.sigmoid(self.head(source + attended).squeeze(-1))


class DINOv2PanoramaFGFS(nn.Module):
    """DINOv2 shared representation with source-only FG and joint FS heads.

    This is the FG/FS successor to :class:`DINOv2PanoramaFS`.  The backbone,
    projection, circular direction encoder, and source-goal alignment are
    shared.  FG is computed from Source tokens only; FS receives Source plus
    aligned Goal tokens, matching the NTS-inspired dependency contract.
    """

    model_name = "dinov2_panorama_fgfs_habitat_gs_v2"

    def __init__(self, backbone: nn.Module, backbone_dim: int = 384, dim: int = 384) -> None:
        super().__init__()
        self.backbone = backbone
        self.project = nn.Identity() if backbone_dim == dim else nn.Linear(backbone_dim, dim)
        self.circular = CircularDirectionEncoder(dim)
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.v = nn.Linear(dim, dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)
        relative = torch.as_tensor(circular_relative_indices(12), dtype=torch.float32)
        self.register_buffer(
            "direction_cosine", torch.cos(relative * 2.0 * math.pi / 12.0), persistent=True
        )
        self.direction_bias_scale = nn.Parameter(torch.tensor(1.0))
        self.fg_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 1))
        self.fs_head = nn.Sequential(
            nn.LayerNorm(2 * dim), nn.Linear(2 * dim, dim), nn.GELU(), nn.Linear(dim, 1)
        )

    def _encode(self, views: torch.Tensor) -> torch.Tensor:
        if views.ndim != 5 or views.shape[1] != 12 or views.shape[2] != 3:
            raise ValueError("views must have shape [B, 12, 3, H, W]")
        batch = views.shape[0]
        features = self.backbone(views.flatten(0, 1))
        if isinstance(features, dict):
            features = features["x_norm_clstoken"]
        return self.circular(self.project(features).reshape(batch, 12, -1))

    def _encode_features(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 3 or features.shape[1] != 12:
            raise ValueError("features must have shape [B, 12, D]")
        return self.circular(self.project(features))

    def _aligned_goal(self, source: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        q = self.q(source)
        alignments = []
        for shift in range(12):
            shifted_goal = torch.roll(goal, shifts=shift, dims=1)
            k, v = self.k(shifted_goal), self.v(shifted_goal)
            logits = torch.matmul(q, k.transpose(1, 2)) / math.sqrt(q.shape[-1])
            logits = logits + self.direction_bias_scale * self.direction_cosine.to(logits.dtype)
            alignments.append(torch.matmul(logits.softmax(dim=-1), v))
        return self.out(torch.stack(alignments, dim=0).mean(dim=0))

    def forward_features(self, source: torch.Tensor, goal: torch.Tensor) -> dict[str, torch.Tensor]:
        source = self._encode_features(source)
        goal = self._encode_features(goal)
        return self._forward_encoded(source, goal)

    def forward(self, source_views: torch.Tensor, goal_views: torch.Tensor) -> dict[str, torch.Tensor]:
        if source_views.ndim == 3 and goal_views.ndim == 3:
            return self.forward_features(source_views, goal_views)
        return self._forward_encoded(self._encode(source_views), self._encode(goal_views))

    def _forward_encoded(self, source: torch.Tensor, goal: torch.Tensor) -> dict[str, torch.Tensor]:
        aligned_goal = self._aligned_goal(source, goal)
        fg_hidden = self.fg_head[:3](source)
        fs_hidden = self.fs_head[:3](torch.cat((source, aligned_goal), dim=-1))
        fg_logits = self.fg_head[3](fg_hidden).squeeze(-1)
        fs_scores = torch.sigmoid(self.fs_head[3](fs_hidden).squeeze(-1))
        return {
            "fg_logits": fg_logits, "fs_scores": fs_scores,
            "fg_hidden": fg_hidden, "fs_hidden": fs_hidden,
        }


def dino_fgfs_loss(
    outputs: dict[str, torch.Tensor],
    fg_target: torch.Tensor,
    fs_target: torch.Tensor,
    fs_valid: torch.Tensor,
    *,
    fg_weight: float = 1.0,
    fs_weight: float = 10.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Joint DINO FG/FS objective with explicit FS validity masking."""
    fg_logits, fs_scores = outputs["fg_logits"], outputs["fs_scores"]
    if fg_logits.shape != fg_target.shape or fs_scores.shape != fs_target.shape or fs_valid.shape != fs_target.shape:
        raise ValueError("DINO FG/FS output, target, and mask shapes must match")
    loss_fg = F.binary_cross_entropy_with_logits(fg_logits, fg_target)
    mask = fs_valid.to(dtype=fs_scores.dtype)
    loss_fs = ((fs_scores - fs_target).square() * mask).sum() / mask.sum().clamp_min(1.0)
    total = fg_weight * loss_fg + fs_weight * loss_fs
    return total, {"loss_fg": loss_fg, "loss_fs": loss_fs, "loss_total": total}
