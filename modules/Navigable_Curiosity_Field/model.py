from __future__ import annotations


# ---------------------------------------------------------------------------
# Curiosity network
# ---------------------------------------------------------------------------

"""Revised DINOv2 panorama FG/FS direction decoder (v3 protocol).

The module consumes frozen ViT-S/14 patch grids shaped [B, 2, 16, 32, 384]
or pre-projected sector features shaped [B, 2, 12, 384].  The decoder keeps
the 384-dimensional DINO representation and uses explicit circular queries,
current/goal type embeddings, periodic longitude positions, and target-
conditioned cosine evidence.  It is deliberately separate from the legacy
rank-v1 implementation and checkpoint namespace.
"""

import math
from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import nn


class RevisedDINOv3Output(NamedTuple):
    fg_logits: torch.Tensor
    fs_scores: torch.Tensor
    direction_features: torch.Tensor
    similarity: torch.Tensor


def _circular_longitude_encoding(width: int, dim: int, device, dtype):
    if dim < 4 or dim % 2:
        raise ValueError("position dimension must be an even number >= 4")
    longitude = torch.arange(width, device=device, dtype=dtype) / max(width, 1)
    frequency = torch.arange(1, dim // 2 + 1, device=device, dtype=dtype)
    phase = 2.0 * math.pi * longitude[:, None] * frequency[None, :]
    return torch.cat((phase.sin(), phase.cos()), dim=-1)


class PeriodicERPPosition(nn.Module):
    def __init__(self, dim: int, height: int = 16, width: int = 32):
        super().__init__()
        self.dim, self.height, self.width = dim, height, width
        self.latitude = nn.Parameter(torch.zeros(height, dim))
        nn.init.normal_(self.latitude, std=0.02)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        b, n, d = tokens.shape
        if n != self.height * self.width or d != self.dim:
            raise ValueError("ERP tokens must have [B,16*32,384] shape")
        lon = _circular_longitude_encoding(self.width, self.dim, tokens.device, tokens.dtype)
        pos = lon[None, :, :] + self.latitude[:, None, :]
        return tokens + pos.reshape(1, n, d)


class RevisedDINOv2PanoramaFGFSV3(nn.Module):
    model_name = "revised_dinov2_panorama_fgfs_v3"

    def __init__(self, dim: int = 384, heads: int = 6, decoder_layers: int = 2,
                 ffn_dim: int = 1536, dropout: float = 0.1, direction_count: int = 12):
        super().__init__()
        if dim != 384 or heads != 6 or ffn_dim != 1536 or decoder_layers != 2:
            raise ValueError("v3 protocol fixes ViT-S/14 decoder dimensions")
        self.dim, self.direction_count = dim, direction_count
        self.position = PeriodicERPPosition(dim)
        self.type_embedding = nn.Embedding(2, dim)
        self.direction_query = nn.Parameter(torch.randn(direction_count, dim) * 0.02)
        angles = torch.arange(direction_count, dtype=torch.float32) * 2 * math.pi / direction_count
        self.register_buffer("direction_angle", torch.stack((angles.sin(), angles.cos()), -1), persistent=True)
        self.angle_projection = nn.Linear(2, dim, bias=False)
        self.target_projection = nn.Linear(dim * 3, dim)
        layer = nn.TransformerDecoderLayer(dim, heads, ffn_dim, dropout=dropout,
                                           activation="gelu", batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(layer, decoder_layers)
        self.fg_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 128), nn.GELU(), nn.Linear(128, 1))
        self.fs_head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 128), nn.GELU(), nn.Linear(128, 1))

    def _grid_tokens(self, grid: torch.Tensor, type_id: int) -> torch.Tensor:
        if grid.ndim != 4 or grid.shape[1:] != (16, 32, self.dim):
            raise ValueError("patch grid must have shape [B,16,32,384]")
        b = grid.shape[0]
        tokens = self.position(grid.reshape(b, 16 * 32, self.dim))
        return tokens + self.type_embedding.weight[type_id].view(1, 1, -1)

    def _sector_tokens(self, grid: torch.Tensor) -> torch.Tensor:
        # Twelve wrapped ERP sectors retain circular order while reducing the
        # decoder memory footprint.  Each sector aggregates a 2- or 3-column
        # band with wrap-around indexing.
        b, h, w, d = grid.shape
        centers = torch.arange(self.direction_count, device=grid.device) * w / self.direction_count
        columns = (centers[:, None] + torch.tensor([-1., 0., 1.], device=grid.device)[None, :]).round().long() % w
        gathered = grid[:, :, columns, :].mean(dim=1)
        return gathered.mean(dim=2)

    def forward(self, current_grid: torch.Tensor, goal_grid: torch.Tensor) -> RevisedDINOv3Output:
        current = self._grid_tokens(current_grid, 0)
        goal = self._grid_tokens(goal_grid, 1)
        current_patch = current_grid.reshape(current_grid.shape[0], 16 * 32, self.dim)
        goal_patch = goal_grid.reshape(goal_grid.shape[0], 16 * 32, self.dim)
        current_norm = F.normalize(current_patch, dim=-1)
        goal_norm = F.normalize(goal_patch, dim=-1)
        similarity = torch.einsum("bid,bjd->bij", current_norm, goal_norm)
        topk = min(4, similarity.shape[-1])
        weights, indices = similarity.softmax(-1).topk(topk, dim=-1)
        goal_bank = goal_patch[:, None, :, :].expand(-1, 16 * 32, -1, -1)
        aligned = torch.gather(goal_bank, 2, indices[..., None].expand(-1, -1, -1, self.dim))
        aligned = (weights[..., None] * aligned).sum(dim=2)
        queries = self.direction_query[None].expand(current_grid.shape[0], -1, -1)
        queries = queries + self.angle_projection(self.direction_angle.to(queries)).unsqueeze(0)
        target_bias = torch.cat((current_norm, F.normalize(aligned, dim=-1),
                                 (current_norm - F.normalize(aligned, dim=-1)).abs()), dim=-1)
        evidence = self.target_projection(target_bias)
        memory = torch.cat((current + evidence, goal), dim=1)
        decoded = self.decoder(queries, memory)
        fg_logits = self.fg_head(decoded).squeeze(-1)
        fs_scores = self.fs_head(decoded).squeeze(-1)
        return RevisedDINOv3Output(fg_logits, fs_scores, decoded, similarity)


def _zero(reference):
    return reference.sum() * 0.0


def revised_dino_v3_loss(output: RevisedDINOv3Output, fg_target: torch.Tensor,
                         fs_target: torch.Tensor, fs_valid: torch.Tensor,
                         *, fg_alpha: float = 0.53, focal_gamma: float = 2.0,
                         huber_delta: float = 0.1, ranking_margin: float = 0.05,
                         topk_temperature: float = 0.1,
                         cyclic_output: RevisedDINOv3Output | None = None,
                         cyclic_shift: int = 0) -> dict[str, torch.Tensor]:
    fg_target = fg_target.float(); fs_target = fs_target.float(); mask = fs_valid.bool()
    alpha = torch.where(fg_target > 0.5, torch.as_tensor(fg_alpha, device=fg_target.device),
                        torch.as_tensor(1.0 - fg_alpha, device=fg_target.device))
    bce = F.binary_cross_entropy_with_logits(output.fg_logits, fg_target, reduction="none")
    p = torch.sigmoid(output.fg_logits)
    focal = (alpha * (1.0 - torch.where(fg_target > 0.5, p, 1.0 - p)).pow(focal_gamma) * bce).mean()
    if mask.any():
        reg = F.smooth_l1_loss(output.fs_scores[mask], fs_target[mask], beta=huber_delta)
    else:
        reg = _zero(output.fs_scores)
    delta = fs_target[:, :, None] - fs_target[:, None, :]
    pred_delta = output.fs_scores[:, :, None] - output.fs_scores[:, None, :]
    pair = mask[:, :, None] & mask[:, None, :] & (delta.abs() >= ranking_margin)
    upper = torch.triu(torch.ones(12, 12, device=pair.device, dtype=torch.bool), diagonal=1)
    pair &= upper
    if pair.any():
        sign = delta[pair].sign(); rank = F.softplus(-sign * pred_delta[pair]).mean()
    else:
        rank = _zero(output.fs_scores)
    rows = mask.any(-1)
    if rows.any():
        target_prob = F.softmax((fs_target[rows].masked_fill(~mask[rows], -1e4)) / topk_temperature, -1)
        pred_logprob = F.log_softmax(output.fs_scores[rows].masked_fill(~mask[rows], -1e4) / topk_temperature, -1)
        topk = -(target_prob * pred_logprob).sum(-1).mean()
    else:
        topk = _zero(output.fs_scores)
    # Cyclic consistency is evaluated on a separately rolled ERP when the
    # caller supplies it.  The inverse roll maps the rolled prediction back
    # to the canonical sector order before comparing probabilities/scores.
    if cyclic_output is not None and cyclic_shift % 12:
        shift = int(cyclic_shift) % 12
        rolled_fg = torch.roll(cyclic_output.fg_logits, shifts=-shift, dims=1)
        rolled_fs = torch.roll(cyclic_output.fs_scores, shifts=-shift, dims=1)
        fg_consistency = F.mse_loss(torch.sigmoid(output.fg_logits), torch.sigmoid(rolled_fg))
        base_prob = F.softmax(output.fs_scores.masked_fill(~mask, -1e4), dim=-1)
        roll_mask = mask
        roll_prob = F.softmax(rolled_fs.masked_fill(~roll_mask, -1e4), dim=-1)
        valid_rows = mask.any(-1) & roll_mask.any(-1)
        if valid_rows.any():
            cyclic = fg_consistency + F.kl_div(
                (base_prob[valid_rows] + 1e-8).log(),
                roll_prob[valid_rows].detach() + 1e-8,
                reduction="batchmean",
            )
        else:
            cyclic = fg_consistency
    else:
        cyclic = _zero(output.fs_scores)

    # Smooth only neighboring valid sectors and weight the penalty by target
    # similarity, preserving genuine discontinuities between distinct exits.
    adjacent = mask & torch.roll(mask, shifts=-1, dims=1)
    target_similarity = torch.exp(-torch.abs(fs_target - torch.roll(fs_target, shifts=-1, dims=1)) / 0.1)
    score_delta = output.fs_scores - torch.roll(output.fs_scores, shifts=-1, dims=1)
    if adjacent.any():
        smooth = (target_similarity[adjacent] * score_delta[adjacent].pow(2)).mean()
    else:
        smooth = _zero(output.fs_scores)
    total = focal + 2.0 * reg + rank + topk + 0.5 * cyclic + 0.1 * smooth
    return {"loss_fg": focal, "loss_fs_reg": reg, "loss_fs_rank": rank,
            "loss_topk": topk, "loss_cyclic": cyclic,
            "loss_smooth": smooth, "loss_total": total}


class RevisedDINOv2PanoramaFGFSV3LastBlockB2(nn.Module):
    """Frozen-prefix wrapper used by the released checkpoint."""

    model_name = "revised_dinov2_panorama_fgfs_v3_last_block_b2"

    def __init__(
        self,
        last_block: nn.Module,
        final_norm: nn.Module,
        decoder: RevisedDINOv2PanoramaFGFSV3,
    ) -> None:
        super().__init__()
        self.last_block = last_block
        self.final_norm = final_norm
        self.decoder = decoder
        for parameter in self.final_norm.parameters():
            parameter.requires_grad_(False)

    def encode_cached_prefix(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 3 or tokens.shape[1:] != (513, 384):
            raise ValueError("cached DINO prefix tokens must have shape [B,513,384]")
        with torch.autocast(device_type="cuda", enabled=False):
            encoded = self.final_norm(self.last_block(tokens.float()))
        return encoded[:, 1:].reshape(tokens.shape[0], 16, 32, 384)

    def forward(
        self, current_tokens: torch.Tensor, goal_tokens: torch.Tensor
    ) -> RevisedDINOv3Output:
        return self.decoder(
            self.encode_cached_prefix(current_tokens),
            self.encode_cached_prefix(goal_tokens),
        )


# ---------------------------------------------------------------------------
# ERP input validation and losses
# ---------------------------------------------------------------------------

"""Input validation and score utilities for final curiosity inference."""

from typing import Mapping

import numpy as np
import torch


FORBIDDEN_ONLINE_FIELDS = frozenset({
    "pose", "gt_pose", "yaw", "gt_yaw", "goal_position", "gt_goal_position",
    "navmesh", "depth", "gps", "compass", "geodesic_distance", "collision",
    "shortest_path", "success", "spl", "oracle_direction", "oracle_action",
})
ALLOWED_ONLINE_FIELDS = frozenset({
    "current_erp_rgb", "goal_erp_rgb", "current_features", "goal_features",
})


def validate_online_inputs(inputs: Mapping[str, object]) -> None:
    keys = set(inputs)
    forbidden = keys & FORBIDDEN_ONLINE_FIELDS
    unknown = keys - ALLOWED_ONLINE_FIELDS
    if forbidden:
        raise ValueError(f"forbidden online curiosity inputs: {sorted(forbidden)}")
    if unknown:
        raise ValueError(f"unrecognized online curiosity inputs: {sorted(unknown)}")
    image_pair = {"current_erp_rgb", "goal_erp_rgb"} <= keys
    feature_pair = {"current_features", "goal_features"} <= keys
    if image_pair == feature_pair:
        raise ValueError("provide exactly one complete Current/Goal RGB or feature pair")


def normalize_erp(image: object, *, device: torch.device) -> torch.Tensor:
    tensor = torch.as_tensor(np.asarray(image) if isinstance(image, np.ndarray) else image)
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 4:
        raise ValueError("ERP RGB must be an unbatched or batched image")
    if tensor.shape[-3:] == (3, 224, 448):
        pass
    elif tensor.shape[-3:] == (224, 448, 3):
        tensor = tensor.permute(0, 3, 1, 2)
    else:
        raise ValueError("ERP RGB must have shape [224,448,3] or [B,3,224,448]")
    if tensor.dtype == torch.uint8:
        tensor = tensor.to(device=device, dtype=torch.float32, non_blocking=True).div_(255.0)
    elif tensor.is_floating_point():
        tensor = tensor.to(device=device, dtype=torch.float32, non_blocking=True)
        if not bool(torch.isfinite(tensor).all()) or float(tensor.min()) < 0.0 or float(tensor.max()) > 1.0:
            raise ValueError("floating-point ERP RGB must be finite and normalized to [0,1]")
    else:
        raise TypeError("ERP RGB must use uint8 or floating-point values")
    return tensor.contiguous()


def fs_scores(
    endpoint_goal_geodesic: np.ndarray,
    valid_direction: np.ndarray,
    *,
    dmax: float = 20.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the training target for the twelve candidate directions."""
    distances = np.asarray(endpoint_goal_geodesic, dtype=np.float64)
    valid = np.asarray(valid_direction, dtype=bool)
    if distances.shape[-1] != 12 or valid.shape != distances.shape:
        raise ValueError("distances and mask must have matching [..., 12] shapes")
    if not np.isfinite(dmax) or dmax <= 0:
        raise ValueError("dmax must be finite and positive")
    unreachable = valid & ~np.isfinite(distances)
    finite_valid = valid & np.isfinite(distances)
    scores = np.zeros_like(distances, dtype=np.float32)
    scores[finite_valid] = np.maximum(1.0 - distances[finite_valid] / dmax, 0.0)
    return scores, unreachable


def masked_mse(prediction: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor):
    mask = valid_mask.to(dtype=prediction.dtype)
    return ((prediction - target).square() * mask).sum() / mask.sum().clamp_min(1.0)


# ---------------------------------------------------------------------------
# Panoramic geometry helpers
# ---------------------------------------------------------------------------

import math

import numpy as np
import torch
import torch.nn.functional as F


def roll_source(erp: np.ndarray, labels: np.ndarray, sectors: int) -> tuple[np.ndarray, np.ndarray]:
    """Roll Source ERP and labels together by an integer sector count."""
    width = erp.shape[-2]
    if width % 12:
        raise ValueError("ERP width must be divisible by 12 for exact sector rolls")
    return np.roll(erp, sectors * (width // 12), axis=-2), np.roll(labels, sectors, axis=-1)


def roll_goal(erp: np.ndarray, sectors: int) -> np.ndarray:
    """Roll Goal ERP independently; Source-coordinate labels do not move."""
    width = erp.shape[-2]
    if width % 12:
        raise ValueError("ERP width must be divisible by 12 for exact sector rolls")
    return np.roll(erp, sectors * (width // 12), axis=-2)


def circular_relative_indices(n: int = 12) -> np.ndarray:
    idx = np.arange(n)
    delta = idx[None, :] - idx[:, None]
    return (delta + n // 2) % n - n // 2


def direction_cosine_matrix(n: int = 12) -> np.ndarray:
    return np.cos(2.0 * np.pi * circular_relative_indices(n) / n).astype(np.float32)


def erp_to_perspective(
    erp: torch.Tensor,
    *,
    yaw_degrees: torch.Tensor,
    output_size: int = 224,
    horizontal_fov_degrees: float = 90.0,
) -> torch.Tensor:
    """Project ERP images to perspective views with horizontal seam wrapping."""
    if erp.ndim != 4 or erp.shape[1] != 3:
        raise ValueError("erp must have shape [B, 3, H, W]")
    if yaw_degrees.ndim != 1:
        raise ValueError("yaw_degrees must be one-dimensional")
    b, _, _, _ = erp.shape
    v = yaw_degrees.numel()
    device, dtype = erp.device, erp.dtype
    extent = math.tan(math.radians(horizontal_fov_degrees) / 2.0)
    xy = torch.linspace(-extent, extent, output_size, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(xy, xy, indexing="ij")
    directions = torch.stack((xx, -yy, torch.ones_like(xx)), dim=-1)
    directions = directions / directions.norm(dim=-1, keepdim=True)
    yaw = torch.deg2rad(yaw_degrees.to(device=device, dtype=dtype))
    cos_yaw, sin_yaw = yaw.cos(), yaw.sin()
    x = directions[..., 0][None] * cos_yaw[:, None, None] + directions[..., 2][None] * sin_yaw[:, None, None]
    z = -directions[..., 0][None] * sin_yaw[:, None, None] + directions[..., 2][None] * cos_yaw[:, None, None]
    latitude = torch.asin(directions[..., 1][None].expand(v, -1, -1).clamp(-1.0, 1.0))
    longitude = torch.atan2(x, z)
    grid = torch.stack((longitude / math.pi / 3.0, -2.0 * latitude / math.pi), dim=-1)
    tiled = torch.cat((erp, erp, erp), dim=-1)
    expanded = tiled[:, None].expand(b, v, 3, tiled.shape[-2], tiled.shape[-1]).reshape(
        b * v, 3, tiled.shape[-2], tiled.shape[-1]
    )
    grid = grid[None].expand(b, v, output_size, output_size, 2).reshape(
        b * v, output_size, output_size, 2
    )
    views = F.grid_sample(expanded, grid, mode="bilinear", padding_mode="border", align_corners=False)
    return views.reshape(b, v, 3, output_size, output_size)


def twelve_sector_views(erp: torch.Tensor, output_size: int = 224) -> torch.Tensor:
    yaws = torch.arange(12, device=erp.device, dtype=erp.dtype) * 30.0
    return erp_to_perspective(erp, yaw_degrees=yaws, output_size=output_size)


# ---------------------------------------------------------------------------
# Checkpoint loading and inference
# ---------------------------------------------------------------------------

import hashlib
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
from torch import nn



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
