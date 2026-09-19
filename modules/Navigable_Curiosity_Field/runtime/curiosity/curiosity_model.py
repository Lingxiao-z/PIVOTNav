"""Revised DINOv2 panorama FG/FS direction decoder (v3 protocol).

The module consumes frozen ViT-S/14 patch grids shaped [B, 2, 16, 32, 384]
or pre-projected sector features shaped [B, 2, 12, 384].  The decoder keeps
the 384-dimensional DINO representation and uses explicit circular queries,
current/goal type embeddings, periodic longitude positions, and target-
conditioned cosine evidence.  It is deliberately separate from the legacy
rank-v1 implementation and checkpoint namespace.
"""
from __future__ import annotations

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
