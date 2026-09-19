"""Shared panoramic descriptor and retrieval model."""
from __future__ import annotations


# ---------------------------------------------------------------------------
# DINOv2 panoramic backbone
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Panoramic descriptor
# ---------------------------------------------------------------------------

import math
from dataclasses import dataclass
from typing import Dict

import torch
from torch import nn
from torch.nn import functional as F


DESCRIPTOR_SCHEMA_VERSION = "pano_vpr_v2_erp224_salad2112_ring32x128_v1"


@dataclass(frozen=True)
class ModelConfig:
    token_dim: int = 384
    salad_local_dim: int = 32
    salad_clusters: int = 64
    ring_dim: int = 128
    ring_bins: int = 32
    token_h: int = 16
    token_w: int = 32
    refinement_layers: int = 2

    @property
    def global_dim(self) -> int:
        return self.salad_local_dim * self.salad_clusters + self.salad_clusters

    @property
    def descriptor_schema_version(self) -> str:
        return DESCRIPTOR_SCHEMA_VERSION


class HorizontalCircularConv2d(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        pad = kernel_size // 2
        self.horizontal_pad = pad
        self.vertical_pad = pad
        self.conv = nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=0, groups=1, bias=False)
        self.norm = nn.LayerNorm(channels)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,C,H,W]. Longitude uses circular padding; latitude uses replicate padding.
        if self.horizontal_pad:
            x = F.pad(x, (self.horizontal_pad, self.horizontal_pad, 0, 0), mode='circular')
            x = F.pad(x, (0, 0, self.vertical_pad, self.vertical_pad), mode='replicate')
        y = self.conv(x)
        y = y.permute(0, 2, 3, 1)
        y = self.act(self.norm(y))
        return y.permute(0, 3, 1, 2).contiguous()


class CyclicTokenRefinement(nn.Module):
    def __init__(self, channels: int, layers: int = 2):
        super().__init__()
        self.layers = nn.ModuleList([HorizontalCircularConv2d(channels) for _ in range(layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = x + layer(x)
        return x


def salad_log_otp_solver(log_a: torch.Tensor, log_b: torch.Tensor, scores: torch.Tensor, num_iters: int = 20, reg: float = 1.0) -> torch.Tensor:
    """SALAD/OpenGlue Sinkhorn solver in log space.

    This is a shape-stable adaptation of the official SALAD implementation at
    v2/third_party/salad/models/aggregators/salad.py. It avoids the reference
    `.squeeze()` calls so batch size 1 keeps explicit batch dimensions.
    """
    scores = scores / float(reg)
    u, v = torch.zeros_like(log_a), torch.zeros_like(log_b)
    for _ in range(int(num_iters)):
        u = log_a - torch.logsumexp(scores + v.unsqueeze(1), dim=2)
        v = log_b - torch.logsumexp(scores + u.unsqueeze(2), dim=1)
    return scores + u.unsqueeze(2) + v.unsqueeze(1)


def salad_get_matching_probs(scores: torch.Tensor, dustbin_score: torch.Tensor | float = 1.0, num_iters: int = 3, reg: float = 1.0) -> torch.Tensor:
    """Return SALAD Sinkhorn log matching probabilities with a dustbin row.

    Args:
        scores: Similarity matrix [B, clusters, features].
        dustbin_score: Scalar learned dustbin score z.
    """
    if scores.ndim != 3:
        raise ValueError("scores must be [B, clusters, features]")
    batch_size, clusters, features = scores.shape
    if features <= clusters:
        raise ValueError(f"SALAD OT expects features > clusters, got features={features}, clusters={clusters}")
    augmented = torch.empty(batch_size, clusters + 1, features, dtype=scores.dtype, device=scores.device)
    augmented[:, :clusters, :] = scores
    augmented[:, clusters, :] = torch.as_tensor(dustbin_score, dtype=scores.dtype, device=scores.device)

    norm = -torch.tensor(math.log(features + clusters), device=scores.device, dtype=scores.dtype)
    log_a = norm.expand(clusters + 1).clone()
    log_b = norm.expand(features).clone()
    log_a[-1] = log_a[-1] + math.log(features - clusters)
    log_a = log_a.expand(batch_size, -1)
    log_b = log_b.expand(batch_size, -1)
    return salad_log_otp_solver(log_a, log_b, augmented, num_iters=num_iters, reg=reg) - norm


def latitude_cosine_weights(height: int, device=None, dtype=None) -> torch.Tensor:
    # ERP latitude centers from north(+pi/2) to south(-pi/2). cos(latitude)
    # downweights polar regions and is normalized to mean 1 for stable scale.
    rows = torch.arange(height, device=device, dtype=dtype or torch.float32) + 0.5
    lat = (0.5 - rows / float(height)) * torch.pi
    weights = torch.cos(lat).clamp_min(1e-4)
    return weights / weights.mean()


class SALADCompatibleOT(nn.Module):
    """SALAD-compatible global head with Sinkhorn OT and a learned dustbin.

    The descriptor keeps the project target shape: 64 clusters x 32 residual
    dims plus 64 cluster-mass features = 2112 dimensions.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.local = nn.Linear(cfg.token_dim, cfg.salad_local_dim, bias=False)
        self.score = nn.Linear(cfg.token_dim, cfg.salad_clusters, bias=True)
        self.dust_bin = nn.Parameter(torch.tensor(1.0))
        self.sinkhorn_iters = 3
        self.sinkhorn_reg = 1.0

    def forward(self, tokens_bchw: torch.Tensor) -> Dict[str, torch.Tensor]:
        b, c, h, w = tokens_bchw.shape
        if c != self.cfg.token_dim or (h, w) != (self.cfg.token_h, self.cfg.token_w):
            raise ValueError(
                f"expected [B,{self.cfg.token_dim},{self.cfg.token_h},{self.cfg.token_w}], "
                f"got {tuple(tokens_bchw.shape)}"
            )
        tokens = tokens_bchw.permute(0, 2, 3, 1).reshape(b, h * w, c)
        lat = latitude_cosine_weights(h, tokens_bchw.device, tokens_bchw.dtype).view(1, h, 1, 1)
        flat_weights = lat.expand(b, h, w, 1).reshape(b, h * w, 1)
        # Sinkhorn is intentionally evaluated in float32 under AMP; fp16
        # log-sum-exp can underflow and silently collapse cluster assignment.
        local = self.local(tokens).float()
        scores = self.score(tokens).transpose(1, 2).contiguous().float()  # [B,K,N]
        log_probs_aug = salad_get_matching_probs(scores, self.dust_bin.float(), self.sinkhorn_iters, self.sinkhorn_reg)
        probs_aug = torch.exp(log_probs_aug)
        probs = probs_aug[:, :-1, :] * flat_weights.transpose(1, 2)
        cluster_mass_raw = probs.sum(dim=-1).clamp_min(1e-6)
        cluster_mass = cluster_mass_raw / cluster_mass_raw.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        dustbin_mass = probs_aug[:, -1, :].sum(dim=-1, keepdim=True)
        assigned_mass = probs.sum(dim=(1, 2), keepdim=False).unsqueeze(-1)
        dustbin_fraction = dustbin_mass / (dustbin_mass + assigned_mass).clamp_min(1e-6)
        weighted_residual = torch.einsum('bkn,bnd->bdk', probs, local)
        residual = F.normalize(weighted_residual, p=2, dim=1).transpose(1, 2).reshape(b, -1)
        descriptor = torch.cat([residual, cluster_mass], dim=-1)
        return {
            'global': F.normalize(descriptor, dim=-1),
            'cluster_mass': cluster_mass,
            'dustbin_mass': dustbin_mass,
            'dustbin_fraction': dustbin_fraction,
            'salad_transport': probs,
        }


class PanoSaladRingV2Head(nn.Module):
    def __init__(self, cfg: ModelConfig | None = None):
        super().__init__()
        self.cfg = cfg or ModelConfig()
        self.refine = CyclicTokenRefinement(self.cfg.token_dim, self.cfg.refinement_layers)
        self.salad = SALADCompatibleOT(self.cfg)
        self.ring = nn.Linear(self.cfg.token_dim, self.cfg.ring_dim, bias=False)

    def forward_tokens(self, tokens: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Accept [B,H,W,C] DINO patch tokens or [B,C,H,W].
        if tokens.ndim != 4:
            raise ValueError('tokens must be [B,H,W,C] or [B,C,H,W]')
        if tokens.shape[-1] == self.cfg.token_dim:
            x = tokens.permute(0, 3, 1, 2).contiguous()
        elif tokens.shape[1] == self.cfg.token_dim:
            x = tokens.contiguous()
        else:
            raise ValueError(f'token dim mismatch for {tuple(tokens.shape)}')
        if x.shape[-2:] != (self.cfg.token_h, self.cfg.token_w):
            raise ValueError(f'expected token grid {(self.cfg.token_h, self.cfg.token_w)}, got {tuple(x.shape[-2:])}')
        x = self.refine(x)
        salad = self.salad(x)
        lat = latitude_cosine_weights(x.shape[2], x.device, x.dtype).view(1, 1, x.shape[2], 1)
        ring_tokens = (x * lat).sum(dim=2) / lat.sum(dim=2).clamp_min(1e-6)  # [B,C,W]
        ring_tokens = ring_tokens.permute(0, 2, 1).contiguous()  # [B,W,C]
        ring = F.normalize(self.ring(ring_tokens), dim=-1)
        return {
            'global': salad['global'],
            'ring': ring,
            'cluster_mass': salad['cluster_mass'],
            'dustbin_mass': salad['dustbin_mass'],
            'dustbin_fraction': salad['dustbin_fraction'],
        }


def circular_correlation(query_ring: torch.Tensor, candidate_ring: torch.Tensor) -> torch.Tensor:
    if query_ring.shape != candidate_ring.shape:
        raise ValueError('query_ring and candidate_ring must have same shape')
    if query_ring.ndim != 3:
        raise ValueError('ring descriptors must be [B,L,D]')
    bins = query_ring.shape[1]
    # C[s] = mean_n dot(query[n], candidate[n-s]). FFT removes 32 Python
    # launches while preserving the original circular-shift convention.
    query_fft = torch.fft.rfft(query_ring.float(), dim=1)
    candidate_fft = torch.fft.rfft(candidate_ring.float(), dim=1)
    correlation = torch.fft.irfft(query_fft * candidate_fft.conj(), n=bins, dim=1)
    return correlation.sum(dim=-1) / float(bins)


# ---------------------------------------------------------------------------
# Open-set retrieval matcher
# ---------------------------------------------------------------------------

from dataclasses import dataclass
from typing import Dict

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class MatcherConfig:
    descriptor_dim: int = 2112
    ring_bins: int = 32
    ring_dim: int = 128
    cluster_count: int = 64
    pair_projection_dim: int = 64
    hidden_dim: int = 192
    top_k: int = 8
    dropout: float = 0.1


def _masked_mean(values: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    weights = mask.to(values.dtype)
    while weights.ndim < values.ndim:
        weights = weights.unsqueeze(-1)
    return (values * weights).sum(dim=dim) / weights.sum(dim=dim).clamp_min(1.0)


def ring_correlation_matrix(query_ring: torch.Tensor, candidate_ring: torch.Tensor) -> torch.Tensor:
    """Compute all circular candidate-to-query alignments.

    Args:
        query_ring: [B,L,R]
        candidate_ring: [B,K,L,R]
    Returns:
        Correlation scores [B,K,L]. A positive shift rolls the candidate ring
        right to match the query frame, matching the project yaw convention.
    """
    if query_ring.ndim != 3 or candidate_ring.ndim != 4:
        raise ValueError("query_ring must be [B,L,R] and candidate_ring [B,K,L,R]")
    if query_ring.shape[0] != candidate_ring.shape[0] or query_ring.shape[1:] != candidate_ring.shape[2:]:
        raise ValueError(f"ring shape mismatch: {tuple(query_ring.shape)} vs {tuple(candidate_ring.shape)}")
    scores = []
    query = F.normalize(query_ring.float(), dim=-1).unsqueeze(1)
    candidates = F.normalize(candidate_ring.float(), dim=-1)
    for shift in range(query_ring.shape[1]):
        aligned = torch.roll(candidates, shifts=shift, dims=2)
        scores.append((query * aligned).sum(dim=-1).mean(dim=-1))
    return torch.stack(scores, dim=-1)


class TopKOpenSetMatcher(nn.Module):
    """Small learned verifier applied only to retrieved Top-K candidates.

    Global descriptors perform scalable retrieval. This module then combines
    descriptor interactions, ring alignment and SALAD assignment statistics to
    classify candidate nodes jointly with one explicit UNKNOWN state.
    """

    scalar_feature_dim = 9

    def __init__(self, cfg: MatcherConfig | None = None):
        super().__init__()
        self.cfg = cfg or MatcherConfig()
        p = self.cfg.pair_projection_dim
        h = self.cfg.hidden_dim
        self.abs_projection = nn.Linear(self.cfg.descriptor_dim, p, bias=False)
        self.product_projection = nn.Linear(self.cfg.descriptor_dim, p, bias=False)
        pair_dim = p * 2 + self.scalar_feature_dim
        self.pair_encoder = nn.Sequential(
            nn.LayerNorm(pair_dim),
            nn.Linear(pair_dim, h),
            nn.GELU(),
            nn.Dropout(self.cfg.dropout),
            nn.Linear(h, h),
            nn.GELU(),
        )
        self.same_place_head = nn.Linear(h, 1)
        self.unknown_head = nn.Sequential(
            nn.LayerNorm(h * 2 + 4),
            nn.Linear(h * 2 + 4, h),
            nn.GELU(),
            nn.Dropout(self.cfg.dropout),
            nn.Linear(h, 1),
        )

    def _validate(self, query: Dict[str, torch.Tensor], candidates: Dict[str, torch.Tensor]) -> None:
        required = {"global", "ring", "cluster_mass", "dustbin_fraction"}
        missing_query = required.difference(query)
        missing_candidates = required.difference(candidates)
        if missing_query or missing_candidates:
            raise KeyError(f"missing matcher tensors: query={missing_query}, candidates={missing_candidates}")
        b, k, d = candidates["global"].shape
        if query["global"].shape != (b, d) or d != self.cfg.descriptor_dim:
            raise ValueError("global descriptor shape mismatch")
        if query["ring"].shape != (b, self.cfg.ring_bins, self.cfg.ring_dim):
            raise ValueError("query ring shape mismatch")
        if candidates["ring"].shape != (b, k, self.cfg.ring_bins, self.cfg.ring_dim):
            raise ValueError("candidate ring shape mismatch")
        if query["cluster_mass"].shape != (b, self.cfg.cluster_count):
            raise ValueError("query cluster_mass shape mismatch")
        if candidates["cluster_mass"].shape != (b, k, self.cfg.cluster_count):
            raise ValueError("candidate cluster_mass shape mismatch")

    def forward(
        self,
        query: Dict[str, torch.Tensor],
        candidates: Dict[str, torch.Tensor],
        candidate_mask: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        self._validate(query, candidates)
        b, k, _ = candidates["global"].shape
        if candidate_mask is None:
            candidate_mask = torch.ones(b, k, dtype=torch.bool, device=candidates["global"].device)
        if candidate_mask.shape != (b, k) or not candidate_mask.any(dim=1).all():
            raise ValueError("candidate_mask must be [B,K] with at least one valid candidate per query")

        q_global = F.normalize(query["global"].float(), dim=-1)
        c_global = F.normalize(candidates["global"].float(), dim=-1)
        q_expanded = q_global.unsqueeze(1).expand_as(c_global)
        global_similarity = (q_expanded * c_global).sum(dim=-1)
        descriptor_abs = self.abs_projection(torch.abs(q_expanded - c_global))
        descriptor_product = self.product_projection(q_expanded * c_global)

        ring_scores = ring_correlation_matrix(query["ring"], candidates["ring"])
        ring_top2 = torch.topk(ring_scores, k=2, dim=-1)
        ring_peak = ring_top2.values[..., 0]
        ring_margin = ring_top2.values[..., 0] - ring_top2.values[..., 1]
        ring_prob = torch.softmax(ring_scores, dim=-1)
        ring_entropy = -(ring_prob * ring_prob.clamp_min(1e-8).log()).sum(dim=-1)
        ring_entropy = ring_entropy / torch.log(torch.tensor(float(self.cfg.ring_bins), device=ring_entropy.device))
        yaw_shift_bins = ring_top2.indices[..., 0]

        q_cluster = F.normalize(query["cluster_mass"].float(), dim=-1).unsqueeze(1)
        c_cluster = F.normalize(candidates["cluster_mass"].float(), dim=-1)
        cluster_similarity = (q_cluster * c_cluster).sum(dim=-1)
        cluster_l1 = torch.abs(query["cluster_mass"].float().unsqueeze(1) - candidates["cluster_mass"].float()).mean(dim=-1)
        q_dust = query["dustbin_fraction"].float().reshape(b, 1).expand(b, k)
        c_dust = candidates["dustbin_fraction"].float().reshape(b, k)
        rank = torch.arange(k, device=q_global.device, dtype=q_global.dtype).view(1, k).expand(b, k)
        normalized_rank = rank / max(1, k - 1)

        scalar_features = torch.stack(
            [
                global_similarity,
                ring_peak,
                ring_margin,
                ring_entropy,
                cluster_similarity,
                cluster_l1,
                q_dust,
                c_dust,
                normalized_rank,
            ],
            dim=-1,
        )
        pair_features = torch.cat([descriptor_abs, descriptor_product, scalar_features], dim=-1)
        pair_embedding = self.pair_encoder(pair_features)
        candidate_logits = self.same_place_head(pair_embedding).squeeze(-1)
        candidate_logits = candidate_logits.masked_fill(~candidate_mask, torch.finfo(candidate_logits.dtype).min)

        set_mean = _masked_mean(pair_embedding, candidate_mask, dim=1)
        set_max = pair_embedding.masked_fill(~candidate_mask.unsqueeze(-1), torch.finfo(pair_embedding.dtype).min).max(dim=1).values
        sorted_similarity = global_similarity.masked_fill(~candidate_mask, -1.0).sort(dim=1, descending=True).values
        top1 = sorted_similarity[:, 0]
        top2 = sorted_similarity[:, 1] if k > 1 else torch.full_like(top1, -1.0)
        valid_count = candidate_mask.sum(dim=1).to(global_similarity.dtype)
        score_mean = _masked_mean(global_similarity, candidate_mask, dim=1)
        set_scalars = torch.stack([top1, top1 - top2, score_mean, valid_count / float(k)], dim=-1)
        unknown_logit = self.unknown_head(torch.cat([set_mean, set_max, set_scalars], dim=-1)).squeeze(-1)
        logits_with_unknown = torch.cat([candidate_logits, unknown_logit.unsqueeze(-1)], dim=-1)
        probabilities = torch.softmax(logits_with_unknown, dim=-1)
        node_probabilities = probabilities[:, :k] * candidate_mask.to(probabilities.dtype)
        unknown_probability = probabilities[:, -1]

        return {
            "candidate_logits": candidate_logits,
            "unknown_logit": unknown_logit,
            "logits_with_unknown": logits_with_unknown,
            "node_probabilities": node_probabilities,
            "unknown_probability": unknown_probability,
            "present_probability": 1.0 - unknown_probability,
            "best_candidate_index": node_probabilities.argmax(dim=-1),
            "global_similarity": global_similarity,
            "ring_scores": ring_scores,
            "ring_peak": ring_peak,
            "ring_peak_second_margin": ring_margin,
            "ring_entropy": ring_entropy,
            "yaw_shift_bins": yaw_shift_bins,
            "pair_embedding": pair_embedding,
        }


# ---------------------------------------------------------------------------
# Training phase policy used by the checkpoint architecture
# ---------------------------------------------------------------------------

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Dict, Iterable, List

from torch import nn



class TrainingPhase(str, Enum):
    A = "A_frozen_backbone"
    B = "B_partial_backbone"
    C = "C_open_set"


@dataclass(frozen=True)
class PhasePolicy:
    phase: TrainingPhase
    train_backbone_blocks: int
    train_descriptor_head: bool
    train_matcher: bool
    backbone_lr_scale: float
    descriptor_lr_scale: float
    matcher_lr_scale: float


PHASE_POLICIES = {
    TrainingPhase.A: PhasePolicy(TrainingPhase.A, 0, True, False, 0.0, 1.0, 0.0),
    TrainingPhase.B: PhasePolicy(TrainingPhase.B, 4, True, False, 0.1, 1.0, 0.0),
    TrainingPhase.C: PhasePolicy(TrainingPhase.C, 0, True, True, 0.0, 0.1, 1.0),
}


def _set_trainable(module: nn.Module, enabled: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad_(enabled)


def configure_training_phase(
    backbone: DINOv2S14Backbone,
    descriptor_head: nn.Module,
    matcher: nn.Module,
    phase: TrainingPhase | str,
    phase_b_unfreeze_blocks: int = 4,
) -> Dict[str, object]:
    phase = TrainingPhase(phase)
    policy = PHASE_POLICIES[phase]
    if phase is TrainingPhase.B:
        backbone.unfreeze_last_blocks(phase_b_unfreeze_blocks)
    else:
        backbone.freeze_all()
    _set_trainable(descriptor_head, policy.train_descriptor_head)
    _set_trainable(matcher, policy.train_matcher)

    backbone.train(phase is TrainingPhase.B)
    descriptor_head.train(policy.train_descriptor_head)
    matcher.train(policy.train_matcher)
    return {
        "policy": asdict(policy),
        "phase_b_unfreeze_blocks": int(phase_b_unfreeze_blocks),
        "trainable_parameters": {
            "backbone": sum(p.numel() for p in backbone.parameters() if p.requires_grad),
            "descriptor_head": sum(p.numel() for p in descriptor_head.parameters() if p.requires_grad),
            "matcher": sum(p.numel() for p in matcher.parameters() if p.requires_grad),
        },
        "trainable_backbone_names": backbone.trainable_parameter_names(),
    }


def optimizer_parameter_groups(
    backbone: nn.Module,
    descriptor_head: nn.Module,
    matcher: nn.Module,
    phase: TrainingPhase | str,
    base_lr: float,
    weight_decay: float = 1e-4,
) -> List[Dict[str, object]]:
    phase = TrainingPhase(phase)
    policy = PHASE_POLICIES[phase]
    groups: List[Dict[str, object]] = []
    specifications: Iterable[tuple[str, nn.Module, float]] = (
        ("backbone", backbone, policy.backbone_lr_scale),
        ("descriptor_head", descriptor_head, policy.descriptor_lr_scale),
        ("matcher", matcher, policy.matcher_lr_scale),
    )
    for name, module, scale in specifications:
        parameters = [p for p in module.parameters() if p.requires_grad]
        if parameters:
            groups.append(
                {
                    "name": name,
                    "params": parameters,
                    "lr": float(base_lr) * float(scale),
                    "weight_decay": float(weight_decay),
                }
            )
    if not groups:
        raise RuntimeError(f"phase {phase.value} has no trainable parameters")
    return groups


# ---------------------------------------------------------------------------
# Unified panoramic retrieval system
# ---------------------------------------------------------------------------

from typing import Dict, Tuple

import torch
from torch import nn
from torch.nn import functional as F



ENCODING_KEYS = ("global", "ring", "cluster_mass", "dustbin_fraction")


class PanoramicVPRV2System(nn.Module):
    """Unified encoder, scalable retrieval and Top-K open-set verifier."""

    def __init__(
        self,
        model_config: ModelConfig | None = None,
        matcher_config: MatcherConfig | None = None,
        load_backbone: bool = True,
    ):
        super().__init__()
        self.model_config = model_config or ModelConfig()
        self.matcher_config = matcher_config or MatcherConfig(descriptor_dim=self.model_config.global_dim)
        if self.matcher_config.descriptor_dim != self.model_config.global_dim:
            raise ValueError("matcher descriptor_dim must equal model global_dim")
        self.backbone = DINOv2S14Backbone(freeze=True) if load_backbone else None
        self.descriptor_head = PanoSaladRingV2Head(self.model_config)
        self.matcher = TopKOpenSetMatcher(self.matcher_config)
        self._configured_phase: TrainingPhase | None = None

    def encode_tokens(self, tokens: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.descriptor_head.forward_tokens(tokens)

    def encode(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        if self.backbone is None:
            raise RuntimeError("system was constructed without a backbone")
        return self.encode_tokens(self.backbone.forward_tokens(images))

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.encode(images)

    @staticmethod
    def retrieve_topk(query_global: torch.Tensor, gallery_global: torch.Tensor, top_k: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if query_global.ndim != 2 or gallery_global.ndim != 2:
            raise ValueError("query_global and gallery_global must be [B,D] and [N,D]")
        if query_global.shape[1] != gallery_global.shape[1] or gallery_global.shape[0] == 0:
            raise ValueError("invalid gallery descriptor shape")
        similarities = F.normalize(query_global.float(), dim=-1) @ F.normalize(gallery_global.float(), dim=-1).T
        return torch.topk(similarities, k=min(int(top_k), gallery_global.shape[0]), dim=-1)

    @staticmethod
    def gather_topk(gallery: Dict[str, torch.Tensor], indices: torch.Tensor) -> Dict[str, torch.Tensor]:
        if indices.ndim != 2:
            raise ValueError("indices must be [B,K]")
        gathered: Dict[str, torch.Tensor] = {}
        for key in ENCODING_KEYS:
            tensor = gallery[key]
            gathered[key] = tensor[indices]
        return gathered

    def match_gallery(
        self,
        query: Dict[str, torch.Tensor],
        gallery: Dict[str, torch.Tensor],
        top_k: int | None = None,
    ) -> Dict[str, torch.Tensor]:
        k = int(top_k or self.matcher_config.top_k)
        retrieval_scores, indices = self.retrieve_topk(query["global"], gallery["global"], k)
        candidates = self.gather_topk(gallery, indices)
        output = self.matcher(query, candidates)
        output["retrieval_scores"] = retrieval_scores
        output["candidate_indices"] = indices
        output["best_gallery_index"] = indices.gather(1, output["best_candidate_index"].unsqueeze(1)).squeeze(1)
        return output

    def configure_phase(self, phase: TrainingPhase | str, phase_b_unfreeze_blocks: int = 4) -> Dict[str, object]:
        if self.backbone is None:
            raise RuntimeError("cannot configure training phase without a backbone")
        self._configured_phase = TrainingPhase(phase)
        return configure_training_phase(
            self.backbone,
            self.descriptor_head,
            self.matcher,
            self._configured_phase,
            phase_b_unfreeze_blocks=phase_b_unfreeze_blocks,
        )

    def train(self, mode: bool = True) -> "PanoramicVPRV2System":
        super().train(mode)
        if self._configured_phase is None:
            return self
        if self.backbone is not None:
            self.backbone.train(mode and self._configured_phase is TrainingPhase.B)
        self.descriptor_head.train(mode)
        self.matcher.train(mode and self._configured_phase is TrainingPhase.C)
        return self
