from __future__ import annotations

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
