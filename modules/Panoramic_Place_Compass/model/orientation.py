"""Relative yaw model used by Panoramic Place Compass."""
from __future__ import annotations


# ---------------------------------------------------------------------------
# Circular relative-yaw head
# ---------------------------------------------------------------------------

import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, Tuple

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class R34YawConfig:
    input_dim: int = 384
    hidden_dim: int = 128
    coarse_bins: int = 64
    vertical_bands: Tuple[int, ...] = (1, 2, 4)
    circular_layers: int = 2
    score_hidden_dim: int = 64
    pair_embedding_dim: int = 128
    soft_label_sigma_bins: float = 1.0
    peak_margin: float = 0.20

    @property
    def scale_count(self) -> int:
        return sum(self.vertical_bands)

    @property
    def bin_width_degrees(self) -> float:
        return 360.0 / float(self.coarse_bins)


def _circular_pad_1d(x: torch.Tensor, padding: int) -> torch.Tensor:
    return F.pad(x, (padding, padding), mode="circular") if padding else x


def _erp_pad_2d(x: torch.Tensor, padding: int) -> torch.Tensor:
    if not padding:
        return x
    x = F.pad(x, (padding, padding, 0, 0), mode="circular")
    return F.pad(x, (0, 0, padding, padding), mode="replicate")


class CircularTokenBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        self.padding = kernel_size // 2
        self.depthwise = nn.Conv2d(channels, channels, kernel_size, groups=channels, bias=False)
        self.pointwise = nn.Conv2d(channels, channels, 1, bias=False)
        self.norm = nn.GroupNorm(8, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.depthwise(_erp_pad_2d(x, self.padding))
        x = self.pointwise(x)
        return residual + F.gelu(self.norm(x))


class CircularConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, bias: bool = True):
        super().__init__()
        self.padding = kernel_size // 2
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=0, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(_circular_pad_1d(x, self.padding))


def circular_upsample_2x(x: torch.Tensor) -> torch.Tensor:
    """Interleave samples and circular midpoints along the last dimension."""
    midpoint = 0.5 * (x + torch.roll(x, shifts=-1, dims=-1))
    return torch.stack((x, midpoint), dim=-1).flatten(-2)


class MultiScaleHorizontalRingEncoder(nn.Module):
    def __init__(self, cfg: R34YawConfig):
        super().__init__()
        self.cfg = cfg
        self.projection = nn.Conv2d(cfg.input_dim, cfg.hidden_dim, 1, bias=False)
        self.blocks = nn.ModuleList([CircularTokenBlock(cfg.hidden_dim) for _ in range(cfg.circular_layers)])
        self.output_norm = nn.LayerNorm(cfg.hidden_dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 4:
            raise ValueError("tokens must be [B,H,W,C] or [B,C,H,W]")
        if tokens.shape[-1] == self.cfg.input_dim:
            x = tokens.permute(0, 3, 1, 2).contiguous()
        elif tokens.shape[1] == self.cfg.input_dim:
            x = tokens.contiguous()
        else:
            raise ValueError(f"token dimension mismatch: {tuple(tokens.shape)}")
        if x.shape[-1] * 2 != self.cfg.coarse_bins:
            raise ValueError(f"expected input ring width {self.cfg.coarse_bins // 2}, got {x.shape[-1]}")
        x = self.projection(x)
        for block in self.blocks:
            x = block(x)
        rings = []
        for band_count in self.cfg.vertical_bands:
            pooled = F.adaptive_avg_pool2d(x, (band_count, x.shape[-1]))
            pooled = pooled.permute(0, 2, 1, 3).contiguous()
            pooled = circular_upsample_2x(pooled)
            pooled = pooled.permute(0, 1, 3, 2).contiguous()
            rings.append(F.normalize(self.output_norm(pooled), dim=-1))
        return torch.cat(rings, dim=1)


def multi_scale_circular_correlation(query_rings: torch.Tensor, candidate_rings: torch.Tensor) -> torch.Tensor:
    if query_rings.shape != candidate_rings.shape or query_rings.ndim != 4:
        raise ValueError("ring tensors must have equal [B,S,L,D] shapes")
    bins = query_rings.shape[2]
    query_fft = torch.fft.rfft(query_rings.float(), dim=2)
    candidate_fft = torch.fft.rfft(candidate_rings.float(), dim=2)
    correlation = torch.fft.irfft(query_fft * candidate_fft.conj(), n=bins, dim=2)
    return correlation.sum(dim=-1) / float(bins)


class R34CrossPositionYawHead(nn.Module):
    def __init__(self, cfg: R34YawConfig | None = None):
        super().__init__()
        self.cfg = cfg or R34YawConfig()
        self.ring_encoder = MultiScaleHorizontalRingEncoder(self.cfg)
        self.scale_logits = nn.Parameter(torch.zeros(self.cfg.scale_count))
        self.score_encoder = nn.Sequential(
            CircularConv1d(self.cfg.scale_count, self.cfg.score_hidden_dim, 5),
            nn.GroupNorm(8, self.cfg.score_hidden_dim),
            nn.GELU(),
            CircularConv1d(self.cfg.score_hidden_dim, self.cfg.score_hidden_dim, 3),
            nn.GroupNorm(8, self.cfg.score_hidden_dim),
            nn.GELU(),
        )
        self.logit_delta = CircularConv1d(self.cfg.score_hidden_dim, 1, 1)
        self.residual_head = CircularConv1d(self.cfg.score_hidden_dim, 1, 1)
        confidence_dim = 4
        self.confidence_calibrator = nn.Sequential(
            nn.LayerNorm(confidence_dim),
            nn.Linear(confidence_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )
        pair_summary_dim = self.cfg.scale_count * 2 + 4
        self.pair_projection = nn.Sequential(
            nn.LayerNorm(pair_summary_dim),
            nn.Linear(pair_summary_dim, self.cfg.pair_embedding_dim),
            nn.GELU(),
            nn.Linear(self.cfg.pair_embedding_dim, self.cfg.pair_embedding_dim),
        )
        nn.init.zeros_(self.logit_delta.conv.weight)
        nn.init.zeros_(self.logit_delta.conv.bias)
        nn.init.zeros_(self.residual_head.conv.weight)
        nn.init.zeros_(self.residual_head.conv.bias)

    @property
    def architecture_record(self) -> Dict[str, Any]:
        return {
            "schema_version": "r34_yaw_head_v1",
            "config": asdict(self.cfg),
            "coarse_bin_width_degrees": self.cfg.bin_width_degrees,
            "horizontal_wrap": "all token and score convolutions use circular longitude padding",
            "vertical_scales": list(self.cfg.vertical_bands),
            "alignment": "multi-scale FFT circular correlation plus circular score refinement",
            "continuous_refinement": "per-coarse-bin residual bounded to half a bin",
            "confidence": "learned calibration multiplied by entropy and top1/top2 peak-separation evidence",
        }

    def forward_rings(self, query_rings: torch.Tensor, candidate_rings: torch.Tensor) -> Dict[str, torch.Tensor]:
        scale_scores = multi_scale_circular_correlation(query_rings, candidate_rings)
        scale_weights = torch.softmax(self.scale_logits.float(), dim=0)
        raw_logits = (scale_scores * scale_weights.view(1, -1, 1)).sum(dim=1)
        encoded_scores = self.score_encoder(scale_scores)
        logits = raw_logits + self.logit_delta(encoded_scores).squeeze(1)
        half_bin = self.cfg.bin_width_degrees / 2.0
        residual_by_bin = torch.tanh(self.residual_head(encoded_scores).squeeze(1)) * half_bin

        probabilities = torch.softmax(logits.float(), dim=-1)
        top2 = torch.topk(probabilities, k=2, dim=-1)
        predicted_bin = top2.indices[:, 0]
        chosen_residual = residual_by_bin.gather(1, predicted_bin[:, None]).squeeze(1)
        predicted_yaw = torch.remainder(
            predicted_bin.float() * self.cfg.bin_width_degrees + chosen_residual,
            360.0,
        )
        normalized_entropy = -(
            probabilities * probabilities.clamp_min(1e-8).log()
        ).sum(dim=-1) / math.log(float(self.cfg.coarse_bins))
        peak_margin = top2.values[:, 0] - top2.values[:, 1]
        confidence_features = torch.stack(
            (1.0 - normalized_entropy, peak_margin, top2.values[:, 0], top2.values[:, 1]), dim=-1
        )
        learned_confidence = torch.sigmoid(self.confidence_calibrator(confidence_features).squeeze(-1))
        evidence_gate = (1.0 - normalized_entropy).clamp(0.0, 1.0) * torch.sigmoid(12.0 * peak_margin)
        yaw_confidence = learned_confidence * evidence_gate

        scale_max = scale_scores.max(dim=-1).values
        scale_mean = scale_scores.mean(dim=-1)
        pair_summary = torch.cat(
            (scale_max, scale_mean, confidence_features), dim=-1
        )
        pair_embedding = self.pair_projection(pair_summary)
        return {
            "logits": logits,
            "probabilities": probabilities,
            "residual_by_bin_degrees": residual_by_bin,
            "predicted_bin": predicted_bin,
            "predicted_yaw_degrees": predicted_yaw,
            "yaw_confidence": yaw_confidence,
            "normalized_entropy": normalized_entropy,
            "peak_margin": peak_margin,
            "top1_probability": top2.values[:, 0],
            "top2_probability": top2.values[:, 1],
            "scale_scores": scale_scores,
            "scale_weights": scale_weights,
            "query_rings": query_rings,
            "candidate_rings": candidate_rings,
            "pair_embedding": pair_embedding,
        }

    def forward(self, query_tokens: torch.Tensor, candidate_tokens: torch.Tensor) -> Dict[str, torch.Tensor]:
        query_rings = self.ring_encoder(query_tokens)
        candidate_rings = self.ring_encoder(candidate_tokens)
        return self.forward_rings(query_rings, candidate_rings)


def circular_error_degrees(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.abs(torch.remainder(predicted.float() - target.float() + 180.0, 360.0) - 180.0)


def wrapped_soft_labels(target_degrees: torch.Tensor, cfg: R34YawConfig) -> torch.Tensor:
    target_position = torch.remainder(target_degrees.float(), 360.0) / cfg.bin_width_degrees
    indices = torch.arange(cfg.coarse_bins, device=target_degrees.device, dtype=torch.float32).view(1, -1)
    distance = torch.abs(indices - target_position.view(-1, 1))
    distance = torch.minimum(distance, cfg.coarse_bins - distance)
    labels = torch.exp(-0.5 * (distance / cfg.soft_label_sigma_bins) ** 2)
    return labels / labels.sum(dim=-1, keepdim=True).clamp_min(1e-8)


def r34_yaw_losses(
    output: Dict[str, torch.Tensor],
    target_degrees: torch.Tensor,
    cfg: R34YawConfig,
    reverse_output: Dict[str, torch.Tensor] | None = None,
    teacher_query_ring: torch.Tensor | None = None,
    teacher_candidate_ring: torch.Tensor | None = None,
) -> Dict[str, torch.Tensor]:
    labels = wrapped_soft_labels(target_degrees, cfg)
    classification = -(labels * F.log_softmax(output["logits"].float(), dim=-1)).sum(dim=-1).mean()
    target_bin = torch.remainder(torch.round(target_degrees.float() / cfg.bin_width_degrees).long(), cfg.coarse_bins)
    bin_center = target_bin.float() * cfg.bin_width_degrees
    target_residual = torch.remainder(target_degrees.float() - bin_center + 180.0, 360.0) - 180.0
    predicted_residual = output["residual_by_bin_degrees"].gather(1, target_bin[:, None]).squeeze(1)
    residual = (1.0 - torch.cos(torch.deg2rad(predicted_residual - target_residual))).mean()

    target_logits = output["logits"].gather(1, target_bin[:, None]).squeeze(1)
    mask = F.one_hot(target_bin, cfg.coarse_bins).bool()
    highest_wrong = output["logits"].masked_fill(mask, -torch.inf).max(dim=-1).values
    peak_margin = F.relu(cfg.peak_margin - target_logits + highest_wrong).mean()

    predicted_error = circular_error_degrees(output["predicted_yaw_degrees"].detach(), target_degrees)
    confidence_target = (predicted_error <= 11.25).float()
    # BCE itself is rejected by PyTorch autocast; the explicit log form is
    # equivalent and keeps the confidence term valid under BF16 workload smoke.
    confidence_probability = output["yaw_confidence"].float().clamp(1e-6, 1.0 - 1e-6)
    confidence = -(
        confidence_target.float() * confidence_probability.log()
        + (1.0 - confidence_target.float()) * (1.0 - confidence_probability).log()
    ).mean()

    translation_consistency = output["logits"].sum() * 0.0
    if reverse_output is not None:
        wrapped_sum = torch.remainder(
            output["predicted_yaw_degrees"] + reverse_output["predicted_yaw_degrees"] + 180.0,
            360.0,
        ) - 180.0
        translation_consistency = (1.0 - torch.cos(torch.deg2rad(wrapped_sum))).mean()

    ring_distillation = output["logits"].sum() * 0.0
    if teacher_query_ring is not None and teacher_candidate_ring is not None:
        student_query = output["query_rings"][:, 0, ::2]
        student_candidate = output["candidate_rings"][:, 0, ::2]
        ring_distillation = 0.5 * (
            (1.0 - F.cosine_similarity(student_query.float(), teacher_query_ring.float(), dim=-1)).mean()
            + (1.0 - F.cosine_similarity(student_candidate.float(), teacher_candidate_ring.float(), dim=-1)).mean()
        )

    total = (
        classification
        + 0.5 * residual
        + 0.25 * peak_margin
        + 0.10 * translation_consistency
        + 0.10 * ring_distillation
        + 0.05 * confidence
        # Keep the future Track C pair embedding in the DDP graph during
        # Track Y-r1; it is intentionally not optimized by the yaw objective.
        + output["pair_embedding"].sum() * 0.0
    )
    return {
        "loss": total,
        "wrapped_classification": classification,
        "angular_residual": residual,
        "peak_margin": peak_margin,
        "translation_consistency": translation_consistency,
        "ring_distillation": ring_distillation,
        "confidence_calibration": confidence,
        "yaw_error_mean_degrees": predicted_error.mean(),
        "accuracy_le_11_25deg": (predicted_error <= 11.25).float().mean(),
        "accuracy_le_22_5deg": (predicted_error <= 22.5).float().mean(),
        "catastrophic_gt_45deg": (predicted_error > 45.0).float().mean(),
    }


# ---------------------------------------------------------------------------
# Final relative-yaw refinement
# ---------------------------------------------------------------------------

import math
from dataclasses import asdict, dataclass
from typing import Any, Dict

import torch
from torch import nn
from torch.nn import functional as F



@dataclass(frozen=True)
class R34YawR3Config:
    descriptor_ring_bins: int = 32
    descriptor_ring_dim: int = 128
    resolver_hidden_dim: int = 32
    opposite_peak_margin: float = 1.0
    quarter_turn_peak_margin: float = 0.5
    opposite_peak_weight: float = 0.35
    quarter_turn_peak_weight: float = 0.15
    descriptor_delta_l2_weight: float = 0.01


class DescriptorRingAmbiguityResolver(nn.Module):
    def __init__(self, cfg: R34YawR3Config):
        super().__init__()
        self.cfg = cfg
        hidden = cfg.resolver_hidden_dim
        if hidden % 8:
            raise ValueError("resolver_hidden_dim must be divisible by 8")
        self.score_encoder = nn.Sequential(
            CircularConv1d(1, hidden, 5),
            nn.GroupNorm(8, hidden),
            nn.GELU(),
            CircularConv1d(hidden, hidden, 3),
            nn.GroupNorm(8, hidden),
            nn.GELU(),
        )
        self.logit_delta = CircularConv1d(hidden, 1, 1)
        self.pair_projection = nn.Sequential(
            nn.LayerNorm(7),
            nn.Linear(7, hidden),
            nn.GELU(),
            nn.Linear(hidden, 128),
        )
        nn.init.zeros_(self.logit_delta.conv.weight)
        nn.init.zeros_(self.logit_delta.conv.bias)
        nn.init.zeros_(self.pair_projection[-1].weight)
        nn.init.zeros_(self.pair_projection[-1].bias)

    def forward(self, query_ring: torch.Tensor, candidate_ring: torch.Tensor) -> Dict[str, torch.Tensor]:
        expected = (self.cfg.descriptor_ring_bins, self.cfg.descriptor_ring_dim)
        if query_ring.shape != candidate_ring.shape or query_ring.ndim != 3:
            raise ValueError("descriptor rings must have equal [B,L,D] shapes")
        if tuple(query_ring.shape[1:]) != expected:
            raise ValueError(f"expected descriptor ring shape [B,{expected[0]},{expected[1]}]")
        correlation_32 = multi_scale_circular_correlation(
            query_ring.unsqueeze(1), candidate_ring.unsqueeze(1)
        )
        correlation_64 = circular_upsample_2x(correlation_32)
        centered = correlation_64 - correlation_64.mean(dim=-1, keepdim=True)
        normalized = centered / centered.std(dim=-1, keepdim=True, unbiased=False).clamp_min(1e-5)
        encoded = self.score_encoder(normalized)
        logit_delta = self.logit_delta(encoded).squeeze(1)
        top2 = torch.topk(torch.softmax(correlation_64.squeeze(1).float(), dim=-1), k=2, dim=-1).values
        summary = torch.stack(
            (
                correlation_64.amax(dim=(1, 2)),
                correlation_64.mean(dim=(1, 2)),
                correlation_64.std(dim=(1, 2), unbiased=False),
                normalized.abs().mean(dim=(1, 2)),
                top2[:, 0],
                top2[:, 1],
                top2[:, 0] - top2[:, 1],
            ),
            dim=-1,
        )
        return {
            "descriptor_correlation_32": correlation_32.squeeze(1),
            "descriptor_correlation_64": correlation_64.squeeze(1),
            "descriptor_logit_delta": logit_delta,
            "descriptor_pair_delta": self.pair_projection(summary),
        }


class R34CrossPositionYawHeadR3(nn.Module):
    def __init__(
        self,
        base_head: R34CrossPositionYawHead | None = None,
        cfg: R34YawR3Config | None = None,
    ) -> None:
        super().__init__()
        self.base_head = base_head or R34CrossPositionYawHead()
        self.cfg = cfg or R34YawR3Config()
        self.resolver = DescriptorRingAmbiguityResolver(self.cfg)

    @property
    def architecture_record(self) -> Dict[str, Any]:
        return {
            "schema_version": "r34_yaw_head_r3_v1",
            "base_architecture": self.base_head.architecture_record,
            "r3_config": asdict(self.cfg),
            "descriptor_fusion": "32-bin SALAD ring correlation, circular 2x interpolation, zero-initialized 64-bin residual logits",
            "ambiguity_control": "explicit opposite-peak and quarter-turn peak margins",
            "initial_equivalence": "zero-initialized resolver exactly preserves the frozen Y-r2 logits and pair embedding",
        }

    def freeze_base_head(self) -> None:
        for parameter in self.base_head.parameters():
            parameter.requires_grad_(False)

    def forward_rings(
        self,
        query_rings: torch.Tensor,
        candidate_rings: torch.Tensor,
        query_descriptor_ring: torch.Tensor,
        candidate_descriptor_ring: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        base = self.base_head.forward_rings(query_rings, candidate_rings)
        resolved = self.resolver(query_descriptor_ring, candidate_descriptor_ring)
        logits = base["logits"] + resolved["descriptor_logit_delta"]
        probabilities = torch.softmax(logits.float(), dim=-1)
        top2 = torch.topk(probabilities, k=2, dim=-1)
        predicted_bin = top2.indices[:, 0]
        chosen_residual = base["residual_by_bin_degrees"].gather(1, predicted_bin[:, None]).squeeze(1)
        predicted_yaw = torch.remainder(
            predicted_bin.float() * self.base_head.cfg.bin_width_degrees + chosen_residual,
            360.0,
        )
        normalized_entropy = -(
            probabilities * probabilities.clamp_min(1e-8).log()
        ).sum(dim=-1) / math.log(float(self.base_head.cfg.coarse_bins))
        peak_margin = top2.values[:, 0] - top2.values[:, 1]
        confidence_features = torch.stack(
            (1.0 - normalized_entropy, peak_margin, top2.values[:, 0], top2.values[:, 1]), dim=-1
        )
        learned_confidence = torch.sigmoid(
            self.base_head.confidence_calibrator(confidence_features).squeeze(-1)
        )
        evidence_gate = (1.0 - normalized_entropy).clamp(0.0, 1.0) * torch.sigmoid(12.0 * peak_margin)
        output = dict(base)
        output.update(resolved)
        output.update(
            {
                "logits": logits,
                "probabilities": probabilities,
                "predicted_bin": predicted_bin,
                "predicted_yaw_degrees": predicted_yaw,
                "yaw_confidence": learned_confidence * evidence_gate,
                "normalized_entropy": normalized_entropy,
                "peak_margin": peak_margin,
                "top1_probability": top2.values[:, 0],
                "top2_probability": top2.values[:, 1],
                "pair_embedding": base["pair_embedding"] + resolved["descriptor_pair_delta"],
            }
        )
        return output

    def forward(
        self,
        query_tokens: torch.Tensor,
        candidate_tokens: torch.Tensor,
        query_descriptor_ring: torch.Tensor,
        candidate_descriptor_ring: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        return self.forward_rings(
            self.base_head.ring_encoder(query_tokens),
            self.base_head.ring_encoder(candidate_tokens),
            query_descriptor_ring,
            candidate_descriptor_ring,
        )


def r34_yaw_r3_losses(
    output: Dict[str, torch.Tensor],
    target_degrees: torch.Tensor,
    head: R34CrossPositionYawHeadR3,
    *,
    reverse_output: Dict[str, torch.Tensor] | None = None,
    teacher_query_ring: torch.Tensor | None = None,
    teacher_candidate_ring: torch.Tensor | None = None,
) -> Dict[str, torch.Tensor]:
    losses = r34_yaw_losses(
        output,
        target_degrees,
        head.base_head.cfg,
        reverse_output=reverse_output,
        teacher_query_ring=teacher_query_ring,
        teacher_candidate_ring=teacher_candidate_ring,
    )
    bins = head.base_head.cfg.coarse_bins
    target_bin = torch.remainder(
        torch.round(target_degrees.float() / head.base_head.cfg.bin_width_degrees).long(), bins
    )
    target_logits = output["logits"].gather(1, target_bin[:, None]).squeeze(1)
    opposite_bin = torch.remainder(target_bin + bins // 2, bins)
    opposite_logits = output["logits"].gather(1, opposite_bin[:, None]).squeeze(1)
    quarter_a = torch.remainder(target_bin + bins // 4, bins)
    quarter_b = torch.remainder(target_bin - bins // 4, bins)
    quarter_logits = torch.maximum(
        output["logits"].gather(1, quarter_a[:, None]).squeeze(1),
        output["logits"].gather(1, quarter_b[:, None]).squeeze(1),
    )
    opposite_peak_margin = F.relu(
        head.cfg.opposite_peak_margin - target_logits + opposite_logits
    ).mean()
    quarter_turn_peak_margin = F.relu(
        head.cfg.quarter_turn_peak_margin - target_logits + quarter_logits
    ).mean()
    descriptor_delta_l2 = output["descriptor_logit_delta"].float().square().mean()
    losses["base_yaw_loss"] = losses["loss"]
    losses["opposite_peak_margin"] = opposite_peak_margin
    losses["quarter_turn_peak_margin"] = quarter_turn_peak_margin
    losses["descriptor_delta_l2"] = descriptor_delta_l2
    losses["loss"] = (
        losses["base_yaw_loss"]
        + head.cfg.opposite_peak_weight * opposite_peak_margin
        + head.cfg.quarter_turn_peak_weight * quarter_turn_peak_margin
        + head.cfg.descriptor_delta_l2_weight * descriptor_delta_l2
    )
    losses["yaw_error_mean_degrees"] = circular_error_degrees(
        output["predicted_yaw_degrees"].detach(), target_degrees
    ).mean()
    return losses
