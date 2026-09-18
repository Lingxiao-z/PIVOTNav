from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Dict

import torch
from torch import nn
from torch.nn import functional as F

from .r34_yaw_head import (
    CircularConv1d,
    R34CrossPositionYawHead,
    circular_error_degrees,
    circular_upsample_2x,
    multi_scale_circular_correlation,
    r34_yaw_losses,
)


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
