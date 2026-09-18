from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Dict, Iterable, List

from torch import nn

from .backbone import DINOv2S14Backbone


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
