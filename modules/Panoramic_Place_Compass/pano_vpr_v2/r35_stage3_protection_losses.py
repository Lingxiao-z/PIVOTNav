from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class R35Stage3ProtectionLossWeights:
    descriptor_global_cosine: float = 1.0
    descriptor_ring_cosine: float = 0.5
    retrieval_distribution: float = 0.5
    track_y_distribution: float = 0.125
    track_y_angle: float = 0.10
    track_y_confidence: float = 0.025


def _finite_pair(
    student: torch.Tensor,
    teacher: torch.Tensor,
    label: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if student.shape != teacher.shape or student.numel() < 1:
        raise ValueError(f"{label} student/teacher shape不匹配")
    student = student.float()
    teacher = teacher.detach().float()
    if not bool(torch.isfinite(student).all()) or not bool(
        torch.isfinite(teacher).all()
    ):
        raise FloatingPointError(f"{label}包含非有限值")
    return student, teacher


def descriptor_distillation_losses(
    student: Dict[str, torch.Tensor],
    teacher: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    student_global, teacher_global = _finite_pair(
        student["global"], teacher["global"], "global descriptor"
    )
    student_ring, teacher_ring = _finite_pair(
        student["ring"], teacher["ring"], "ring descriptor"
    )
    global_cosine = (
        1.0
        - F.cosine_similarity(
            student_global,
            teacher_global,
            dim=-1,
            eps=1e-8,
        )
    ).mean()
    ring_cosine = (
        1.0
        - F.cosine_similarity(
            student_ring,
            teacher_ring,
            dim=-1,
            eps=1e-8,
        )
    ).mean()
    return {
        "descriptor_global_cosine": global_cosine,
        "descriptor_ring_cosine": ring_cosine,
    }


def retrieval_distribution_preservation_loss(
    student_global: torch.Tensor,
    teacher_global: torch.Tensor,
    *,
    temperature: float = 0.05,
) -> torch.Tensor:
    student, teacher = _finite_pair(
        student_global,
        teacher_global,
        "retrieval global descriptor",
    )
    if student.ndim != 2 or student.shape[0] < 2:
        raise ValueError("retrieval蒸馏至少需要两个二维descriptor")
    if not 0.0 < float(temperature) <= 1.0:
        raise ValueError("retrieval temperature必须在(0,1]内")
    student = F.normalize(student, dim=-1)
    teacher = F.normalize(teacher, dim=-1)
    student_similarity = student @ student.T
    teacher_similarity = teacher @ teacher.T
    diagonal = torch.eye(
        student.shape[0],
        device=student.device,
        dtype=torch.bool,
    )
    mask_value = torch.finfo(student_similarity.dtype).min
    student_logits = (student_similarity / temperature).masked_fill(
        diagonal, mask_value
    )
    teacher_logits = (teacher_similarity / temperature).masked_fill(
        diagonal, mask_value
    )
    teacher_probability = torch.softmax(
        teacher_logits.detach(), dim=-1
    )
    teacher_log_probability = torch.log_softmax(
        teacher_logits.detach(), dim=-1
    ).masked_fill(diagonal, 0.0)
    student_log_probability = torch.log_softmax(
        student_logits, dim=-1
    ).masked_fill(diagonal, 0.0)
    divergence = teacher_probability * (
        teacher_log_probability - student_log_probability
    )
    return divergence.sum(dim=-1).mean().clamp_min(0.0)


def _circular_difference_degrees(
    first: torch.Tensor,
    second: torch.Tensor,
) -> torch.Tensor:
    return torch.remainder(first.float() - second.float() + 180.0, 360.0) - 180.0


def track_y_consistency_losses(
    student: Dict[str, torch.Tensor],
    teacher: Dict[str, torch.Tensor],
    *,
    temperature: float = 1.0,
) -> Dict[str, torch.Tensor]:
    student_logits, teacher_logits = _finite_pair(
        student["logits"], teacher["logits"], "Track Y logits"
    )
    if student_logits.ndim != 2 or student_logits.shape[1] < 2:
        raise ValueError("Track Y logits必须是[N,L]且L>=2")
    if float(temperature) <= 0.0:
        raise ValueError("Track Y temperature必须为正数")
    teacher_probability = torch.softmax(
        teacher_logits / temperature,
        dim=-1,
    )
    student_log_probability = torch.log_softmax(
        student_logits / temperature,
        dim=-1,
    )
    teacher_log_probability = torch.log_softmax(
        teacher_logits / temperature,
        dim=-1,
    )
    distribution = (
        teacher_probability
        * (teacher_log_probability - student_log_probability)
    ).sum(dim=-1).mean() * (temperature**2)
    student_angle, teacher_angle = _finite_pair(
        student["predicted_yaw_degrees"],
        teacher["predicted_yaw_degrees"],
        "Track Y angle",
    )
    angle = (
        1.0
        - torch.cos(
            torch.deg2rad(
                _circular_difference_degrees(
                    student_angle,
                    teacher_angle,
                )
            )
        )
    ).mean()
    student_confidence, teacher_confidence = _finite_pair(
        student["yaw_confidence"],
        teacher["yaw_confidence"],
        "Track Y confidence",
    )
    confidence = F.smooth_l1_loss(
        student_confidence,
        teacher_confidence,
    )
    return {
        "track_y_distribution": distribution.clamp_min(0.0),
        "track_y_angle": angle,
        "track_y_confidence": confidence,
    }


def r35_stage3_protection_losses(
    *,
    student_descriptor: Dict[str, torch.Tensor],
    teacher_descriptor: Dict[str, torch.Tensor],
    student_yaw: Dict[str, torch.Tensor],
    teacher_yaw: Dict[str, torch.Tensor],
    weights: R35Stage3ProtectionLossWeights | None = None,
    retrieval_temperature: float = 0.05,
    yaw_temperature: float = 1.0,
) -> Dict[str, torch.Tensor]:
    selected = weights or R35Stage3ProtectionLossWeights()
    descriptor = descriptor_distillation_losses(
        student_descriptor,
        teacher_descriptor,
    )
    retrieval = retrieval_distribution_preservation_loss(
        student_descriptor["global"],
        teacher_descriptor["global"],
        temperature=retrieval_temperature,
    )
    yaw = track_y_consistency_losses(
        student_yaw,
        teacher_yaw,
        temperature=yaw_temperature,
    )
    total = (
        selected.descriptor_global_cosine
        * descriptor["descriptor_global_cosine"]
        + selected.descriptor_ring_cosine
        * descriptor["descriptor_ring_cosine"]
        + selected.retrieval_distribution * retrieval
        + selected.track_y_distribution * yaw["track_y_distribution"]
        + selected.track_y_angle * yaw["track_y_angle"]
        + selected.track_y_confidence * yaw["track_y_confidence"]
    )
    result = {
        **descriptor,
        "retrieval_distribution": retrieval,
        **yaw,
        "loss": total,
    }
    if not all(bool(torch.isfinite(value)) for value in result.values()):
        raise FloatingPointError("Stage 3保护损失包含非有限值")
    return result


def stage3_protection_architecture_record() -> Dict[str, Any]:
    return {
        "schema_version": "r35_stage3_protection_losses_v1",
        "default_weights": asdict(R35Stage3ProtectionLossWeights()),
        "teacher_gradient_disabled": True,
        "descriptor_terms": [
            "global cosine distillation",
            "ring cosine distillation",
        ],
        "retrieval_term": (
            "off-diagonal in-batch similarity-distribution KL divergence"
        ),
        "track_y_terms": [
            "circular logit-distribution KL divergence",
            "wrapped angular consistency",
            "confidence smooth-L1 consistency",
        ],
        "test_r32_confirmation_accessed": False,
    }
