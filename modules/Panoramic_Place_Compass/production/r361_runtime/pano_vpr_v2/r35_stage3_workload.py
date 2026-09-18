from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .r35_bearing_head import r35_bearing_losses
from .r35_multitask_system import R35MultitaskSystem
from .r35_open_set_heads import r35_candidate_match_losses
from .r35_stage2_revision2 import (
    R35SceneRobustPresenceHead,
    r35_scene_robust_presence_losses,
)
from .r35_stage2_revision3 import (
    R35DualExpertPresenceHead,
    r35_dual_expert_presence_losses,
)
from .r35_stage2_revision4 import (
    R35RawPairVerifierPresenceHead,
    r35_raw_pair_verifier_presence_losses,
)
from .r35_stage2_workload import r35_stage2_losses
from .r35_stage3_protection_losses import r35_stage3_protection_losses


class R35Stage3StudentWorkload(nn.Module):
    """One DDP-visible student forward covering every Stage 3 trainable scope."""

    def __init__(self, student: R35MultitaskSystem) -> None:
        super().__init__()
        self.student = student

    def forward(
        self,
        bearing_source: torch.Tensor,
        bearing_target: torch.Tensor,
        bearing_rolled_source: torch.Tensor,
        track_query: torch.Tensor,
        track_candidates: torch.Tensor,
        track_candidate_mask: torch.Tensor,
        track_reciprocal_rank: torch.Tensor,
    ) -> dict[str, Any]:
        if not (
            bearing_source.shape
            == bearing_target.shape
            == bearing_rolled_source.shape
        ):
            raise ValueError("Stage 3 Bearing source/target/rolled shape不一致")
        encoded = self.student.encode(
            torch.cat(
                (bearing_source, bearing_target, bearing_rolled_source),
                dim=0,
            )
        )
        source, target, rolled = (
            {
                key: chunks[index]
                for key, value in encoded.items()
                for chunks in (value.chunk(3, dim=0),)
            }
            for index in range(3)
        )
        yaw = self.student._forward_track_y_encoded(
            source["dense_ring"],
            target["dense_ring"],
            source["descriptor_ring"],
            target["descriptor_ring"],
        )
        bearing = {
            "forward": self.student.bearing_head(
                source["tokens"], target["tokens"]
            ),
            "reverse": self.student.bearing_head(
                target["tokens"], source["tokens"]
            ),
            "rolled_source": self.student.bearing_head(
                rolled["tokens"], target["tokens"]
            ),
        }
        track_c = self.student.match_candidate_set(
            track_query,
            track_candidates,
            candidate_mask=track_candidate_mask,
            reciprocal_rank_score=track_reciprocal_rank,
        )
        return {
            "bearing": bearing,
            "track_c": track_c,
            "protection": {
                "descriptor": {
                    "global": torch.cat(
                        (source["global"], target["global"]), dim=0
                    ),
                    "ring": torch.cat(
                        (source["descriptor_ring"], target["descriptor_ring"]),
                        dim=0,
                    ),
                },
                "track_y": yaw,
            },
        }


@torch.no_grad()
def compute_stage3_teacher_targets(
    *,
    r3_teacher: nn.Module,
    track_y_teacher: R35MultitaskSystem,
    bearing_source: torch.Tensor,
    bearing_target: torch.Tensor,
) -> dict[str, Any]:
    images = torch.cat((bearing_source, bearing_target), dim=0)
    tokens = r3_teacher.backbone.forward_tokens(images)
    descriptor = r3_teacher.descriptor_head.forward_tokens(tokens.float())
    track_y_output = track_y_teacher.forward_pairs(
        bearing_source, bearing_target
    )["yaw"]
    targets = {
        "descriptor": {
            "global": descriptor["global"].detach(),
            "ring": descriptor["ring"].detach(),
        },
        "track_y": {
            key: value.detach()
            for key, value in track_y_output.items()
            if key in (
                "logits",
                "predicted_yaw_degrees",
                "yaw_confidence",
            )
        },
    }
    required = {"logits", "predicted_yaw_degrees", "yaw_confidence"}
    if set(targets["track_y"]) != required:
        raise RuntimeError(
            f"Stage 3 Track Y teacher输出字段不完整: {sorted(set(targets['track_y']))}"
        )
    return targets


def compute_stage3_joint_losses(
    *,
    output: dict[str, Any],
    teacher: dict[str, Any],
    bearing_batch: dict[str, torch.Tensor],
    track_batch: dict[str, torch.Tensor],
    student: R35MultitaskSystem,
) -> dict[str, torch.Tensor]:
    bearing = r35_bearing_losses(
        output["bearing"]["forward"],
        bearing_batch["bearing_degrees"],
        bearing_batch["bearing_valid"],
        student.bearing_head.cfg,
        rolled_source_output=output["bearing"]["rolled_source"],
        source_roll_degrees=bearing_batch["source_roll_degrees"],
        reverse_output=output["bearing"]["reverse"],
        source_yaw_degrees=bearing_batch["source_yaw_degrees"],
        target_yaw_degrees=bearing_batch["target_yaw_degrees"],
    )
    if isinstance(student.set_presence_head, R35RawPairVerifierPresenceHead):
        candidate = r35_candidate_match_losses(
            output["track_c"]["candidate_match"],
            track_batch["same_targets"],
            track_batch["hard_positive_candidate_mask"],
            track_batch["hard_negative_targets"],
            cfg=student.candidate_match_head.cfg,
        )
        verifier = output["track_c"].get("candidate_verifier")
        if verifier is None:
            raise RuntimeError("Stage 3 revision4缺少raw-pair Candidate Verifier输出")
        presence = r35_raw_pair_verifier_presence_losses(
            output["track_c"]["set_presence"],
            target_present=track_batch["target_present"],
            hard_positive_set=track_batch["hard_positive_set"],
            ordinary_positive_set=track_batch["ordinary_positive_set"],
            near_wrong_set=track_batch["near_wrong_set"],
            scene_group_ids=track_batch["scene_group_id"],
            verifier_output=verifier,
            candidate_same_targets=track_batch["same_targets"],
            hard_positive_candidate_mask=track_batch[
                "hard_positive_candidate_mask"
            ],
            hard_negative_candidate_mask=track_batch[
                "hard_negative_targets"
            ],
            config=student.set_presence_head.robust_cfg,
        )
        track_c = {
            "loss": candidate["loss"] + presence["loss"],
            "candidate_loss": candidate["loss"],
            "presence_loss": presence["loss"],
            **{
                f"candidate_{key}": value
                for key, value in candidate.items()
                if key != "loss"
            },
            **{
                f"presence_{key}": value
                for key, value in presence.items()
                if key != "loss"
            },
        }
    elif isinstance(
        student.set_presence_head,
        (R35SceneRobustPresenceHead, R35DualExpertPresenceHead),
    ):
        candidate = r35_candidate_match_losses(
            output["track_c"]["candidate_match"],
            track_batch["same_targets"],
            track_batch["hard_positive_candidate_mask"],
            track_batch["hard_negative_targets"],
            cfg=student.candidate_match_head.cfg,
        )
        loss_function = (
            r35_dual_expert_presence_losses
            if isinstance(student.set_presence_head, R35DualExpertPresenceHead)
            else r35_scene_robust_presence_losses
        )
        presence = loss_function(
            output["track_c"]["set_presence"],
            target_present=track_batch["target_present"],
            hard_positive_set=track_batch["hard_positive_set"],
            ordinary_positive_set=track_batch["ordinary_positive_set"],
            near_wrong_set=track_batch["near_wrong_set"],
            scene_group_ids=track_batch["scene_group_id"],
            config=student.set_presence_head.robust_cfg,
        )
        track_c = {
            "loss": candidate["loss"] + presence["loss"],
            "candidate_loss": candidate["loss"],
            "presence_loss": presence["loss"],
            **{
                f"candidate_{key}": value
                for key, value in candidate.items()
                if key != "loss"
            },
            **{
                f"presence_{key}": value
                for key, value in presence.items()
                if key != "loss"
            },
        }
    else:
        track_c = r35_stage2_losses(
            {
                "candidate_match": output["track_c"]["candidate_match"],
                "set_presence": output["track_c"]["set_presence"],
            },
            candidate_same_place=track_batch["same_targets"],
            hard_positive_candidate_mask=track_batch[
                "hard_positive_candidate_mask"
            ],
            hard_negative_candidate_mask=track_batch[
                "hard_negative_targets"
            ],
            target_present=track_batch["target_present"],
            target_candidate_index=track_batch["target_candidate_index"],
            hard_positive_set=track_batch["hard_positive_set"],
            ordinary_positive_set=track_batch["ordinary_positive_set"],
            near_wrong_set=track_batch["near_wrong_set"],
            candidate_cfg=student.candidate_match_head.cfg,
            presence_cfg=student.set_presence_head.cfg,
        )
    protection = r35_stage3_protection_losses(
        student_descriptor=output["protection"]["descriptor"],
        teacher_descriptor=teacher["descriptor"],
        student_yaw=output["protection"]["track_y"],
        teacher_yaw=teacher["track_y"],
    )
    total = (
        bearing["loss"]
        + track_c["candidate_loss"]
        + track_c["presence_loss"]
        + protection["loss"]
    )
    losses = {
        "loss": total,
        "bearing_loss": bearing["loss"],
        "candidate_loss": track_c["candidate_loss"],
        "presence_loss": track_c["presence_loss"],
        "protection_loss": protection["loss"],
        **{
            f"bearing_{key}": value
            for key, value in bearing.items()
            if key != "loss"
        },
        **{
            f"track_c_{key}": value
            for key, value in track_c.items()
            if key not in ("loss", "candidate_loss", "presence_loss")
        },
        **{
            f"protection_{key}": value
            for key, value in protection.items()
            if key != "loss"
        },
    }
    if not all(bool(torch.isfinite(value).all()) for value in losses.values()):
        raise FloatingPointError("Stage 3 joint workload出现非有限loss")
    return losses


def stage3_workload_architecture_record() -> dict[str, Any]:
    return {
        "schema_version": "r35_stage3_joint_workload_v1",
        "task_loss_weights": {
            "bearing": 1.0,
            "candidate_match": 1.0,
            "set_presence": 1.0,
        },
        "protection_inputs": (
            "Bearing source和target原始图像同时送入student、冻结R3 teacher和冻结Track Y teacher"
        ),
        "teacher_online_forward": True,
        "teacher_gradient_disabled": True,
        "candidate_and_presence_heads_separate": True,
        "explicit_candidate_competition": True,
        "dedicated_near_wrong_auxiliary": True,
        "scene_robust_presence_revision2_supported": True,
        "dual_expert_presence_revision3_supported": True,
        "raw_pair_verifier_presence_revision4_supported": True,
        "scene_group_is_training_loss_only": True,
        "cached_stage2_student_features_used_for_teacher": False,
        "test_r32_confirmation_accessed": False,
    }
