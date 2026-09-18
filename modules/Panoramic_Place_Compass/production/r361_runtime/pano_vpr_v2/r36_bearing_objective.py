from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch.nn import functional as F

from .r35_bearing_head import (
    R35BearingConfig,
    R35BearingLossWeights,
    circular_error_degrees_tensor,
    r35_bearing_losses,
)


@dataclass(frozen=True)
class R36BearingLossWeights:
    direct_accuracy_5deg: float = 0.75
    close_range_5deg: float = 0.50
    high_confidence_catastrophic: float = 0.25


def r36_bearing_losses(
    output: dict[str, torch.Tensor],
    target_degrees: torch.Tensor,
    target_valid: torch.Tensor,
    distance_bucket: torch.Tensor,
    cfg: R35BearingConfig,
    *,
    rolled_source_output: dict[str, torch.Tensor] | None = None,
    source_roll_degrees: torch.Tensor | None = None,
    reverse_output: dict[str, torch.Tensor] | None = None,
    source_yaw_degrees: torch.Tensor | None = None,
    target_yaw_degrees: torch.Tensor | None = None,
    weights: R36BearingLossWeights | None = None,
) -> dict[str, torch.Tensor]:
    extra = weights or R36BearingLossWeights()
    base = r35_bearing_losses(
        output,
        target_degrees,
        target_valid,
        cfg,
        weights=R35BearingLossWeights(
            circular_soft_label=1.25,
            sin_cos_regression=.75,
            geodesic=.75,
            rotation_equivariance=.25,
            reciprocal_consistency=.25,
            confidence_calibration=.15,
            valid_classification=.25,
        ),
        rolled_source_output=rolled_source_output,
        source_roll_degrees=source_roll_degrees,
        reverse_output=reverse_output,
        source_yaw_degrees=source_yaw_degrees,
        target_yaw_degrees=target_yaw_degrees,
    )
    valid = target_valid.bool()
    error = circular_error_degrees_tensor(output["bearing_angle_degrees"], target_degrees)
    # Smooth hinge is zero inside 5 degrees and directly penalizes misses beyond the formal tolerance.
    direct = F.smooth_l1_loss(
        (error[valid] / 5.0).clamp_min(1.0),
        torch.ones_like(error[valid]),
    ) if bool(valid.any()) else error.sum() * 0.0
    close = valid & (distance_bucket == 0)
    close_direct = F.smooth_l1_loss(
        (error[close] / 5.0).clamp_min(1.0),
        torch.ones_like(error[close]),
    ) if bool(close.any()) else error.sum() * 0.0
    confidence = output["bearing_raw_learned_confidence"].float()
    catastrophic = F.relu(confidence[valid] - .5) * F.relu((error[valid] - 45.0) / 45.0)
    catastrophic_loss = catastrophic.mean() if catastrophic.numel() else error.sum() * 0.0
    total = base["loss"] + extra.direct_accuracy_5deg * direct + extra.close_range_5deg * close_direct + extra.high_confidence_catastrophic * catastrophic_loss
    return {
        "loss": total,
        **{f"base_{name}": value for name, value in base.items() if name != "loss"},
        "direct_accuracy_5deg": direct,
        "close_range_accuracy_5deg": close_direct,
        "high_confidence_catastrophic": catastrophic_loss,
        "valid_accuracy_le_5deg": ((error <= 5.0) & valid).float().sum() / valid.float().sum().clamp_min(1),
    }


def r36_bearing_metrics(errors: torch.Tensor, confidence: torch.Tensor, distance_bucket: torch.Tensor, scene_ids: list[str]) -> dict[str, Any]:
    import numpy as np
    e=errors.float().cpu().numpy();c=confidence.float().cpu().numpy();d=distance_bucket.long().cpu().numpy()
    def summary(values):
        return {"count":int(len(values)),"accuracy_le_5deg":float((values<=5).mean()),"accuracy_le_10deg":float((values<=10).mean()),"mae_degrees":float(values.mean()),"median_degrees":float(np.median(values)),"p75_degrees":float(np.percentile(values,75)),"p90_degrees":float(np.percentile(values,90)),"catastrophic_gt_45deg_rate":float((values>45).mean())}
    names={0:"0.5-1m",1:"1-2m",2:"2-3m",3:"3-4m"};by_distance={names[i]:summary(e[d==i]) for i in range(4)};one_three=e[np.isin(d,[1,2])]
    per_scene={scene:summary(e[np.asarray([value==scene for value in scene_ids])]) for scene in sorted(set(scene_ids))}
    risk=[]
    order=np.argsort(-c)
    for coverage in (1.0,.9,.8,.6,.4,.2):
        count=max(1,int(np.ceil(len(order)*coverage)));risk.append({"coverage":coverage,"confidence_threshold":float(c[order[count-1]]),**summary(e[order[:count]])})
    monotonic=all(risk[i+1]["accuracy_le_5deg"]+1e-12>=risk[i]["accuracy_le_5deg"] for i in range(len(risk)-1))
    gates={"distance_0_5_1m_accuracy_le_5_ge_80pct":by_distance["0.5-1m"]["accuracy_le_5deg"]>=.8,"distance_1_2m_accuracy_le_5_ge_85pct":by_distance["1-2m"]["accuracy_le_5deg"]>=.85,"distance_2_3m_accuracy_le_5_ge_85pct":by_distance["2-3m"]["accuracy_le_5deg"]>=.85,"distance_3_4m_accuracy_le_5_ge_80pct":by_distance["3-4m"]["accuracy_le_5deg"]>=.8,"distance_1_3m_p90_le_10deg":float(np.percentile(one_three,90))<=10,"catastrophic_gt_45_le_1pct":float((e>45).mean())<=.01,"confidence_risk_monotonic":monotonic,"all_outputs_finite":bool(np.isfinite(e).all() and np.isfinite(c).all())}
    score=sum(by_distance[name]["accuracy_le_5deg"] for name in names.values())-float(np.percentile(one_three,90))/90-2*float((e>45).mean())
    return {"overall":summary(e),"by_distance":by_distance,"distance_1_3m":summary(one_three),"risk_coverage":risk,"per_scene":per_scene,"gates":gates,"all_bearing_gates_passed":all(gates.values()),"primary_checkpoint_selection_score":score,"checkpoint_selection_rule_zh":"四距离层Accuracy@5度之和，惩罚1-3m P90和严重错误；不使用阈值搜索。","test_r32_confirmation_accessed":False}
