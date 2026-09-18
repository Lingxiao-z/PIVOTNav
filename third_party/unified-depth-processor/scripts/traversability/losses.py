"""通行性预测损失函数。

All losses are masked to only compute within the camera FOV.
Distance Huber loss is computed on raw (pre-combination) dist output to avoid
coupling gradients between the distance and existence heads.
An optional angular smoothness (TV) regulariser penalises large differences
between adjacent angle bins, encouraging locally smooth distance profiles.
"""

import torch
import torch.nn.functional as F


def angular_smoothness_loss(pred_dist: torch.Tensor,
                            continuous_has_data: torch.Tensor = None) -> torch.Tensor:
    """Total-variation smoothness penalty over the 360-degree angle dimension.

    Penalises large differences between adjacent angle bins, including the
    wrap-around boundary between bin 359 and bin 0.  When continuous_has_data is provided,
    only adjacent pairs where both bins are inside the FOV are penalised.

    Args:
        pred_dist: (B, 360) predicted radar distance.
        continuous_has_data:  (B, 360) bool, True for FOV bins.  None = all bins.

    Returns:
        Scalar mean squared angular difference.
    """
    diff = pred_dist[:, 1:] - pred_dist[:, :-1]          # (B, 359)
    wrap = pred_dist[:, 0:1] - pred_dist[:, -1:]          # (B, 1)
    sq = torch.cat([diff, wrap], dim=1) ** 2              # (B, 360)

    if continuous_has_data is not None:
        # Both neighbours must be in FOV: pair (i, i+1) valid if continuous_has_data[i] & continuous_has_data[i+1]
        pair_mask = continuous_has_data[:, 1:] & continuous_has_data[:, :-1]                    # (B, 359)
        wrap_mask = continuous_has_data[:, 0:1] & continuous_has_data[:, -1:]                   # (B, 1)
        pair_mask = torch.cat([pair_mask, wrap_mask], dim=1).float()      # (B, 360)
        n = pair_mask.sum().clamp(min=1)
        return (sq * pair_mask).sum() / n

    return sq.mean()


def total_loss(raw_dist: torch.Tensor,
               pred_exist: torch.Tensor,
               target_dist: torch.Tensor,
               continuous_has_data: torch.Tensor,
               has_data: torch.Tensor,
               dist_weight: float = 1.0,
               exist_pos_weight: torch.Tensor = None,
               smooth_weight: float = 0.0,
               huber_beta: float = 1.0,
               dist_decay: str = 'none',
               dist_decay_scale: float = 20.0) -> dict:
    """Compute total training loss.

    Args:
        raw_dist:         (B, 360) raw distance from Softplus head.
        pred_exist:       (B, 360) existence logit (predicts FOV membership).
        target_dist:      (B, 360) ground truth distance.
        continuous_has_data: (B, 360) bool, True for FOV bins (continuous has_data).
        has_data:         (B, 360) bool, True for bins with actual point cloud data.
        dist_weight:      Weight for distance Huber term.
        exist_pos_weight: Optional BCE pos_weight tensor.
        smooth_weight:    Weight for angular TV smoothness term (0 = disabled).
        huber_beta:       Huber loss transition point in metres (default 1.0).
        dist_decay:       Distance-based weighting for Huber loss.
            'none': uniform (default).
            'inverse': w = 1/d, closer bins weighted more.
            'linear': w = 1 - d/max_dist.
            'exp': w = exp(-d / dist_decay_scale).
        dist_decay_scale: Scale (metres) for exponential decay (default 20.0).

    Returns:
        Dict with 'total', 'radar_dist', 'exist', and 'smooth' loss values.
    """
    if continuous_has_data.sum() == 0:
        zero = raw_dist.new_tensor(0.0)
        return {'total': zero, 'radar_dist': zero, 'exist': zero, 'smooth': zero}

    max_dist = 100.0

    # Existence: predict FOV membership (continuous_has_data) on all 360 bins
    # Target: 1 if bin is in FOV, 0 otherwise
    exist_target = continuous_has_data.float()  # (360,)
    exist_loss = F.binary_cross_entropy_with_logits(
        pred_exist,  # All 360 bins
        exist_target,
        reduction='mean',
        pos_weight=exist_pos_weight,
    )

    # Distance Huber loss: compute on all FOV bins with actual data
    has_data_mask = continuous_has_data & has_data
    if has_data_mask.sum() > 0:
        target_clamped = target_dist[has_data_mask].clamp(max=max_dist)
        per_loss = F.huber_loss(raw_dist[has_data_mask], target_clamped,
                                delta=huber_beta, reduction='none')
        if dist_decay == 'inverse':
            w = 1.0 / target_clamped.clamp(min=1.0)
            dist_loss = (per_loss * w).sum() / w.sum()
        elif dist_decay == 'linear':
            w = 1.0 - target_clamped / max_dist
            dist_loss = (per_loss * w).sum() / w.sum().clamp(min=1e-6)
        elif dist_decay == 'exp':
            w = torch.exp(-target_clamped / dist_decay_scale)
            dist_loss = (per_loss * w).sum() / w.sum()
        else:
            dist_loss = per_loss.mean()
    else:
        dist_loss = raw_dist.new_tensor(0.0)

    # Angular smoothness regulariser (FOV-masked)
    if smooth_weight > 0.0:
        smooth_loss = angular_smoothness_loss(raw_dist, continuous_has_data)
    else:
        smooth_loss = raw_dist.new_tensor(0.0)

    total = dist_weight * dist_loss + exist_loss + smooth_weight * smooth_loss

    return {
        'total': total,
        'radar_dist': dist_loss.detach(),
        'exist': exist_loss.detach(),
        'smooth': smooth_loss.detach(),
    }
