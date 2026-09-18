from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class YawConvention:
    positive_yaw: str = 'Habitat yaw increases with turn_left/quaternion +Y rotation'
    positive_ring_shift: str = 'positive shift rolls candidate ring right to match query frame'
    bin_count: int = 32


def circular_error_degrees(pred: torch.Tensor | float, target: torch.Tensor | float) -> torch.Tensor:
    pred_t = torch.as_tensor(pred, dtype=torch.float32)
    target_t = torch.as_tensor(target, dtype=torch.float32, device=pred_t.device)
    return torch.abs(torch.remainder(pred_t - target_t + 180.0, 360.0) - 180.0)


def degrees_to_bins(yaw_degrees: torch.Tensor | float, bins: int = 32) -> torch.Tensor:
    yaw = torch.as_tensor(yaw_degrees, dtype=torch.float32)
    return torch.remainder(torch.round(yaw / (360.0 / bins)).long(), bins)


def bins_to_degrees(bins_idx: torch.Tensor | int, bins: int = 32) -> torch.Tensor:
    idx = torch.as_tensor(bins_idx, dtype=torch.float32)
    return torch.remainder(idx * (360.0 / bins), 360.0)


def pixel_roll_to_yaw_degrees(pixel_shift: int | torch.Tensor, width: int) -> torch.Tensor:
    # Positive torch.roll shift moves content right. In our ring convention this is
    # the positive candidate roll needed by circular correlation.
    shift = torch.as_tensor(pixel_shift, dtype=torch.float32)
    return torch.remainder(shift * (360.0 / float(width)), 360.0)


def yaw_degrees_to_pixel_roll(yaw_degrees: float | torch.Tensor, width: int) -> torch.Tensor:
    yaw = torch.as_tensor(yaw_degrees, dtype=torch.float32)
    return torch.remainder(torch.round(yaw / 360.0 * float(width)).long(), width)



def habitat_quat_yaw_to_render_yaw(quat_yaw_degrees: torch.Tensor | float) -> torch.Tensor:
    """Convert yaw_from_quat() output to the render/action yaw convention.

    Habitat-GS audit shows set_agent(yaw=+30) and turn_left(+30) produce
    yaw_from_quat()==330, while the panorama shifts by +30 degrees. V2 uses
    render/action yaw as the canonical label and converts quaternion-derived yaw
    by negation modulo 360.
    """
    yaw = torch.as_tensor(quat_yaw_degrees, dtype=torch.float32)
    return torch.remainder(-yaw, 360.0)


def render_yaw_to_habitat_quat_yaw(render_yaw_degrees: torch.Tensor | float) -> torch.Tensor:
    yaw = torch.as_tensor(render_yaw_degrees, dtype=torch.float32)
    return torch.remainder(-yaw, 360.0)
