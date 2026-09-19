from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F


def roll_source(erp: np.ndarray, labels: np.ndarray, sectors: int) -> tuple[np.ndarray, np.ndarray]:
    """Roll Source ERP and labels together by an integer sector count."""
    width = erp.shape[-2]
    if width % 12:
        raise ValueError("ERP width must be divisible by 12 for exact sector rolls")
    return np.roll(erp, sectors * (width // 12), axis=-2), np.roll(labels, sectors, axis=-1)


def roll_goal(erp: np.ndarray, sectors: int) -> np.ndarray:
    """Roll Goal ERP independently; Source-coordinate labels do not move."""
    width = erp.shape[-2]
    if width % 12:
        raise ValueError("ERP width must be divisible by 12 for exact sector rolls")
    return np.roll(erp, sectors * (width // 12), axis=-2)


def circular_relative_indices(n: int = 12) -> np.ndarray:
    idx = np.arange(n)
    delta = idx[None, :] - idx[:, None]
    return (delta + n // 2) % n - n // 2


def direction_cosine_matrix(n: int = 12) -> np.ndarray:
    return np.cos(2.0 * np.pi * circular_relative_indices(n) / n).astype(np.float32)


def erp_to_perspective(
    erp: torch.Tensor,
    *,
    yaw_degrees: torch.Tensor,
    output_size: int = 224,
    horizontal_fov_degrees: float = 90.0,
) -> torch.Tensor:
    """Project ERP images to perspective views with horizontal seam wrapping."""
    if erp.ndim != 4 or erp.shape[1] != 3:
        raise ValueError("erp must have shape [B, 3, H, W]")
    if yaw_degrees.ndim != 1:
        raise ValueError("yaw_degrees must be one-dimensional")
    b, _, _, _ = erp.shape
    v = yaw_degrees.numel()
    device, dtype = erp.device, erp.dtype
    extent = math.tan(math.radians(horizontal_fov_degrees) / 2.0)
    xy = torch.linspace(-extent, extent, output_size, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(xy, xy, indexing="ij")
    directions = torch.stack((xx, -yy, torch.ones_like(xx)), dim=-1)
    directions = directions / directions.norm(dim=-1, keepdim=True)
    yaw = torch.deg2rad(yaw_degrees.to(device=device, dtype=dtype))
    cos_yaw, sin_yaw = yaw.cos(), yaw.sin()
    x = directions[..., 0][None] * cos_yaw[:, None, None] + directions[..., 2][None] * sin_yaw[:, None, None]
    z = -directions[..., 0][None] * sin_yaw[:, None, None] + directions[..., 2][None] * cos_yaw[:, None, None]
    latitude = torch.asin(directions[..., 1][None].expand(v, -1, -1).clamp(-1.0, 1.0))
    longitude = torch.atan2(x, z)
    grid = torch.stack((longitude / math.pi / 3.0, -2.0 * latitude / math.pi), dim=-1)
    tiled = torch.cat((erp, erp, erp), dim=-1)
    expanded = tiled[:, None].expand(b, v, 3, tiled.shape[-2], tiled.shape[-1]).reshape(
        b * v, 3, tiled.shape[-2], tiled.shape[-1]
    )
    grid = grid[None].expand(b, v, output_size, output_size, 2).reshape(
        b * v, output_size, output_size, 2
    )
    views = F.grid_sample(expanded, grid, mode="bilinear", padding_mode="border", align_corners=False)
    return views.reshape(b, v, 3, output_size, output_size)


def twelve_sector_views(erp: torch.Tensor, output_size: int = 224) -> torch.Tensor:
    yaws = torch.arange(12, device=erp.device, dtype=erp.dtype) * 30.0
    return erp_to_perspective(erp, yaw_degrees=yaws, output_size=output_size)


def nts_wrapped_crops(erp: torch.Tensor) -> torch.Tensor:
    """Reproduce NTS's 12 wrapped 128x128 crops from a 128x512 panorama."""
    if erp.ndim != 4 or erp.shape[1:] != (3, 128, 512):
        raise ValueError("NTS input must have shape [B, 3, 128, 512]")
    half = 64
    centers = torch.arange(12, device=erp.device) * (512 / 12.0)
    columns = (centers[:, None].round().long() + torch.arange(-half, half, device=erp.device)[None]) % 512
    expanded = erp[:, None].expand(-1, 12, -1, -1, -1)
    index = columns[None, :, None, None, :].expand(erp.shape[0], 12, 3, 128, 128)
    return torch.gather(expanded, dim=-1, index=index)
