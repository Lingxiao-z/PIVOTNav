from __future__ import annotations

import math

import torch
import torch.nn.functional as F


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
    zz = torch.ones_like(xx)
    directions = torch.stack((xx, -yy, zz), dim=-1)
    directions = directions / directions.norm(dim=-1, keepdim=True)
    yaw = torch.deg2rad(yaw_degrees.to(device=device, dtype=dtype))
    cos_yaw, sin_yaw = yaw.cos(), yaw.sin()
    x = directions[..., 0][None] * cos_yaw[:, None, None] + directions[..., 2][None] * sin_yaw[:, None, None]
    z = -directions[..., 0][None] * sin_yaw[:, None, None] + directions[..., 2][None] * cos_yaw[:, None, None]
    y = directions[..., 1][None].expand(v, -1, -1)
    longitude = torch.atan2(x, z)
    latitude = torch.asin(y.clamp(-1.0, 1.0))
    grid_x = longitude / math.pi
    grid_y = -2.0 * latitude / math.pi
    grid = torch.stack((grid_x, grid_y), dim=-1)
    tiled = torch.cat((erp, erp, erp), dim=-1)
    grid = grid.clone()
    # [-1, 1] longitude maps to the middle copy of a 3x tiled ERP.
    grid[..., 0] = grid[..., 0] / 3.0
    expanded = tiled[:, None].expand(b, v, 3, tiled.shape[-2], tiled.shape[-1]).reshape(b * v, 3, tiled.shape[-2], tiled.shape[-1])
    grid = grid[None].expand(b, v, output_size, output_size, 2).reshape(b * v, output_size, output_size, 2)
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
