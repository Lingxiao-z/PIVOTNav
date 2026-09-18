#!/usr/bin/env python3
"""Habitat-GS ERP to perspective projection for V3.3.8.

The module intentionally contains no arrival decision logic. It only maps an
ERP RGB image to a deterministic, shared query/reference perspective layout.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class PerspectiveViewSpec:
    view_index: int
    center_yaw_deg: float
    hfov_deg: float = 100.0
    vfov_deg: float = 90.0
    width: int = 256
    height: int = 256


def layout(view_count: int, hfov_deg: float = 100.0, vfov_deg: float = 90.0,
           width: int = 256, height: int = 256) -> tuple[PerspectiveViewSpec, ...]:
    if view_count not in (8, 12, 16):
        raise ValueError("V3.3.8 comparison requires 8, 12, or 16 views")
    return tuple(PerspectiveViewSpec(i, 360.0 * i / view_count, hfov_deg,
                                     vfov_deg, width, height)
                 for i in range(view_count))


def _mapping(spec: PerspectiveViewSpec, erp_shape: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray]:
    height, width = erp_shape[:2]
    fx = (spec.width / 2.0) / math.tan(math.radians(spec.hfov_deg) / 2.0)
    fy = (spec.height / 2.0) / math.tan(math.radians(spec.vfov_deg) / 2.0)
    cx = (spec.width - 1.0) / 2.0
    cy = (spec.height - 1.0) / 2.0
    u, v = np.meshgrid(np.arange(spec.width, dtype=np.float32),
                       np.arange(spec.height, dtype=np.float32))
    ray_x = (u - cx) / fx
    ray_y = -(v - cy) / fy
    longitude = np.arctan2(ray_x, np.ones_like(ray_x)) + math.radians(spec.center_yaw_deg)
    latitude = np.arctan2(ray_y, np.sqrt(ray_x * ray_x + 1.0))
    map_x = ((longitude / (2.0 * math.pi) + 0.5) * width) % width
    map_y = np.clip((0.5 - latitude / math.pi) * height, 0.0, height - 1.0)
    return map_x.astype(np.float32), map_y.astype(np.float32)


def project(erp_rgb: np.ndarray, spec: PerspectiveViewSpec) -> tuple[np.ndarray, dict]:
    if erp_rgb.ndim != 3 or erp_rgb.shape[2] != 3:
        raise ValueError("ERP must be HxWx3 RGB")
    map_x, map_y = _mapping(spec, erp_rgb.shape)
    view = cv2.remap(erp_rgb, map_x, map_y, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_WRAP)
    width = erp_rgb.shape[1]
    half = spec.hfov_deg / 2.0
    seam_crossing = (spec.center_yaw_deg - half < 0.0 or
                     spec.center_yaw_deg + half >= 360.0)
    metadata = asdict(spec)
    metadata.update({
        "erp_resolution": [int(width), int(erp_rgb.shape[0])],
        "mapping": "gnomonic pinhole ray to ERP longitude/latitude; horizontal modulo wrap",
        "seam_crossing": bool(seam_crossing),
        "map_x_min": float(map_x.min()),
        "map_x_max": float(map_x.max()),
        "map_y_min": float(map_y.min()),
        "map_y_max": float(map_y.max()),
        "view_sha256": hashlib.sha256(view.tobytes()).hexdigest(),
    })
    return view, metadata


def project_layout(erp_rgb: np.ndarray, view_count: int) -> tuple[list[np.ndarray], list[dict]]:
    views, metadata = [], []
    for spec in layout(view_count):
        view, info = project(erp_rgb, spec)
        views.append(view)
        metadata.append(info)
    return views, metadata


def save_layout(erp_path: Path, output_dir: Path, view_count: int) -> dict:
    erp = cv2.cvtColor(cv2.imread(str(erp_path)), cv2.COLOR_BGR2RGB)
    views, metadata = project_layout(erp, view_count)
    output_dir.mkdir(parents=True, exist_ok=True)
    for image, info in zip(views, metadata):
        target = output_dir / f"view_{info['view_index']:02d}.png"
        cv2.imwrite(str(target), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        info["path"] = str(target)
    contact = np.concatenate(views, axis=1)
    contact_path = output_dir / "contact_sheet.png"
    cv2.imwrite(str(contact_path), cv2.cvtColor(contact, cv2.COLOR_RGB2BGR))
    return {"layout": metadata, "contact_sheet": str(contact_path),
            "contact_sheet_sha256": hashlib.sha256(contact.tobytes()).hexdigest()}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("erp")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--views", type=int, choices=(8, 12, 16), default=8)
    args = parser.parse_args()
    print(json.dumps(save_layout(Path(args.erp), args.output_dir, args.views), indent=2))
