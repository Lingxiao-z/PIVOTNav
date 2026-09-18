from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from ..geometry import signed_model_degrees
from ..models.unik3d_runtime import resize_rays, resize_scalar_map


def azimuth_to_rgb(azimuth: np.ndarray, valid_mask: np.ndarray | None = None) -> np.ndarray:
    az = np.asarray(azimuth, dtype=np.float32)
    hue = ((az + np.pi) / (2.0 * np.pi) * 179.0).astype(np.uint8)
    hsv = np.zeros((*az.shape, 3), dtype=np.uint8)
    hsv[..., 0] = hue
    hsv[..., 1] = 255
    hsv[..., 2] = 230
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    if valid_mask is not None:
        rgb = rgb.copy()
        rgb[~np.asarray(valid_mask, dtype=bool)] = np.array([18, 18, 18], dtype=np.uint8)
    return rgb


def build_pixel_geometry(
    *,
    image_hw: tuple[int, int],
    camera_model: str,
    intrinsics: dict[str, Any],
) -> dict[str, np.ndarray]:
    h, w = int(image_hw[0]), int(image_hw[1])
    u, v = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    camera_model = str(camera_model).strip().lower()

    if camera_model == "fisheye":
        if "rays" not in intrinsics:
            raise RuntimeError("Cannot build fisheye pixel geometry without rays.")
        rays = resize_rays(np.asarray(intrinsics["rays"], dtype=np.float32), (h, w))
        valid = np.isfinite(rays).all(axis=2) & (np.linalg.norm(rays, axis=2) > 1e-6)
        azimuth = np.arctan2(rays[:, :, 0], rays[:, :, 2]).astype(np.float32)
    elif camera_model == "equirectangular":
        azimuth = ((u / max(float(w), 1.0)) - 0.5) * (2.0 * np.pi)
        elevation = (0.5 - v / max(float(h), 1.0)) * np.pi
        valid = np.isfinite(azimuth) & np.isfinite(elevation)
    elif camera_model == "pinhole":
        fx = float(intrinsics["fx"])
        cx = float(intrinsics["cx"])
        x = (u - cx) / max(fx, 1e-6)
        azimuth = np.arctan2(x, 1.0).astype(np.float32)
        valid = np.isfinite(azimuth)
    else:
        raise ValueError(f"Unsupported camera model for pixel geometry: {camera_model}")

    return {
        "azimuth_map": azimuth.astype(np.float32, copy=False),
        "valid_mask": valid.astype(bool, copy=False),
    }


def build_projection_context(
    *,
    depth_m: np.ndarray,
    frame_bgr: np.ndarray,
    camera_model: str,
    intrinsics: dict[str, Any],
) -> dict[str, np.ndarray | None]:
    """Build per-pixel metric geometry for RGB radar-distance projection."""
    rgb = cv2.cvtColor(np.asarray(frame_bgr, dtype=np.uint8), cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]
    depth = resize_scalar_map(np.asarray(depth_m, dtype=np.float32), (h, w))
    valid = np.isfinite(depth) & (depth > 0)

    x_map = np.full((h, w), np.nan, dtype=np.float32)
    y_map = np.full((h, w), np.nan, dtype=np.float32)
    z_map = np.full((h, w), np.nan, dtype=np.float32)
    u, v = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    camera_model = str(camera_model).strip().lower()

    if camera_model == "fisheye":
        if "rays" not in intrinsics:
            raise RuntimeError("Fisheye geo_interp overlay requires per-pixel rays.")
        rays = resize_rays(np.asarray(intrinsics["rays"], dtype=np.float32), (h, w))
        valid &= np.isfinite(rays).all(axis=2)
        x_map[valid] = rays[:, :, 0][valid] * depth[valid]
        y_map[valid] = rays[:, :, 1][valid] * depth[valid]
        z_map[valid] = rays[:, :, 2][valid] * depth[valid]
    elif camera_model == "equirectangular":
        azimuth = ((u / max(float(w), 1.0)) - 0.5) * (2.0 * np.pi)
        elevation = (0.5 - v / max(float(h), 1.0)) * np.pi
        x_map[valid] = (np.cos(elevation) * np.sin(azimuth) * depth)[valid]
        y_map[valid] = (-np.sin(elevation) * depth)[valid]
        z_map[valid] = (np.cos(elevation) * np.cos(azimuth) * depth)[valid]
    elif camera_model == "pinhole":
        required = ("fx", "fy", "cx", "cy")
        missing = [key for key in required if intrinsics.get(key) is None]
        if missing:
            raise RuntimeError(f"Pinhole geo_interp overlay requires intrinsics: missing {missing}.")
        fx = float(intrinsics["fx"])
        fy = float(intrinsics["fy"])
        cx = float(intrinsics["cx"])
        cy = float(intrinsics["cy"])
        z = depth[valid]
        x_map[valid] = (u[valid] - cx) * z / max(fx, 1e-6)
        y_map[valid] = (v[valid] - cy) * z / max(fy, 1e-6)
        z_map[valid] = z
    else:
        raise ValueError(f"Unsupported camera model for projection context: {camera_model}")

    valid &= np.isfinite(x_map) & np.isfinite(y_map) & np.isfinite(z_map)
    if not valid.any():
        raise RuntimeError("Depth projection produced no valid RGB pixels.")

    return {
        "valid_mask": valid.astype(bool, copy=False),
        "dist_map": np.sqrt(x_map * x_map + z_map * z_map).astype(np.float32, copy=False),
        "angle_map": np.arctan2(x_map, z_map).astype(np.float32, copy=False),
        "x_map": x_map,
        "y_map": y_map,
        "z_map": z_map,
        "depth_m": depth.astype(np.float32, copy=False),
        "rgb_aligned": rgb,
    }


def traversability_overlay_rgb(
    *,
    frame_bgr: np.ndarray,
    depth_m: np.ndarray,
    effective_distance_m: np.ndarray,
    exist_probability: np.ndarray,
    config: dict[str, Any],
    camera_model: str,
    intrinsics: dict[str, Any],
    observed_angle_mask: np.ndarray | None = None,
    point_radius_px: int = 3,
    distance_tolerance_m: float | None = None,
    lower_envelope_margin_px: int = 10,
) -> np.ndarray:
    """Project predicted radar-distance endpoints back onto RGB as a clearance line."""
    context = build_projection_context(
        depth_m=depth_m,
        frame_bgr=frame_bgr,
        camera_model=camera_model,
        intrinsics=intrinsics,
    )
    rgb = np.asarray(context["rgb_aligned"], dtype=np.uint8)
    valid = np.asarray(context["valid_mask"], dtype=bool)
    az = np.asarray(context["angle_map"], dtype=np.float32)
    hdist = np.asarray(context["dist_map"], dtype=np.float32)

    distances = np.asarray(effective_distance_m, dtype=np.float32)
    probabilities = np.asarray(exist_probability, dtype=np.float32)
    n_bins = int(distances.shape[0])
    if n_bins <= 0:
        raise ValueError("effective_distance_m must contain at least one angle bin.")
    max_range = float(config["controllers"]["navigation"]["polar_esdf"]["max_range"])
    angles_deg = signed_model_degrees(n_bins)
    if observed_angle_mask is None:
        fov_half = float(config["controllers"]["navigation"]["polar_esdf"].get("fov_deg", 360.0)) * 0.5
        observed_bins = np.abs(angles_deg) <= fov_half + 1e-3
    else:
        observed_bins = np.asarray(observed_angle_mask, dtype=bool).reshape(-1)
        if observed_bins.shape[0] != n_bins:
            raise ValueError(f"observed_angle_mask has {observed_bins.shape[0]} bins, expected {n_bins}")

    bin_width = 2.0 * np.pi / float(n_bins)
    bin_idx = np.floor((az + np.pi) / bin_width).astype(np.int32)
    bin_idx = np.clip(bin_idx, 0, n_bins - 1)
    in_model_fov = observed_bins[bin_idx]

    overlay = rgb.copy()
    legacy_tolerance = float(distance_tolerance_m) if distance_tolerance_m is not None else 0.5
    radius = max(1, int(point_radius_px))

    candidate_mask = valid & in_model_fov & np.isfinite(hdist)
    if not candidate_mask.any():
        raise RuntimeError("No valid RGB pixels are available for radar-distance point projection.")

    bin_pixels: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for bin_id in range(n_bins):
        mask = candidate_mask & (bin_idx == bin_id)
        if not mask.any():
            continue
        rows, cols = np.nonzero(mask)
        dist_values = hdist[rows, cols].astype(np.float32, copy=False)

        # A radar endpoint is a 2D ground-plane measurement.  The RGB/depth image
        # contains many pixels with the same horizontal distance on vertical
        # surfaces, so follow OmniTrav's ground-support idea without a semantic
        # mask: keep the visible lower envelope in each distance slice.
        finite = np.isfinite(dist_values)
        if finite.any():
            rows_i = rows[finite].astype(np.int32, copy=False)
            cols_i = cols[finite].astype(np.int32, copy=False)
            dist_i = dist_values[finite]
            bucket_m = max(0.04, 0.006 * max_range)
            bucket_idx = np.round(dist_i / bucket_m).astype(np.int32)
            best_by_bucket: dict[int, tuple[int, int, float]] = {}
            for row, col, dist_value, bucket in zip(rows_i, cols_i, dist_i, bucket_idx):
                current = best_by_bucket.get(int(bucket))
                if current is None or int(row) > current[0]:
                    best_by_bucket[int(bucket)] = (int(row), int(col), float(dist_value))
            support_rows = np.asarray([item[0] for item in best_by_bucket.values()], dtype=np.float32)
            support_cols = np.asarray([item[1] for item in best_by_bucket.values()], dtype=np.float32)
            support_dist = np.asarray([item[2] for item in best_by_bucket.values()], dtype=np.float32)
            if support_dist.size:
                keep_margin = max(0, int(lower_envelope_margin_px))
                if keep_margin > 0:
                    keep = np.zeros_like(dist_i, dtype=bool)
                    for support_row, support_dist_value in zip(support_rows, support_dist):
                        near_dist = np.abs(dist_i - support_dist_value) <= bucket_m
                        near_row = rows_i >= int(round(support_row)) - keep_margin
                        keep |= near_dist & near_row
                    if keep.any():
                        support_rows = rows_i[keep].astype(np.float32, copy=False)
                        support_cols = cols_i[keep].astype(np.float32, copy=False)
                        support_dist = dist_i[keep].astype(np.float32, copy=False)
                bin_pixels[bin_id] = (support_rows, support_cols, support_dist)

    height, width = rgb.shape[:2]

    def _split_polyline(
        bin_ids: np.ndarray,
        u_arr: np.ndarray,
        v_arr: np.ndarray,
        *,
        max_jump_px: float,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Split the projected clearance curve when neighboring bins jump too far."""
        if u_arr.size < 2:
            return []
        segments: list[tuple[np.ndarray, np.ndarray]] = []
        start = 0
        for i in range(1, u_arr.size):
            du = float(u_arr[i] - u_arr[i - 1])
            dv = float(v_arr[i] - v_arr[i - 1])
            bin_gap = int(bin_ids[i]) != int(bin_ids[i - 1]) + 1
            seam_jump = camera_model == "equirectangular" and abs(du) > (0.5 * float(width))
            jump = (du * du + dv * dv) ** 0.5
            if bin_gap or seam_jump or jump > max_jump_px:
                if i - start >= 2:
                    segments.append((u_arr[start:i], v_arr[start:i]))
                start = i
        if u_arr.size - start >= 2:
            segments.append((u_arr[start:], v_arr[start:]))
        return segments

    def _sample_line_uv(bin_id: int, target_m: float) -> tuple[float, float] | None:
        """OmniTrav-style geometry-aware distance interpolation inside one angle bin."""
        if bin_id not in bin_pixels:
            return None
        rows, cols, bin_dist = bin_pixels[bin_id]
        finite = np.isfinite(bin_dist)
        if not finite.any():
            return None

        row = rows[finite].astype(np.float32, copy=False)
        col = cols[finite].astype(np.float32, copy=False)
        dist = bin_dist[finite].astype(np.float32, copy=False)

        order = np.argsort(dist)
        dist = dist[order]
        row = row[order]
        col = col[order]
        if dist.size == 0:
            return None

        if camera_model == "equirectangular" and (np.nanmax(col) - np.nanmin(col)) > (0.5 * float(width)):
            col = np.where(col < (0.5 * float(width)), col + float(width), col)

        target = float(target_m)
        if target <= float(dist[0]):
            uu = float(col[0])
            vv = float(row[0])
        elif target >= float(dist[-1]):
            uu = float(col[-1])
            vv = float(row[-1])
        else:
            idx = int(np.searchsorted(dist, target, side="left"))
            idx = int(np.clip(idx, 1, dist.size - 1))
            d0 = float(dist[idx - 1])
            d1 = float(dist[idx])
            if abs(d1 - d0) < 1e-6:
                interp = 0.0
            else:
                interp = (target - d0) / (d1 - d0)
            uu = float((1.0 - interp) * col[idx - 1] + interp * col[idx])
            vv = float((1.0 - interp) * row[idx - 1] + interp * row[idx])

        if camera_model == "equirectangular":
            uu = uu % float(width)
        uu = float(np.clip(uu, 0.0, max(0.0, float(width - 1))))
        vv = float(np.clip(vv, 0.0, max(0.0, float(height - 1))))
        return uu, vv

    projected_bins: list[int] = []
    projected_cols: list[float] = []
    projected_rows: list[float] = []
    for bin_id in range(n_bins):
        if not bool(observed_bins[bin_id]):
            continue
        target_m = float(np.clip(distances[bin_id], 0.0, max_range))
        if not np.isfinite(target_m) or target_m <= 0.0:
            continue

        point = _sample_line_uv(bin_id, target_m)
        if point is None:
            continue
        col_f, row_f = point
        if distance_tolerance_m is not None:
            nearest_dist = hdist[int(round(row_f)), int(round(col_f))]
            if np.isfinite(nearest_dist) and abs(float(nearest_dist) - target_m) > legacy_tolerance:
                continue
        row = int(round(row_f))
        col = int(round(col_f))
        projected_bins.append(int(bin_id))
        projected_cols.append(float(col))
        projected_rows.append(float(row))

    if not projected_bins:
        raise RuntimeError("Radar-distance line projection found no drawable angle bins.")

    bin_array = np.asarray(projected_bins, dtype=np.int32)
    col_array = np.asarray(projected_cols, dtype=np.float32)
    row_array = np.asarray(projected_rows, dtype=np.float32)
    jump_thresh = max(12.0, 0.03 * float(np.hypot(height, width)))
    segments = _split_polyline(bin_array, col_array, row_array, max_jump_px=jump_thresh)

    line_color = (0, 255, 0)
    outline_thickness = max(3, radius + 2)
    line_thickness = max(2, radius + 1)
    drawn = 0
    for seg_u, seg_v in segments:
        pts = np.stack([np.rint(seg_u), np.rint(seg_v)], axis=1).astype(np.int32, copy=False)
        if pts.shape[0] < 2:
            continue
        pts = pts.reshape(-1, 1, 2)
        cv2.polylines(overlay, [pts], isClosed=False, color=(0, 0, 0), thickness=outline_thickness, lineType=cv2.LINE_AA)
        cv2.polylines(overlay, [pts], isClosed=False, color=line_color, thickness=line_thickness, lineType=cv2.LINE_AA)
        drawn += 1

    if drawn == 0 and col_array.size > 0:
        for col, row in zip(col_array, row_array):
            cv2.circle(
                overlay,
                (int(round(col)), int(round(row))),
                radius + 1,
                (0, 0, 0),
                thickness=-1,
                lineType=cv2.LINE_AA,
            )
            cv2.circle(
                overlay,
                (int(round(col)), int(round(row))),
                radius,
                line_color,
                thickness=-1,
                lineType=cv2.LINE_AA,
            )
        drawn = int(col_array.size)

    if drawn == 0:
        raise RuntimeError("Radar-distance line projection found no drawable segments.")

    return overlay
