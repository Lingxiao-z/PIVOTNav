"""Small online evidence helpers shared by the RGB arrival state machine."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def pair_from_encoding(runtime: Any, query: Any, target: Any) -> dict[str, float]:
    """Build the scalar arrival evidence from two cached R36 encodings."""
    import torch
    from torch.nn import functional as F

    with torch.inference_mode():
        pair = runtime._pair_outputs(query, target)
        feature = pair["arrival_feature"].float().unsqueeze(1)
        visual = runtime.arrival_head.visual_encoder(feature)
        probability = torch.sigmoid(runtime.arrival_head.frame_logit(visual).squeeze(-1))[:, 0]
        similarity = F.cosine_similarity(
            query["global_descriptor"].float(),
            target["global_descriptor"].float(),
            dim=-1,
        )
        yaw = pair["yaw"]
    return {
        "arrival_probability": float(probability[0]),
        "vpr_similarity": float(similarity[0]),
        "yaw_degrees": float(yaw["predicted_yaw_degrees"][0]),
        "yaw_confidence": float(yaw["yaw_confidence"][0]),
    }


def wrap_degrees(value: float) -> float:
    return (float(value) + 180.0) % 360.0 - 180.0


def feature_names() -> list[str]:
    names = []
    for view in ("restore", "approach1", "approach2"):
        for field in (
            "raw_matches", "essential_inliers", "essential_ratio",
            "homography_inlier_ratio", "homography_scale", "flow_mean",
            "flow_median", "flow_p25", "flow_p75", "flow_p90",
            "flow_x_median", "flow_y_median",
        ):
            names.append(f"{view}_{field}")
    names.extend([
        "flow_decrease_01", "flow_decrease_12", "distance_estimate_1",
        "distance_estimate_2", "distance_estimate_median",
        "monotonic_flow_decrease_count", "current_arrival_probability",
        "current_vpr_similarity", "current_e_inliers", "current_e_inlier_ratio",
        "current_grid_coverage", "current_hull_area", "current_reprojection_error",
        "current_supported_sectors", "current_h_dominance", "target_top1_fraction",
        "target_top1_current",
    ])
    return names


def make_features(parallax: dict, row: dict) -> np.ndarray:
    common_sector = int(parallax["causal"]["common_sector_id"])
    values = []
    for view in parallax["views"]:
        sector = view["sectors"][common_sector]
        values.extend([
            sector["raw_matches"], sector["essential_inliers"], sector["essential_ratio"],
            sector["homography_inlier_ratio"], sector["homography_scale"],
            sector["flow"]["mean"], sector["flow"]["median"], sector["flow"]["p25"],
            sector["flow"]["p75"], sector["flow"]["p90"], sector["flow_x"]["median"],
            sector["flow_y"]["median"],
        ])
    causal = parallax["causal"]
    estimates = [min(float(value), 10.0) for value in causal["remaining_distance_estimate_sequence_m"]]
    evidence = row["views"][-1]["evidence"]
    top1 = [bool(view["target_top1"]) for view in row["views"]]
    values.extend([
        *causal["flow_decrease_sequence"], *estimates,
        min(float(causal["remaining_distance_estimate_median_m"]), 10.0),
        causal["monotonic_flow_decrease_count"], evidence["arrival_probability"],
        evidence["vpr_similarity"], evidence["e_inliers"], evidence["e_inlier_ratio"],
        evidence["grid_coverage"], evidence["hull_area"], evidence["reprojection_error"],
        evidence["supported_sectors"], evidence["h_dominance"], np.mean(top1), float(top1[-1]),
    ])
    result = np.asarray(values, dtype=np.float32)
    if len(result) != len(feature_names()) or not np.isfinite(result).all():
        raise ValueError(f"invalid parallax feature vector for {row['trial_id']}")
    return result


def hard_safety(row: dict, config: dict) -> tuple[bool, list[str]]:
    by_label = {view["view_label"]: view for view in row["views"]}
    required = ["ORIGINAL", "LEFT_10", "RIGHT_20", "RESTORE_10"]
    reasons = []
    if [view["view_label"] for view in row["views"]][:4] != required:
        reasons.append("missing_rotation_cycle")
    else:
        original = float(by_label["ORIGINAL"]["r361"]["yaw_degrees"])
        errors = [
            abs(wrap_degrees(float(by_label["LEFT_10"]["r361"]["yaw_degrees"]) - original + 10.0)),
            abs(wrap_degrees(float(by_label["RIGHT_20"]["r361"]["yaw_degrees"]) - original - 10.0)),
            abs(wrap_degrees(float(by_label["RESTORE_10"]["r361"]["yaw_degrees"]) - original)),
        ]
        if max(errors) > config["maximum_yaw_cycle_error_degrees"]:
            reasons.append("yaw_cycle_inconsistent")
    evidence = row["views"][-1]["evidence"]
    if int(evidence["e_inliers"]) < config["minimum_current_e_inliers"]:
        reasons.append("insufficient_current_inliers")
    if int(evidence["supported_sectors"]) < config["minimum_current_supported_sectors"]:
        reasons.append("insufficient_current_sector_support")
    return not reasons, reasons


def _view_rgb(view: dict) -> np.ndarray:
    image_rgb = view.get("image_rgb")
    if image_rgb is not None:
        array = np.asarray(image_rgb)
    else:
        image_path = view.get("image_path")
        if not image_path or not Path(image_path).is_file():
            raise RuntimeError("MISSING_RGB_EVIDENCE")
        array = np.asarray(Image.open(image_path).convert("RGB"))
    if array.ndim != 3 or array.shape[-1] < 3:
        raise RuntimeError("INVALID_RGB_EVIDENCE_SHAPE")
    return np.asarray(array[..., :3], dtype=np.float32)


def rgb_change(first: dict, second: dict) -> float:
    first_rgb = _view_rgb(first)
    second_rgb = _view_rgb(second)
    first_aligned = np.roll(
        first_rgb,
        int(round(-float(first["r361"]["yaw_degrees"]) / 360.0 * first_rgb.shape[1])),
        axis=1,
    )
    second_aligned = np.roll(
        second_rgb,
        int(round(-float(second["r361"]["yaw_degrees"]) / 360.0 * second_rgb.shape[1])),
        axis=1,
    )
    return float(np.mean(np.abs(first_aligned - second_aligned)))
