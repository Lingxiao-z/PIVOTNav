"""Habitat-GS/Habitat-Lab environment contract used by the formal runner."""
from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np


def env_config(
    episode_path: str | Path,
    *,
    habitat_root: str | Path,
    gpu: int,
    dt: float = 0.25,
    max_v: float = 0.4,
    max_w: float = 0.3,
):
    """Build the same Habitat-Lab config used by the server runner."""
    root = Path(habitat_root).expanduser().resolve()
    os.chdir(root)
    _patch_habitat_opencv_compatibility()
    scripts_root = root / "scripts_gs"
    if str(scripts_root) not in sys.path:
        sys.path.insert(0, str(scripts_root))
    from run_panoramic_imagenav_eval_loop import build_env_config
    from habitat.config.default_structured_configs import VelocityControlActionConfig
    from habitat.config.read_write import read_write
    from omegaconf import OmegaConf

    config_path = root / "data/scene_datasets/gs_scenes/configs/ddppo_panoramic_rgb_imagenav_gs_eval.yaml"
    scene_config = root / "data/scene_datasets/gs_scenes/hm3d_annotated_basis.scene_dataset_config.json"
    args = argparse.Namespace(
        config=str(config_path),
        # The Habitat-GS builder resolves its bootstrap episode relative to
        # the Habitat-GS checkout. The formal runner replaces the dataset
        # path immediately after construction with the frozen task dataset.
        episode_data=str(root / "data/v3312_episode_manifests/development_velocity_v5_sparse_shard_96/v3312-dev-07-near_1p25_1p5m-02/navigation.json.gz"),
        dataset_split="train",
        scene_dataset_config=str(scene_config),
        seed=8102026,
        max_steps=500,
        gpu=int(gpu),
        height=256,
        width=512,
    )
    config = build_env_config(args)
    action = VelocityControlActionConfig(
        lin_vel_range=[0.0, float(max_v)],
        ang_vel_range=[-math.degrees(float(max_w)), math.degrees(float(max_w))],
        min_abs_lin_speed=-1.0,
        min_abs_ang_speed=-1.0,
        time_step=float(dt),
    )
    with read_write(config):
        config.habitat.dataset.data_path = str(Path(episode_path).expanduser().resolve())
        config.habitat.task.actions.velocity_control = OmegaConf.structured(action)
    return config


def _patch_habitat_opencv_compatibility() -> None:
    """Normalize OpenCV's colormap shape for the frozen Habitat-Lab maps code."""
    import cv2

    if getattr(cv2, "_pivotnav_colormap_compat", False):
        return
    original = cv2.applyColorMap

    def apply_color_map_compat(source, colormap):
        value = original(source, colormap)
        if getattr(value, "ndim", 0) == 3 and value.shape[0] == 1:
            value = value.transpose(1, 0, 2)
        elif getattr(value, "ndim", 0) == 2:
            value = value[:, None, :]
        return value

    cv2.applyColorMap = apply_color_map_compat
    cv2._pivotnav_colormap_compat = True


def velocity_action(
    linear_mps: float,
    angular_rps: float,
    *,
    max_v: float = 0.4,
    max_w: float = 0.3,
    dt: float = 0.25,
) -> dict[str, Any]:
    """Create the official Habitat-Lab velocity action payload."""
    return {
        "action": "velocity_control",
        "action_args": {
            "linear_velocity": float(np.clip(2.0 * linear_mps / max_v - 1.0, -1.0, 1.0)),
            "angular_velocity": float(np.clip(-angular_rps / max_w, -1.0, 1.0)),
            "time_step": float(dt),
            "allow_sliding": False,
        },
    }
