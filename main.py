#!/usr/bin/env python3
"""Public CLI for the released PIVOTNav navigation pipeline."""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path


DIFFICULTY_BUDGETS = {
    "easy": 1200,
    "medium": 1500,
    "hard": 2000,
    "hard+": 3000,
    "hard++": 4000,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the formal PIVOTNav navigation pipeline")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--scene", required=False, help="Frozen task JSON")
    parser.add_argument("--goal", required=False, help="Goal ERP image; overrides task runtime input")
    parser.add_argument("--episode", help="Episode JSON.GZ when the task JSON has no runtime_inputs")
    parser.add_argument("--weights-root", required=False)
    parser.add_argument("--habitat-root", required=False)
    parser.add_argument("--phase-root", required=False)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def _build_single_task_manifest(task_path: Path, goal: str | None, episode: str | None, phase_root: Path) -> tuple[Path, dict]:
    task = json.loads(task_path.read_text(encoding="utf-8"))
    runtime = dict(task.get("runtime_inputs") or {})
    if episode:
        runtime["episode_path"] = str(Path(episode).expanduser().resolve())
    if goal:
        runtime["goal_erp_path"] = str(Path(goal).expanduser().resolve())
    if not runtime.get("episode_path") or not runtime.get("goal_erp_path"):
        raise SystemExit("task JSON must provide runtime_inputs.episode_path and runtime_inputs.goal_erp_path, or pass --episode/--goal")
    difficulty = str(task.get("difficulty", "Medium"))
    budget = DIFFICULTY_BUDGETS.get(difficulty.lower(), 1500)
    task_id = str(task.get("task_id", task_path.stem))
    task["task_id"] = task_id
    task["runtime_inputs"] = runtime
    task["budget_steps"] = int(task.get("budget_steps", budget))
    manifest = phase_root / "task_manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps({"schema": "pivotnav_single_task_manifest_v1", "tasks": [task]}, indent=2) + "\n", encoding="utf-8")
    return manifest, task


def main() -> None:
    args = parse_args()
    if args.smoke:
        print("PIVOTNav CLI smoke test passed")
        return
    if not args.scene:
        raise SystemExit("--scene is required unless --smoke is used")
    config = {}
    config_path = Path(args.config).resolve()
    if config_path.is_file():
        try:
            import yaml
            config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        except ImportError as exc:
            raise SystemExit("Install PyYAML before running PIVOTNav") from exc
    habitat_root = args.habitat_root or config.get("habitat_root") or os.environ.get("PIVOTNAV_HABITAT_ROOT")
    weights_root = args.weights_root or config.get("weights_root") or os.environ.get("PIVOTNAV_WEIGHTS_ROOT")
    if not habitat_root or not weights_root:
        raise SystemExit("configure habitat_root and weights_root or pass --habitat-root/--weights-root")
    os.environ["PIVOTNAV_HABITAT_ROOT"] = str(Path(habitat_root).expanduser().resolve())
    os.environ["PIVOTNAV_WEIGHTS_ROOT"] = str(Path(weights_root).expanduser().resolve())
    external_env = {
        "fs_worker": "PIVOTNAV_FS_WORKER",
        "fs_package_root": "PIVOTNAV_FS_PACKAGE_ROOT",
        "fs_checkpoint": "PIVOTNAV_FS_CHECKPOINT",
        "dinov2_weight": "PIVOTNAV_DINOV2_WEIGHT",
        "dinov2_repo": "PIVOTNAV_DINOV2_REPO",
        "omniguard_root": "PIVOTNAV_OMNIGUARD_ROOT",
        "omniguard_worker": "PIVOTNAV_OMNIGUARD_WORKER",
        "omniguard_checkpoint": "PIVOTNAV_OMNIGUARD_CHECKPOINT",
        "omniguard_python": "PIVOTNAV_OMNIGUARD_PYTHON",
        "arrival_frozen_root": "PIVOTNAV_ARRIVAL_FROZEN_ROOT",
        "arrival_dependency_root": "PIVOTNAV_ARRIVAL_DEPENDENCY_ROOT",
        "arrival_protocol": "PIVOTNAV_ARRIVAL_PROTOCOL",
        "arrival_model": "PIVOTNAV_ARRIVAL_MODEL",
    }
    for key, env_key in external_env.items():
        value = config.get(key) or os.environ.get(env_key)
        if value:
            os.environ[env_key] = str(Path(value).expanduser().resolve())
    if os.environ.get("PIVOTNAV_DINOV2_WEIGHT"):
        os.environ["PANORAMIC_VPR_DINOV2_WEIGHT"] = os.environ["PIVOTNAV_DINOV2_WEIGHT"]
    if os.environ.get("PIVOTNAV_DINOV2_REPO"):
        os.environ["PANORAMIC_VPR_DINOV2_CHECKOUT"] = os.environ["PIVOTNAV_DINOV2_REPO"]
    phase_root = Path(args.phase_root or config.get("phase_root", tempfile.mkdtemp(prefix="pivotnav_run_"))).expanduser().resolve()
    manifest, task = _build_single_task_manifest(Path(args.scene).resolve(), args.goal, args.episode, phase_root)
    from navigation import main as formal_main
    sys.argv = [
        "pivotnav-formal",
        "--manifest", str(manifest),
        "--phase-root", str(phase_root),
        "--scene-id", str(task["scene_id"]),
        "--gpu", str(args.gpu),
        "--bearing-mode", "visual",
        "--goal-verifier",
    ]
    if args.max_steps is not None:
        task["budget_steps"] = int(args.max_steps)
        manifest.write_text(json.dumps({"schema": "pivotnav_single_task_manifest_v1", "tasks": [task]}, indent=2) + "\n", encoding="utf-8")
    formal_main()


if __name__ == "__main__":
    main()
