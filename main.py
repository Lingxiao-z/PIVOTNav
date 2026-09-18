#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from navigation import NavigationSystem, run_smoke


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PIVOTNav in Habitat-GS")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--scene", help="Habitat-GS scene configuration")
    parser.add_argument("--goal", help="Goal ERP image")
    parser.add_argument("--weights-root", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--topology-backend", choices=("original", "accelerated"), default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoke:
        run_smoke()
        return
    if not args.scene or not args.goal:
        raise SystemExit("--scene and --goal are required unless --smoke is used")
    try:
        import yaml
    except ImportError as exc:
        raise SystemExit("Install requirements.txt before a non-smoke run") from exc
    config_path = Path(args.config).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if args.weights_root is not None:
        config["weights_root"] = args.weights_root
    if args.device is not None:
        config["device"] = args.device
    if args.topology_backend is not None:
        config["topology_backend"] = args.topology_backend
    if args.max_steps is not None:
        config["max_steps"] = args.max_steps
    system = NavigationSystem.from_habitat_gs(config, Path(args.scene), Path(args.goal))
    result = system.run()
    print(result)


if __name__ == "__main__":
    main()
