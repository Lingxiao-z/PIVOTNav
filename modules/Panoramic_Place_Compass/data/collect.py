"""Collect ERP observations from a user-provided Habitat-GS adapter."""

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.parse_args()
    raise SystemExit("Provide a project-specific Habitat-GS factory to collect data")


if __name__ == "__main__":
    main()
