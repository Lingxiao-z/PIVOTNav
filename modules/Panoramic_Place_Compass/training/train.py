"""Training entry point placeholder; dataset-specific training stays reproducible."""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.parse_args()
    raise SystemExit("Connect this entry point to the collected Compass manifest before training")


if __name__ == "__main__":
    main()
