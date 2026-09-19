from __future__ import annotations

import argparse


def circular_error_deg(prediction: float, target: float) -> float:
    return abs((float(prediction) - float(target) + 180.0) % 360.0 - 180.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.parse_args()
    print("Compass evaluation requires the supplied dataset manifest")


if __name__ == "__main__":
    main()
