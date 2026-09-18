from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.parse_args()
    print("Compass evaluation requires the supplied dataset manifest")


if __name__ == "__main__":
    main()
