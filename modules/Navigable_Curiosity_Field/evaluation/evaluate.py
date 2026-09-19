import argparse

import numpy as np


def top1_direction(scores: np.ndarray, valid: np.ndarray) -> int | None:
    masked = np.where(np.asarray(valid, dtype=bool), np.asarray(scores, dtype=float), -np.inf)
    return int(np.argmax(masked)) if np.isfinite(masked).any() else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()
    print(f"Curiosity evaluation input: {args.manifest}")


if __name__ == "__main__":
    main()
