import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.parse_args()
    raise SystemExit("Set a Habitat-GS factory and task manifest before collecting data")


if __name__ == "__main__":
    main()
