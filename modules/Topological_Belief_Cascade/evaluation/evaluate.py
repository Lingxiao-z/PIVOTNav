import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", required=True)
    args = parser.parse_args()
    print(f"Topology benchmark input: {args.benchmark}")


if __name__ == "__main__":
    main()
