import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()
    print(f"Curiosity evaluation input: {args.manifest}")


if __name__ == "__main__":
    main()
