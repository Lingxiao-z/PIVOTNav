"""Export absolute paths of all raw images in validation set.

Usage:
    python -m scripts.traversability.export_image_paths \
        --val-dir /data1/renhao/datasets/UniNav-DB/val \
        --data-root /data1/renhao/datasets/UniNav-DB/dataset \
        --output image_paths.txt
"""

import argparse
from pathlib import Path
import numpy as np


def main():
    parser = argparse.ArgumentParser(description='Export raw image paths from validation set')
    parser.add_argument('--val-dir', required=True,
                       help='Path to val directory')
    parser.add_argument('--data-root', required=True,
                       help='Path to dataset root')
    parser.add_argument('--output', default='image_paths.txt',
                       help='Output text file')

    args = parser.parse_args()

    val_dir = Path(args.val_dir)
    data_root = Path(args.data_root)
    gt_dir = val_dir / 'ground_truth'

    if not gt_dir.exists():
        print(f"ERROR: Ground truth directory not found: {gt_dir}")
        return

    # Collect all image paths
    image_paths = []
    sample_ids = sorted([f.stem for f in gt_dir.glob('*.npz')])

    print(f"Processing {len(sample_ids)} samples...")

    for sample_id in sample_ids:
        gt_path = gt_dir / f'{sample_id}.npz'
        gt_npz = np.load(gt_path)
        rgb_path_rel = str(gt_npz['rgb_path'])

        # Convert relative path to absolute
        # rgb_path_rel format: ../UniNav-DB/dataset/raw_images/...
        # Extract the part after 'dataset/'
        parts = Path(rgb_path_rel).parts
        try:
            dataset_idx = parts.index('dataset')
            rel_from_dataset = Path(*parts[dataset_idx+1:])
            abs_path = data_root / rel_from_dataset

            if abs_path.exists():
                image_paths.append(str(abs_path.resolve()))
            else:
                print(f"  WARN: Image not found: {abs_path}")
        except (ValueError, IndexError):
            print(f"  WARN: Cannot parse path: {rgb_path_rel}")

    # Write to output file
    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        for path in image_paths:
            f.write(f"{path}\n")

    print(f"\nDone. Exported {len(image_paths)} image paths to {output_path}")


if __name__ == '__main__':
    main()
