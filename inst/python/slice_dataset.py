#!/usr/bin/env python3
"""Slice COCO dataset for dense detection with varying image sizes."""

import argparse
from pathlib import Path
from sahi.slicing import slice_coco
import shutil
import json


def fix_coco_metadata(annotation_path):
    """Ensure COCO annotation has all required metadata fields."""
    with open(annotation_path, 'r') as f:
        coco_data = json.load(f)

    # Add 'info' field if missing
    if 'info' not in coco_data:
        coco_data['info'] = {
            'year': 2024,
            'version': '1.0',
            'description': 'Sliced COCO dataset',
            'contributor': '',
            'url': '',
            'date_created': ''
        }

    # Add 'licenses' field if missing
    if 'licenses' not in coco_data:
        coco_data['licenses'] = []

    # Write back
    with open(annotation_path, 'w') as f:
        json.dump(coco_data, f)


def main():
    parser = argparse.ArgumentParser(description="Slice COCO dataset")
    parser.add_argument("--input-dir", required=True, help="Input directory with train/valid splits")
    parser.add_argument("--output-dir", required=True, help="Output directory for sliced dataset")
    parser.add_argument("--slice-size", type=int, default=1024, help="Slice size (default: 1024)")
    parser.add_argument("--overlap", type=float, default=0.2, help="Overlap ratio (default: 0.2)")
    parser.add_argument("--min-area-ratio", type=float, default=0.1, help="Min area ratio for fragments")
    parser.add_argument("--output-format", type=str, default=".jpg", help="Output image format: .jpg or .png (default: .jpg)")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    # Slice training set
    print(f"\n{'='*60}")
    print(f"Slicing training set...")
    print(f"{'='*60}")
    train_dict, train_path = slice_coco(
        coco_annotation_file_path=str(input_dir / "train" / "_annotations.coco.json"),
        image_dir=str(input_dir / "train"),
        output_coco_annotation_file_name="_annotations",
        output_dir=str(output_dir / "train"),
        slice_height=args.slice_size,
        slice_width=args.slice_size,
        overlap_height_ratio=args.overlap,
        overlap_width_ratio=args.overlap,
        min_area_ratio=args.min_area_ratio,
        out_ext=args.output_format,
        ignore_negative_samples=True,
        verbose=False
    )

    # Rename annotation file to match expected format
    sahi_anno = output_dir / "train" / "_annotations_coco.json"
    target_anno = output_dir / "train" / "_annotations.coco.json"
    if sahi_anno.exists():
        shutil.move(str(sahi_anno), str(target_anno))
        train_path = str(target_anno)

    # Fix COCO metadata to ensure all required fields are present
    fix_coco_metadata(target_anno)

    print(f"\n✓ Train set complete:")
    print(f"  - Images: {len(train_dict['images'])}")
    print(f"  - Annotations: {len(train_dict['annotations'])}")
    print(f"  - Output: {train_path}")

    # Slice validation set
    print(f"\n{'='*60}")
    print(f"Slicing validation set...")
    print(f"{'='*60}")
    val_dict, val_path = slice_coco(
        coco_annotation_file_path=str(input_dir / "valid" / "_annotations.coco.json"),
        image_dir=str(input_dir / "valid"),
        output_coco_annotation_file_name="_annotations",
        output_dir=str(output_dir / "valid"),
        slice_height=args.slice_size,
        slice_width=args.slice_size,
        overlap_height_ratio=args.overlap,
        overlap_width_ratio=args.overlap,
        min_area_ratio=args.min_area_ratio,
        out_ext=args.output_format,
        ignore_negative_samples=True,
        verbose=False
    )

    # Rename annotation file to match expected format
    sahi_anno = output_dir / "valid" / "_annotations_coco.json"
    target_anno = output_dir / "valid" / "_annotations.coco.json"
    if sahi_anno.exists():
        shutil.move(str(sahi_anno), str(target_anno))
        val_path = str(target_anno)

    # Fix COCO metadata to ensure all required fields are present
    fix_coco_metadata(target_anno)

    print(f"\n✓ Validation set complete:")
    print(f"  - Images: {len(val_dict['images'])}")
    print(f"  - Annotations: {len(val_dict['annotations'])}")
    print(f"  - Output: {val_path}")

    # Slice test set if it exists
    test_anno = input_dir / "test" / "_annotations.coco.json"
    if test_anno.exists():
        print(f"\n{'='*60}")
        print(f"Slicing test set...")
        print(f"{'='*60}")
        test_dict, test_path = slice_coco(
            coco_annotation_file_path=str(test_anno),
            image_dir=str(input_dir / "test"),
            output_coco_annotation_file_name="_annotations",
            output_dir=str(output_dir / "test"),
            slice_height=args.slice_size,
            slice_width=args.slice_size,
            overlap_height_ratio=args.overlap,
            overlap_width_ratio=args.overlap,
            min_area_ratio=args.min_area_ratio,
            out_ext=args.output_format,
            ignore_negative_samples=True,
            verbose=False
        )

        # Rename annotation file to match expected format
        sahi_anno = output_dir / "test" / "_annotations_coco.json"
        target_anno = output_dir / "test" / "_annotations.coco.json"
        if sahi_anno.exists():
            shutil.move(str(sahi_anno), str(target_anno))
            test_path = str(target_anno)

        # Fix COCO metadata to ensure all required fields are present
        fix_coco_metadata(target_anno)

        print(f"\n✓ Test set complete:")
        print(f"  - Images: {len(test_dict['images'])}")
        print(f"  - Annotations: {len(test_dict['annotations'])}")
        print(f"  - Output: {test_path}")

    print(f"\n{'='*60}")
    print(f"✓ Sliced dataset saved to: {output_dir}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
