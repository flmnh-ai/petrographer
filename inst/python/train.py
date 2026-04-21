#!/usr/bin/env python3
"""
RF-DETR Training Script for Petrographer
Uses RF-DETR's PyTorch Lightning training loop (rfdetr >= 1.6.0).
"""

import argparse
import json
import os
from pathlib import Path

# Reduce CUDA memory fragmentation. PyTorch's default caching allocator
# accumulates unusable holes across long training runs with variable tensor
# sizes (aux losses + multi-scale training + occasional dense-GT tiles), which
# has been responsible for the `generalized_box_iou` OOMs we keep seeing even
# when nominal memory headroom should be enough. `expandable_segments` lets
# freed memory compact and grow on demand, typically recovering 5-30% of
# "reserved but unallocated" memory. Must be set BEFORE torch is imported —
# `import rfdetr` below will pull in torch, so this block stays above it.
# `setdefault` preserves any user override passed in from the environment.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import rfdetr
from rfdetr.datasets.aug_config import AUG_AERIAL

# Remove PyTorch DDP variables that SLURM sets automatically.
# We handle GPU allocation via PTL's accelerator/devices args instead.
for key in ['RANK', 'WORLD_SIZE', 'LOCAL_RANK',
            'SLURM_PROCID', 'SLURM_LOCALID', 'SLURM_NTASKS']:
    os.environ.pop(key, None)

# Variant -> class name mapping
VARIANT_TO_CLASS = {
    'nano': 'RFDETRNano', 'small': 'RFDETRSmall',
    'medium': 'RFDETRMedium', 'large': 'RFDETRLarge',
    'seg_preview': 'RFDETRSegPreview',
    'seg_nano': 'RFDETRSegNano', 'seg_small': 'RFDETRSegSmall',
    'seg_medium': 'RFDETRSegMedium', 'seg_large': 'RFDETRSegLarge',
    'seg_xlarge': 'RFDETRSegXLarge', 'seg_2xlarge': 'RFDETRSeg2XLarge',
}

# Max objects per variant (num_queries from rfdetr configs)
VARIANT_MAX_OBJECTS = {
    'nano': 300, 'small': 300, 'medium': 300, 'large': 300,
    'seg_preview': 200,
    'seg_nano': 100, 'seg_small': 100,
    'seg_medium': 200, 'seg_large': 200,
    'seg_xlarge': 300, 'seg_2xlarge': 300,
}


def main():
    parser = argparse.ArgumentParser(description='Train RF-DETR model for petrography')

    # Dataset
    parser.add_argument('--dataset-dir', type=str, required=True,
                        help='Path to dataset directory (contains train/ and valid/ subdirs)')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory for trained model')

    # Model
    parser.add_argument('--model-variant', type=str, default='nano',
                        choices=list(VARIANT_TO_CLASS.keys()),
                        help='RF-DETR model variant')
    parser.add_argument('--resolution', type=int, default=None,
                        help='Image resolution for training (auto-detected from variant if not specified)')

    # Training parameters
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs (default: 100)')
    parser.add_argument('--batch-size', type=str, default='auto',
                        help='Batch size (integer or "auto" for auto-detection)')
    parser.add_argument('--grad-accum-steps', type=int, default=4,
                        help='Gradient accumulation steps (default: 4)')
    parser.add_argument('--learning-rate', type=float, default=None,
                        help='Learning rate (uses model default if not specified)')

    # Memory optimization
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cpu', 'cuda', 'mps'],
                        help='Device for training')
    parser.add_argument('--use-amp', action='store_true', default=False,
                        help='Use automatic mixed precision training when supported')
    parser.add_argument('--amp-dtype', type=str, default='bf16',
                        choices=['bf16', 'fp16'],
                        help='AMP dtype for mixed precision training')
    parser.add_argument('--gradient-checkpointing', action='store_true', default=False,
                        help='Enable gradient checkpointing (reduces memory usage)')

    # System
    parser.add_argument('--num-workers', type=int, default=8,
                        help='Number of data loading workers (default: 8)')

    # Validation, checkpointing & early stopping
    # Fixed in rfdetr 1.6.1 (PR #848) — eval_interval > 1 now works
    parser.add_argument('--eval-interval', type=int, default=None,
                        help='Evaluate every N epochs (omit for rfdetr default: 1)')
    parser.add_argument('--checkpoint-interval', type=int, default=None,
                        help='Save checkpoint every N epochs (omit for rfdetr default: 10)')
    parser.add_argument('--early-stopping-patience', type=int, default=None,
                        help='Stop if val metric stalls for N evals (disabled if not set)')
    parser.add_argument('--eval-max-dets', type=int, default=500,
                        help='Max detections for COCO eval (default: 500)')

    # Performance tuning
    parser.add_argument('--fp16-eval', action='store_true', default=True,
                        help='Run validation in half precision (default: True)')
    parser.add_argument('--no-fp16-eval', action='store_false', dest='fp16_eval',
                        help='Disable half-precision validation')
    parser.add_argument('--pin-memory', action='store_true', default=True,
                        help='Pin memory for faster CPU->GPU transfer (default: True)')
    parser.add_argument('--persistent-workers', action='store_true', default=True,
                        help='Keep dataloader workers alive between epochs (default: True)')
    parser.add_argument('--prefetch-factor', type=int, default=4,
                        help='Number of batches each worker pre-loads (default: 4)')

    args = parser.parse_args()

    # Parse batch size (integer or "auto")
    if args.batch_size == 'auto':
        batch_size = 'auto'
    else:
        batch_size = int(args.batch_size)

    # Extract class names from COCO annotations
    train_anno = Path(args.dataset_dir) / 'train' / '_annotations.coco.json'
    with open(train_anno, 'r') as f:
        anno = json.load(f)

    categories = sorted(anno['categories'], key=lambda x: x['id'])
    class_names = [cat['name'] for cat in categories]
    category_ids = [int(cat['id']) for cat in categories]
    num_classes = len(class_names)

    print(f"Dataset: {args.dataset_dir}")
    print(f"Classes ({num_classes}): {class_names}")
    print(f"Model: {args.model_variant} (max {VARIANT_MAX_OBJECTS.get(args.model_variant, '?')} objects/image)")
    print(f"Device: {args.device}")
    print(f"Eval every {args.eval_interval} epochs, checkpoint every {args.checkpoint_interval} epochs")

    # Initialize model — don't pass num_classes, let training handle it
    model_class_name = VARIANT_TO_CLASS[args.model_variant]
    model_class = getattr(rfdetr, model_class_name)
    model_kwargs = {}

    if args.resolution is not None:
        model_kwargs['resolution'] = args.resolution

    if args.gradient_checkpointing:
        model_kwargs['gradient_checkpointing'] = True

    model = model_class(**model_kwargs)

    # Build train() kwargs — PTL-compatible (rfdetr >= 1.6.0)
    # Explicitly set single-GPU training to avoid SLURM/DDP auto-detection
    accelerator = {
        'cuda': 'gpu',
        'cpu': 'cpu',
        'mps': 'mps',
    }[args.device]

    train_kwargs = {
        'dataset_dir': args.dataset_dir,
        'output_dir': args.output_dir,
        'epochs': args.epochs,
        'batch_size': batch_size,
        'grad_accum_steps': args.grad_accum_steps,
        'num_workers': args.num_workers,
        'eval_max_dets': args.eval_max_dets,
        'log_per_class_metrics': True,
        'accelerator': accelerator,
        'devices': 1,
        'strategy': 'auto',
        # Performance tuning
        'fp16_eval': bool(args.fp16_eval and args.device == 'cuda' and args.use_amp),
        'pin_memory': args.pin_memory,
        'persistent_workers': args.persistent_workers,
        'prefetch_factor': args.prefetch_factor,
        # Augmentation: AUG_AERIAL gives full D4 symmetry (horizontal + vertical
        # flip + 90° rotation) which matches thin-section / petrography data —
        # rotationally symmetric, no preferred orientation. Plus mild brightness
        # / contrast for the mixed photomicrograph + slide-scanner sources.
        # The naming is for aerial imagery but the math is identical.
        # rfdetr's default is just HorizontalFlip(p=0.5), which leaves symmetries
        # on the table for this domain.
        'aug_config': AUG_AERIAL,
    }

    if args.use_amp:
        if args.device == 'cuda':
            train_kwargs['precision'] = 'bf16-mixed' if args.amp_dtype == 'bf16' else '16-mixed'
        else:
            print(f"AMP requested on {args.device}; ignoring because mixed precision is only enabled for CUDA here.")

    # Only pass if explicitly set (otherwise use rfdetr defaults)
    if args.eval_interval is not None:
        train_kwargs['eval_interval'] = args.eval_interval
    if args.checkpoint_interval is not None:
        train_kwargs['checkpoint_interval'] = args.checkpoint_interval
    if args.learning_rate is not None:
        train_kwargs['lr'] = args.learning_rate

    # Early stopping
    if args.early_stopping_patience is not None:
        train_kwargs['early_stopping'] = True
        train_kwargs['early_stopping_patience'] = args.early_stopping_patience

    # Train
    print(f"\nStarting training: {args.epochs} epochs, batch_size={batch_size}")
    model.train(**train_kwargs)

    # Save metadata used by R to assemble the versioned petrographer manifest.
    # The package intentionally keeps this intermediate file simple and lets R
    # own the stable public schema (`manifest.json` / `training_summary.json`).
    # NOTE: RF-DETR predictions currently surface model-local class indices
    # (0..N-1), not the original COCO category ids. Persist both the display
    # names and the original COCO ids here so R can reconstruct overlays and
    # COCO evaluation targets correctly. If RF-DETR or SAHI changes class-id
    # semantics in a future release, re-verify the mapping in
    # from_pretrained() / evaluate_model_sahi() against a dataset with
    # non-consecutive COCO category ids.
    metadata = {
        'thing_classes': class_names,
        'category_ids': category_ids,
        'model_category_names': {
            str(i): name for i, name in enumerate(class_names)
        },
        'model_to_coco_category_id': {
            str(i): cid for i, cid in enumerate(category_ids)
        },
        'categories': [
            {'model_id': i, 'coco_id': cid, 'name': name}
            for i, (cid, name) in enumerate(zip(category_ids, class_names))
        ],
        'num_classes': num_classes,
        'model_variant': args.model_variant,
        'backend': 'rfdetr',
        'rfdetr_version': getattr(rfdetr, '__version__', None),
        'is_segmentation': args.model_variant.startswith('seg'),
        'max_objects': VARIANT_MAX_OBJECTS.get(args.model_variant, 300),
    }

    if args.resolution is not None:
        metadata['training_resolution'] = args.resolution

    metadata_path = Path(args.output_dir) / 'metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\nMetadata saved to {metadata_path}")


if __name__ == '__main__':
    main()
