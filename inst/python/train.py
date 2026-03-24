#!/usr/bin/env python3
"""
RF-DETR Training Script for Petrographer
Simplified training interface using RF-DETR's built-in training loop.
"""

import argparse
import json
import os
from pathlib import Path
import rfdetr

# Remove distributed environment variables
# SLURM sets these automatically, which triggers PyTorch DDP mode
for key in ['RANK', 'WORLD_SIZE', 'LOCAL_RANK', 'SLURM_PROCID', 'SLURM_LOCALID', 'SLURM_NTASKS']:
    os.environ.pop(key, None)


def main():
    parser = argparse.ArgumentParser(description='Train RF-DETR model for petrography')

    # Dataset
    parser.add_argument('--dataset-dir', type=str, required=True,
                        help='Path to dataset directory (contains train/ and valid/ subdirs)')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory for trained model')

    # Model
    parser.add_argument('--model-variant', type=str, default='nano',
                        choices=['nano', 'small', 'medium', 'large',
                                 'seg_nano', 'seg_small', 'seg_medium', 'seg_large',
                                 'seg_xlarge', 'seg_2xlarge', 'seg_preview'],
                        help='RF-DETR model variant')
    parser.add_argument('--resolution', type=int, default=None,
                        help='Image resolution for training (auto-detected from variant if not specified)')

    # Training parameters
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=2,
                        help='Batch size')
    parser.add_argument('--grad-accum-steps', type=int, default=8,
                        help='Gradient accumulation steps')
    parser.add_argument('--learning-rate', type=float, default=None,
                        help='Learning rate (uses model default if not specified)')

    # Memory optimization
    parser.add_argument('--use-amp', action='store_true', default=False,
                        help='Use automatic mixed precision training (reduces memory usage)')
    parser.add_argument('--amp-dtype', type=str, default='bf16',
                        choices=['bf16', 'fp16'],
                        help='AMP dtype (bf16 recommended for modern GPUs)')
    parser.add_argument('--gradient-checkpointing', action='store_true', default=False,
                        help='Enable gradient checkpointing (reduces memory usage)')

    # System
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cpu', 'cuda', 'mps'],
                        help='Device for training')
    parser.add_argument('--num-workers', type=int, default=8,
                        help='Number of data loading workers (default: 8)')

    # Validation
    parser.add_argument('--validate-every', type=int, default=None,
                        help='Validate every N epochs (default: 1 from R, or model default if not specified)')
    parser.add_argument('--early-stopping-patience', type=int, default=None,
                        help='Stop training if validation loss does not improve for N epochs (default: 10 from R, or disabled if not specified)')

    args = parser.parse_args()

    # Extract class names from COCO annotations
    train_anno = Path(args.dataset_dir) / 'train' / '_annotations.coco.json'
    with open(train_anno, 'r') as f:
        anno = json.load(f)

    categories = sorted(anno['categories'], key=lambda x: x['id'])
    class_names = [cat['name'] for cat in categories]

    # Initialize model
    variant_to_class = {
        'nano': 'RFDETRNano', 'small': 'RFDETRSmall',
        'medium': 'RFDETRMedium', 'large': 'RFDETRLarge',
        'seg_preview': 'RFDETRSegPreview',
        'seg_nano': 'RFDETRSegNano', 'seg_small': 'RFDETRSegSmall',
        'seg_medium': 'RFDETRSegMedium', 'seg_large': 'RFDETRSegLarge',
        'seg_xlarge': 'RFDETRSegXLarge', 'seg_2xlarge': 'RFDETRSeg2XLarge',
    }

    model_class_name = variant_to_class[args.model_variant]
    model_class = getattr(rfdetr, model_class_name)
    model_kwargs = {'device': args.device}

    if args.resolution is not None:
        model_kwargs['resolution'] = args.resolution

    if args.learning_rate is not None:
        model_kwargs['learning_rate'] = args.learning_rate

    model = model_class(**model_kwargs)

    # Memory diagnostics (if CUDA available)
    if args.device == 'cuda':
        import torch
        print("\n=== GPU Memory Before Training ===")
        print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
        print(f"Max allocated: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
        print("=" * 40)

    # Build train() kwargs
    train_kwargs = {
        'dataset_dir': args.dataset_dir,
        'output_dir': args.output_dir,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'grad_accum_steps': args.grad_accum_steps,
        'num_workers': args.num_workers,
        'tensorboard': False,
        'wandb': False
    }

    # Add validation settings
    if args.validate_every is not None:
        train_kwargs['validate_every'] = args.validate_every

    # Add early stopping
    if args.early_stopping_patience is not None:
        train_kwargs['early_stopping'] = True
        train_kwargs['early_stopping_patience'] = args.early_stopping_patience
    else:
        train_kwargs['early_stopping'] = False

    # Add memory optimization parameters
    if args.use_amp:
        train_kwargs['use_amp'] = True
        train_kwargs['amp_dtype'] = args.amp_dtype
        print(f"Using AMP with dtype={args.amp_dtype}")

    if args.gradient_checkpointing:
        train_kwargs['gradient_checkpointing'] = True
        print("Using gradient checkpointing")

    # Train
    model.train(**train_kwargs)

    # Memory diagnostics after training (if CUDA available)
    if args.device == 'cuda':
        import torch
        print("\n=== GPU Memory After Training ===")
        print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
        print(f"Max allocated: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
        print("=" * 40)

    # Save metadata
    metadata = {
        'thing_classes': class_names,
        'num_classes': len(class_names),
        'model_variant': args.model_variant,
        'backend': 'rfdetr',
        'is_segmentation': args.model_variant.startswith('seg')
    }

    # Add resolution if specified
    if args.resolution is not None:
        metadata['training_resolution'] = args.resolution

    metadata_path = Path(args.output_dir) / 'metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)


if __name__ == '__main__':
    main()
