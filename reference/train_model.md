# Train a new petrography detection model

Orchestrates local or HPC training using RF-DETR. Models are
automatically pinned to the local board (.petrographer/) for versioning.

## Usage

``` r
train_model(
  dataset_id = NULL,
  data_dir = NULL,
  model_id = NULL,
  model_variant = "nano",
  resolution = NULL,
  epochs = 10,
  batch_size = NA,
  grad_accum_steps = NA,
  learning_rate = NULL,
  device = "cuda",
  use_amp = NULL,
  amp_dtype = "bf16",
  gradient_checkpointing = NULL,
  num_workers = NULL,
  time_hours = 4,
  validate_every = 2L,
  early_stopping_patience = NULL
)
```

## Arguments

- dataset_id:

  Name of pinned dataset to use for training (preferred).

- data_dir:

  Path to dataset directory (alternative to dataset_id; will be
  auto-pinned with temp ID).

- model_id:

  Name for the trained model (used for pins). Defaults to `dataset_id`
  if not provided.

- model_variant:

  RF-DETR model variant. Detection: "nano" (default), "small", "medium",
  "large". Segmentation: "seg_nano", "seg_small", "seg_medium",
  "seg_large", "seg_xlarge", "seg_2xlarge". Legacy: "seg_preview".

- resolution:

  Image resolution for training. Auto-detected from variant if not
  specified.

- epochs:

  Number of training epochs. Default: 10.

- batch_size:

  Batch size for training. If NA (default), uses 2.

- grad_accum_steps:

  Gradient accumulation steps. If NA (default), auto-calculated as 16 /
  batch_size for effective batch size of 16.

- learning_rate:

  Learning rate. If NULL (default), uses model default.

- device:

  Device for local training: 'cpu', 'cuda', or 'mps' (default: 'cuda').

- use_amp:

  Use automatic mixed precision training (default: TRUE for CUDA, FALSE
  otherwise). Reduces memory usage by ~40%.

- amp_dtype:

  AMP dtype: 'bf16' (recommended for modern GPUs) or 'fp16' (default:
  'bf16').

- gradient_checkpointing:

  Enable gradient checkpointing (default: FALSE). Reduces memory usage
  by ~30%.

- num_workers:

  Number of data loading workers (default: 8).

- time_hours:

  Time limit for HPC training in hours (default: 3). Examples: 4 = 4
  hours, 0.5 = 30 minutes, 1.5 = 1.5 hours. Ignored for local training.

- validate_every:

  Validate every N epochs (default: 1). Set to NULL to use model
  default.

- early_stopping_patience:

  Stop training if validation loss doesn't improve for N epochs
  (default: 10). Set to NULL to disable early stopping.

## Value

Model ID (can be loaded with `from_pretrained(model_id)`).

## Details

Training mode (local vs HPC) is auto-detected based on `hipergator`
configuration. For HPC training, call
[`hipergator::hpg_configure()`](https://rdrr.io/pkg/hipergator/man/hpg_configure.html)
before `train_model()` to set connection details (host, user, base_dir).
