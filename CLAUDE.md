# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working
with code in this repository.

## Package Overview

**petrographer** is an R package for automated petrographic thin section
analysis using RF-DETR (instance segmentation) and SAHI (slicing aided
hyper inference). It provides a clean, HuggingFace-like interface for
training models, running inference, and analyzing morphological
properties of detected objects.

**Backend:** - **RF-DETR**: Modern DETR-based transformer detector with
simplified training interface. Supports nano, small, medium, and large
model variants.

**Target Users:** - Primary: Simple HuggingFace-like interface for
researchers running inference with pretrained models - Secondary: Full
training pipeline for developers training custom models (local + HPC)

## Development Commands

### Package Development

``` r
# Load package for development
devtools::load_all()

# Update documentation (always use this, never edit NAMESPACE by hand)
devtools::document()

# Run tests
devtools::test()

# Check package
devtools::check()
```

### Python Dependencies

The package requires Python 3.8+ with: - rfdetr - sahi - torch,
torchvision - opencv-python - scikit-image

Python integration via `reticulate` - the R package manages this
automatically.

## Code Architecture

### Design Philosophy (CRITICAL)

**Simplicity Over Abstraction:** - Fail fast with clear errors - no
defensive try-catch unless truly needed - Delete unused code
aggressively - no “future-proofing” - Single entry point for model
loading:
[`from_pretrained()`](https://flmnh-ai.github.io/petrographer/reference/from_pretrained.md)
(smart: checks local first, then hub) - Minimal dependencies, focused
functionality - Breaking changes OK - this is research code

**Recent Refactorings:**

*2025-10: RF-DETR Migration* - Removed Detectron2 backend entirely
(~1,300 lines of code removed, 30-35% reduction) - Simplified to RF-DETR
only - no more backend abstraction - Removed complex LR scaling, freeze
stages, backbone configuration - Much simpler training interface

*2025-01: General Simplification* - Removed complex pins
catalog/versioning infrastructure - Removed compare_models,
diagnose_annotations wrappers - Streamlined model loading to 2-3 clear
paths - Unified predict() interface via S3 methods

**DO NOT add back removed complexity** without explicit approval.

### Core Workflow

    User Workflow:
    1. from_pretrained("model_id") → PetrographyModel (checks local first, then hub)
    2. predict(model, "image.jpg") → tibble with detections + morphology + class names
    3. Analysis functions (summarize_by_image, get_population_stats)

    Developer Workflow:
    1. validate_dataset("data_dir") → check COCO format
    2. train_model(dataset_id, model_id, model_variant = "nano", ...) → local or HPC training (auto-pins to .petrographer/, auto-infers num_classes from COCO)
    3. from_pretrained("model_id") → load trained model (smart: checks local first)
    4. [Optional] pin_model(..., hub_board) → publish to public hub (maintainers only)

### File Organization

**R/** - All R functions (one file per concern): - `pins.R` - Core pins
integration (versioning, caching, distribution) -
`from_pretrained(model_id, board = NULL)` - Smart loading (checks local
first, then hub) - `pin_model(model_dir, model_id, board)` - Pin model
to board - `list_models(board)`, `model_info(model_id, board)` - Query
boards - `.get_local_board()` - Internal: returns local training board

- `model.R` - Model utilities
  - [`list_trained_models()`](https://flmnh-ai.github.io/petrographer/reference/list_trained_models.md) -
    Convenience wrapper for `list_models(board = "local")`
- `prediction.R` - Inference and evaluation
  - [`predict()`](https://rdrr.io/r/stats/predict.html) - S3 generic for
    PetrographyModel
  - [`predict_image()`](https://flmnh-ai.github.io/petrographer/reference/predict_image.md) -
    Single image inference with SAHI + morphology
  - [`predict_images()`](https://flmnh-ai.github.io/petrographer/reference/predict_images.md) -
    Batch processing
  - [`evaluate_model_sahi()`](https://flmnh-ai.github.io/petrographer/reference/evaluate_model_sahi.md) -
    COCO evaluation metrics
  - [`evaluate_training()`](https://flmnh-ai.github.io/petrographer/reference/evaluate_training.md) -
    Parse training metrics (metrics.csv for PTL \>= 1.6.0, log.txt for
    native loop)
- `training.R` - Training orchestration (LOCAL + HPC)
  - [`train_model()`](https://flmnh-ai.github.io/petrographer/reference/train_model.md) -
    Unified interface, handles local/HPC dispatch
  - Simple parameter passing to RF-DETR (no complex LR/freeze logic)
  - Auto-versioning for model names
  - Manifest creation with metadata
- `dataset.R` - Dataset utilities
  - [`validate_dataset()`](https://flmnh-ai.github.io/petrographer/reference/validate_dataset.md) -
    COCO format validation + diagnostics
  - [`slice_dataset()`](https://flmnh-ai.github.io/petrographer/reference/slice_dataset.md) -
    SAHI dataset slicing for varying image sizes
- `morphology.R` - Extract properties from SAHI results via scikit-image
- `summary.R` - Aggregation functions (by image, population stats)
- `metrics.R` - Parse training metrics (metrics.csv / log.txt)
- `utils.R` - S3 print methods, small helpers
- `visualization.R` - Plot COCO annotations via Roboflow supervision
  (inst/python/visualize.py)

**inst/python/train.py** - RF-DETR training script - Called by
training.R via processx - Simple interface: imports rfdetr and calls
model.train() with metrics=True - Extracts class names from COCO JSON -
RF-DETR automatically saves checkpoints: - `checkpoint_best_total.pth` -
Best checkpoint by total loss (primary, loaded by petrographer) -
`checkpoint_best_regular.pth` - Best checkpoint (non-EMA) -
`checkpoint_best_ema.pth` - Best checkpoint with EMA weights (if EMA
enabled) - `checkpoint.pth` - Latest checkpoint for resuming -
`metrics.csv` + `hparams.yaml` - PyTorch Lightning training metrics
(RF-DETR \>= 1.6.0) - `log.txt` - Native-loop training metrics (RF-DETR
\< 1.6.0) - Creates `metadata.json` with class names and training
config - Supports all RF-DETR model variants (nano, small, medium,
large, xlarge, 2xlarge, preview)

**inst/python/slice_dataset.py** - SAHI dataset slicing utility

### Key Abstractions

**PetrographyModel Object:**

``` r
structure(list(
  sahi_model = <SAHI AutoDetectionModel>,  # SAHI wrapper for inference
  rfdetr_model = <RF-DETR model>,          # Underlying RF-DETR model
  model_path = "path/to/checkpoint_best_total.pth", # Weights path
  model_variant = "nano",                   # RF-DETR variant (nano/small/medium/large)
  confidence = 0.5,                         # Detection threshold
  device = "cpu",                           # cpu/cuda/mps
  manifest = list(...)                      # Model metadata
), class = "PetrographyModel")
```

[`from_pretrained()`](https://flmnh-ai.github.io/petrographer/reference/from_pretrained.md)
returns this structure after loading the RF-DETR model. The `sahi_model`
includes `category_mapping` loaded from metadata.json, enabling class
name predictions. The wrapper is needed because
[`predict_images()`](https://flmnh-ai.github.io/petrographer/reference/predict_images.md)
requires access to model paths for SAHI’s batch function.

**Training Config:**
[`train_model()`](https://flmnh-ai.github.io/petrographer/reference/train_model.md)
validates parameters and builds config, then calls Python training
script. Much simpler than before - just passes parameters directly to
RF-DETR. Auto-calculates `grad_accum_steps` to maintain effective batch
size of 16: `batch_size × grad_accum_steps = 16`.

**HPC Integration:** Uses `hipergator` package for SLURM job submission
with rsync for efficient file transfer.

**Shared Directory Structure** (efficient, avoids re-uploading):

    /blue/base_dir/
      datasets/{dataset_id}/    # Shared across all models (rsync skips if unchanged)
      scripts/train.py          # Shared RF-DETR training script (rsync skips if unchanged)
      models/{model_id}/{run_id}/output/  # Versioned training runs

**Workflow**: Upload dataset/scripts → submit job → wait → download
files → pin to local board.

**Key Points:** - `run_id` (timestamp) ensures multiple training runs
don’t conflict - rsync only uploads changed files (datasets and train.py
reused across runs) - SLURM `working_dir = base_dir` for clean relative
paths - **Downloaded files:** - **Required:**
`checkpoint_best_total.pth`, `metadata.json` - **Optional:**
`metrics.csv` + `hparams.yaml` (PTL \>= 1.6.0), `log.txt` (native loop),
`metrics_plot.png`, `results.json` - RF-DETR creates additional
checkpoints on HPC (not downloaded): `checkpoint.pth`,
`checkpoint_best_regular.pth`, `checkpoint_best_ema.pth`

### Pins Integration (Core Infrastructure)

**Purpose:** Pins is core to the package - handles versioning, caching,
and distribution.

**Architecture (Transparent to Users):** - **Public Hub:**
pkgdown-served board at
`https://flmnh-ai.github.io/petrographer/pins/` - Consumed via
`board_url()`, served from `pkgdown/assets/pins/` in repo - Updated by
maintainers, auto-deployed via GitHub Pages - **Local Dataset Board:**
`.petrographer/datasets/` in project root (auto-created) - Where dataset
pins are stored - Users reference with `dataset_id` in
[`train_model()`](https://flmnh-ai.github.io/petrographer/reference/train_model.md) -
**Local Model Board:** `.petrographer/models/` in project root
(auto-created) - Where
[`train_model()`](https://flmnh-ai.github.io/petrographer/reference/train_model.md)
automatically pins trained models - Users load with
`from_pretrained(model_id, board = "local")` - **Custom Boards:**
Advanced users can specify `board_folder("path")` for either datasets or
models

**Board Separation:** Datasets and models use separate boards to allow
identical names without collision:

``` r
pin_dataset(data_dir, dataset_id = "inclusions_shell")
train_model(
  dataset_id = "inclusions_shell",
  model_id = "inclusions_shell",  # Same name OK - different boards!
  ...
)
```

**Key Functions:** - `from_pretrained(model_id, board = NULL)` - Load
model (NULL = hub, “local” = model board) -
`pin_model(model_dir, model_id, board)` - Pin model to board -
`pin_dataset(data_dir, dataset_id, board)` - Pin dataset to board -
`list_models(board)` /
[`list_trained_models()`](https://flmnh-ai.github.io/petrographer/reference/list_trained_models.md) -
List available models - `list_datasets(board)` - List available
datasets - `model_info(model_id, board)` - Show model metadata -
`get_training_dataset(model_id, board)` - Retrieve the exact dataset
version used to train a model

**Internal Helpers:** - `.get_dataset_board()` - Returns local dataset
board (not exported) - `.get_model_board()` - Returns local model board
(not exported)

**Dataset Version Tracking (Reproducibility):** Models automatically
track the exact dataset version used for training. This ensures
reproducibility and allows you to retrieve the training data later:

``` r
# Train a model (dataset version and num_classes automatically captured)
train_model(dataset_id = "my_dataset", model_variant = "nano", ...)

# Later, retrieve the exact dataset version used for training
dataset_path <- get_training_dataset("my_dataset")

# Use it to retrain or analyze
train_model(data_dir = dataset_path, model_id = "my_dataset_v2", ...)
```

The dataset version ID (e.g., `"20251013T143943Z-8329a"`) is stored in
model metadata and used to retrieve the correct pins version. This works
even if the dataset has been updated since training.

**Maintainer Workflow (Publishing to Hub):** 1. Pin model:
`pin_model(model_dir, model_id, hub_board)` 2. Update manifest:
`pins::write_board_manifest(hub_board)` 3. Rebuild pkgdown:
[`pkgdown::build_site()`](https://pkgdown.r-lib.org/reference/build_site.html)
4. Commit & push to GitHub 5. Model appears on hub after deployment

**Refactored (2025-10):** Separated datasets and models into different
boards to prevent namespace collisions and allow same names for related
datasets/models. Training auto-pins to local boards.

## Coding Conventions

### R Style

- Use native pipe `|>` (not magrittr `%>%`)
- Modern tidyverse: `dplyr`, `purrr`, `fs`, `cli`, `glue`
- Always use [`fs::path()`](https://fs.r-lib.org/reference/path.html)
  for path construction
- Use `cli::cli_*()` for user feedback (not
  [`message()`](https://rdrr.io/r/base/message.html) or
  [`cat()`](https://rdrr.io/r/base/cat.html))
- Roxygen2 for all documentation - **never edit NAMESPACE by hand**

### Function Design

- Fail fast: validate inputs early, abort with clear messages
- Return consistent types (tibbles for data, lists for complex objects)
- Use S3 classes sparingly (PetrographyModel, sahi_evaluation,
  training_evaluation)
- Export only user-facing functions - keep internals private

### Python Integration

- All Python called via
  [`reticulate::import()`](https://rstudio.github.io/reticulate/reference/import.html)
  or [`processx::run()`](http://processx.r-lib.org/reference/run.md)
- Store imported modules in package environment (see zzz.R)
- `sahi` and `skimage` are global:
  `sahi$predict$get_sliced_prediction(...)`

### Error Messages

Use [`cli::cli_abort()`](https://cli.r-lib.org/reference/cli_abort.html)
with helpful context:

``` r
if (!fs::file_exists(model_path)) {
  cli::cli_abort("Model weights not found at {.path {model_path}}")
}
```

## Common Patterns

### Loading Models

``` r
# From public hub (downloads + caches)
model <- from_pretrained("shell_v3", device = "cpu", confidence = 0.5)

# From local training board
model <- from_pretrained("my_model", board = "local", device = "cuda")

# From custom board
my_board <- pins::board_folder("~/shared-models", versioned = TRUE)
model <- from_pretrained("my_model", board = my_board)

# All return identical PetrographyModel objects
```

### Running Predictions

``` r
# Single image (saves visualization by default)
results <- predict(model, "image.jpg")

# Single image (no viz, custom settings)
results <- predict_image("image.jpg", model,
                         use_slicing = TRUE,
                         slice_size = 512,
                         save_visualizations = FALSE)

# Batch processing
results <- predict_images("input_dir/", model, output_dir = "results/")
```

### Training Models

``` r
# Local training (auto-pins to .petrographer/)
# num_classes auto-inferred from COCO annotations
# grad_accum_steps auto-calculated: batch_size × grad_accum_steps = 16
train_model(
  dataset_id = "my_dataset",
  model_id = "my_model",
  model_variant = "nano",  # or "small", "medium", "large"
  epochs = 10,
  batch_size = 2,  # grad_accum_steps will be 8 (effective batch = 16)
  device = "mps"  # or "cuda", "cpu"
)

# HPC training with custom gradient accumulation
train_model(
  dataset_id = "my_dataset",
  model_id = "my_model",
  model_variant = "small",
  epochs = 20,
  batch_size = 4,  # grad_accum_steps will be 4 (effective batch = 16)
  grad_accum_steps = 8  # Override auto-calculation if needed
)

# Load trained model
model <- from_pretrained("my_model", board = "local")
```

### Publishing Models (Maintainers Only)

``` r
# Create hub board
hub_board <- pins::board_folder(here::here("pkgdown/assets/pins"), versioned = TRUE)

# Pin model to hub
pin_model(
  model_dir = ".petrographer/models/my_model/output",
  model_id = "my_model",
  board = hub_board,
  metadata = list(description = "Shell detector v3")
)

# Update manifest
pins::write_board_manifest(hub_board)

# Then: pkgdown::build_site(), git commit, git push
# See data-raw/publish-to-hub.R for complete workflow
```

## Testing

Tests in `tests/testthat/`: - `test-metrics.R` - Metrics parsing -
`test-training.R` - Training config validation

Run with `devtools::test()` or
`testthat::test_file("tests/testthat/test-*.R")`.

**Testing philosophy:** Test core logic, not wrappers. Most functions
expect real data/models.

## Important Notes

### When Adding Features

**Before adding complexity, ask:** 1. Is this actually needed, or
nice-to-have? 2. Can existing functions handle this with minor tweaks?
3. Will this be maintained, or become technical debt?

**Preference hierarchy:** 1. Use existing functions differently 2. Add a
simple parameter to existing function 3. Create new function (only if
truly distinct use case)

### Removed Features (Do Not Re-add)

**Removed in 2025-10 (RF-DETR migration):** - Detectron2 backend
entirely (~1,300 lines) - Complex LR scaling based on batch size and
freeze_at - Backbone configuration (ResNet50/101, ResNeXt) - Freeze
stages for transfer learning - Differential learning rates (0.1x
backbone, 1.0x head) - All Detectron2-specific training logic

**Removed in 2025-01 (general simplification):** - `compare_models()` -
model comparison plots - `diagnose_annotations()` - wrapper around
annotation_diagnostics - `summarize_dataset()` - redundant with
validate_dataset - `pg_dataset_*` - dataset publishing infrastructure -
`pg_install_*` - installation utilities - Complex pins
catalog/versioning

**Rationale:** Not core to the 80/20 use case. RF-DETR provides simpler,
modern alternative. Can add back if genuine need emerges.

### Environment Variables

Optional configuration via `.Renviron`: - `PETROGRAPHER_HUB_URL` -
Custom model hub URL - `PETROGRAPHER_BOARD_PATH` - Custom pins board
location - `PETROGRAPHER_HPC_HOST` - Default HPC hostname -
`PETROGRAPHER_HPC_BASE_DIR` - Default HPC working directory

## External Dependencies

### R Packages

- **Core:** `reticulate`, `processx` (R-Python bridge)
- **Tidyverse:** `dplyr`, `purrr`, `tibble`, `readr`, `stringr`
- **System:** `fs`, `cli`, `glue`, `jsonlite`
- **Optional:** `pins` (model sharing), `hipergator` (HPC), `png` +
  `grid` (in-chunk image display)

### Python Packages

- **Deep Learning:** `torch`, `rfdetr`
- **Detection:** `sahi` (sliced inference)
- **Visualization:** `supervision` (annotation + prediction overlays,
  via `inst/python/visualize.py`)
- **Processing:** `opencv-python`, `scikit-image`, `pycocotools`

## Resources

- **Package docs:** <https://flmnh-ai.github.io/petrographer/>
- **RF-DETR:** <https://github.com/om-ai-lab/RF-DETR>
- **SAHI:** <https://github.com/obss/sahi>
- **Pins:** <https://pins.rstudio.com/>
