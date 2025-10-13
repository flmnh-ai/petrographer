# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Package Overview

**petrographer** is an R package for automated petrographic thin section analysis using Detectron2 (instance segmentation) and SAHI (slicing aided hyper inference). It provides a clean interface for training models, running inference, and analyzing morphological properties of detected objects.

**Target Users:**
- Primary: Simple HuggingFace-like interface for researchers running inference with pretrained models
- Secondary: Full training pipeline for developers training custom models (local + HPC)

## Development Commands

### Package Development
```r
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
The package requires Python 3.8+ with:
- detectron2
- sahi
- torch, torchvision
- opencv-python
- scikit-image

Python integration via `reticulate` - the R package manages this automatically.

## Code Architecture

### Design Philosophy (CRITICAL)

**Simplicity Over Abstraction:**
- Fail fast with clear errors - no defensive try-catch unless truly needed
- Delete unused code aggressively - no "future-proofing"
- Prefer explicit over implicit (e.g., separate `from_pretrained()` vs `load_model()`)
- Minimal dependencies, focused functionality
- Breaking changes OK - this is research code

**Recent Refactoring (2025-01):**
The package was dramatically simplified, removing ~40% of code:
- Removed complex pins catalog/versioning infrastructure
- Removed compare_models, diagnose_annotations wrappers
- Streamlined model loading to 2-3 clear paths
- Unified predict() interface via S3 methods

**DO NOT add back removed complexity** without explicit approval.

### Core Workflow

```
User Workflow:
1. from_pretrained("model_id") OR load_model("local/dir") → PetrographyModel
2. predict(model, "image.jpg") → tibble with detections + morphology
3. Analysis functions (summarize_by_image, get_population_stats)

Developer Workflow:
1. validate_dataset("data_dir") → check COCO format
2. train_model(...) → local or HPC training
3. publish_model(...) → optional pins publishing
```

### File Organization

**R/** - All R functions (one file per concern):
- `pins.R` - Model hub integration (download/publish via pins package)
  - `from_pretrained()` - Load model from hub, returns PetrographyModel
  - `publish_model()` - Publish trained model to pins board
  - `board_user()` - Create writable board for publishing
  - Internal helpers: `pg_coalesce`, `pg_model_iso_time`, etc.

- `model.R` - Local model loading
  - `load_model()` - Load from directory, returns PetrographyModel
  - `list_trained_models()` - Scan Detectron2_Models/ directory

- `prediction.R` - Inference and evaluation
  - `predict()` - S3 generic for PetrographyModel
  - `predict_image()` - Single image inference with SAHI + morphology
  - `predict_images()` - Batch processing
  - `evaluate_model_sahi()` - COCO evaluation metrics
  - `evaluate_training()` - Parse training metrics.json

- `training.R` - Training orchestration (LOCAL + HPC)
  - `train_model()` - Unified interface, handles local/HPC dispatch
  - Smart LR scaling based on batch size + freeze_at
  - Auto-versioning for model names
  - Manifest creation with metadata
  - **DO NOT MODIFY without approval** - this was heavily tuned

- `dataset.R` - Dataset utilities
  - `validate_dataset()` - COCO format validation + diagnostics
  - `slice_dataset()` - SAHI dataset slicing for varying image sizes

- `morphology.R` - Extract properties from SAHI results via scikit-image
- `summary.R` - Aggregation functions (by image, population stats)
- `metrics.R` - Parse Detectron2 metrics.json
- `utils.R` - S3 print methods, small helpers
- `visualization.R` - Plot COCO annotations with magick

**inst/python/train.py** - Detectron2 training script
- Called by training.R via processx
- Handles dataset registration, augmentations, training loop
- Uses WarmupCosineLR schedule, differential LR (0.1x backbone, 1.0x head)
- BestCheckpointer saves model_best.pth based on segm/AP
- **Stable - don't modify unless training improvements needed**

**inst/python/slice_dataset.py** - SAHI dataset slicing utility

### Key Abstractions

**PetrographyModel Object:**
```r
structure(list(
  sahi_model = <SAHI AutoDetectionModel>,  # The actual detector
  model_path = "path/to/weights.pth",      # Needed for predict_images
  config_path = "path/to/config.yaml",     # Needed for predict_images
  confidence = 0.5,                         # Threshold
  device = "cpu",                           # cpu/cuda/mps
  manifest = NULL or list()                 # Optional metadata
), class = "PetrographyModel")
```

Both `from_pretrained()` and `load_model()` return this structure.
The wrapper is necessary because `predict_images()` needs paths to call SAHI's batch function.

**Training Config:**
`train_model()` has extensive parameter validation and config building.
R computes batch sizes, learning rates, workers, then calls Python script.
The display/config logic is verbose but **do not simplify** - it was heavily refined.

**HPC Integration:**
Uses `hipergator` package for SLURM job submission.
Workflow: upload data → submit job → wait → download results.
Auto-versioning checks both local and remote directories.

### Pins Integration

**Purpose:** HuggingFace-like model sharing via `pins` package (optional, not required).

**Architecture:**
- Public hub: `board_url()` pointing to hosted Dropbox _pins.yaml
- User board: `board_folder()` in ~/Dropbox/petrographer-pins (writable)
- Models stored as pin bundles with manifest.json metadata

**Key Functions:**
- `from_pretrained(model_id)` - Download from hub, cache locally
- `publish_model(model_dir, model_id, board)` - Upload to board
- `list_models(board)` - List available models
- `model_info(model_id, board)` - Show metadata

**Simplified (2025-01):** Removed complex catalog, dataset publishing, install infrastructure.
Keep minimal hub functionality only.

## Coding Conventions

### R Style
- Use native pipe `|>` (not magrittr `%>%`)
- Modern tidyverse: `dplyr`, `purrr`, `fs`, `cli`, `glue`
- Always use `fs::path()` for path construction
- Use `cli::cli_*()` for user feedback (not `message()` or `cat()`)
- Roxygen2 for all documentation - **never edit NAMESPACE by hand**

### Function Design
- Fail fast: validate inputs early, abort with clear messages
- Return consistent types (tibbles for data, lists for complex objects)
- Use S3 classes sparingly (PetrographyModel, sahi_evaluation, training_evaluation)
- Export only user-facing functions - keep internals private

### Python Integration
- All Python called via `reticulate::import()` or `processx::run()`
- Store imported modules in package environment (see zzz.R)
- `sahi` and `skimage` are global: `sahi$predict$get_sliced_prediction(...)`

### Error Messages
Use `cli::cli_abort()` with helpful context:
```r
if (!fs::file_exists(model_path)) {
  cli::cli_abort("Model weights not found at {.path {model_path}}")
}
```

## Common Patterns

### Loading Models
```r
# From hub (downloads + caches)
model <- from_pretrained("shell_v3", device = "cpu", confidence = 0.5)

# From local directory
model <- load_model("Detectron2_Models/my_model", device = "cuda")

# Both return identical PetrographyModel objects
```

### Running Predictions
```r
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
```r
# Local training
train_model(
  data_dir = "data/processed/dataset",
  output_name = "my_model",
  num_classes = 5,
  max_iter = 12000,
  device = "cuda"
)

# HPC training (reads PETROGRAPHER_HPC_HOST, PETROGRAPHER_HPC_BASE_DIR from .Renviron)
train_model(
  data_dir = "data/processed/dataset",
  output_name = "my_model",
  num_classes = 5,
  hpc_user = "username"
)
```

### Publishing Models
```r
# After training
publish_model(
  model_dir = "Detectron2_Models/my_model",
  model_id = "my_model",
  board = board_user(),
  metadata = list(description = "Shell detector v3")
)

# Or auto-publish during training
train_model(..., publish_after_train = TRUE, model_board = board_user())
```

## Testing

Tests in `tests/testthat/`:
- `test-metrics.R` - Metrics parsing
- `test-training.R` - Training config validation

Run with `devtools::test()` or `testthat::test_file("tests/testthat/test-*.R")`.

**Testing philosophy:** Test core logic, not wrappers. Most functions expect real data/models.

## Important Notes

### When Adding Features

**Before adding complexity, ask:**
1. Is this actually needed, or nice-to-have?
2. Can existing functions handle this with minor tweaks?
3. Will this be maintained, or become technical debt?

**Preference hierarchy:**
1. Use existing functions differently
2. Add a simple parameter to existing function
3. Create new function (only if truly distinct use case)

### Training.R is Sacred

The training pipeline (`training.R`, `train.py`) was heavily refined through iteration:
- Smart LR scaling based on batch size and freeze_at
- Differential learning rates (0.1x backbone, 1.0x head)
- Auto-versioning with local + remote checks
- Extensive config validation and display

**Do not refactor without explicit approval.**

### Removed Features (Do Not Re-add)

The following were removed in the 2025-01 simplification:
- `compare_models()` - model comparison plots
- `diagnose_annotations()` - wrapper around annotation_diagnostics
- `summarize_dataset()` - redundant with validate_dataset
- `pg_dataset_*` - dataset publishing infrastructure
- `pg_install_*` - installation utilities
- Complex pins catalog/versioning

**Rationale:** Not core to the 80/20 use case. Can add back if genuine need emerges.

### Environment Variables

Optional configuration via `.Renviron`:
- `PETROGRAPHER_HUB_URL` - Custom model hub URL
- `PETROGRAPHER_BOARD_PATH` - Custom pins board location
- `PETROGRAPHER_HPC_HOST` - Default HPC hostname
- `PETROGRAPHER_HPC_BASE_DIR` - Default HPC working directory

## External Dependencies

### R Packages
- **Core:** `reticulate`, `processx` (R-Python bridge)
- **Tidyverse:** `dplyr`, `purrr`, `tibble`, `readr`, `stringr`
- **System:** `fs`, `cli`, `glue`, `jsonlite`
- **Optional:** `pins` (model sharing), `hipergator` (HPC), `magick` (visualization)

### Python Packages
- **Deep Learning:** `torch`, `detectron2`
- **Detection:** `sahi` (sliced inference)
- **Processing:** `opencv-python`, `scikit-image`, `pycocotools`

## Resources

- **Package docs:** https://flmnh-ai.github.io/petrographer/
- **Detectron2:** https://github.com/facebookresearch/detectron2
- **SAHI:** https://github.com/obss/sahi
- **Pins:** https://pins.rstudio.com/
