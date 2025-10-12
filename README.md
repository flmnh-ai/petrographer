# petrographer: Petrographic Image Analysis with Detectron2 and SAHI

Automated instance segmentation and morphological analysis of petrographic thin sections using state-of-the-art computer vision models.

## Overview

This R package provides a complete workflow for training, evaluating, and analyzing petrographic thin section images using Detectron2 with SAHI (Slicing Aided Hyper Inference) for improved detection of small objects. The workflow combines Python-based machine learning with R-based analysis and visualization through a clean, modern interface.

Quick start (local)

```
library(petrographer)

# Validate dataset
validate_dataset("data/processed/shell_mixed")

# Train locally
model_dir <- train_model(
  data_dir = "data/processed/shell_mixed",
  output_name = "shell_detector_v3",
  num_classes = 5,
  device = "cpu" # or "cuda"/"mps"
)

evaluate_training(model_dir)
```

Quick start (HPC)

**Setup**: First configure your HPC defaults in `.Renviron`:

```r
# Add to your .Renviron file (edit with usethis::edit_r_environ())
usethis::edit_r_environ("project")

# Add these lines:
PETROGRAPHER_HPC_HOST="hpg"
PETROGRAPHER_HPC_BASE_DIR="/blue/your_lab/your_user"
```

Then restart R and train:

```r
library(petrographer)

# HPC settings are read from environment variables
model_dir <- train_model(
  data_dir = "data/processed/shell_mixed",
  output_name = "shell_detector_v3",
  num_classes = 5,
  hpc_user = "your_user"  # optional if different from system user
)
```

**Alternative**: Override environment variables if needed:

```r
model_dir <- train_model(
  data_dir = "data/processed/shell_mixed", 
  output_name = "shell_detector_v3",
  num_classes = 5,
  hpc_host = "different.cluster.edu",
  hpc_base_dir = "/different/path"
)
```

Safety

- No remote deletion is performed automatically.

### Key Features

- **Advanced Instance Segmentation**: Uses Detectron2 Mask R-CNN with SAHI for high-quality detection
- **Morphological Analysis**: Comprehensive particle characterization (size, shape, orientation, etc.)
- **Unified Training Interface**: Local and HPC training through R functions with automatic job management
- **Interactive Analysis**: Modern R workflow using `cli`, `fs`, and `glue` for professional output
- **Batch Processing**: Efficient processing of large image collections
- **Model Management**: Automatic model download and caching system

## Installation and Setup

### Prerequisites

- **Python 3.8+** with detectron2, SAHI, and dependencies
- **R 4.0+** with required packages
- **CUDA-capable GPU** (recommended for training)

### R Dependencies

```r
install.packages(c("reticulate", "tidyverse", "magick", "scico", 
                   "patchwork", "glue", "cli", "fs", "future"))
```

### Python Dependencies

The package automatically manages Python dependencies. Required packages:
- detectron2
- sahi  
- torch, torchvision
- opencv-python
- scikit-image

## Quick Start

### 1. Load the Package Functions

```r
# Load all package functions
source("R/model.R")
source("R/training.R") 
source("R/prediction.R")
source("R/data_utils.R")
source("R/summary.R")
```

### 2. Data Preparation

Organize your data in COCO format:

```
data/processed/
├── shell_mixed/         # 5-class shell detection
├── inclusions/          # 2-class inclusion detection  
├── background_removal/  # 2-class background removal
│   ├── train/
│   │   ├── _annotations.coco.json
│   │   └── [training images]
│   └── val/
│       ├── _annotations.coco.json
│       └── [validation images]
```

### 3. Model Training

#### Local Training
```r
model_path <- train_model(
  data_dir = "data/processed/shell_mixed",
  output_name = "shell_detector_v3",
  max_iter = 2000,
  num_classes = 5,
  device = "cuda"  # or "cpu", "mps"
)
```

#### HPC Training (SLURM)
```r
model_path <- train_model(
  data_dir = "data/processed/inclusions", 
  output_name = "inclusions_v2",
  max_iter = 4000,
  num_classes = 2,
  hpc_host = "hpg.rc.ufl.edu",
  hpc_user = "your.username",
  hpc_base_dir = "/blue/your.group/your.username"
)
```

### 4. Model Loading and Prediction

```r
# Load trained model (automatically downloads if needed)
model <- load_model(confidence = 0.5, device = "cpu")

# Analyze single image
result <- predict_image(
  image_path = "data/raw/sample_image.jpg",
  model = model,
  use_slicing = TRUE,
  slice_size = 512,
  overlap = 0.2
)

# Batch processing
batch_result <- predict_images(
  input_dir = "data/raw/batch_images/",
  model = model,
  output_dir = "results/batch_analysis"
)
```

### 5. Analysis and Visualization

Use the analysis notebooks:
- `petrography_analysis.qmd` - Main analysis workflow
- `training/training_shell.qmd` - Shell detection training
- `training/training_inclusions.qmd` - Inclusion detection training  
- `training/training_background_removal.qmd` - Background removal training

## Core Functions

### Model Management
- `load_model()` - Load detection model with caching
- `download_model()` - Download pretrained models

### Training
- `train_model()` - Unified training interface (local or HPC)
- `evaluate_training()` - Analyze training metrics and logs

### Prediction  
- `predict_image()` - Analyze single image with morphological analysis
- `predict_images()` - Process multiple images efficiently

### Analysis
- `enhance_results()` - Add derived morphological properties  
- `summarize_by_image()` - Per-image statistical summaries
- `get_population_stats()` - Overall population metrics

## Output Files

### Prediction Results
- **CSV files**: Detailed morphological measurements per object
- **Visualizations**: Images with detected objects and confidence scores
- **Summary statistics**: Per-image and population-level metrics

### Morphological Properties
Each detected object includes:
- **Basic metrics**: Area, perimeter, centroid coordinates
- **Shape descriptors**: Eccentricity, orientation, circularity, aspect ratio  
- **Advanced features**: Solidity, extent, major/minor axis lengths
- **Derived metrics**: Log area, size categories, shape categories

### Training Evaluation
- **Training curves**: Loss progression over iterations
- **Validation metrics**: Model performance on validation set
- **Learning rate schedule**: LR changes during training

## Configuration

### Training Parameters
Key parameters for `train_model()`:
- `max_iter`: Training iterations (2000-4000 typical)
- `learning_rate`: Base learning rate (0.00025-0.001)
- `num_classes`: Number of object classes
- `eval_period`: Validation frequency (100-500 iterations)
- `device`: "cpu", "cuda", or "mps"

### SAHI Parameters  
Optimize for your data:
- `slice_size`: Slice dimensions (512 recommended)
- `overlap`: Overlap between slices (0.2 typical)
- `confidence`: Detection threshold (0.3-0.7)

### HPC Configuration
For SLURM training:
- `hpc_host`: SSH hostname (e.g., "hpg.rc.ufl.edu")  
- `hpc_user`: Your username
- `hpc_base_dir`: Remote working directory
- Automatic file sync and job monitoring

## Performance Optimization

### For Dense Small Objects (200+ per image)
- Keep `ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512`
- Use moderate batch sizes (`IMS_PER_BATCH = 4`)
- Enable mixed precision training (AMP)
- Consider higher `TEST.DETECTIONS_PER_IMAGE` values

### Training Speed
- Use `IMS_PER_BATCH = 4` for good speed/accuracy balance
- Enable AMP for 30-50% speedup on modern GPUs
- Optimize `NUM_WORKERS` to match available CPU cores
- Use 4-hour SLURM time limits for safety

## Troubleshooting

### Training Issues
- **CUDA out of memory**: Reduce `IMS_PER_BATCH` (try 2-4)
- **Slow training**: Check GPU utilization, enable AMP
- **Job timeouts**: Training time scales with image resolution and object density

### Detection Issues  
- **Missing small objects**: Lower confidence threshold, use smaller slice sizes
- **False positives**: Increase confidence threshold, check training data quality
- **Poor segmentation**: Verify annotation quality, increase training iterations

### R-Python Integration
- **Import errors**: Check `py_require()` calls in notebooks
- **Environment issues**: Restart R session, verify Python environment
- **Path problems**: Use absolute paths, check file existence

## File Structure

```
petrography/
├── R/                          # Package functions
│   ├── model.R                 # Model loading and management
│   ├── training.R              # Training orchestration  
│   ├── hpc_utils.R             # HPC/SLURM utilities
│   ├── prediction.R            # Prediction and analysis
│   ├── data_utils.R            # Data processing utilities
│   └── summary.R               # Summary statistics
├── src/
│   └── train.py                # Python training script
├── training/                   # Training notebooks
│   ├── training_shell.qmd
│   ├── training_inclusions.qmd
│   └── training_background_removal.qmd
├── petrography_analysis.qmd    # Main analysis workflow
└── data/processed/             # Training datasets (gitignored)
```

## Citation

If you use this workflow in your research, please cite:

```bibtex
@software{petrography_analysis,
  title={Petrographic Image Analysis with Detectron2 and SAHI},
  author={Nicolas Gauthier},
  year={2025},
  url={https://github.com/your-repo/petrography}
}
```

## Acknowledgments

- [Detectron2](https://github.com/facebookresearch/detectron2) for instance segmentation
- [SAHI](https://github.com/obss/sahi) for sliced inference
- [reticulate](https://rstudio.github.io/reticulate/) for R-Python integration
- Modern R utilities: [cli](https://cli.r-lib.org/), [fs](https://fs.r-lib.org/), [glue](https://glue.tidyverse.org/)

## Model Registry (pins)

Use the pins package to publish and load trained models by name. This is optional and off by default.

Basic flow:

```r
library(petrographer)

# 1) Configure a writable board (defaults to ~/Dropbox/petrographer-pins)
board <- pg_board_user()  # override with PETRO_PINS_PATH if you prefer a different folder

# 2) Publish a trained model directory
publish_model(
  model_dir = "Detectron2_Models/shell_detector_v3",
  name = "shell_detector_v3",
  board = board,
  metadata = list(owner = Sys.info()[["user"]]),
  include_metrics = TRUE
)

# 3) Load a model by name (uses PETRO_PINS_URL if set, otherwise the local board)
mdl <- load_model(model_name = "shell_detector_v3", device = "cpu")

# 4) Discover pins (optional; uses pins directly)
if (requireNamespace("pins", quietly = TRUE)) {
  pins::pin_list(board)
  pins::pin_meta(board, "shell_detector_v3")
}
```

Publish automatically after training:

```r
train_model(
  data_dir = "data/processed/shell_mixed",
  output_name = "shell_detector_v4",
  num_classes = 5,
  publish_after_train = TRUE,
  model_board = pg_board_user()
)
```

Download a hosted pretrained model (read-only hub served from a shared URL):

```r
Sys.setenv(PETRO_PINS_URL = "https://www.dropbox.com/scl/fo/.../pins?dl=1")
pg_install_pretrained("petrography/inclusions")
mdl <- load_model(model_name = "petrography/inclusions", device = "cpu")
```

After updating the board that feeds your pkgdown or Dropbox site, run:

```r
pg_publish_pkgdown_manifest()
# then rebuild your pkgdown site so the new _pins.yaml is deployed
```

Publish a dataset (images + COCO annotations) the same way:
Before building the pkgdown site, generate the catalog data used on the
Model Library page:

```r
pg_pkgdown_refresh()
```


```r
pg_dataset_publish(
  dataset_dir = "data/processed/inclusions_shell_sliced",
  dataset_id = "datasets/inclusions_shell"
)
```

Consumers can install the dataset bundle from the hosted hub:

```r
Sys.setenv(PETRO_PINS_URL = "https://www.dropbox.com/scl/fo/.../pins?dl=1")
pg_install_dataset("datasets/inclusions_shell")
installed <- pg_dataset_from_pretrained("datasets/inclusions_shell")
installed$dataset_dir
```
Add an optional `preview.png` inside your dataset directory so the catalog shows a thumbnail.

```r
pg_save_dataset_preview("data/processed/inclusions_shell_sliced", split = "valid")
```

Use `pg_plot_dataset_image()` or `pg_plot_annotations()` to inspect samples interactively.

