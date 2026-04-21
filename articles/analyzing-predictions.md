# Analyzing Predictions

This vignette shows what to do **after** you have a trained or
pretrained model. It focuses on the analysis side of the workflow:

- loading a model
- running prediction on one image or a directory
- summarizing detections across images
- exploring morphology for segmentation models

It covers both supported prediction paths:

- **Detection**: bounding boxes with SAHI sliced inference
- **Segmentation**: masks with direct morphology analysis

## Load the Toolkit

``` r
library(petrographer)
library(dplyr)
library(ggplot2)
library(fs)
```

## Load a Model

Use
[`from_pretrained()`](https://flmnh-ai.github.io/petrographer/reference/from_pretrained.md)
to load either a local model pin or a published model from the public
hub.

``` r
model <- from_pretrained(
  model_id = "inclusions_small",
  device = "cpu",
  confidence = 0.5
)

model
```

The loaded object retains the parsed manifest and training summary:

``` r
model$manifest$model
model$manifest$categories
model$training_summary$training
```

## Detection Workflow

Detection models are best for object counts, class labels, and
location-based analysis. They also support SAHI sliced inference, which
helps on large images.

### Predict a Single Image

``` r
detector <- from_pretrained("inclusions_small", board = "local", device = "cpu")

det_result <- predict_image(
  image_path = "path/to/image.jpg",
  model = detector,
  use_slicing = TRUE,
  slice_size = 1024,
  overlap = 0.2,
  output_dir = "results/detection_single"
)

det_result
```

### Review the Output Columns

Detection results still come back as a tidy tibble, including class
labels, confidence, geometry, and derived summary fields.

``` r
det_result |>
  select(image_name, class_name, confidence, area, aspect_ratio,
         log_area, size_category, shape_category)
```

### Summarize a Single Image

``` r
get_population_stats(det_result)
```

### Batch Process a Directory

``` r
det_batch <- predict_images(
  input_dir = "path/to/image_directory",
  model = detector,
  use_slicing = TRUE,
  slice_size = 1024,
  overlap = 0.2,
  output_dir = "results/detection_batch",
  save_visualizations = TRUE
)

det_batch
```

### Summarize Across Images

``` r
det_by_image <- summarize_by_image(det_batch)
det_by_image

get_population_stats(det_batch)
```

### Plot Detection-Level Distributions

``` r
det_batch |>
  ggplot(aes(log_area, fill = class_name)) +
  geom_histogram(bins = 30, alpha = 0.7, position = "identity") +
  labs(
    title = "Detection size distribution",
    x = "log10(Area)",
    y = "Count"
  ) +
  theme_minimal()
```

``` r
det_by_image |>
  ggplot(aes(reorder(image_name, n_objects), n_objects)) +
  geom_col(fill = "steelblue") +
  coord_flip() +
  labs(
    title = "Objects detected per image",
    x = "Image",
    y = "Object count"
  ) +
  theme_minimal()
```

## Segmentation Workflow

Segmentation models are best when you need mask-derived morphology such
as area, eccentricity, circularity, orientation, or solidity.

### Predict a Single Image

``` r
segmenter <- from_pretrained("inclusions_shell_seg_small", board = "local", device = "cpu")

seg_result <- predict_image(
  image_path = "path/to/image.jpg",
  model = segmenter,
  use_slicing = FALSE,
  output_dir = "results/segmentation_single"
)

seg_result
```

### Inspect Morphology Columns

``` r
seg_result |>
  select(
    image_name, class_name, confidence, area, eccentricity, circularity,
    orientation_deg, solidity, extent, size_category, shape_category
  )
```

### Single-Image Morphology Summary

``` r
get_population_stats(seg_result)
```

### Plot Morphology

``` r
seg_result |>
  ggplot(aes(log_area, fill = class_name)) +
  geom_histogram(bins = 30, alpha = 0.7, position = "identity") +
  labs(
    title = "Segmented object size distribution",
    x = "log10(Area)",
    y = "Count"
  ) +
  theme_minimal()
```

``` r
seg_result |>
  ggplot(aes(eccentricity, circularity, color = class_name, size = area)) +
  geom_point(alpha = 0.6) +
  scale_size_continuous(trans = "log10") +
  labs(
    title = "Shape space",
    x = "Eccentricity",
    y = "Circularity"
  ) +
  theme_minimal()
```

### Batch Segmentation Analysis

For segmentation, the highest-level helper is
[`analyze_segmentation_dir()`](https://flmnh-ai.github.io/petrographer/reference/analyze_segmentation_dir.md).
It runs prediction on a directory and writes overlays, per-object
measurements, per-image summaries, and population statistics in one
call.

``` r
seg_batch <- analyze_segmentation_dir(
  input_dir = "path/to/image_directory",
  model = segmenter,
  output_dir = "results/segmentation_batch"
)

seg_batch$summary
seg_batch$population_stats
```

### Summaries from Batch Segmentation

``` r
seg_batch$summary |>
  ggplot(aes(reorder(image_name, n_objects), n_objects)) +
  geom_col(fill = "darkolivegreen3") +
  coord_flip() +
  labs(
    title = "Segmented objects per image",
    x = "Image",
    y = "Object count"
  ) +
  theme_minimal()
```

## Choosing an Analysis Path

Use **detection** when you need:

- counts
- locations
- class labels
- scalable whole-slide inference with SAHI slicing

Use **segmentation** when you need:

- mask-derived morphology
- shape measurements
- richer object-by-object physical interpretation

## Next Steps

- use `whole-slide-basics.qmd` for a more focused detection-first
  inference walkthrough
- use `training-models.qmd` for the conceptual training workflow
- use `inst/notebooks/templates/train_model.qmd` when you want a
  runnable training template
