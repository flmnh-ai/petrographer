# Run a segmentation batch workflow on a directory of images

This is the higher-level segmentation path: predict a folder, save
overlays, write per-object measurements, and write per-image summaries
in one call. It currently uses direct RF-DETR segmentation inference;
revisit once SAHI supports Roboflow segmentation models cleanly.

## Usage

``` r
analyze_segmentation_dir(
  input_dir,
  model,
  output_dir = "results/segmentation_batch",
  save_visualizations = TRUE,
  save_measurements = TRUE,
  save_summary = TRUE,
  save_population_stats = TRUE
)
```

## Arguments

- input_dir:

  Directory containing images

- model:

  Segmentation `PetrographyModel` from
  [`from_pretrained()`](https://flmnh-ai.github.io/petrographer/reference/from_pretrained.md)

- output_dir:

  Output directory for overlays and tables

- save_visualizations:

  Whether to save overlay images

- save_measurements:

  Whether to write per-object measurements CSV

- save_summary:

  Whether to write per-image summary CSV

- save_population_stats:

  Whether to write a JSON population summary

## Value

A list with detections, per-image summary, population stats, and output
directory
