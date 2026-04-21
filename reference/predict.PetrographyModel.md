# Predict objects in an image

S3 method for [`stats::predict()`](https://rdrr.io/r/stats/predict.html)
that runs inference on an image with a `PetrographyModel`. Delegates to
[`predict_image()`](https://flmnh-ai.github.io/petrographer/reference/predict_image.md).

## Usage

``` r
# S3 method for class 'PetrographyModel'
predict(
  object,
  image_path,
  use_slicing = TRUE,
  slice_size = NULL,
  overlap = 0.2,
  save_visualizations = FALSE,
  output_dir = NULL,
  ...
)
```

## Arguments

- object:

  A `PetrographyModel` from
  [`from_pretrained()`](https://flmnh-ai.github.io/petrographer/reference/from_pretrained.md).
  (Named `object` rather than `model` to match the
  [`stats::predict`](https://rdrr.io/r/stats/predict.html) generic.)

- image_path:

  Path to image file.

- use_slicing:

  Whether to use SAHI sliced inference (default `TRUE`).

- slice_size:

  Slice size in pixels (default: model's resolution).

- overlap:

  Overlap ratio between slices (default `0.2`).

- save_visualizations:

  Whether to save prediction visualization.

- output_dir:

  Output directory (auto-generated if `NULL`).

- ...:

  Unused; present for generic compatibility.

## Value

Tibble with detection results.
