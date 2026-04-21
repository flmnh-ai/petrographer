# Predict objects in a single image

Predict objects in a single image

## Usage

``` r
predict_image(
  image_path,
  model,
  use_slicing = TRUE,
  slice_size = NULL,
  overlap = 0.2,
  output_dir = NULL,
  save_visualizations = TRUE
)
```

## Arguments

- image_path:

  Path to image file

- model:

  PetrographyModel object from from_pretrained()

- use_slicing:

  Whether to use SAHI sliced inference (default: TRUE)

- slice_size:

  Size of slices for SAHI in pixels (default: use model's resolution).
  Must be divisible by 56 for RF-DETR.

- overlap:

  Overlap ratio between slices (default: 0.2)

- output_dir:

  Output directory (auto-generated if NULL)

- save_visualizations:

  Whether to save prediction visualization (default: TRUE)

## Value

Tibble with detection results and morphological properties
