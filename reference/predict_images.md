# Predict objects in multiple images (directory)

Predict objects in multiple images (directory)

## Usage

``` r
predict_images(
  input_dir,
  model,
  use_slicing = TRUE,
  slice_size = NULL,
  overlap = 0.2,
  output_dir = "results/batch",
  save_visualizations = TRUE
)
```

## Arguments

- input_dir:

  Directory containing images

- model:

  PetrographyModel object from from_pretrained()

- use_slicing:

  Whether to use SAHI sliced inference (default: TRUE)

- slice_size:

  Size of slices for SAHI in pixels (default: use model's resolution)

- overlap:

  Overlap ratio between slices (default: 0.2)

- output_dir:

  Output directory (default: 'results/batch')

- save_visualizations:

  Whether to save prediction visualizations (default: TRUE)

## Value

Tibble with detection results for all images
