# Save a preview image for a dataset

Convenience helper that writes a `preview.png` alongside the dataset so
the pkgdown catalog can embed a thumbnail.

## Usage

``` r
pg_save_dataset_preview(
  dataset_dir,
  split = "valid",
  output = fs::path(dataset_dir, "preview.png"),
  overwrite = TRUE,
  ...
)
```

## Arguments

- dataset_dir:

  Dataset directory (containing split subfolders).

- split:

  Which split to sample (default `"valid"`).

- output:

  Path for the preview image (default `<dataset_dir>/preview.png`).

- overwrite:

  Whether to overwrite an existing preview.

- ...:

  Additional arguments passed to
  [`pg_plot_dataset_image()`](https://flmnh-ai.github.io/petrographer/reference/pg_plot_dataset_image.md).

## Value

The path to the written preview image (invisibly).
