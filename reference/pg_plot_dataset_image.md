# Plot a sample image from a dataset directory

Plot a sample image from a dataset directory

## Usage

``` r
pg_plot_dataset_image(
  dataset_dir,
  split = c("train", "valid", "test"),
  image_name = NULL,
  annotation_json = NULL,
  ...
)
```

## Arguments

- dataset_dir:

  Directory containing dataset splits (train, valid, or optionally
  test).

- split:

  Which split to draw from (`"train"`, `"valid"`, or `"test"`).

- image_name:

  Optional specific image file name; otherwise a random one.

- annotation_json:

  Optional explicit path to annotation JSON.

- ...:

  Additional arguments passed to
  [`pg_plot_annotations()`](https://flmnh-ai.github.io/petrographer/reference/pg_plot_annotations.md).

## Value

The path to the written PNG, invisibly.
