# Validate a COCO-style dataset directory

Performs existence checks for expected splits and annotations, then runs
annotation diagnostics (counts, size distribution, potential issues) for
each split.

## Usage

``` r
validate_dataset(data_dir, quiet = FALSE)
```

## Arguments

- data_dir:

  Directory containing 'train' and 'valid' subdirectories, or a
  `.tar.gz` / `.tgz` archive thereof (e.g. the path returned by
  [`get_training_dataset()`](https://flmnh-ai.github.io/petrographer/reference/get_training_dataset.md)).
  Archives are transparently extracted into a session-scoped cache.

- quiet:

  If TRUE, suppress CLI output while still returning diagnostics

## Value

A list with validation flags, counts, size metrics, and diagnostics
