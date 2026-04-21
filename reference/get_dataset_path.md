# Get path to pinned dataset

Returns filesystem path to a pinned dataset tar.gz file. The tar.gz
should be extracted at training time.

## Usage

``` r
get_dataset_path(dataset_id, board = "local", version = NULL)
```

## Arguments

- dataset_id:

  Dataset name

- board:

  Pins board (or board object)

- version:

  Specific version to retrieve (NULL = latest)

## Value

Path to dataset tar.gz file
