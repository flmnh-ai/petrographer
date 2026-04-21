# Pin a dataset to a board

Pins a COCO-format dataset directory to a pins board for versioning and
reuse. The dataset is compressed as tar.gz before pinning.

## Usage

``` r
pin_dataset(data_dir, dataset_id, board = NULL, metadata = list())
```

## Arguments

- data_dir:

  Path to dataset directory, or a `.tar.gz` / `.tgz` archive
  (transparently extracted before re-pinning).

- dataset_id:

  Name for the pinned dataset

- board:

  Pins board (NULL = local board at .petrographer/)

- metadata:

  Optional metadata list
