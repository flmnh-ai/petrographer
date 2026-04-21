# Pin a trained RF-DETR model to a board

Uploads model files to a pins board for versioning and sharing.
Maintainers should call
[`pins::write_board_manifest()`](https://pins.rstudio.com/reference/write_board_manifest.html)
after pinning to update the board manifest for board_url() consumers.

## Usage

``` r
pin_model(model_dir, model_id, board, metadata = list())
```

## Arguments

- model_dir:

  Directory with model files

- model_id:

  Name for the model

- board:

  Pins board to pin to

- metadata:

  Optional metadata list
