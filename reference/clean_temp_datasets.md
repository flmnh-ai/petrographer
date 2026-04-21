# Delete auto-pinned temp datasets

When
[`train_model()`](https://flmnh-ai.github.io/petrographer/reference/train_model.md)
is called with `data_dir` rather than `dataset_id`, it auto-pins the
dataset as `_temp_<timestamp>` (tagged `temp = TRUE` in metadata) so
training is reproducible. Those pins persist in
`.petrographer/datasets/` and accumulate over time — each is a tar.gz of
the full dataset. This helper removes them.

## Usage

``` r
clean_temp_datasets(board = NULL, confirm = interactive())
```

## Arguments

- board:

  Pins board (`NULL`/`"local"` = local dataset board).

- confirm:

  If `TRUE`, prompt before deleting. Defaults to `TRUE` in interactive
  sessions, `FALSE` otherwise (e.g. scripted cleanup).

## Value

Character vector of deleted dataset ids (invisibly).
