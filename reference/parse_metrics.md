# Parse RF-DETR training metrics into tibbles

Reads RF-DETR training artifacts and returns tibbles for training
losses, validation metrics, and per-class AP when available.

## Usage

``` r
parse_metrics(log_file)
```

## Arguments

- log_file:

  Path to `metrics.csv` (PTL) or `log.txt` (native-loop JSONL).

## Value

list(training, validation, classwise)

## Details

Dispatches on filename:

- `metrics.csv` -\> PyTorch Lightning CSVLogger (RF-DETR \>= 1.6.0).

- `log.txt` -\> native-loop JSONL (RF-DETR \< 1.6.0).

Both paths produce the same output shape so callers can stay agnostic.

Returned tibbles use a common, stable schema (prefix-stripped column
names):

- `training` - keyed on `epoch`, with numeric columns for loss
  components, `lr`, etc.

- `validation`- keyed on `epoch`, with validation metrics. Where
  possible we expose a canonical `ap` alias (maps to `mAP_50_95` on PTL
  and to COCO `AP` on legacy) plus `ap50`, `map`, `precision`, `recall`
  when available.

- `classwise` - long format, one row per (`epoch`, `class_name`) with
  `map_50_95`, `map_50`, `precision`, `recall` when provided. On PTL
  runs, we also populate `map_50_95_ema` and `map_50_ema` when EMA AP
  columns are present.
