# Evaluate model training

Parses RF-DETR training metrics and exports:

- training_metrics.csv: per-epoch losses, learning rate, class error

- validation_metrics.csv: per-epoch validation losses + AP/AR + summary
  map

- validation_classwise.csv: per-class AP/precision/recall (when logged)

## Usage

``` r
evaluate_training(
  model_id = NULL,
  model_dir = NULL,
  board = NULL,
  output_dir = "results/evaluation"
)
```

## Arguments

- model_id:

  Model ID (will be resolved from local board)

- model_dir:

  Directory containing trained model (alternative to model_id)

- board:

  Pins board (NULL = local board, only used if model_id provided)

- output_dir:

  Output directory for results (default: 'results/evaluation')

## Value

List with parsed tibbles and summary statistics

## Details

Source-file preference (first one found wins):

1.  `metrics.csv` — RF-DETR \>= 1.6.0 (PyTorch Lightning CSVLogger)

2.  `log.txt` — RF-DETR \< 1.6.0 (native training loop, JSONL)
