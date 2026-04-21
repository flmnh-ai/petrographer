# Train model on HPC

Train model on HPC

## Usage

``` r
train_model_hpc(
  data_dir,
  dataset_id,
  model_id,
  run_id,
  model_variant,
  resolution,
  epochs,
  batch_size,
  grad_accum_steps,
  learning_rate,
  workspace_dir,
  use_amp,
  amp_dtype,
  gradient_checkpointing,
  num_workers,
  time_hours,
  validate_every,
  early_stopping_patience
)
```
