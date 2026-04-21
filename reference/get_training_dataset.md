# Get the dataset used to train a model

Retrieves the exact dataset version that was used to train a model. This
ensures reproducibility by downloading the specific version from pins.

## Usage

``` r
get_training_dataset(model_id, board = NULL)
```

## Arguments

- model_id:

  Model name

- board:

  Pins board to load model from (NULL = local training board)

## Value

Path to the dataset tar.gz file

## Examples

``` r
if (FALSE) { # \dontrun{
# Retrieve the exact dataset version used to train a model. Returns a
# .tar.gz path; `train_model()` and `validate_dataset()` accept this directly
# and extract transparently.
dataset_path <- get_training_dataset("my_model")

# Retrain on the same data
train_model(data_dir = dataset_path, model_id = "my_model_v2", ...)

# Or inspect without retraining
validate_dataset(dataset_path)
} # }
```
