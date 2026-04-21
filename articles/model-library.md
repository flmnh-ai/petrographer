# Model & Dataset Library

This page is the website-facing catalog for the petrographer model and
dataset library. It is designed for browsing what is available, checking
high-level training details, and copying the right
[`from_pretrained()`](https://flmnh-ai.github.io/petrographer/reference/from_pretrained.md)
or
[`train_model()`](https://flmnh-ai.github.io/petrographer/reference/train_model.md)
calls into your own workflow.

## Available Models

## Available Datasets

## Loading a Model

All models can be loaded with a single line:

``` r
library(petrographer)

# Load any model from the hub
model <- from_pretrained("model_id")

# Run predictions
results <- predict(model, "path/to/image.jpg")
```

## Publishing to the Library

Maintainers can publish a finished local model pin to the shared library
board with
[`pin_model()`](https://flmnh-ai.github.io/petrographer/reference/pin_model.md).
The exact destination board depends on the deployment environment, so
the example below is intentionally generic:

``` r
# Download the local pin so you can locate the model directory
local_board <- pins::board_folder(here::here(".petrographer/models"), versioned = TRUE)
files <- pins::pin_download(local_board, "my_model")
model_dir <- fs::path_dir(files[[1]])

# Configure the destination board
# board <- pins::board_url("https://example.com/petrographer/models/")
# board <- pins::board_folder("~/shared/petrographer-models", versioned = TRUE)

# Publish the local pin to that board
pin_model(
  model_dir = model_dir,
  model_id  = "my_model",
  board     = board,
  metadata  = list(
    description   = "Brief description",
    model_variant = "small"
  )
)

# Refresh the destination board manifest for board_url() consumers
pins::write_board_manifest(board)
```

Contact the package maintainers to have your models added to the public
hub.
