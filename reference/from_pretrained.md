# Load a pretrained RF-DETR model

Loads RF-DETR models from local training board, hub, or custom board. By
default, checks local models first, then falls back to the public hub.

## Usage

``` r
from_pretrained(
  model_id,
  version = NULL,
  board = NULL,
  device = "cpu",
  confidence = 0.5,
  resolution = NULL
)
```

## Arguments

- model_id:

  Model name (e.g., "shell_v3")

- version:

  Specific version (NULL for latest)

- board:

  Board to load from:

  - `NULL` (default): check local first (.petrographer/models/), then
    hub

  - `"local"`: only check locally trained models (.petrographer/models/)

  - Custom board object

- device:

  Device: "cpu", "cuda", or "mps"

- confidence:

  Detection threshold

- resolution:

  Input image resolution for RF-DETR (default: auto-detect from
  variant). Defaults by variant:

  - Detection (patch_size=16, divisible by 32): nano=384, small=512,
    medium=576, large=704

  - Segmentation (patch_size=12, divisible by 12): seg_nano=312,
    seg_small=384, seg_medium=432, seg_large=504, seg_xlarge=624,
    seg_2xlarge=768, seg_preview=432

  Override only if you know your dataset wants a different input size.

## Value

PetrographyModel object containing RF-DETR model

## Examples

``` r
if (FALSE) { # \dontrun{
# Smart loading (checks local first, then hub)
model <- from_pretrained("my_model")

# Force local only
model <- from_pretrained("my_model", board = "local")

# Force hub only
hub_board <- pins::board_url("https://flmnh-ai.s3.us-east-1.amazonaws.com/.petrographer/models/")
model <- from_pretrained("public_model", board = hub_board)
} # }
```
