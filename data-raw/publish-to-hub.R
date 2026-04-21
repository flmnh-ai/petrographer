# ==============================================================================
# Publishing Models to Petrographer Hub (Maintainers Only)
# ==============================================================================
#
# This script demonstrates how to publish trained models to the public
# petrographer hub, which is served via S3 (flmnh-ai bucket).
#
# Workflow:
# 1. Train model locally (automatically pinned to .petrographer/)
# 2. Pin model to S3 hub board
# 3. Update board manifest
# 4. Model becomes available via from_pretrained()
#
# Prerequisites:
#   - AWS credentials configured (access to flmnh-ai bucket)
#   - Trained model pinned locally

library(petrographer)
library(pins)

# Create hub board (S3-backed, matches .hub_models_url in pins.R)
hub_board <- pins::board_s3(
  bucket = "flmnh-ai",
  prefix = ".petrographer/models",
  versioned = TRUE
)

# Example: Pin a trained model to the hub
# Replace with your actual model directory and ID. After train_model(),
# the output lives at .petrographer/models/<model_id>/<run_id>/output/.
model_dir <- ".petrographer/models/my_model/<run_id>/output"
model_id <- "my_model_v1"

pin_model(
  model_dir = model_dir,
  model_id = model_id,
  board = hub_board,
  metadata = list(
    description  = "Description of what this model does",
    model_variant = "small",
    notes        = "Any additional notes"
  )
)

# IMPORTANT: Update the manifest so board_url() consumers can discover pins
pins::write_board_manifest(hub_board)

# Verify the pin was created
pins::pin_list(hub_board)
pins::pin_meta(hub_board, model_id)

# Next steps:
# Model will be available at: from_pretrained("my_model_v1")
# (from_pretrained checks local first, then falls back to S3 hub)

# ==============================================================================
# Publishing the shell_species_large model (Apr 2026)
# ==============================================================================
#
# hub_board <- pins::board_s3(
#   bucket = "flmnh-ai",
#   prefix = ".petrographer/models",
#   versioned = TRUE
# )
#
# pin_model(
#   model_dir = ".petrographer/models/shell_species_large/20260421T003804Z-a1870/output",
#   model_id = "shell_species_large",
#   board = hub_board,
#   metadata = list(
#     description   = "Shell species detector (Clam/Mussel/Oyster) trained on Bloch experimental samples",
#     model_variant = "large",
#     dataset       = "bloch_shell_species (post-QA, 2203 images, 62835 annotations)",
#     training      = "RF-DETR large, 120 epochs, HiPerGator B200",
#     metrics       = "mAP 0.762 (SAHI eval), AP50 0.870, F1 0.917",
#     notes         = "Trained on experimental samples only. Archaeological validation pending."
#   )
# )
#
# pins::write_board_manifest(hub_board)
