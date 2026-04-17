# ==============================================================================
# Publishing Models to Petrographer Hub (Maintainers Only)
# ==============================================================================
#
# This script demonstrates how to publish trained models to the public
# petrographer hub, which is served via pkgdown/GitHub Pages.
#
# Workflow:
# 1. Train model locally (automatically pinned to .petrographer/)
# 2. Pin model to pkgdown hub board
# 3. Update board manifest
# 4. Rebuild pkgdown site
# 5. Commit and push to GitHub
# 6. Model becomes available via from_pretrained()

library(petrographer)
library(pins)
library(here)

# Create hub board (points to pkgdown assets directory)
hub_board <- pins::board_folder(
  here("pkgdown/assets/pins"),
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
# 1. Rebuild pkgdown site: pkgdown::build_site()
# 2. Commit changes: git add pkgdown/assets/pins && git commit -m "Add model_id to hub"
# 3. Push to GitHub: git push
# 4. Model will be available at: from_pretrained("my_model_v1")
