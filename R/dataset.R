# ============================================================================
# Dataset validation and summary helpers
# ============================================================================

#' Validate a COCO-style dataset directory
#' @param data_dir Directory containing 'train' and 'valid' subdirectories
#' @return A list with validation flags, counts, and size metrics
#' @export
validate_dataset <- function(data_dir) {
  data_dir <- fs::path_abs(fs::path_norm(data_dir))
  train_dir <- fs::path(data_dir, "train")
  val_dir <- fs::path(data_dir, "valid")
  train_ok <- fs::dir_exists(train_dir)
  val_ok <- fs::dir_exists(val_dir)
  train_anno <- fs::file_exists(fs::path(train_dir, "_annotations.coco.json"))
  val_anno <- fs::file_exists(fs::path(val_dir, "_annotations.coco.json"))
  train_images <- if (train_ok) length(fs::dir_ls(train_dir, regexp = "(?i)\\.(jpg|jpeg|png)$")) else 0
  val_images <- if (val_ok) length(fs::dir_ls(val_dir, regexp = "(?i)\\.(jpg|jpeg|png)$")) else 0

  # Compute size
  files <- fs::dir_ls(data_dir, recurse = TRUE, type = "file")
  total_bytes <- sum(fs::file_size(files), na.rm = TRUE)
  total_mb <- as.numeric(total_bytes) / (1024^2)

  valid <- train_ok && val_ok && train_anno && val_anno && (train_images + val_images) > 0

  # Always show validation results
  cli::cli_h2("Dataset Validation")
  cli::cli_dl(c(
    "Data directory" = data_dir,
    "Train images" = train_images,
    "Val images" = val_images,
    "Train annotations" = if (isTRUE(train_anno)) cli::symbol$tick else cli::symbol$cross,
    "Val annotations" = if (isTRUE(val_anno)) cli::symbol$tick else cli::symbol$cross,
    "Total size" = glue::glue("{round(total_mb, 1)} MB")
  ))

  if (!valid) {
    cli::cli_alert_danger("Dataset invalid")
    cli::cli_abort("Dataset validation failed")
  }

  cli::cli_alert_success("Dataset valid")

  # Return simple list
  invisible(list(
    data_dir = data_dir,
    train_images = train_images,
    val_images = val_images,
    size_mb = round(total_mb, 1),
    valid = valid
  ))
}

#' Summarize a dataset directory
#' @param data_dir Directory containing 'train' and 'valid'
#' @return A tibble with counts for train and val
#' @export
summarize_dataset <- function(data_dir) {
  data_dir <- fs::path_abs(fs::path_norm(data_dir))
  dirs <- c("train", "valid")
  out <- tibble::tibble(
    split = dirs,
    images = vapply(dirs, function(d) {
      p <- fs::path(data_dir, d)
      if (!fs::dir_exists(p)) return(0L)
      length(fs::dir_ls(p, regexp = "(?i)\\.(jpg|jpeg|png)$"))
    }, integer(1)),
    annotations = vapply(dirs, function(d) {
      fs::file_exists(fs::path(data_dir, d, "_annotations.coco.json"))
    }, logical(1))
  )

  # Print summary
  tot <- sum(out$images, na.rm = TRUE)
  cli::cli_h2("Dataset Summary")
  cli::cli_dl(c(
    "Total images" = tot,
    "Splits" = paste(out$split, collapse = ", ")
  ))

  # Show the tibble
  print(out)
  invisible(out)
}


#' Slice COCO dataset for varying image sizes
#'
#' Uses SAHI to slice images and annotations into tiles. Images smaller than
#' `slice_size` are treated as single slices (no fragmentation). Larger images
#' are split into overlapping tiles, which increases training samples and
#' improves detection of small objects in large images.
#'
#' This is particularly useful for dense detection datasets with varying image
#' sizes. For the inclusions dataset, slicing with `slice_size = 1024` will:
#' - Keep small images (<1024px) intact as single slices
#' - Split large images (>1024px) into 2-4 overlapping tiles
#' - Result: ~2x more training images with better small object coverage
#'
#' @param input_dir Input directory with `train/` and `valid/` subdirectories
#'   containing COCO annotations. If a `test/` directory exists, it will also be sliced.
#' @param output_dir Output directory for sliced dataset (will be created)
#' @param slice_size Slice size in pixels (default: 1024). Images smaller than
#'   this will not be fragmented.
#' @param overlap Overlap ratio between adjacent slices (default: 0.2, or 20%)
#' @param min_area_ratio Minimum area ratio to keep object fragments (default: 0.1).
#'   Objects cut by slice boundaries with less than 10% visible area are dropped.
#' @param output_format Output image format: ".jpg" or ".png" (default: ".jpg").
#'   JPG minimizes storage (~10x smaller) with minimal quality loss. Use PNG for lossless slicing.
#' @return Path to sliced dataset directory (invisibly)
#' @export
#' @examples
#' \dontrun{
#' # Slice dataset for training
#' sliced_dir <- slice_dataset(
#'   input_dir = "data/processed/inclusions",
#'   output_dir = "data/processed/inclusions_sliced",
#'   slice_size = 1024,
#'   overlap = 0.2
#' )
#'
#' # Train on sliced dataset
#' train_model(data_dir = sliced_dir, ...)
#' }
slice_dataset <- function(input_dir,
                         output_dir,
                         slice_size = 1024,
                         overlap = 0.2,
                         min_area_ratio = 0.1,
                         output_format = ".jpg") {

  # Validate inputs
  if (!fs::dir_exists(input_dir)) {
    cli::cli_abort("Input directory not found: {.path {input_dir}}")
  }

  if (!output_format %in% c(".jpg", ".png")) {
    cli::cli_abort("output_format must be '.jpg' or '.png', got: {.val {output_format}}")
  }

  input_dir <- fs::path_abs(fs::path_norm(input_dir))
  output_dir <- fs::path_abs(fs::path_norm(output_dir))

  train_dir <- fs::path(input_dir, "train")
  val_dir <- fs::path(input_dir, "valid")

  if (!fs::dir_exists(train_dir)) {
    cli::cli_abort("Train directory not found: {.path {train_dir}}")
  }
  if (!fs::dir_exists(val_dir)) {
    cli::cli_abort("Valid directory not found: {.path {val_dir}}")
  }

  train_anno <- fs::path(train_dir, "_annotations.coco.json")
  val_anno <- fs::path(val_dir, "_annotations.coco.json")

  if (!fs::file_exists(train_anno)) {
    cli::cli_abort("Train annotations not found: {.path {train_anno}}")
  }
  if (!fs::file_exists(val_anno)) {
    cli::cli_abort("Valid annotations not found: {.path {val_anno}}")
  }

  cli::cli_h2("Dataset Slicing")
  cli::cli_alert_info("Input: {.path {input_dir}}")
  cli::cli_alert_info("Output: {.path {output_dir}}")
  cli::cli_dl(c(
    "Slice size" = paste0(slice_size, "px"),
    "Overlap" = paste0(round(100 * overlap), "%"),
    "Min area ratio" = paste0(round(100 * min_area_ratio), "%"),
    "Output format" = output_format
  ))

  # Create output directory
  fs::dir_create(output_dir)

  # Get Python script
  slice_script <- system.file("python", "slice_dataset.py", package = "petrographer")
  if (!fs::file_exists(slice_script)) {
    cli::cli_abort("Slicing script not found. Package installation may be incomplete.")
  }

  # Run slicing
  python_exe <- reticulate::py_config()$python
  args <- c(
    slice_script,
    "--input-dir", input_dir,
    "--output-dir", output_dir,
    "--slice-size", as.character(slice_size),
    "--overlap", as.character(overlap),
    "--min-area-ratio", as.character(min_area_ratio),
    "--output-format", output_format
  )

  cli::cli_alert_info("Running SAHI slicing (this may take a few minutes)...")
  res <- processx::run(python_exe, args = args, echo = TRUE, error_on_status = FALSE)

  if (!identical(res$status, 0L)) {
    cli::cli_abort("Slicing failed with exit code: {res$status}")
  }

  cli::cli_alert_success("Dataset sliced successfully!")
  cli::cli_alert_info("Sliced dataset: {.path {output_dir}}")

  # Validate sliced dataset
  cli::cli_h2("Validating Sliced Dataset")
  validate_dataset(output_dir)

  invisible(output_dir)
}
