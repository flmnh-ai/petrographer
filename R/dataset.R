# ============================================================================
# Dataset validation and summary helpers
# ============================================================================

#' Validate a COCO-style dataset directory
#' @param data_dir Directory containing 'train' and 'val' subdirectories
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
#' @param data_dir Directory containing 'train' and 'val'
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
