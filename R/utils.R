# ============================================================================
# S3 Print Methods
# ============================================================================

#' @export
print.PetrographyModel <- function(x, ...) {
  cli::cli_h2("Petrography Model")

  # Try to get class names from config if available
  class_names <- NULL
  if (fs::file_exists(x$config_path)) {
    cfg <- yaml::read_yaml(x$config_path)
    # Try to extract class names from various possible locations in config
    if (!is.null(cfg$DATASETS$TRAIN) && length(cfg$DATASETS$TRAIN) > 0) {
      class_names <- paste0("<from dataset: ", cfg$DATASETS$TRAIN[1], ">")
    }
  }

  cli::cli_dl(c(
    "Model path" = x$model_path,
    "Config path" = x$config_path,
    "Confidence threshold" = x$confidence,
    "Device" = x$device,
    "Classes" = class_names %||% "<unknown>"
  ))
  invisible(x)
}

#' @export
print.sahi_evaluation <- function(x, ...) {
  cli::cli_h2("SAHI COCO Evaluation Results")

  n_images <- length(unique(x$predictions$image_id))
  n_detections <- nrow(x$predictions)

  cli::cli_alert_info("{n_detections} detection{?s} across {n_images} image{?s}")

  # Show key metrics
  key_metrics <- x$summary |>
    dplyr::filter(metric %in% c("AP", "AP50", "AP75", "AR@100"))

  cli::cli_h3("Key Metrics")
  for (i in seq_len(nrow(key_metrics))) {
    m <- key_metrics[i, ]
    cli::cli_text("{m$metric}: {round(m$value, 3)}")
  }

  invisible(x)
}

#' @export
print.training_evaluation <- function(x, ...) {
  cli::cli_h2("Training Evaluation Results")

  cli::cli_dl(c(
    "Total iterations" = x$summary$total_iterations,
    "Validation evaluations" = x$summary$validation_evaluations,
    "Segm metrics available" = if (isTRUE(x$summary$validation_segm_available)) "yes" else "no",
    "Training records" = nrow(x$training_data),
    "Output directory" = x$output_dir
  ))

  if (nrow(x$training_data) > 0) {
    final_metrics <- tail(x$training_data, 1)
    if ("total_loss" %in% names(final_metrics)) {
      cli::cli_alert_info("Final training loss: {round(final_metrics$total_loss, 4)}")
    }
  }

  if (!is.null(x$validation_data) && nrow(x$validation_data) > 0) {
    final_val <- tail(x$validation_data, 1)
    if ("segm_AP" %in% names(final_val)) {
      cli::cli_alert_info("Final segm/AP: {round(final_val$segm_AP, 4)}")
    }
  }

  invisible(x)
}

# ============================================================================
# Small data helpers
# ============================================================================

clean_names <- function(.data) {
  names(.data) <- names(.data) |>
    stringr::str_replace_all("-", "_") |>
    stringr::str_replace_all("\\s+", "_") |>
    stringr::str_to_lower()
  .data
}

enhance_results <- function(.data) {
  if (nrow(.data) == 0) return(.data)
  .data |>
    dplyr::mutate(
      log_area = log10(area),
      orientation_deg = orientation * 180 / pi,
      size_category = dplyr::case_when(
        area < stats::quantile(area, 0.33, na.rm = TRUE) ~ "small",
        area < stats::quantile(area, 0.67, na.rm = TRUE) ~ "medium",
        TRUE ~ "large"
      ),
      shape_category = dplyr::case_when(
        circularity > 0.8 ~ "circular",
        aspect_ratio > 2 ~ "elongated",
        eccentricity > 0.8 ~ "eccentric",
        TRUE ~ "irregular"
      )
    )
}

