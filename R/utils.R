# ============================================================================
# S3 Print Methods
# ============================================================================

#' @export
print.PetrographyModel <- function(x, ...) {
  cli::cli_h2("Petrography Model")

  category_map <- .manifest_category_map(x$manifest)
  class_display <- if (length(category_map$model_id_to_name) > 0) {
    paste(unname(category_map$model_id_to_name), collapse = ", ")
  } else {
    "<unknown>"
  }

  cli::cli_dl(c(
    "Model path" = x$model_path,
    "Variant" = x$model_variant %||% "<unknown>",
    "Resolution" = if (!is.null(x$resolution)) as.character(x$resolution) else "<unknown>",
    "Type" = if (isTRUE(x$is_segmentation)) "segmentation" else "detection",
    "Confidence threshold" = x$confidence,
    "Device" = x$device,
    "Classes" = class_display
  ))
  invisible(x)
}

#' @export
print.sahi_evaluation <- function(x, ...) {
  cli::cli_h2("SAHI COCO Evaluation Results")

  n_images <- length(unique(x$predictions$image_id))
  n_detections <- nrow(x$predictions)

  cli::cli_alert_info("{n_detections} detection{?s} across {n_images} image{?s}")

  # Show key metrics. The AR@N row name depends on the max_dets used at
  # evaluation time, so pick it up dynamically rather than hardcoding AR@100.
  ar_metric <- grep("^AR@", x$summary$metric, value = TRUE)
  key_metrics <- x$summary |>
    dplyr::filter(metric %in% c("AP", "AP50", "AP75", ar_metric))

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

  dl <- c(
    x$summary$metrics_source %||% "<none>",
    x$summary$total_epochs %||% 0,
    x$summary$validation_evaluations,
    nrow(x$training_data),
    x$output_dir
  )
  names(dl) <- c("Metrics source", "Total epochs", "Validation evaluations",
                 "Training records", "Output directory")
  cli::cli_dl(dl)

  if (nrow(x$training_data) > 0 && "loss" %in% names(x$training_data)) {
    final_loss <- utils::tail(x$training_data$loss, 1)
    if (!is.na(final_loss)) {
      cli::cli_alert_info("Final training loss: {round(final_loss, 4)}")
    }
  }

  if (!is.null(x$validation_data) && nrow(x$validation_data) > 0) {
    final_val <- utils::tail(x$validation_data, 1)
    # Preference order across RF-DETR source formats:
    #   PTL metrics.csv : mAP_50_95, ema_mAP_50_95
    #   log.txt JSONL   : map (from test_results_json), ap (from COCO array)
    pref <- c("mAP_50_95", "map", "ap", "ema_mAP_50_95")
    chosen <- intersect(pref, names(final_val))
    for (nm in chosen) {
      v <- final_val[[nm]]
      if (!is.na(v)) {
        cli::cli_alert_info("Final validation {nm}: {round(v, 4)}")
        break
      }
    }
    if ("loss" %in% names(final_val) && !is.na(final_val$loss)) {
      cli::cli_alert_info("Final validation loss: {round(final_val$loss, 4)}")
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

  # shape_category depends on mask-based morphology (circularity, eccentricity).
  # Detection-only models fall back to bbox morphology and return NA for those
  # columns — in that case every row would drop through case_when() to
  # "irregular", which is misleading. Return NA instead when we have no real
  # signal. (aspect_ratio alone isn't enough to justify a shape call.)
  shape_available <- any(!is.na(.data$circularity)) ||
    any(!is.na(.data$eccentricity))

  out <- .data |>
    dplyr::mutate(
      log_area = log10(area),
      orientation_deg = orientation * 180 / pi,
      size_category = dplyr::case_when(
        area < stats::quantile(area, 0.33, na.rm = TRUE) ~ "small",
        area < stats::quantile(area, 0.67, na.rm = TRUE) ~ "medium",
        TRUE ~ "large"
      )
    )

  if (shape_available) {
    out |>
      dplyr::mutate(
        shape_category = dplyr::case_when(
          circularity > 0.8 ~ "circular",
          aspect_ratio > 2 ~ "elongated",
          eccentricity > 0.8 ~ "eccentric",
          TRUE ~ "irregular"
        )
      )
  } else {
    out |>
      dplyr::mutate(shape_category = NA_character_)
  }
}
