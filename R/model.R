# ============================================================================
# Model Loading and Management
# ============================================================================

#' Load petrography detection model from directory
#'
#' Loads a Detectron2/SAHI model from a local directory containing model files.
#' For pretrained models from the hub, use [from_pretrained()] instead.
#'
#' @param model_dir Path to directory containing model files.
#'   Will look for `model_best.pth`.
#' @param config_path Optional path to model config (auto-detected if NULL).
#' @param confidence Confidence threshold (default: 0.3).
#' @param device Device to use: 'cpu', 'cuda', 'mps' (default: 'cpu').
#' @return A `PetrographyModel` object.
#' @export
load_model <- function(model_dir,
                       config_path = NULL,
                       confidence = 0.3,
                       device = "cpu") {

  if (!fs::dir_exists(model_dir)) {
    cli::cli_abort("Model directory not found: {.path {model_dir}}")
  }

  model_dir <- fs::path_abs(fs::path_norm(model_dir))

  # Read manifest if available
  manifest_path <- fs::path(model_dir, "manifest.json")
  manifest <- if (fs::file_exists(manifest_path)) {
    jsonlite::read_json(manifest_path)
  } else {
    NULL
  }

  # Get model path
  model_path <- fs::path(model_dir, "model_best.pth")

  if (!fs::file_exists(model_path)) {
    cli::cli_abort("Model weights not found at {.path {model_path}}")
  }

  # Determine config path
  if (is.null(config_path)) {
    config_path <- fs::path(model_dir, "config.yaml")
  }

  if (!fs::file_exists(config_path)) {
    cli::cli_abort("Config not found at {.path {config_path}}")
  }

  # Load with SAHI
  sahi_model <- sahi$AutoDetectionModel$from_pretrained(
    model_type = 'detectron2',
    model_path = as.character(model_path),
    config_path = as.character(config_path),
    confidence_threshold = confidence,
    device = device
  )

  model <- list(
    sahi_model = sahi_model,
    model_path = as.character(model_path),
    config_path = as.character(config_path),
    confidence = confidence,
    device = device,
    manifest = manifest
  )
  class(model) <- "PetrographyModel"
  return(model)
}


# ============================================================================
# Model Management Utilities
# ============================================================================

#' List all trained models with metadata
#'
#' Scans the model output directory and returns a tibble with information
#' about each trained model, including creation time, size, and final metrics
#' if available.
#'
#' @param output_dir Directory containing trained models (default: 'Detectron2_Models')
#' @return A tibble with model metadata
#' @export
list_trained_models <- function(output_dir = "Detectron2_Models") {
  if (!fs::dir_exists(output_dir)) {
    cli::cli_warn("Model directory not found: {.path {output_dir}}")
    return(tibble::tibble())
  }

  dirs <- fs::dir_ls(output_dir, type = "directory")

  if (length(dirs) == 0) {
    cli::cli_warn("No models found in {.path {output_dir}}")
    return(tibble::tibble())
  }

  models <- purrr::map_dfr(dirs, function(d) {
    cfg_path <- fs::path(d, "config.yaml")
    metrics_path <- fs::path(d, "metrics.json")
    model_path <- fs::path(d, "model_best.pth")

    # Get final AP if metrics available
    final_ap <- NA_real_
    final_loss <- NA_real_
    max_iter <- NA_integer_

    if (fs::file_exists(metrics_path)) {
      tryCatch({
        parsed <- parse_metrics(metrics_path)
        if (nrow(parsed$validation) > 0) {
          final_val <- tail(parsed$validation, 1)
          if ("segm_AP" %in% names(final_val)) {
            final_ap <- final_val$segm_AP
          }
        }
        if (nrow(parsed$training) > 0) {
          final_train <- tail(parsed$training, 1)
          if ("total_loss" %in% names(final_train)) {
            final_loss <- final_train$total_loss
          }
          max_iter <- max(parsed$training$iteration, na.rm = TRUE)
        }
      }, error = function(e) NULL)
    }

    # Get model size
    all_files <- fs::dir_ls(d, recurse = TRUE, type = "file")
    total_size_mb <- sum(fs::file_size(all_files), na.rm = TRUE) / (1024^2)

    tibble::tibble(
      name = fs::path_file(d),
      created = fs::file_info(d)$modification_time,
      has_config = fs::file_exists(cfg_path),
      has_metrics = fs::file_exists(metrics_path),
      has_model = fs::file_exists(model_path),
      size_mb = round(total_size_mb, 1),
      max_iter = max_iter,
      final_loss = final_loss,
      final_ap = final_ap,
      path = as.character(d)
    )
  }) |>
    dplyr::arrange(dplyr::desc(created))

  cli::cli_h2("Trained Models")
  cli::cli_alert_info("Found {nrow(models)} model{?s} in {.path {output_dir}}")

  print(models |> dplyr::select(-path), n = Inf)

  invisible(models)
}


