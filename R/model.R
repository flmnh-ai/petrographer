# ============================================================================
# Model Loading and Management
# ============================================================================

#' Load petrography detection model
#'
#' Loads a Detectron2/SAHI model either from file paths or by pin name via
#' the pins registry. When `model_name` is supplied (or `model_path` does not
#' exist and does not end with `.pth`), the model is retrieved from a board
#' using [get_model()].
#'
#' @param model_path Path to trained model weights (ignored if `model_name` is supplied or `model_path` appears to be a pin name).
#'   If a directory path is provided, will look for `model_best.pth` (if `use_best = TRUE`) or `model_final.pth`.
#' @param config_path Path to model config (ignored if loading from a pin).
#' @param confidence Confidence threshold (default: 0.3).
#' @param device Device to use: 'cpu', 'cuda', 'mps' (default: 'cpu').
#' @param model_name Optional pin name to load from a board.
#' @param version Specific version to load when using pins (NULL for latest).
#' @param board Optional pins board override; when NULL, the hosted hub is used if available.
#' @param use_best If TRUE (default), load `model_best.pth` when available instead of `model_final.pth`.
#'   The best model is selected by BestCheckpointer during training based on validation segm/AP.
#' @return A `PetrographyModel` object.
#' @export
load_model <- function(model_path = NULL,
                       config_path = NULL,
                       confidence = 0.3,
                       device = "cpu",
                       model_name = NULL,
                       version = NULL,
                       board = NULL,
                       use_best = TRUE) {
  manifest <- NULL
  pin_meta <- NULL
  cache_dir <- NULL

  # Resolve pin-based loading
  if (!is.null(model_name)) {
    resolved <- pg_model_from_pretrained(model_name, version = version, board = board, use_best = use_best)
    model_path <- resolved$model_path
    config_path <- resolved$config_path
    manifest <- resolved$manifest
    pin_meta <- resolved$pin_meta
    cache_dir <- resolved$cache_dir
  } else if (!is.null(model_path) && !fs::file_exists(model_path) && !grepl("\\.pth$", model_path)) {
    resolved <- pg_model_from_pretrained(model_path, version = version, board = board, use_best = use_best)
    model_path <- resolved$model_path
    config_path <- resolved$config_path
    manifest <- resolved$manifest
    pin_meta <- resolved$pin_meta
    cache_dir <- resolved$cache_dir
  } else if (!is.null(model_path) && fs::is_dir(model_path)) {
    dir_manifest <- pg_model_resolve_manifest(model_path)
    if (!is.null(dir_manifest)) {
      manifest <- dir_manifest
      model_dir <- fs::path_abs(fs::path_norm(model_path))
      if (isTRUE(use_best) && !is.null(manifest$artifacts$model_best)) {
        candidate <- fs::path(model_dir, manifest$artifacts$model_best)
        if (fs::file_exists(candidate)) {
          cli::cli_alert_info("Using best checkpoint: {.path {fs::path_file(candidate)}}")
          model_path <- candidate
        } else {
          cli::cli_alert_info("Best checkpoint not found, using final checkpoint")
          model_path <- fs::path(model_dir, pg_coalesce(manifest$artifacts$model_final, "model_final.pth"))
        }
      } else {
        model_path <- fs::path(model_dir, pg_coalesce(manifest$artifacts$model_final, "model_final.pth"))
      }
      if (is.null(config_path)) {
        config_path <- fs::path(model_dir, pg_coalesce(manifest$artifacts$config, "config.yaml"))
      }
    } else {
      # Legacy folder layout
      if (use_best) {
        best_model <- fs::path(model_path, "model_best.pth")
        if (fs::file_exists(best_model)) {
          cli::cli_alert_info("Using best checkpoint: {.path {fs::path_file(best_model)}}")
          model_path <- best_model
        } else {
          cli::cli_alert_info("Best checkpoint not found, using final checkpoint")
          model_path <- fs::path(model_path, "model_final.pth")
        }
      } else {
        model_path <- fs::path(model_path, "model_final.pth")
      }
      if (is.null(config_path)) {
        config_path <- fs::path(fs::path_dir(model_path), "config.yaml")
      }
    }
  }

  # Standard file-based fallback
  cache <- get_model_cache_dir()
  default_model <- fs::path(cache, "model_final.pth")
  default_config <- fs::path(cache, "config.yaml")

  if (is.null(model_path)) {
    model_path <- default_model
  }

  if (is.null(config_path)) config_path <- default_config

  if (!fs::file_exists(model_path) || !fs::file_exists(config_path)) {
    cli::cli_alert_info("Model files not found. Downloading...")
    download_model()
  }

  sahi_model <- sahi$AutoDetectionModel$from_pretrained(
    model_type = 'detectron2',
    model_path = model_path,
    config_path = config_path,
    confidence_threshold = confidence,
    device = device
  )

  model <- list(
    sahi_model = sahi_model,
    model_path = model_path,
    config_path = config_path,
    confidence = confidence,
    device = device,
    manifest = manifest,
    pin_meta = pin_meta,
    cache_dir = cache_dir
  )
  class(model) <- "PetrographyModel"
  return(model)
}


get_model_cache_dir <- function() {
  tools::R_user_dir("petrographer", which = "cache")
}


download_model <- function(force = FALSE) {
  cache_dir <- get_model_cache_dir()
  fs::dir_create(cache_dir)

  model_url <- "https://www.dropbox.com/scl/fi/3ilo6msi7r1d9fmfn1zq2/model_final.pth?rlkey=6x2ielfy0fr7kijkysa0i3b3l&st=wbfz9k50&dl=1"
  config_url <- "https://www.dropbox.com/scl/fi/kjlggms8k1x4ghhjiph39/config.yaml?rlkey=8lqiu9eeh6xtjcoj2v7ksyb3k&st=haqn63up&dl=1"

  model_path <- fs::path(cache_dir, "model_final.pth")
  config_path <- fs::path(cache_dir, "config.yaml")

  if (!fs::file_exists(model_path) || force) {
    cli::cli_alert_info("Downloading model weights...")
    download.file(model_url, model_path, mode = "wb")
    cli::cli_alert_success("Model weights saved to: {.path {model_path}}")
  } else {
    cli::cli_alert_info("Model weights already present at: {.path {model_path}}")
  }

  if (!fs::file_exists(config_path) || force) {
    cli::cli_alert_info("Downloading model config...")
    download.file(config_url, config_path, mode = "wb")
    cli::cli_alert_success("Model config saved to: {.path {config_path}}")
  } else {
    cli::cli_alert_info("Model config already present at: {.path {config_path}}")
  }

  return(list(model_path = model_path, config_path = config_path))
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
list_models <- function(output_dir = "Detectron2_Models") {
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
    model_path <- fs::path(d, "model_final.pth")

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


#' Compare metrics across multiple models
#'
#' Loads training metrics from multiple models and creates comparison plots.
#'
#' @param model_names Character vector of model names (directory names)
#' @param output_dir Directory containing trained models (default: 'Detectron2_Models')
#' @param metrics Character vector of metrics to compare (default: c("total_loss", "segm_AP"))
#' @return A list with training data and validation data for all models
#' @export
compare_models <- function(model_names,
                          output_dir = "Detectron2_Models",
                          metrics = c("total_loss", "segm_AP")) {

  if (length(model_names) == 0) {
    cli::cli_abort("Please provide at least one model name")
  }

  # Load metrics for each model
  all_training <- list()
  all_validation <- list()

  cli::cli_h2("Loading model metrics")
  for (model_name in model_names) {
    model_dir <- fs::path(output_dir, model_name)
    metrics_path <- fs::path(model_dir, "metrics.json")

    if (!fs::file_exists(metrics_path)) {
      cli::cli_warn("No metrics found for {.val {model_name}}")
      next
    }

    parsed <- parse_metrics(metrics_path)
    if (nrow(parsed$training) > 0) {
      all_training[[model_name]] <- parsed$training |>
        dplyr::mutate(model = model_name)
    }
    if (nrow(parsed$validation) > 0) {
      all_validation[[model_name]] <- parsed$validation |>
        dplyr::mutate(model = model_name)
    }
    cli::cli_alert_success("Loaded {.val {model_name}}")
  }

  # Combine all data
  training_combined <- if (length(all_training) > 0) {
    dplyr::bind_rows(all_training)
  } else {
    tibble::tibble()
  }

  validation_combined <- if (length(all_validation) > 0) {
    dplyr::bind_rows(all_validation)
  } else {
    tibble::tibble()
  }

  result <- list(
    training = training_combined,
    validation = validation_combined,
    models = model_names
  )

  # Print summary
  cli::cli_h2("Comparison Summary")
  cli::cli_dl(c(
    "Models compared" = length(model_names),
    "Training records" = nrow(training_combined),
    "Validation records" = nrow(validation_combined)
  ))

  # Show final metrics comparison
  if (nrow(validation_combined) > 0 && "segm_AP" %in% names(validation_combined)) {
    cli::cli_h3("Final segm/AP by Model")
    final_aps <- validation_combined |>
      dplyr::group_by(model) |>
      dplyr::slice_tail(n = 1) |>
      dplyr::select(model, segm_AP) |>
      dplyr::arrange(dplyr::desc(segm_AP))
    print(final_aps, n = Inf)
  }

  invisible(result)
}
