# ============================================================================
# Core Prediction Functions - Using Direct Reticulate Calls
# ============================================================================

#' Predict objects in an image
#'
#' S3 generic for running predictions with a PetrographyModel.
#'
#' @param model A PetrographyModel object
#' @param image_path Path to image file
#' @param ... Additional arguments passed to methods
#' @return Tibble with detection results
#' @export
predict <- function(model, image_path, ...) {
  UseMethod("predict")
}

#' @export
predict.PetrographyModel <- function(model, image_path,
                                      use_slicing = TRUE,
                                      slice_size = NULL,
                                      overlap = 0.2,
                                      save_visualizations = FALSE,
                                      output_dir = NULL,
                                      ...) {
  predict_image(
    image_path = image_path,
    model = model,
    use_slicing = use_slicing,
    slice_size = slice_size %||% model$resolution,
    overlap = overlap,
    save_visualizations = save_visualizations,
    output_dir = output_dir
  )
}

#' Predict objects in a single image
#' @param image_path Path to image file
#' @param model PetrographyModel object from from_pretrained()
#' @param use_slicing Whether to use SAHI sliced inference (default: TRUE)
#' @param slice_size Size of slices for SAHI in pixels (default: use model's resolution). Must be divisible by 56 for RF-DETR.
#' @param overlap Overlap ratio between slices (default: 0.2)
#' @param output_dir Output directory (auto-generated if NULL)
#' @param save_visualizations Whether to save prediction visualization (default: TRUE)
#' @return Tibble with detection results and morphological properties
#' @export
predict_image <- function(image_path, model, use_slicing = TRUE,
                         slice_size = NULL, overlap = 0.2, output_dir = NULL,
                         save_visualizations = TRUE) {

  # Validate inputs
  if (!fs::file_exists(image_path)) {
    cli::cli_abort("Image file not found: {.path {image_path}}")
  }

  if (!inherits(model, "PetrographyModel")) {
    cli::cli_abort("model must be a PetrographyModel object from from_pretrained()")
  }

  # Set up output directory
  if (is.null(output_dir)) {
    output_dir <- fs::path("results", tools::file_path_sans_ext(fs::path_file(image_path)))
  }

  if (save_visualizations) {
    fs::dir_create(output_dir)
  }

  if (model$is_segmentation) {
    # Segmentation models: direct inference (no SAHI)
    PIL <- reticulate::import("PIL")
    img_pil <- PIL$Image$open(image_path)
    detections <- model$direct_model$predict(img_pil, threshold = model$confidence)

    n_det <- nrow(detections$xyxy)
    if (n_det == 0) return(tibble::tibble())

    if (save_visualizations) {
      image_name <- tools::file_path_sans_ext(basename(image_path))
      visualize_py$render_sv_overlay(
        image_path      = as.character(image_path),
        detections      = detections,
        class_names_map = .category_map(model),
        output_path     = as.character(fs::path(output_dir, paste0(image_name, "_prediction.png"))),
        draw_labels     = TRUE,
        mask_opacity    = 0.4
      )
    }

    # Extract morphology from masks
    calculate_morphology_from_detections(
      detections,
      image_path,
      class_names_map = .category_map(model)
    ) |>
      clean_names() |>
      enhance_results()
  } else {
    # Detection models: SAHI sliced inference
    if (use_slicing) {
      actual_slice_size <- slice_size %||% model$resolution %||% 512

      result <- sahi$predict$get_sliced_prediction(
        image = image_path,
        detection_model = model$sahi_model,
        slice_height = as.integer(actual_slice_size),
        slice_width = as.integer(actual_slice_size),
        overlap_height_ratio = overlap,
        overlap_width_ratio = overlap
      )
    } else {
      result <- sahi$predict$get_prediction(
        image = image_path,
        detection_model = model$sahi_model
      )
    }

    if (length(result$object_prediction_list) == 0) {
      return(tibble::tibble())
    }

    if (save_visualizations) {
      image_name <- tools::file_path_sans_ext(basename(image_path))
      visualize_py$render_sahi_overlay(
        image_path      = as.character(image_path),
        sahi_result     = result,
        output_path     = as.character(fs::path(output_dir, paste0(image_name, "_prediction.png"))),
        class_names_map = .category_map(model),
        draw_labels     = TRUE,
        mask_opacity    = 0.4
      )
    }

    calculate_morphology_from_result(result, image_path) |>
      clean_names() |>
      enhance_results()
  }
}

# Internal: map a prediction-time class id back to the source COCO category id.
.prediction_category_id <- function(model, class_id) {
  class_id <- as.integer(class_id)
  id_map <- .manifest_category_id_map(model$manifest)
  mapped <- unname(id_map[as.character(class_id)])
  if (length(mapped) == 0L || is.na(mapped)) class_id else as.integer(mapped)
}

# Internal: best-effort map of class_id -> class_name for a PetrographyModel.
# Prefers the SAHI wrapper's category_mapping (populated from manifest.json)
# and falls back to an empty dict so Python code treats class_ids as strings.
.category_map <- function(model) {
  cm <- NULL
  if (!is.null(model$sahi_model)) {
    cm <- tryCatch(model$sahi_model$category_mapping, error = function(e) NULL)
  }
  if (is.null(cm)) {
    cm <- .manifest_category_name_map(model$manifest)
  }
  if (is.null(cm)) reticulate::dict() else cm
}

#' Predict objects in multiple images (directory)
#' @param input_dir Directory containing images
#' @param model PetrographyModel object from from_pretrained()
#' @param use_slicing Whether to use SAHI sliced inference (default: TRUE)
#' @param slice_size Size of slices for SAHI in pixels (default: use model's resolution)
#' @param overlap Overlap ratio between slices (default: 0.2)
#' @param output_dir Output directory (default: 'results/batch')
#' @param save_visualizations Whether to save prediction visualizations (default: TRUE)
#' @return Tibble with detection results for all images
#' @export
predict_images <- function(input_dir, model, use_slicing = TRUE,
                          slice_size = NULL, overlap = 0.2,
                          output_dir = "results/batch",
                          save_visualizations = TRUE) {

  # Validate inputs
  if (!fs::dir_exists(input_dir)) {
    cli::cli_abort("Input directory not found: {.path {input_dir}}")
  }

  if (!inherits(model, "PetrographyModel")) {
    cli::cli_abort("model must be a PetrographyModel object from from_pretrained()")
  }

  image_files <- fs::dir_ls(
    input_dir,
    regexp = "(?i)\\.(jpg|jpeg|png|tif|tiff)$"
  )
  if (length(image_files) == 0) {
    cli::cli_abort("No image files (jpg/jpeg/png/tif/tiff) found in {.path {input_dir}}")
  }

  if (save_visualizations) {
    fs::dir_create(output_dir)
  }

  # Per-image inference via predict_image(), which handles both the SAHI
  # detection path and the direct segmentation path and writes per-image
  # visualizations into output_dir.
  all_results <- list()
  cli::cli_progress_bar("Running predictions", total = length(image_files))
  for (img_path in image_files) {
    res <- predict_image(
      image_path = img_path,
      model = model,
      use_slicing = use_slicing,
      slice_size = slice_size,
      overlap = overlap,
      output_dir = output_dir,
      save_visualizations = save_visualizations
    )
    if (nrow(res) > 0) {
      all_results[[length(all_results) + 1]] <- res
    }
    cli::cli_progress_update()
  }
  cli::cli_progress_done()

  if (length(all_results) == 0) return(tibble::tibble())

  # Re-run enhance_results() on the combined batch so size_category quantiles
  # reflect the whole batch rather than per-image.
  purrr::map_dfr(all_results, identity) |>
    enhance_results()
}

#' Run a segmentation batch workflow on a directory of images
#'
#' This is the higher-level segmentation path: predict a folder, save overlays,
#' write per-object measurements, and write per-image summaries in one call.
#' It currently uses direct RF-DETR segmentation inference; revisit once SAHI
#' supports Roboflow segmentation models cleanly.
#'
#' @param input_dir Directory containing images
#' @param model Segmentation `PetrographyModel` from [from_pretrained()]
#' @param output_dir Output directory for overlays and tables
#' @param save_visualizations Whether to save overlay images
#' @param save_measurements Whether to write per-object measurements CSV
#' @param save_summary Whether to write per-image summary CSV
#' @param save_population_stats Whether to write a JSON population summary
#' @return A list with detections, per-image summary, population stats, and output directory
#' @export
analyze_segmentation_dir <- function(input_dir,
                                     model,
                                     output_dir = "results/segmentation_batch",
                                     save_visualizations = TRUE,
                                     save_measurements = TRUE,
                                     save_summary = TRUE,
                                     save_population_stats = TRUE) {
  if (!inherits(model, "PetrographyModel")) {
    cli::cli_abort("model must be a PetrographyModel object from from_pretrained()")
  }
  if (!isTRUE(model$is_segmentation)) {
    cli::cli_abort("{.fn analyze_segmentation_dir} requires a segmentation model.")
  }

  fs::dir_create(output_dir)
  overlay_dir <- fs::path(output_dir, "overlays")

  detections <- predict_images(
    input_dir = input_dir,
    model = model,
    use_slicing = FALSE,
    output_dir = overlay_dir,
    save_visualizations = save_visualizations
  )

  image_summary <- if (nrow(detections) > 0) {
    summarize_by_image(detections)
  } else {
    tibble::tibble()
  }
  population_stats <- get_population_stats(detections)

  if (save_measurements) {
    readr::write_csv(detections, fs::path(output_dir, "measurements.csv"))
  }
  if (save_summary) {
    readr::write_csv(image_summary, fs::path(output_dir, "image_summary.csv"))
  }
  if (save_population_stats) {
    jsonlite::write_json(
      population_stats,
      fs::path(output_dir, "population_stats.json"),
      pretty = TRUE,
      auto_unbox = TRUE,
      null = "null"
    )
  }

  list(
    detections = detections,
    summary = image_summary,
    population_stats = population_stats,
    output_dir = output_dir
  )
}

#' Evaluate detections with SAHI and COCO metrics
#'
#' Runs SAHI inference across the validation set described by a COCO-style
#' annotation file and computes COCO metrics using `pycocotools`. The function
#' returns a tidy summary of the standard 12 bbox metrics alongside the raw
#' prediction table for further analysis.
#'
#' @param model A `PetrographyModel` from [from_pretrained()].
#' @param annotation_json Path to COCO annotation JSON (e.g. `valid/_annotations.coco.json`).
#' @param image_dir Directory containing the images referenced in the
#'   annotation file. If `NULL`, image paths are resolved relative to the
#'   annotation file.
#' @param use_slicing Whether to use SAHI sliced inference (default `TRUE`).
#' @param slice_size Slice size for SAHI inference (pixels, default: use model's resolution)
#' @param overlap Overlap ratio between slices (default 0.2).
#' @param max_images Optional maximum number of images to evaluate (useful for
#'   smoke tests).
#' @param save_predictions Optional path to write COCO-format predictions JSON.
#' @param iou_type IoU type to evaluate (`"bbox"` by default).
#' @param max_dets Maximum detections per image for evaluation. For dense detection
#'   (100+ objects), set to 300 or higher (default: 100).
#' @return A list with elements `summary` (tibble of COCO metrics),
#'   `predictions` (tibble of detections), and `coco_eval` (pycocotools object).
#' @export
evaluate_model_sahi <- function(model,
                                annotation_json,
                                image_dir = NULL,
                                use_slicing = TRUE,
                                slice_size = NULL,
                                overlap = 0.2,
                                max_images = NULL,
                                save_predictions = NULL,
                                iou_type = "bbox",
                                max_dets = 100) {

  if (!inherits(model, "PetrographyModel")) {
    cli::cli_abort("model must be a PetrographyModel object from from_pretrained().")
  }
  if (isTRUE(model$is_segmentation) || is.null(model$sahi_model)) {
    cli::cli_abort(c(
      "{.fn evaluate_model_sahi} currently supports detection models only",
      "i" = "Segmentation models are loaded without a SAHI wrapper, so this helper cannot evaluate them yet."
    ))
  }
  if (!fs::file_exists(annotation_json)) {
    cli::cli_abort("Annotation file not found: {.path {annotation_json}}")
  }

  # Load COCO metadata
  coco_mod <- reticulate::import("pycocotools.coco", convert = FALSE)
  coco_eval_mod <- reticulate::import("pycocotools.cocoeval", convert = FALSE)

  coco_gt <- coco_mod$COCO(annotation_json)
  images_info <- reticulate::py_to_r(coco_gt$dataset$images)

  images_df <- tibble::tibble(
    image_id = purrr::map_int(images_info, ~ .x$id),
    file_name = purrr::map_chr(images_info, ~ .x$file_name)
  )

  base_dir <- if (!is.null(image_dir)) {
    image_dir
  } else {
    fs::path_dir(annotation_json)
  }

  images_df <- images_df |>
    dplyr::mutate(full_path = fs::path(base_dir, file_name))

  missing_files <- images_df |> dplyr::filter(!fs::file_exists(full_path))
  if (nrow(missing_files) > 0) {
    cli::cli_abort(c(
      "Image files referenced in annotations were not found",
      paste0("- ", missing_files$file_name)
    ))
  }

  if (!is.null(max_images) && max_images < nrow(images_df)) {
    images_df <- dplyr::slice_head(images_df, n = max_images)
  }

  results_list <- list()
  detections_rows <- list()

  cli::cli_progress_bar("Evaluating images", total = nrow(images_df))
  for (idx in seq_len(nrow(images_df))) {
    img_row <- images_df[idx, ]
    img_path <- img_row$full_path

    if (use_slicing) {
      pred <- sahi$predict$get_sliced_prediction(
        image = img_path,
        detection_model = model$sahi_model,
        slice_height = as.integer(slice_size %||% model$resolution %||% 512),
        slice_width = as.integer(slice_size %||% model$resolution %||% 512),
        overlap_height_ratio = overlap,
        overlap_width_ratio = overlap
      )
    } else {
      pred <- sahi$predict$get_prediction(
        image = img_path,
        detection_model = model$sahi_model
      )
    }

    cli::cli_progress_update()

    if (length(pred$object_prediction_list) == 0) {
      next
    }

    preds <- pred$object_prediction_list
    for (obj in preds) {
      # SAHI's ObjectPrediction has .mask set only when the underlying model
      # emits segmentation masks. Detection-only models (RF-DETR bbox head)
      # leave .mask = None, so we can't run regionprops — read bbox directly.
      has_mask <- !is.null(obj$mask) && !is.null(obj$mask$bool_mask)

      if (has_mask) {
        mask <- obj$mask$bool_mask
        labeled_mask <- skimage$measure$label(mask)
        storage.mode(labeled_mask) <- "integer"
        props <- skimage$measure$regionprops(labeled_mask)
        if (length(props) == 0) {
          next
        }
        prop <- props[[1]]
        bbox <- prop$bbox  # (min_row, min_col, max_row, max_col)
        min_row <- as.numeric(bbox[[1]])
        min_col <- as.numeric(bbox[[2]])
        max_row <- as.numeric(bbox[[3]])
        max_col <- as.numeric(bbox[[4]])
      } else {
        # Detection: SAHI's BoundingBox exposes minx/miny/maxx/maxy directly.
        bb <- obj$bbox
        if (is.null(bb)) next
        min_col <- as.numeric(bb$minx)
        min_row <- as.numeric(bb$miny)
        max_col <- as.numeric(bb$maxx)
        max_row <- as.numeric(bb$maxy)
      }

      width <- max_col - min_col
      height <- max_row - min_row

      coco_det <- list(
        image_id = as.integer(img_row$image_id),
        category_id = .prediction_category_id(model, obj$category$id),
        bbox = c(min_col, min_row, width, height),
        score = as.numeric(obj$score$value)
      )

      results_list[[length(results_list) + 1]] <- coco_det
      detections_rows[[length(detections_rows) + 1]] <- tibble::tibble(
        image_id = img_row$image_id,
        file_name = img_row$file_name,
        category_id = obj$category$id,
        category_name = obj$category$name,
        score = as.numeric(obj$score$value),
        xmin = min_col,
        ymin = min_row,
        width = width,
        height = height
      )
    }
  }
  cli::cli_progress_done()

  if (!length(results_list)) {
    cli::cli_warn("No detections were produced; returning empty evaluation results.")
    return(list(
      summary = tibble::tibble(metric = character(), value = numeric()),
      predictions = tibble::tibble(),
      coco_eval = NULL
    ))
  }

  if (!is.null(save_predictions)) {
    jsonlite::write_json(results_list, save_predictions, auto_unbox = TRUE, digits = 6)
  }

  results_py <- reticulate::r_to_py(results_list, convert = FALSE)
  coco_dt <- coco_gt$loadRes(results_py)
  coco_eval <- coco_eval_mod$COCOeval(coco_gt, coco_dt, iou_type)

  # Configure maxDets for dense detection scenarios
  if (max_dets > 100) {
    coco_eval$params$maxDets <- reticulate::r_to_py(as.integer(c(1, 10, max_dets)))
  }

  coco_eval$evaluate()
  coco_eval$accumulate()
  coco_eval$summarize()

  stats <- reticulate::py_to_r(coco_eval$stats)
  ar_label <- if (max_dets > 100) paste0("AR@", max_dets) else "AR@100"
  names(stats) <- c(
    "AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large",
    "AR@1", "AR@10", ar_label, "AR_small", "AR_medium", "AR_large"
  )

  summary_tbl <- tibble::tibble(
    metric = names(stats),
    value = as.numeric(stats)
  )

  predictions_tbl <- dplyr::bind_rows(detections_rows)

  result <- list(
    summary = summary_tbl,
    predictions = predictions_tbl,
    coco_eval = coco_eval
  )
  class(result) <- "sahi_evaluation"
  result
}

#' Evaluate model training
#'
#' Parses RF-DETR training metrics and exports:
#' - training_metrics.csv: per-epoch losses, learning rate, class error
#' - validation_metrics.csv: per-epoch validation losses + AP/AR + summary map
#' - validation_classwise.csv: per-class AP/precision/recall (when logged)
#'
#' Source-file preference (first one found wins):
#'   1. `metrics.csv` — RF-DETR >= 1.6.0 (PyTorch Lightning CSVLogger)
#'   2. `log.txt`     — RF-DETR <  1.6.0 (native training loop, JSONL)
#' @param model_id Model ID (will be resolved from local board)
#' @param model_dir Directory containing trained model (alternative to model_id)
#' @param board Pins board (NULL = local board, only used if model_id provided)
#' @param output_dir Output directory for results (default: 'results/evaluation')
#' @return List with parsed tibbles and summary statistics
#' @export
evaluate_training <- function(model_id = NULL,
                             model_dir = NULL,
                             board = NULL,
                             output_dir = "results/evaluation") {

  # Resolve model directory from model_id or use provided path
  if (!is.null(model_id)) {
    if (is.null(board)) {
      board <- .get_model_board()
    }
    files <- pins::pin_download(board, model_id)
    model_dir <- fs::path_dir(files[1])
    cli::cli_h2("Training Evaluation")
    cli::cli_alert_info("Model: {.val {model_id}}")
    cli::cli_alert_info("Loading metrics from: {.path {model_dir}}")
  } else if (!is.null(model_dir)) {
    cli::cli_h2("Training Evaluation")
    cli::cli_alert_info("Loading training metrics from: {.path {model_dir}}")
  } else {
    cli::cli_abort("Must provide either {.arg model_id} or {.arg model_dir}")
  }

  # Create output directory
  fs::dir_create(output_dir)

  # Prefer the normalized petrographer-owned summary and only fall back to raw
  # RF-DETR artifacts when it is unavailable.
  training_summary_file <- fs::path(model_dir, "training_summary.json")
  csv_file     <- fs::path(model_dir, "metrics.csv")
  log_file     <- fs::path(model_dir, "log.txt")
  results_file <- fs::path(model_dir, "results.json")

  parsed <- list(training = tibble::tibble(),
                 validation = tibble::tibble(),
                 classwise = tibble::tibble())
  metrics_source <- NA_character_
  final_results <- NULL
  if (fs::file_exists(training_summary_file)) {
    training_summary <- jsonlite::read_json(training_summary_file, simplifyVector = FALSE)
    parsed$training <- .records_to_tibble(training_summary$history$training)
    parsed$validation <- .records_to_tibble(training_summary$history$validation)
    parsed$classwise <- .records_to_tibble(training_summary$history$classwise)
    metrics_source <- training_summary$metrics_source %||% "training_summary.json"
    final_results <- training_summary$final_results %||% NULL
  } else if (fs::file_exists(csv_file)) {
    parsed <- parse_metrics(csv_file)
    metrics_source <- "metrics.csv"
  } else if (fs::file_exists(log_file)) {
    parsed <- parse_metrics(log_file)
    metrics_source <- "log.txt"
  }

  if (nrow(parsed$training)  > 0) readr::write_csv(parsed$training,   fs::path(output_dir, "training_metrics.csv"))
  if (nrow(parsed$validation) > 0) readr::write_csv(parsed$validation, fs::path(output_dir, "validation_metrics.csv"))
  if (nrow(parsed$classwise) > 0) readr::write_csv(parsed$classwise,  fs::path(output_dir, "validation_classwise.csv"))

  # Parse RF-DETR results.json (final evaluation metrics) if the normalized
  # summary did not already provide it.
  if (is.null(final_results) && fs::file_exists(results_file)) {
    final_results <- jsonlite::read_json(results_file, simplifyVector = TRUE)
    # Save to CSV if it's a data frame
    if (is.data.frame(final_results) || is.list(final_results)) {
      readr::write_csv(tibble::as_tibble(final_results), fs::path(output_dir, "final_results.csv"))
    }
  }

  # total_epochs is the count of epoch rows parsed from metrics.csv or log.txt.
  total_epochs <- nrow(parsed$training)

  summary <- list(
    total_epochs = total_epochs,
    metrics_source = metrics_source,
    metrics_available = nrow(parsed$training) > 0,
    validation_metrics_available = nrow(parsed$validation) > 0,
    classwise_available = nrow(parsed$classwise) > 0,
    validation_evaluations = nrow(parsed$validation)
  )

  result <- list(
    training_data = parsed$training,
    summary = summary,
    output_dir = output_dir
  )

  # Add validation data if available
  if (nrow(parsed$validation) > 0) result$validation_data <- parsed$validation
  if (nrow(parsed$classwise) > 0) result$validation_classwise <- parsed$classwise
  if (!is.null(final_results)) result$final_results <- final_results

  class(result) <- "training_evaluation"

  # Print summary
  cli::cli_dl(c(
    "Metrics source" = metrics_source %||% "<none>",
    "Training epochs" = summary$total_epochs,
    "Validation evaluations" = summary$validation_evaluations,
    "Classwise metrics available" = if (isTRUE(summary$classwise_available)) "yes" else "no",
    "Training records" = nrow(parsed$training),
    "Final results available" = if (!is.null(final_results)) "yes" else "no",
    "Output directory" = output_dir
  ))

  if (nrow(parsed$training) > 0 && "loss" %in% names(parsed$training)) {
    final_loss <- utils::tail(parsed$training$loss, 1)
    if (!is.na(final_loss)) {
      cli::cli_alert_info("Final training loss: {round(final_loss, 4)}")
    }
  }

  if (nrow(parsed$validation) > 0) {
    final_val <- utils::tail(parsed$validation, 1)
    # Canonical preference order across both source formats (metrics.csv, log.txt):
    #   map / mAP_50_95 / ap  -> AP@[50:95]
    #   ap50 / mAP_50         -> AP@50
    pref <- c("mAP_50_95", "map", "ap", "ema_mAP_50_95")
    chosen <- intersect(pref, names(final_val))
    val <- NULL
    label <- NULL
    for (nm in chosen) {
      v <- final_val[[nm]]
      if (!is.na(v)) { val <- v; label <- nm; break }
    }
    if (!is.null(val)) {
      cli::cli_alert_info("Final validation {label}: {round(val, 4)}")
    }
  }

  # Print final results if available
  if (!is.null(final_results)) {
    cli::cli_h3("Final Evaluation Results")
    if (is.list(final_results) && !is.data.frame(final_results)) {
      # Print as key-value pairs
      for (metric_name in names(final_results)) {
        value <- final_results[[metric_name]]
        if (is.numeric(value)) {
          cli::cli_alert_info("{metric_name}: {round(value, 4)}")
        } else {
          cli::cli_alert_info("{metric_name}: {value}")
        }
      }
    }
  }

  cli::cli_alert_success("Training evaluation completed")
  return(result)
}
