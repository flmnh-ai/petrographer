# ============================================================================
# Core Prediction Functions - Using Direct Reticulate Calls
# ============================================================================

#' Predict objects in a single image
#' @param image_path Path to image file
#' @param model PetrographyModel object from load_model()
#' @param use_slicing Whether to use SAHI sliced inference (default: TRUE)
#' @param slice_size Size of slices for SAHI in pixels (default: 512)
#' @param overlap Overlap ratio between slices (default: 0.2)
#' @param output_dir Output directory (auto-generated if NULL)
#' @param save_visualizations Whether to save prediction visualization (default: TRUE)
#' @return Tibble with detection results and morphological properties
#' @export
predict_image <- function(image_path, model, use_slicing = TRUE,
                         slice_size = 512, overlap = 0.2, output_dir = NULL,
                         save_visualizations = TRUE) {

  # Validate inputs
  if (!fs::file_exists(image_path)) {
    cli::cli_abort("Image file not found: {.path {image_path}}")
  }

  if (!inherits(model, "PetrographyModel")) {
    cli::cli_abort("model must be a PetrographyModel object from load_model()")
  }

  # Set up output directory
  if (is.null(output_dir)) {
    output_dir <- fs::path("results", tools::file_path_sans_ext(fs::path_file(image_path)))
  }

  if (save_visualizations) {
    fs::dir_create(output_dir)
  }

  # Run SAHI prediction
  if (use_slicing) {
    result <- sahi$predict$get_sliced_prediction(
      image = image_path,
      detection_model = model$sahi_model,
      slice_height = as.integer(slice_size),
      slice_width = as.integer(slice_size),
      overlap_height_ratio = overlap,
      overlap_width_ratio = overlap
    )
  } else {
    result <- sahi$predict$get_prediction(
      image = image_path,
      detection_model = model$sahi_model
    )
  }

  # Check if any objects detected
  if (length(result$object_prediction_list) == 0) {
    return(tibble::tibble())
  }

  # Save visualization if requested
  if (save_visualizations) {
    image_name <- tools::file_path_sans_ext(basename(image_path))
    result$export_visuals(
      export_dir = output_dir,
      file_name = paste0(image_name, "_prediction"),
      hide_conf = TRUE,
      rect_th = 2L
    )
  }
  # Calculate morphological properties and return formatted tibble
  calculate_morphology_from_result(result, image_path) |>
    clean_names() |>
    enhance_results()
}

#' Predict objects in multiple images (directory)
#' @param input_dir Directory containing images
#' @param model PetrographyModel object from load_model()
#' @param use_slicing Whether to use SAHI sliced inference (default: TRUE)
#' @param slice_size Size of slices for SAHI in pixels (default: 512)
#' @param overlap Overlap ratio between slices (default: 0.2)
#' @param output_dir Output directory (default: 'results/batch')
#' @param save_visualizations Whether to save prediction visualizations (default: TRUE)
#' @return Tibble with detection results for all images
#' @export
predict_images <- function(input_dir, model, use_slicing = TRUE,
                          slice_size = 512, overlap = 0.2,
                          output_dir = "results/batch",
                          save_visualizations = TRUE) {

  # Validate inputs
  if (!fs::dir_exists(input_dir)) {
    cli::cli_abort("Input directory not found: {.path {input_dir}}")
  }

  if (!inherits(model, "PetrographyModel")) {
    cli::cli_abort("model must be a PetrographyModel object from load_model()")
  }

  # Create output directory
  if (save_visualizations) {
    fs::dir_create(output_dir)
  }

  # Use SAHI's native batch prediction - much more efficient!
  result <- sahi$predict$predict(
    model_type = 'detectron2',
    model_path = model$model_path,
    model_config_path = model$config_path,
    model_confidence_threshold = model$confidence,
    model_device = model$device,
    source = input_dir,
    no_standard_prediction = use_slicing,  # If using slicing, disable standard
    no_sliced_prediction = !use_slicing,   # If not using slicing, disable sliced
    slice_height = as.integer(slice_size),
    slice_width = as.integer(slice_size),
    overlap_height_ratio = overlap,
    overlap_width_ratio = overlap,
    export_pickle = FALSE,
    export_crop = FALSE,
    export_visuals = save_visualizations,
    export_dir = if (save_visualizations) output_dir else NULL
  )

  # Extract all predictions from batch result
  all_predictions <- list()

  cli::cli_progress_bar("Processing predictions", total = length(result$object_prediction_list))
  for (i in seq_along(result$object_prediction_list)) {
    pred_result <- result$object_prediction_list[[i]]
    image_path <- pred_result$image$file_name

    if (length(pred_result$object_prediction_list) > 0) {
      # Create a temporary result object for morphology calculation
      temp_result <- list(object_prediction_list = pred_result$object_prediction_list)

      # Calculate morphology for this image
      morph_data <- calculate_morphology_from_result(temp_result, image_path)
      all_predictions[[length(all_predictions) + 1]] <- morph_data
    }
    cli::cli_progress_update()
  }
  cli::cli_progress_done()

  # Combine all results
  if (length(all_predictions) > 0) {
    combined_results <- purrr::map_dfr(all_predictions, identity) |>
      clean_names() |>
      enhance_results()
    return(combined_results)
  } else {
    return(tibble::tibble())
  }
}

#' Evaluate detections with SAHI and COCO metrics
#'
#' Runs SAHI inference across the validation set described by a COCO-style
#' annotation file and computes COCO metrics using `pycocotools`. The function
#' returns a tidy summary of the standard 12 bbox metrics alongside the raw
#' prediction table for further analysis.
#'
#' @param model A `PetrographyModel` from [load_model()].
#' @param annotation_json Path to COCO annotation JSON (e.g. `valid/_annotations.coco.json`).
#' @param image_dir Directory containing the images referenced in the
#'   annotation file. If `NULL`, image paths are resolved relative to the
#'   annotation file.
#' @param use_slicing Whether to use SAHI sliced inference (default `TRUE`).
#' @param slice_size Slice size for SAHI inference (pixels, default 512).
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
                                slice_size = 512,
                                overlap = 0.2,
                                max_images = NULL,
                                save_predictions = NULL,
                                iou_type = "bbox",
                                max_dets = 100) {

  if (!inherits(model, "PetrographyModel")) {
    cli::cli_abort("model must be a PetrographyModel object from load_model().")
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
        slice_height = as.integer(slice_size),
        slice_width = as.integer(slice_size),
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

      width <- max_col - min_col
      height <- max_row - min_row

      coco_det <- list(
        image_id = as.integer(img_row$image_id),
        category_id = as.integer(obj$category$id),
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
#' Reads Detectron2 metrics.json and exports:
#' - training_metrics.csv: losses, lr, etc.
#' - validation_metrics.csv: aggregate COCO bbox and segm AP metrics
#' - validation_classwise.csv: per-class AP metrics (when logged by evaluator)
#' @param model_dir Directory containing trained model (default: 'Detectron2_Models')
#' @param output_dir Output directory for results (default: 'results/evaluation')
#' @return List with parsed tibbles and summary statistics
#' @export
evaluate_training <- function(model_dir = "Detectron2_Models",
                             output_dir = "results/evaluation") {
  
  cli::cli_h2("Training Evaluation")
  cli::cli_alert_info("Loading training metrics from: {.path {model_dir}}")

  # Create output directory
  fs::dir_create(output_dir)

  # Look for training metrics
  metrics_file <- fs::path(model_dir, "metrics.json")
  log_file <- fs::path(model_dir, "log.txt")

  parsed <- list(training = tibble::tibble(), validation = tibble::tibble(), classwise = tibble::tibble())
  if (fs::file_exists(metrics_file)) {
    parsed <- parse_metrics(metrics_file)
    # Save to CSV files
    readr::write_csv(parsed$training, fs::path(output_dir, "training_metrics.csv"))
    if (nrow(parsed$validation) > 0) readr::write_csv(parsed$validation, fs::path(output_dir, "validation_metrics.csv"))
    if (nrow(parsed$classwise) > 0) readr::write_csv(parsed$classwise, fs::path(output_dir, "validation_classwise.csv"))
  } else if (fs::file_exists(log_file)) {
    # Could add log parsing here if needed
    warning("Only log.txt found - metrics.json preferred for analysis")
  }

  # Generate enhanced summary
  summary <- list(
    total_iterations = if (nrow(parsed$training) > 0) max(parsed$training$iteration, na.rm = TRUE) else 0,
    metrics_available = nrow(parsed$training) > 0,
    validation_metrics_available = nrow(parsed$validation) > 0,
    validation_segm_available = any(grepl("segm", names(parsed$validation))),
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

  class(result) <- "training_evaluation"

  # Print summary
  cli::cli_dl(c(
    "Training iterations" = summary$total_iterations,
    "Validation evaluations" = summary$validation_evaluations,
    "Segm metrics available" = if (isTRUE(summary$validation_segm_available)) "yes" else "no",
    "Classwise metrics available" = if (isTRUE(summary$classwise_available)) "yes" else "no",
    "Training records" = nrow(parsed$training),
    "Output directory" = output_dir
  ))

  if (nrow(parsed$training) > 0) {
    final_metrics <- tail(parsed$training, 1)
    if ("total_loss" %in% names(final_metrics)) {
      cli::cli_alert_info("Final training loss: {round(final_metrics$total_loss, 4)}")
    }
  }

  cli::cli_alert_success("Training evaluation completed")
  return(result)
}

#' Diagnose annotation dataset for potential issues
#'
#' Analyzes a COCO annotation file to identify potential data quality issues
#' such as incomplete annotations, class imbalance, or unusual object distributions.
#'
#' @param annotation_json Path to COCO annotation JSON file
#' @param image_dir Optional directory containing images (for file checks)
#' @return List with diagnostic statistics and warnings
#' @export
diagnose_annotations <- function(annotation_json, image_dir = NULL) {
  annotation_diagnostics(
    annotation_json = annotation_json,
    image_dir = image_dir,
    emit_header = TRUE
  )
}
