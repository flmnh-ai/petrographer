# ============================================================================
# Core Prediction Functions - Using Direct Reticulate Calls
# ============================================================================

#' Predict objects in an image
#'
#' S3 method for [stats::predict()] that runs inference on an image with a
#' `PetrographyModel`. Delegates to [predict_image()].
#'
#' @param object A `PetrographyModel` from [from_pretrained()]. (Named
#'   `object` rather than `model` to match the `stats::predict` generic.)
#' @param image_path Path to image file.
#' @param use_slicing Whether to use SAHI sliced inference (default `TRUE`).
#' @param slice_size Slice size in pixels (default: model's resolution).
#' @param overlap Overlap ratio between slices (default `0.2`).
#' @param save_visualizations Whether to save prediction visualization.
#' @param output_dir Output directory (auto-generated if `NULL`).
#' @param ... Unused; present for generic compatibility.
#' @return Tibble with detection results.
#' @importFrom stats predict
#' @method predict PetrographyModel
#' @export
predict.PetrographyModel <- function(object, image_path,
                                      use_slicing = TRUE,
                                      slice_size = NULL,
                                      overlap = 0.2,
                                      save_visualizations = FALSE,
                                      output_dir = NULL,
                                      ...) {
  predict_image(
    image_path = image_path,
    model = object,
    use_slicing = use_slicing,
    slice_size = slice_size %||% object$resolution,
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
# Always returns a Python dict with integer keys to match SAHI convention.
.category_map <- function(model) {
  cm <- NULL
  if (!is.null(model$sahi_model)) {
    cm <- tryCatch(model$sahi_model$category_mapping, error = function(e) NULL)
  }
  if (is.null(cm)) {
    r_map <- .manifest_category_name_map(model$manifest)
    if (!is.null(r_map)) {
      cm <- reticulate::py_dict(
        keys = as.integer(names(r_map)),
        values = unname(unlist(r_map)),
        convert = FALSE
      )
    }
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
  #
  # TODO(perf): `all_results[[length(all_results) + 1]] <- res` copies the
  # whole list on each append — O(n^2) for large directories. Preallocate
  # `vector("list", length(image_files))` and drop empty slots at the end, or
  # use `purrr::map()` + `compact()`. Only matters at scale (hundreds of
  # images); the loop is fine at typical thin-section volumes.
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

  # For dense detection (user passes max_dets > 100), override params$maxDets so
  # stats[1-5] (AP50/AP75/AP_small/medium/large) and stats[6-11] (AR@... rows)
  # all report at max_dets. Keeps the reported metrics internally consistent.
  coco_eval <- coco_eval_mod$COCOeval(coco_gt, coco_dt, iou_type)
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

  # pycocotools' summarize() hardcodes maxDets=100 for the first row (AP
  # averaged over IoU, area=all). When we override params$maxDets to drop 100
  # from the list, stats[1] (R) comes back as -1. Recompute it manually at
  # max_dets by slicing the precision tensor directly — single eval pass, no
  # need to run evaluate/accumulate twice.
  precision <- reticulate::py_to_r(coco_eval$eval$precision)
  recall <- reticulate::py_to_r(coco_eval$eval$recall)
  # Tensors:
  #   precision: [T, R, K, A, M] = [iou_thresh, recall_thresh, categories, area_ranges, maxDets]
  #   recall:    [T, K, A, M]
  max_dets_list <- as.integer(reticulate::py_to_r(coco_eval$params$maxDets))
  mind <- which(max_dets_list == max(max_dets_list))[1]  # use the largest maxDets we configured
  aind <- 1L  # 'all' area range (Python index 0 = R index 1)
  iou_thrs <- as.numeric(reticulate::py_to_r(coco_eval$params$iouThrs))
  iou50_idx <- which(abs(iou_thrs - 0.5) < 1e-6)[1]

  if (max_dets != 100 && max_dets > 100) {
    s <- precision[, , , aind, mind]
    valid <- s > -1
    stats[1] <- if (any(valid)) mean(s[valid]) else -1
  }

  summary_tbl <- tibble::tibble(
    metric = names(stats),
    value = as.numeric(stats)
  )

  predictions_tbl <- dplyr::bind_rows(detections_rows)

  # Per-class metrics. Slice precision/recall along the K (category) axis to
  # get AP / AP50 / AR per class at the configured maxDets. Category order in
  # the tensors matches coco_eval$params$catIds.
  cat_ids <- as.integer(reticulate::py_to_r(coco_eval$params$catIds))
  cats_raw <- reticulate::py_to_r(coco_gt$loadCats(coco_eval$params$catIds))
  class_name_lookup <- vapply(cats_raw, function(c) as.character(c$name), character(1))
  names(class_name_lookup) <- as.character(vapply(cats_raw, function(c) as.integer(c$id), integer(1)))

  per_class_tbl <- purrr::map_dfr(seq_along(cat_ids), function(k) {
    # AP averaged over IoU: precision[, , k, aind, mind] — shape [T, R]
    p_k <- precision[, , k, aind, mind]
    valid_p <- p_k > -1
    ap <- if (any(valid_p)) mean(p_k[valid_p]) else NA_real_

    # AP50: precision[iou50_idx, , k, aind, mind] — shape [R]
    ap50 <- if (!is.na(iou50_idx)) {
      p_k_50 <- precision[iou50_idx, , k, aind, mind]
      valid_p50 <- p_k_50 > -1
      if (any(valid_p50)) mean(p_k_50[valid_p50]) else NA_real_
    } else NA_real_

    # AR averaged over IoU: recall[, k, aind, mind] — shape [T]
    r_k <- recall[, k, aind, mind]
    valid_r <- r_k > -1
    ar <- if (any(valid_r)) mean(r_k[valid_r]) else NA_real_

    tibble::tibble(
      class_id = as.integer(cat_ids[k]),
      class_name = unname(class_name_lookup[as.character(cat_ids[k])]),
      AP = ap,
      AP50 = ap50,
      AR = ar
    )
  })

  result <- list(
    summary = summary_tbl,
    per_class = per_class_tbl,
    predictions = predictions_tbl,
    coco_eval = coco_eval
  )
  class(result) <- "sahi_evaluation"
  result
}

#' Compute a confusion matrix from SAHI predictions + COCO ground truth
#'
#' For each image, greedily match predicted boxes to ground-truth boxes by IoU
#' (highest-scoring prediction matches first). Predictions that match a GT box
#' with IoU >= `iou_threshold` contribute a (gt_class, pred_class) cell.
#' Unmatched predictions land in the "background" *row* (false positives);
#' unmatched GT boxes land in the "background" *column* (false negatives).
#'
#' Detection doesn't have a clean confusion-matrix definition (unmatched
#' predictions and unmatched GT don't map onto classification confusion
#' cleanly), so treat this as a diagnostic heatmap rather than a calibrated
#' metric.
#'
#' @param predictions Tibble of predictions with `image_id`, `category_id`,
#'   `score`, `xmin`, `ymin`, `width`, `height` (the `predictions` element of
#'   [evaluate_model_sahi()] output works directly).
#' @param annotation_json Path to the COCO annotation JSON the predictions
#'   were produced against.
#' @param iou_threshold Minimum IoU for a prediction to count as matching a
#'   GT box (default 0.5).
#' @param score_threshold Filter predictions below this score before matching
#'   (default 0; keep all).
#' @return A list with:
#'   \itemize{
#'     \item `long`: long-format tibble `(gt_label, pred_label, count)`
#'     \item `matrix`: wide tibble, one row per GT label
#'     \item `class_names`: character vector of class labels (includes "background")
#'     \item `iou_threshold`: the threshold used
#'   }
#' @export
confusion_matrix <- function(predictions,
                             annotation_json,
                             iou_threshold = 0.5,
                             score_threshold = 0) {
  if (!fs::file_exists(annotation_json)) {
    cli::cli_abort("Annotation file not found: {.path {annotation_json}}")
  }
  required <- c("image_id", "category_id", "score", "xmin", "ymin", "width", "height")
  missing_cols <- setdiff(required, names(predictions))
  if (length(missing_cols) > 0) {
    cli::cli_abort("Predictions tibble missing columns: {.val {missing_cols}}")
  }

  coco_mod <- reticulate::import("pycocotools.coco", convert = FALSE)
  coco_gt <- coco_mod$COCO(annotation_json)

  cat_ids_py <- coco_gt$getCatIds()
  cats <- reticulate::py_to_r(coco_gt$loadCats(cat_ids_py))
  class_names <- vapply(cats, function(c) as.character(c$name), character(1))
  class_ids <- vapply(cats, function(c) as.integer(c$id), integer(1))
  class_name_lookup <- setNames(class_names, as.character(class_ids))

  preds <- predictions |>
    dplyr::filter(.data$score >= score_threshold) |>
    dplyr::mutate(xmax = .data$xmin + .data$width, ymax = .data$ymin + .data$height)

  all_image_ids <- sort(unique(c(
    as.integer(preds$image_id),
    as.integer(reticulate::py_to_r(coco_gt$getImgIds()))
  )))

  match_records <- purrr::map_dfr(all_image_ids, function(img_id) {
    .match_predictions_to_gt(
      img_id = img_id,
      preds_img = preds |> dplyr::filter(.data$image_id == img_id),
      coco_gt = coco_gt,
      iou_threshold = iou_threshold
    )
  })

  labels <- c(class_names, "background")

  matches_labeled <- match_records |>
    dplyr::mutate(
      gt_label = ifelse(
        is.na(.data$gt_class),
        "background",
        unname(class_name_lookup[as.character(.data$gt_class)])
      ),
      pred_label = ifelse(
        is.na(.data$pred_class),
        "background",
        unname(class_name_lookup[as.character(.data$pred_class)])
      )
    )

  long_tbl <- matches_labeled |>
    dplyr::count(.data$gt_label, .data$pred_label, name = "count") |>
    dplyr::mutate(
      gt_label = factor(.data$gt_label, levels = labels),
      pred_label = factor(.data$pred_label, levels = labels)
    ) |>
    tidyr::complete(
      gt_label = factor(labels, levels = labels),
      pred_label = factor(labels, levels = labels),
      fill = list(count = 0L)
    )

  wide_tbl <- long_tbl |>
    tidyr::pivot_wider(names_from = "pred_label", values_from = "count")

  structure(
    list(
      long = long_tbl,
      matrix = wide_tbl,
      class_names = labels,
      iou_threshold = iou_threshold
    ),
    class = "petrographer_confusion_matrix"
  )
}

# Internal: greedy IoU-matching between predictions and GT for one image.
.match_predictions_to_gt <- function(img_id, preds_img, coco_gt, iou_threshold) {
  ann_ids <- coco_gt$getAnnIds(imgIds = reticulate::r_to_py(as.integer(img_id)))
  gt_anns <- reticulate::py_to_r(coco_gt$loadAnns(ann_ids))

  gt_n <- length(gt_anns)
  gt_boxes <- if (gt_n > 0) {
    xmin <- vapply(gt_anns, function(a) as.numeric(a$bbox[[1]]), numeric(1))
    ymin <- vapply(gt_anns, function(a) as.numeric(a$bbox[[2]]), numeric(1))
    w    <- vapply(gt_anns, function(a) as.numeric(a$bbox[[3]]), numeric(1))
    h    <- vapply(gt_anns, function(a) as.numeric(a$bbox[[4]]), numeric(1))
    cbind(xmin = xmin, ymin = ymin, xmax = xmin + w, ymax = ymin + h)
  } else {
    matrix(numeric(), 0, 4, dimnames = list(NULL, c("xmin", "ymin", "xmax", "ymax")))
  }
  gt_classes <- if (gt_n > 0) {
    vapply(gt_anns, function(a) as.integer(a$category_id), integer(1))
  } else integer(0)

  # Sort predictions by descending score so highest-scoring matches first
  preds_img <- preds_img |> dplyr::arrange(dplyr::desc(.data$score))
  pred_n <- nrow(preds_img)
  pred_boxes <- if (pred_n > 0) {
    cbind(
      xmin = preds_img$xmin,
      ymin = preds_img$ymin,
      xmax = preds_img$xmax,
      ymax = preds_img$ymax
    )
  } else {
    matrix(numeric(), 0, 4, dimnames = list(NULL, c("xmin", "ymin", "xmax", "ymax")))
  }
  pred_classes <- as.integer(preds_img$category_id)

  iou_mat <- .iou_matrix(pred_boxes, gt_boxes)

  matched_gt   <- integer(gt_n)   # 0 = unmatched, else the pred index
  matched_pred <- integer(pred_n) # 0 = unmatched, else the gt index

  for (i in seq_len(pred_n)) {
    if (gt_n == 0) break
    candidates <- which(matched_gt == 0)
    if (!length(candidates)) break
    best_j <- candidates[which.max(iou_mat[i, candidates])]
    if (iou_mat[i, best_j] >= iou_threshold) {
      matched_pred[i] <- best_j
      matched_gt[best_j]   <- i
    }
  }

  matched_rows <- if (pred_n > 0) {
    tibble::tibble(
      gt_class = ifelse(matched_pred > 0,
                        gt_classes[pmax(matched_pred, 1L)],
                        NA_integer_),
      pred_class = pred_classes
    ) |> dplyr::mutate(
      gt_class = ifelse(matched_pred == 0, NA_integer_, .data$gt_class)
    )
  } else tibble::tibble(gt_class = integer(), pred_class = integer())

  unmatched_gt_rows <- if (gt_n > 0 && any(matched_gt == 0)) {
    tibble::tibble(
      gt_class = gt_classes[matched_gt == 0],
      pred_class = NA_integer_
    )
  } else tibble::tibble(gt_class = integer(), pred_class = integer())

  dplyr::bind_rows(matched_rows, unmatched_gt_rows)
}

# Internal: vectorized IoU between two sets of boxes (rows of [xmin, ymin, xmax, ymax]).
.iou_matrix <- function(a, b) {
  n <- nrow(a); m <- nrow(b)
  if (n == 0 || m == 0) return(matrix(0, n, m))

  axmin <- matrix(a[, 1], n, m); aymin <- matrix(a[, 2], n, m)
  axmax <- matrix(a[, 3], n, m); aymax <- matrix(a[, 4], n, m)
  bxmin <- matrix(b[, 1], n, m, byrow = TRUE); bymin <- matrix(b[, 2], n, m, byrow = TRUE)
  bxmax <- matrix(b[, 3], n, m, byrow = TRUE); bymax <- matrix(b[, 4], n, m, byrow = TRUE)

  ix1 <- pmax(axmin, bxmin); iy1 <- pmax(aymin, bymin)
  ix2 <- pmin(axmax, bxmax); iy2 <- pmin(aymax, bymax)
  iw <- pmax(0, ix2 - ix1);  ih <- pmax(0, iy2 - iy1)
  inter <- iw * ih

  area_a <- (axmax - axmin) * (aymax - aymin)
  area_b <- (bxmax - bxmin) * (bymax - bymin)
  union <- area_a + area_b - inter

  ifelse(union > 0, inter / union, 0)
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
