# ============================================================================
# Dataset validation and summary helpers
# ============================================================================

#' Validate a COCO-style dataset directory
#'
#' Performs existence checks for expected splits and annotations, then runs
#' annotation diagnostics (counts, size distribution, potential issues) for
#' each split.
#'
#' @param data_dir Directory containing 'train' and 'valid' subdirectories
#' @return A list with validation flags, counts, size metrics, and diagnostics
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

  cli::cli_h2("Annotation Diagnostics")
  train_diag <- annotation_diagnostics(
    annotation_json = fs::path(train_dir, "_annotations.coco.json"),
    image_dir = train_dir,
    split_label = "Train"
  )
  val_diag <- annotation_diagnostics(
    annotation_json = fs::path(val_dir, "_annotations.coco.json"),
    image_dir = val_dir,
    split_label = "Validation"
  )

  invisible(list(
    data_dir = data_dir,
    train_images = train_images,
    val_images = val_images,
    size_mb = round(total_mb, 1),
    valid = valid,
    diagnostics = list(train = train_diag, valid = val_diag)
  ))
}

annotation_diagnostics <- function(annotation_json, image_dir = NULL, split_label = NULL, emit_header = FALSE) {
  if (!fs::file_exists(annotation_json)) {
    cli::cli_abort("Annotation file not found: {.path {annotation_json}}")
  }

  label <- split_label
  if (is.null(label) || !nzchar(label)) {
    parent_dir <- fs::path_dir(annotation_json)
    label <- stringr::str_to_title(fs::path_file(parent_dir))
  }

  if (isTRUE(emit_header)) {
    cli::cli_h2("Annotation Diagnostics")
  }

  cli::cli_h3(glue::glue("{label} Annotations"))
  cli::cli_alert_info("Analyzing: {.path {annotation_json}}")

  anno <- jsonlite::read_json(annotation_json)
  images <- anno$images
  annotations <- if (is.null(anno$annotations)) list() else anno$annotations
  categories <- anno$categories

  n_images <- length(images)
  n_annotations <- length(annotations)
  n_categories <- length(categories)

  image_ids <- if (length(annotations) == 0) integer(0) else purrr::map_int(annotations, "image_id")
  annos_per_image <- if (length(image_ids) == 0) integer(0) else table(image_ids)
  annos_per_image_vec <- as.numeric(annos_per_image)

  bbox_areas <- if (length(annotations) == 0) {
    numeric(0)
  } else {
    purrr::map_dbl(annotations, function(a) {
      bbox <- a$bbox
      if (is.null(bbox) || length(bbox) < 4) return(NA_real_)
      bbox[[3]] * bbox[[4]]
    })
  }

  category_ids <- if (length(annotations) == 0) integer(0) else purrr::map_int(annotations, "category_id")
  cat_counts <- if (length(category_ids) == 0) integer(0) else table(category_ids)

  mean_annos <- if (length(annos_per_image_vec) > 0) round(mean(annos_per_image_vec), 1) else 0
  median_annos <- if (length(annos_per_image_vec) > 0) stats::median(annos_per_image_vec) else 0
  max_annos <- if (length(annos_per_image_vec) > 0) max(annos_per_image_vec) else 0

  cli::cli_dl(c(
    "Total images" = n_images,
    "Total annotations" = n_annotations,
    "Annotations per image (mean)" = mean_annos,
    "Annotations per image (median)" = median_annos,
    "Annotations per image (max)" = max_annos,
    "Categories" = n_categories
  ))

  cli::cli_h3("Object Size Distribution")
  cli::cli_text("Area (pixels²):")
  print(summary(bbox_areas))

  cli::cli_h3("Potential Issues")

  warnings <- list()

  sparse_images <- if (length(annos_per_image_vec) == 0) 0 else sum(annos_per_image_vec < 10)
  if (sparse_images > n_images * 0.1) {
    msg <- paste0(sparse_images, " images have <10 annotations (",
                  if (n_images > 0) round(100 * sparse_images / n_images, 1) else 0, "%)")
    cli::cli_alert_warning(msg)
    warnings <- c(warnings, list(sparse_annotations = msg))
  }

  dense_threshold <- 100
  dense_images <- if (length(annos_per_image_vec) == 0) 0 else sum(annos_per_image_vec > dense_threshold)
  if (dense_images > 0) {
    msg <- paste0(dense_images, " images have >", dense_threshold,
                  " annotations (max: ", max_annos, ")")
    cli::cli_alert_info(msg)
  }

  tiny_threshold <- 100
  tiny_objects <- if (length(bbox_areas) == 0) 0 else sum(bbox_areas < tiny_threshold, na.rm = TRUE)
  if (tiny_objects > n_annotations * 0.2) {
    msg <- paste0(tiny_objects, " objects are very small (<", tiny_threshold,
                  "px²) - ", if (n_annotations > 0) round(100 * tiny_objects / n_annotations, 1) else 0, "%")
    cli::cli_alert_warning(msg)
    warnings <- c(warnings, list(tiny_objects = msg))
  }

  if (!is.null(image_dir)) {
    image_files <- if (n_images == 0) character(0) else purrr::map_chr(images, "file_name", .default = NA_character_)
    image_paths <- fs::path(image_dir, image_files)
    missing <- if (length(image_paths) == 0) 0 else sum(!fs::file_exists(image_paths))
    if (missing > 0) {
      msg <- paste0(missing, " image files not found in ", image_dir)
      cli::cli_alert_danger(msg)
      warnings <- c(warnings, list(missing_images = msg))
    } else {
      cli::cli_alert_success("All image files found")
    }
  }

  result <- list(
    split = split_label,
    n_images = n_images,
    n_annotations = n_annotations,
    n_categories = n_categories,
    annos_per_image = annos_per_image_vec,
    bbox_areas = bbox_areas,
    category_counts = as.numeric(cat_counts),
    warnings = warnings,
    summary_stats = list(
      mean_annos_per_image = if (length(annos_per_image_vec) > 0) mean(annos_per_image_vec) else NA_real_,
      median_annos_per_image = if (length(annos_per_image_vec) > 0) stats::median(annos_per_image_vec) else NA_real_,
      max_annos_per_image = if (length(annos_per_image_vec) > 0) max_annos else NA_real_,
      mean_bbox_area = if (length(bbox_areas) > 0) mean(bbox_areas, na.rm = TRUE) else NA_real_,
      median_bbox_area = if (length(bbox_areas) > 0) stats::median(bbox_areas, na.rm = TRUE) else NA_real_
    )
  )

  invisible(result)
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
