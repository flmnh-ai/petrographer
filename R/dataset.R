# ============================================================================
# Dataset validation and summary helpers
# ============================================================================

#' Resolve a dataset source (directory or .tar.gz) to an extracted directory
#'
#' Accepts either a dataset directory or a `.tar.gz` / `.tgz` archive and
#' returns a path to an extracted dataset directory. Archives are extracted
#' once per session into a cache dir keyed on path + mtime, so subsequent
#' calls on the same archive reuse the extraction instead of re-untarring.
#'
#' @param data_source Path to a dataset directory or archive
#' @return Absolute path to a dataset directory
#' @keywords internal
.resolve_data_dir <- function(data_source) {
  data_source <- as.character(data_source)

  if (fs::dir_exists(data_source)) {
    return(fs::path_abs(fs::path_norm(data_source)))
  }

  if (!fs::file_exists(data_source)) {
    cli::cli_abort("Dataset source not found: {.path {data_source}}")
  }

  if (!grepl("\\.tar\\.gz$|\\.tgz$", data_source)) {
    cli::cli_abort(c(
      "Dataset source must be a directory or a .tar.gz/.tgz archive",
      "x" = "Got: {.path {data_source}}"
    ))
  }

  # Session-scoped cache keyed on path + mtime. A freshly repinned archive
  # will have a new mtime and so will trigger a re-extract.
  mtime <- as.integer(fs::file_info(data_source)$modification_time)
  base <- tools::file_path_sans_ext(tools::file_path_sans_ext(fs::path_file(data_source)))
  cache_dir <- fs::path(tempdir(), sprintf("petrographer-dataset-%s-%d", base, mtime))
  marker <- fs::path(cache_dir, ".extracted")

  if (fs::file_exists(marker)) {
    cli::cli_alert_info("Reusing extracted dataset cache: {.path {cache_dir}}")
    return(fs::path_abs(cache_dir))
  }

  cli::cli_alert_info("Extracting {.path {fs::path_file(data_source)}}...")
  fs::dir_create(cache_dir, recurse = TRUE)
  untar_result <- utils::untar(data_source, exdir = cache_dir, tar = "internal")
  if (untar_result != 0) {
    cli::cli_abort("Failed to extract archive: {.path {data_source}}")
  }
  fs::file_create(marker)

  fs::path_abs(cache_dir)
}

#' Validate a COCO-style dataset directory
#'
#' Performs existence checks for expected splits and annotations, then runs
#' annotation diagnostics (counts, size distribution, potential issues) for
#' each split.
#'
#' @param data_dir Directory containing 'train' and 'valid' subdirectories, or
#'   a `.tar.gz` / `.tgz` archive thereof (e.g. the path returned by
#'   [get_training_dataset()]). Archives are transparently extracted into a
#'   session-scoped cache.
#' @param quiet If TRUE, suppress CLI output while still returning diagnostics
#' @return A list with validation flags, counts, size metrics, and diagnostics
#' @export
validate_dataset <- function(data_dir, quiet = FALSE) {
  data_dir <- .resolve_data_dir(data_dir)

  expected_splits <- c("train", "valid")
  missing_dirs <- expected_splits[!fs::dir_exists(fs::path(data_dir, expected_splits))]
  if (length(missing_dirs) > 0) {
    cli::cli_abort("Required split{?s} missing: {toString(missing_dirs)}")
  }

  all_split_dirs <- fs::dir_ls(data_dir, type = "directory", fail = FALSE)
  all_split_names <- fs::path_file(all_split_dirs)
  ordered_splits <- unique(c(expected_splits, setdiff(all_split_names, expected_splits)))

  split_info <- purrr::map(ordered_splits, function(split) {
    split_dir <- fs::path(data_dir, split)
    if (!fs::dir_exists(split_dir)) {
      return(list(
        name = split,
        path = split_dir,
        images = 0L,
        has_annotations = FALSE,
        annotation_path = NULL
      ))
    }

    images <- length(fs::dir_ls(split_dir, regexp = "(?i)\\.(jpg|jpeg|png)$"))
    annotation_path <- fs::path(split_dir, "_annotations.coco.json")
    has_annotations <- fs::file_exists(annotation_path)

    list(
      name = split,
      path = split_dir,
      images = images,
      has_annotations = has_annotations,
      annotation_path = if (has_annotations) annotation_path else NULL
    )
  })
  names(split_info) <- ordered_splits

  files <- fs::dir_ls(data_dir, recurse = TRUE, type = "file")
  total_bytes <- sum(fs::file_size(files), na.rm = TRUE)
  total_mb <- as.numeric(total_bytes) / (1024^2)

  summary_items <- c("Data directory" = data_dir)
  for (split in ordered_splits) {
    info <- split_info[[split]]
    label <- stringr::str_to_title(split)
    summary_items[paste(label, "images")] <- info$images
    summary_items[paste(label, "annotations")] <- if (isTRUE(info$has_annotations)) cli::symbol$tick else cli::symbol$cross
  }
  summary_items["Total size"] <- glue::glue("{round(total_mb, 1)} MB")

  if (!quiet) {
    cli::cli_h2("Dataset Validation")
    cli::cli_dl(summary_items)
  }

  train_info <- split_info[["train"]]
  val_info <- split_info[["valid"]]
  valid <- isTRUE(train_info$has_annotations) && isTRUE(val_info$has_annotations) &&
    (train_info$images + val_info$images) > 0

  if (!valid) {
    cli::cli_alert_danger("Dataset invalid")
    cli::cli_abort("Dataset validation failed")
  }

  if (!quiet) cli::cli_alert_success("Dataset valid")

  if (!quiet) cli::cli_h2("Annotation Diagnostics")
  diagnostics <- purrr::imap(split_info, function(info, split) {
    if (!isTRUE(info$has_annotations)) return(NULL)
    annotation_diagnostics(
      annotation_json = info$annotation_path,
      image_dir = info$path,
      split_label = stringr::str_to_title(split),
      verbose = !quiet
    )
  })
  diagnostics <- diagnostics[!vapply(diagnostics, is.null, logical(1))]

  invisible(list(
    data_dir = data_dir,
    splits = split_info,
    size_mb = round(total_mb, 1),
    valid = valid,
    diagnostics = diagnostics
  ))
}


annotation_diagnostics <- function(annotation_json,
                                     image_dir = NULL,
                                     split_label = NULL,
                                     emit_header = FALSE,
                                     verbose = TRUE) {
  if (!fs::file_exists(annotation_json)) {
    cli::cli_abort("Annotation file not found: {.path {annotation_json}}")
  }

  label <- split_label
  if (is.null(label) || !nzchar(label)) {
    parent_dir <- fs::path_dir(annotation_json)
    label <- stringr::str_to_title(fs::path_file(parent_dir))
  }

  if (isTRUE(verbose) && isTRUE(emit_header)) {
    cli::cli_h2("Annotation Diagnostics")
  }

  if (isTRUE(verbose)) {
    cli::cli_h3(glue::glue("{label} Annotations"))
    cli::cli_alert_info("Analyzing: {.path {annotation_json}}")
  }

  anno <- jsonlite::read_json(annotation_json)

  images_list <- anno$images %||% list()
  images_tbl <- if (length(images_list) > 0) {
    tibble::tibble(
      id = purrr::map_int(images_list, ~{
        val <- .x$id
        if (is.null(val)) NA_integer_ else as.integer(val)
      }),
      file_name = purrr::map_chr(images_list, ~{
        val <- .x$file_name
        if (is.null(val)) NA_character_ else as.character(val)
      })
    )
  } else {
    tibble::tibble(id = integer(), file_name = character())
  }

  annotations_list <- anno$annotations %||% list()
  annotations_tbl <- if (length(annotations_list) > 0) {
    tibble::tibble(
      id = purrr::map_int(annotations_list, ~{
        val <- .x$id
        if (is.null(val)) NA_integer_ else as.integer(val)
      }),
      image_id = purrr::map_int(annotations_list, ~{
        val <- .x$image_id
        if (is.null(val)) NA_integer_ else as.integer(val)
      }),
      category_id = purrr::map_int(annotations_list, ~{
        val <- .x$category_id
        if (is.null(val)) NA_integer_ else as.integer(val)
      }),
      area = purrr::map_dbl(annotations_list, function(a) {
        bbox <- a$bbox
        if (is.null(bbox) || length(bbox) < 4) return(NA_real_)
        bbox[[3]] * bbox[[4]]
      })
    )
  } else {
    tibble::tibble(id = integer(), image_id = integer(), category_id = integer(), area = numeric())
  }

  categories_list <- anno$categories %||% list()
  categories_tbl <- if (length(categories_list) > 0) {
    tibble::tibble(
      id = purrr::map_int(categories_list, ~{
        val <- .x$id
        if (is.null(val)) NA_integer_ else as.integer(val)
      }),
      name = purrr::map_chr(categories_list, ~{
        val <- .x$name
        if (is.null(val)) NA_character_ else as.character(val)
      })
    )
  } else {
    tibble::tibble(id = integer(), name = character())
  }

  n_images <- nrow(images_tbl)
  n_annotations <- nrow(annotations_tbl)
  n_categories <- nrow(categories_tbl)

  annos_per_image_tbl <- if (n_annotations > 0) {
    annotations_tbl |>
      dplyr::count(image_id, name = "annotation_count")
  } else {
    tibble::tibble(image_id = integer(), annotation_count = integer())
  }

  category_counts_tbl <- if (n_annotations > 0) {
    annotations_tbl |>
      dplyr::count(category_id, name = "count")
  } else {
    tibble::tibble(category_id = integer(), count = integer())
  }

  category_summary_tbl <- categories_tbl |>
    dplyr::left_join(category_counts_tbl, by = c("id" = "category_id")) |>
    dplyr::mutate(
      count = dplyr::coalesce(count, 0L)
    ) |>
    dplyr::arrange(dplyr::desc(count))

  annos_per_image_vec <- annos_per_image_tbl$annotation_count
  bbox_areas <- annotations_tbl$area

  mean_annos <- if (length(annos_per_image_vec) > 0) round(mean(annos_per_image_vec), 1) else 0
  median_annos <- if (length(annos_per_image_vec) > 0) stats::median(annos_per_image_vec) else 0
  max_annos <- if (length(annos_per_image_vec) > 0) max(annos_per_image_vec) else 0

  bbox_summary_tbl <- if (length(bbox_areas) > 0) {
    summary_vals <- summary(bbox_areas)
    tibble::tibble(stat = names(summary_vals), value = as.numeric(summary_vals))
  } else {
    tibble::tibble(stat = character(0), value = numeric(0))
  }

  if (isTRUE(verbose)) {
    cli::cli_dl(c(
      "Total images" = n_images,
      "Total annotations" = n_annotations,
      "Annotations per image (mean)" = mean_annos,
      "Annotations per image (median)" = median_annos,
      "Annotations per image (max)" = max_annos,
      "Categories" = n_categories
    ))

    if (nrow(category_summary_tbl) > 0) {
      cli::cli_h3("Category Counts")
      cli::cli_dl(stats::setNames(as.character(category_summary_tbl$count), category_summary_tbl$name))
    }

    cli::cli_h3("Object Size Distribution")
    if (nrow(bbox_summary_tbl) > 0) {
      formatted <- stats::setNames(format(round(bbox_summary_tbl$value, 1), trim = TRUE), bbox_summary_tbl$stat)
      cli::cli_dl(formatted)
    } else {
      cli::cli_alert_info("No bounding boxes available")
    }
  }

  warnings <- list()

  sparse_images <- if (length(annos_per_image_vec) == 0) 0 else sum(annos_per_image_vec < 10)
  if (sparse_images > n_images * 0.1) {
    msg <- paste0(sparse_images, " images have <10 annotations (",
                  if (n_images > 0) round(100 * sparse_images / n_images, 1) else 0, "%)")
    if (isTRUE(verbose)) cli::cli_alert_warning(msg)
    warnings <- c(warnings, list(sparse_annotations = msg))
  }

  dense_threshold <- 100
  dense_images <- if (length(annos_per_image_vec) == 0) 0 else sum(annos_per_image_vec > dense_threshold)
  if (dense_images > 0) {
    msg <- paste0(dense_images, " images have >", dense_threshold,
                  " annotations (max: ", max_annos, ")")
    if (isTRUE(verbose)) cli::cli_alert_info(msg)
  }

  tiny_threshold <- 100
  tiny_objects <- if (length(bbox_areas) == 0) 0 else sum(bbox_areas < tiny_threshold, na.rm = TRUE)
  if (tiny_objects > n_annotations * 0.2) {
    msg <- paste0(tiny_objects, " objects are very small (<", tiny_threshold,
                  "px^2) - ", if (n_annotations > 0) round(100 * tiny_objects / n_annotations, 1) else 0, "%")
    if (isTRUE(verbose)) cli::cli_alert_warning(msg)
    warnings <- c(warnings, list(tiny_objects = msg))
  }

  if (!is.null(image_dir)) {
    image_paths <- fs::path(image_dir, images_tbl$file_name)
    missing <- if (length(image_paths) == 0) 0 else sum(!fs::file_exists(image_paths))
    if (missing > 0) {
      msg <- paste0(missing, " image file{?s} not found in ", image_dir)
      if (isTRUE(verbose)) cli::cli_alert_danger(msg)
      warnings <- c(warnings, list(missing_images = msg))
    } else if (isTRUE(verbose)) {
      cli::cli_alert_success("All image files found")
    }
  }

  result <- list(
    split = label,
    n_images = n_images,
    n_annotations = n_annotations,
    n_categories = n_categories,
    annos_per_image_tbl = annos_per_image_tbl,
    bbox_summary = bbox_summary_tbl,
    category_counts_tbl = category_summary_tbl,
    annotations = annotations_tbl,
    warnings = warnings,
    summary_stats = list(
      mean_annos_per_image = if (length(annos_per_image_vec) > 0) mean(annos_per_image_vec) else NA_real_,
      median_annos_per_image = if (length(annos_per_image_vec) > 0) stats::median(annos_per_image_vec) else NA_real_,
      max_annos_per_image = if (length(annos_per_image_vec) > 0) max_annos else NA_real_,
      mean_bbox_area = if (length(bbox_areas) > 0) mean(bbox_areas, na.rm = TRUE) else NA_real_,
      median_bbox_area = if (length(bbox_areas) > 0) stats::median(bbox_areas, na.rm = TRUE) else NA_real_
    ),
    images = images_tbl,
    categories = categories_tbl
  )

  invisible(result)
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


# ============================================================================
# Dataset Pinning Functions
# ============================================================================

#' Pin a dataset to a board
#'
#' Pins a COCO-format dataset directory to a pins board for versioning and reuse.
#' The dataset is compressed as tar.gz before pinning.
#'
#' @param data_dir Path to dataset directory, or a `.tar.gz` / `.tgz` archive
#'   (transparently extracted before re-pinning).
#' @param dataset_id Name for the pinned dataset
#' @param board Pins board (NULL = local board at .petrographer/)
#' @param metadata Optional metadata list
#' @export
pin_dataset <- function(data_dir, dataset_id, board = NULL, metadata = list()) {
  # dataset_id is interpolated into filenames and (for HPC training) remote
  # shell commands, so restrict it to the same safe alphabet we use for
  # model_id.
  if (!grepl("^[A-Za-z0-9._-]{1,64}$", dataset_id)) {
    cli::cli_abort("Invalid dataset_id. Use only letters, numbers, ., _, - (max 64 chars).")
  }

  data_dir <- .resolve_data_dir(data_dir)

  # Validate dataset structure FIRST and capture statistics
  cli::cli_alert_info("Validating dataset structure...")
  dataset_info <- validate_dataset(data_dir, quiet = TRUE)

  if (is.null(board)) {
    board <- .get_dataset_board()
  }

  # Create tar.gz in temp directory
  cli::cli_alert_info("Compressing dataset...")
  temp_dir <- fs::path_temp(paste0("pin_dataset_", dataset_id))
  fs::dir_create(temp_dir)
  on.exit(fs::dir_delete(temp_dir), add = TRUE)

  tar_file <- fs::path(temp_dir, paste0(dataset_id, ".tar.gz"))

  # Find all dataset splits (train, valid, test) that exist
  # Include test/ if present (required by RF-DETR)
  dataset_splits <- c("train", "valid", "test")
  existing_splits <- dataset_splits[fs::dir_exists(fs::path(data_dir, dataset_splits))]

  cli::cli_alert_info("Archiving splits: {paste(existing_splits, collapse = ', ')}")

  # Use R's tar() with -C flag to tar dataset subdirectories
  # Result: tar contains train/, valid/, test/ (if present) at root
  tar_result <- utils::tar(
    tarfile = tar_file,
    files = existing_splits,
    compression = "gzip",
    tar = sprintf("tar -C %s", shQuote(data_dir))
  )

  if (tar_result != 0) {
    cli::cli_abort("Failed to create tar.gz archive")
  }

  # Add dataset statistics to metadata
  metadata$pinned <- Sys.time()
  metadata$compressed <- TRUE
  metadata$original_path <- as.character(data_dir)
  metadata$size_mb <- dataset_info$size_mb
  metadata$splits <- existing_splits

  # Add image counts per split
  for (split in existing_splits) {
    if (!is.null(dataset_info$splits[[split]])) {
      metadata[[paste0("num_images_", split)]] <- dataset_info$splits[[split]]$images
    }
  }

  # Pin the tar.gz file
  cli::cli_alert_info("Pinning to board...")
  pins::pin_upload(board, tar_file, name = dataset_id, metadata = metadata)

  cli::cli_alert_success("Pinned dataset {.strong {dataset_id}} ({format(fs::file_size(tar_file))})")
  invisible(dataset_id)
}

#' List pinned datasets
#'
#' Lists all pinned datasets on a board.
#'
#' @param board Pins board (NULL = local board, "local" = local board)
#' @export
list_datasets <- function(board = "local") {
  if (identical(board, "local") || is.null(board)) {
    board <- .get_dataset_board()
  }

  # Get all pins
  all_pins <- pins::pin_list(board)

  # Filter for dataset pins (could use naming convention if needed)
  # For now, return all pins - user can inspect with pin_meta()
  all_pins
}

#' Delete auto-pinned temp datasets
#'
#' When [train_model()] is called with `data_dir` rather than `dataset_id`, it
#' auto-pins the dataset as `_temp_<timestamp>` (tagged `temp = TRUE` in
#' metadata) so training is reproducible. Those pins persist in
#' `.petrographer/datasets/` and accumulate over time — each is a tar.gz of
#' the full dataset. This helper removes them.
#'
#' @param board Pins board (`NULL`/`"local"` = local dataset board).
#' @param confirm If `TRUE`, prompt before deleting. Defaults to `TRUE` in
#'   interactive sessions, `FALSE` otherwise (e.g. scripted cleanup).
#' @return Character vector of deleted dataset ids (invisibly).
#' @export
clean_temp_datasets <- function(board = NULL, confirm = interactive()) {
  if (is.null(board) || identical(board, "local")) {
    board <- .get_dataset_board()
  }

  all_pins <- pins::pin_list(board)
  temp_pins <- character(0)
  for (id in all_pins) {
    meta <- tryCatch(pins::pin_meta(board, id), error = function(e) NULL)
    if (!is.null(meta) && isTRUE(meta$user$temp)) {
      temp_pins <- c(temp_pins, id)
    }
  }

  if (length(temp_pins) == 0) {
    cli::cli_alert_info("No temp datasets to clean")
    return(invisible(character(0)))
  }

  cli::cli_alert_info(
    "Found {length(temp_pins)} temp dataset{?s}: {.val {temp_pins}}"
  )

  if (isTRUE(confirm)) {
    answer <- utils::askYesNo("Delete them?", default = FALSE)
    if (!isTRUE(answer)) {
      cli::cli_alert_info("Cancelled")
      return(invisible(character(0)))
    }
  }

  for (id in temp_pins) {
    pins::pin_delete(board, id)
  }
  cli::cli_alert_success("Deleted {length(temp_pins)} temp dataset{?s}")
  invisible(temp_pins)
}

#' Get path to pinned dataset
#'
#' Returns filesystem path to a pinned dataset tar.gz file.
#' The tar.gz should be extracted at training time.
#'
#' @param dataset_id Dataset name
#' @param board Pins board (or board object)
#' @param version Specific version to retrieve (NULL = latest)
#' @return Path to dataset tar.gz file
#' @export
get_dataset_path <- function(dataset_id, board = "local", version = NULL) {
  if (identical(board, "local") || is.null(board)) {
    board <- .get_dataset_board()
  }

  # Get pin paths (returns vector of file paths)
  paths <- pins::pin_download(board, dataset_id, version = version)

  # Find the tar.gz file
  tar_files <- paths[grepl("\\.tar\\.gz$", paths)]

  if (length(tar_files) == 0) {
    cli::cli_abort("No tar.gz file found in pinned dataset {.val {dataset_id}}")
  }

  if (length(tar_files) > 1) {
    cli::cli_warn("Multiple tar.gz files found, using first: {.path {tar_files[1]}}")
  }

  as.character(tar_files[1])
}
