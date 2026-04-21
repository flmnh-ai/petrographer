# ============================================================================
# Manifest and training-summary helpers
# ============================================================================

.manifest_version <- 1L
.training_summary_version <- 1L

.normalize_manifest_categories <- function(categories = NULL,
                                           class_names = NULL,
                                           category_ids = NULL) {
  out <- list()

  if (!is.null(categories) && length(categories) > 0) {
    out <- lapply(seq_along(categories), function(i) {
      category <- categories[[i]]
      list(
        model_id = as.integer(category$model_id %||% (i - 1L)),
        coco_id = as.integer(category$coco_id %||% category$id %||% NA_integer_),
        name = as.character(category$name %||% category$class_name %||% "")
      )
    })
  } else if (!is.null(class_names) && length(class_names) > 0) {
    class_names <- unlist(class_names, use.names = FALSE)
    if (is.null(category_ids)) {
      category_ids <- seq_along(class_names) - 1L
    } else {
      category_ids <- as.integer(unlist(category_ids, use.names = FALSE))
    }

    out <- lapply(seq_along(class_names), function(i) {
      list(
        model_id = as.integer(i - 1L),
        coco_id = as.integer(category_ids[[i]] %||% NA_integer_),
        name = as.character(class_names[[i]])
      )
    })
  }

  out
}

# NOTE: Class naming and ID mapping are still an easy place for upstream
# breakage if RF-DETR or SAHI changes how prediction-time category ids are
# emitted. This helper is the single place to re-verify when that happens.
.manifest_category_map <- function(manifest) {
  categories <- .normalize_manifest_categories(
    categories = manifest$categories,
    class_names = manifest$thing_classes %||% manifest$class_names,
    category_ids = manifest$category_ids
  )

  if (!length(categories)) {
    return(list(
      categories = list(),
      model_id_to_name = stats::setNames(character(), character()),
      model_id_to_coco_id = stats::setNames(integer(), character()),
      coco_id_to_name = stats::setNames(character(), character())
    ))
  }

  model_ids <- vapply(categories, function(cat) as.character(cat$model_id), character(1))
  coco_ids <- vapply(categories, function(cat) as.integer(cat$coco_id), integer(1))
  names_out <- vapply(categories, function(cat) as.character(cat$name), character(1))

  list(
    categories = categories,
    model_id_to_name = stats::setNames(names_out, model_ids),
    model_id_to_coco_id = stats::setNames(coco_ids, model_ids),
    coco_id_to_name = stats::setNames(names_out, as.character(coco_ids))
  )
}

.manifest_category_name_map <- function(manifest) {
  mapping <- .manifest_category_map(manifest)$model_id_to_name
  if (!length(mapping)) NULL else as.list(mapping)
}

.manifest_category_id_map <- function(manifest) {
  .manifest_category_map(manifest)$model_id_to_coco_id
}

.artifact_basename <- function(path) {
  if (is.null(path) || !length(path) || is.na(path)) return(NULL)
  fs::path_file(path)
}

.find_downloaded_artifact <- function(files, relative_path) {
  base <- .artifact_basename(relative_path)
  if (is.null(base)) return(NA_character_)
  match_idx <- match(base, fs::path_file(files))
  if (is.na(match_idx)) NA_character_ else files[[match_idx]]
}

.tibble_to_records <- function(x) {
  if (is.null(x) || !nrow(x)) return(list())
  jsonlite::fromJSON(
    jsonlite::toJSON(x, dataframe = "rows", auto_unbox = TRUE, null = "null"),
    simplifyVector = FALSE
  )
}

.records_to_tibble <- function(x) {
  if (is.null(x) || !length(x)) return(tibble::tibble())
  tibble::as_tibble(dplyr::bind_rows(x))
}

.catalog_metric <- function(validation_metrics, keys) {
  if (is.null(validation_metrics)) return(NA_real_)
  for (key in keys) {
    value <- validation_metrics[[key]]
    if (!is.null(value) && is.numeric(value) && length(value) == 1 && !is.na(value)) {
      return(as.numeric(value))
    }
  }
  NA_real_
}

.build_catalog_summary <- function(config,
                                   duration_mins,
                                   python_metadata,
                                   parsed_metrics,
                                   training_summary) {
  summary_validation <- training_summary$final_metrics$validation
  if (is.list(summary_validation) && length(summary_validation) > 0) {
    summary_validation <- summary_validation[[1]]
  } else {
    summary_validation <- NULL
  }

  validation_metrics <- if (nrow(parsed_metrics$validation) > 0) {
    as.list(parsed_metrics$validation[nrow(parsed_metrics$validation), , drop = FALSE])
  } else {
    summary_validation
  }

  list(
    created = training_summary$training$created_at,
    run_id = config$run_id,
    model_id = config$model_id,
    task = if (startsWith(config$model_variant, "seg")) "segmentation" else "detection",
    model_variant = config$model_variant,
    num_classes = python_metadata$num_classes %||% length(python_metadata$categories %||% list()),
    dataset_id = config$dataset_id,
    dataset_version = config$dataset_version,
    epochs = as.integer(config$epochs),
    batch_size = config$batch_size,
    grad_accum_steps = as.integer(config$grad_accum_steps),
    learning_rate = if (is.na(config$learning_rate)) NULL else config$learning_rate,
    training_duration_mins = duration_mins,
    training_mode = if (identical(config$mode, "hpc")) "HPC" else "Local",
    final_metrics = list(
      ap_50_95 = .catalog_metric(validation_metrics, c("mAP_50_95", "map", "ap", "ema_mAP_50_95")),
      ap_50 = .catalog_metric(validation_metrics, c("mAP_50", "ap50", "ema_mAP_50")),
      ap_75 = .catalog_metric(validation_metrics, c("ap75")),
      f1 = .catalog_metric(validation_metrics, c("F1")),
      epoch = validation_metrics$epoch %||% NA_integer_
    )
  )
}

.build_training_summary <- function(model_dir,
                                    config,
                                    duration_mins,
                                    parsed_metrics,
                                    metrics_source,
                                    python_metadata) {
  results_path <- fs::path(model_dir, "results.json")
  final_results <- if (fs::file_exists(results_path)) {
    jsonlite::read_json(results_path, simplifyVector = FALSE)
  } else {
    NULL
  }

  list(
    training_summary_version = .training_summary_version,
    model = list(
      id = config$model_id,
      variant = config$model_variant,
      task = if (startsWith(config$model_variant, "seg")) "segmentation" else "detection"
    ),
    dataset = list(
      id = config$dataset_id,
      version = config$dataset_version
    ),
    training = list(
      mode = config$mode,
      device = config$device,
      resolution = config$resolution,
      epochs_requested = as.integer(config$epochs),
      batch_size = config$batch_size,
      grad_accum_steps = as.integer(config$grad_accum_steps),
      learning_rate = if (is.na(config$learning_rate)) NULL else config$learning_rate,
      duration_mins = duration_mins,
      run_id = config$run_id,
      created_at = format(Sys.time(), "%Y-%m-%dT%H:%M:%SZ", tz = "UTC")
    ),
    metrics_source = if (is.null(metrics_source)) NULL else fs::path_file(metrics_source),
    final_metrics = list(
      training = if (nrow(parsed_metrics$training) > 0) {
        .tibble_to_records(utils::tail(parsed_metrics$training, 1))
      } else {
        list()
      },
      validation = if (nrow(parsed_metrics$validation) > 0) {
        .tibble_to_records(utils::tail(parsed_metrics$validation, 1))
      } else {
        list()
      }
    ),
    history = list(
      training = .tibble_to_records(parsed_metrics$training),
      validation = .tibble_to_records(parsed_metrics$validation),
      classwise = .tibble_to_records(parsed_metrics$classwise)
    ),
    artifacts = list(
      weights = "checkpoint_best_total.pth",
      metrics = if (is.null(metrics_source)) NULL else fs::path_file(metrics_source),
      results = if (fs::file_exists(results_path)) "results.json" else NULL
    ),
    software_versions = list(
      petrographer = as.character(utils::packageVersion("petrographer")),
      rfdetr = python_metadata$rfdetr_version %||% NULL
    ),
    final_results = final_results
  )
}

.write_training_summary <- function(model_dir,
                                    config,
                                    duration_mins,
                                    parsed_metrics,
                                    metrics_source,
                                    python_metadata) {
  training_summary <- .build_training_summary(
    model_dir = model_dir,
    config = config,
    duration_mins = duration_mins,
    parsed_metrics = parsed_metrics,
    metrics_source = metrics_source,
    python_metadata = python_metadata
  )
  path <- fs::path(model_dir, "training_summary.json")
  jsonlite::write_json(
    training_summary,
    path,
    pretty = TRUE,
    auto_unbox = TRUE,
    null = "null"
  )
  training_summary
}

.build_model_manifest <- function(model_dir,
                                  config,
                                  duration_mins,
                                  python_metadata,
                                  training_summary) {
  categories <- .normalize_manifest_categories(
    categories = python_metadata$categories,
    class_names = python_metadata$thing_classes,
    category_ids = python_metadata$category_ids
  )

  artifacts <- Filter(Negate(is.null), list(
    weights = "checkpoint_best_total.pth",
    training_summary = "training_summary.json",
    metrics_csv = if (fs::file_exists(fs::path(model_dir, "metrics.csv"))) "metrics.csv" else NULL,
    log_txt = if (fs::file_exists(fs::path(model_dir, "log.txt"))) "log.txt" else NULL,
    results_json = if (fs::file_exists(fs::path(model_dir, "results.json"))) "results.json" else NULL,
    hparams_yaml = if (fs::file_exists(fs::path(model_dir, "hparams.yaml"))) "hparams.yaml" else NULL,
    legacy_metadata_json = if (fs::file_exists(fs::path(model_dir, "metadata.json"))) "metadata.json" else NULL
  ))

  list(
    manifest_version = .manifest_version,
    model = list(
      id = config$model_id,
      backend = "rfdetr",
      variant = config$model_variant,
      task = if (startsWith(config$model_variant, "seg")) "segmentation" else "detection",
      weights = artifacts$weights,
      resolution = config$resolution %||% python_metadata$training_resolution %||% NULL,
      max_objects = python_metadata$max_objects %||% NULL
    ),
    # NOTE: Name/id alignment is intentionally explicit here because it is one
    # of the easiest places for silent breakage if upstream class ordering or
    # label normalization changes. Re-check this when upgrading RF-DETR/SAHI.
    categories = categories,
    training = list(
      dataset_id = config$dataset_id,
      dataset_version = config$dataset_version,
      mode = config$mode,
      device = config$device,
      epochs = as.integer(config$epochs),
      batch_size = config$batch_size,
      grad_accum_steps = as.integer(config$grad_accum_steps),
      learning_rate = if (is.na(config$learning_rate)) NULL else config$learning_rate,
      duration_mins = duration_mins,
      run_id = config$run_id,
      created_at = training_summary$training$created_at
    ),
    artifacts = artifacts,
    software_versions = list(
      petrographer = as.character(utils::packageVersion("petrographer")),
      rfdetr = python_metadata$rfdetr_version %||% NULL
    )
  )
}

.write_model_manifest <- function(model_dir,
                                  config,
                                  duration_mins,
                                  python_metadata,
                                  training_summary) {
  manifest <- .build_model_manifest(
    model_dir = model_dir,
    config = config,
    duration_mins = duration_mins,
    python_metadata = python_metadata,
    training_summary = training_summary
  )
  path <- fs::path(model_dir, "manifest.json")
  jsonlite::write_json(
    manifest,
    path,
    pretty = TRUE,
    auto_unbox = TRUE,
    null = "null"
  )
  manifest
}

.validate_model_manifest <- function(manifest, files = NULL) {
  if (is.null(manifest) || !length(manifest)) {
    cli::cli_abort("Model manifest is missing or empty.")
  }

  required_sections <- c("manifest_version", "model", "categories", "training", "artifacts", "software_versions")
  missing_sections <- required_sections[vapply(required_sections, function(field) {
    is.null(manifest[[field]])
  }, logical(1))]
  if (length(missing_sections) > 0) {
    cli::cli_abort("Model manifest is missing required fields: {.val {missing_sections}}")
  }

  if (!identical(as.integer(manifest$manifest_version), .manifest_version)) {
    cli::cli_abort(
      "Unsupported manifest_version {.val {manifest$manifest_version}}. This package expects {.val {.manifest_version}}."
    )
  }

  required_model_fields <- c("backend", "variant", "task", "weights")
  missing_model_fields <- required_model_fields[vapply(required_model_fields, function(field) {
    is.null(manifest$model[[field]]) || !nzchar(as.character(manifest$model[[field]]))
  }, logical(1))]
  if (length(missing_model_fields) > 0) {
    cli::cli_abort("Manifest model section is missing required fields: {.val {missing_model_fields}}")
  }

  category_map <- .manifest_category_map(manifest)
  if (!length(category_map$categories)) {
    cli::cli_abort("Manifest categories are missing or empty.")
  }
  category_fields_ok <- vapply(category_map$categories, function(cat) {
    !is.null(cat$model_id) && !is.null(cat$coco_id) &&
      !is.null(cat$name) && nzchar(as.character(cat$name))
  }, logical(1))
  if (!all(category_fields_ok)) {
    cli::cli_abort("Manifest categories must include model_id, coco_id, and name for every class.")
  }

  required_artifacts <- c("weights", "training_summary")
  missing_artifacts <- required_artifacts[vapply(required_artifacts, function(field) {
    is.null(manifest$artifacts[[field]]) || !nzchar(as.character(manifest$artifacts[[field]]))
  }, logical(1))]
  if (length(missing_artifacts) > 0) {
    cli::cli_abort("Manifest artifacts section is missing required fields: {.val {missing_artifacts}}")
  }

  if (!is.null(files) && length(files) > 0) {
    missing_files <- required_artifacts[vapply(required_artifacts, function(field) {
      is.na(.find_downloaded_artifact(files, manifest$artifacts[[field]]))
    }, logical(1))]
    if (length(missing_files) > 0) {
      cli::cli_abort("Required model artifacts were not downloaded: {.val {missing_files}}")
    }
    if (is.na(.find_downloaded_artifact(files, manifest$model$weights))) {
      cli::cli_abort("Model weights file referenced by the manifest was not downloaded.")
    }
  }

  if (is.null(manifest$software_versions$rfdetr)) {
    cli::cli_warn("Manifest is missing RF-DETR version metadata. Re-check class/id mapping if predictions look mislabeled.")
  }
  if (is.null(manifest$training$dataset_version)) {
    cli::cli_warn("Manifest is missing dataset version metadata. Reproducibility may be limited.")
  }

  invisible(manifest)
}

.read_training_summary <- function(path) {
  if (is.null(path) || is.na(path) || !fs::file_exists(path)) return(NULL)
  jsonlite::read_json(path, simplifyVector = FALSE)
}
