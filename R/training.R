# ============================================================================
# Model Training Functions
# Simplified RF-DETR training for dense, scale-diverse thin-section microscopy.
# R handles configuration and calls RF-DETR's built-in training loop.
# ============================================================================

# Utility function
`%||%` <- function(x, y) if (is.null(x)) y else x

#' Train a new petrography detection model
#'
#' Orchestrates local or HPC training using RF-DETR.
#' Models are automatically pinned to the local board (.petrographer/) for versioning.
#'
#' Training mode (local vs HPC) is auto-detected based on `hipergator` configuration.
#' For HPC training, call `hipergator::hpg_configure()` before `train_model()` to set
#' connection details (host, user, base_dir).
#'
#' @param dataset_id Name of pinned dataset to use for training (preferred).
#' @param data_dir Path to dataset directory (alternative to dataset_id; will be auto-pinned with temp ID).
#' @param model_id Name for the trained model (used for pins). Defaults to `dataset_id` if not provided.
#' @param model_variant RF-DETR model variant. Detection: "nano" (default), "small", "medium", "large". Segmentation: "seg_nano", "seg_small", "seg_medium", "seg_large", "seg_xlarge", "seg_2xlarge". Legacy: "seg_preview".
#' @param resolution Image resolution for training. Auto-detected from variant if not specified.
#' @param epochs Number of training epochs. Default: 10.
#' @param batch_size Batch size for training. If NA (default), uses 2.
#' @param grad_accum_steps Gradient accumulation steps. If NA (default), auto-calculated as 16 / batch_size for effective batch size of 16.
#' @param learning_rate Learning rate. If NULL (default), uses model default.
#' @param device Device for local training: 'cpu', 'cuda', or 'mps' (default: 'cuda').
#' @param use_amp Use automatic mixed precision training (default: TRUE for CUDA, FALSE otherwise). Reduces memory usage by ~40%.
#' @param amp_dtype AMP dtype: 'bf16' (recommended for modern GPUs) or 'fp16' (default: 'bf16').
#' @param gradient_checkpointing Enable gradient checkpointing (default: FALSE). Reduces memory usage by ~30%.
#' @param num_workers Number of data loading workers (default: 8).
#' @param time_hours Time limit for HPC training in hours (default: 3). Examples: 4 = 4 hours, 0.5 = 30 minutes, 1.5 = 1.5 hours. Ignored for local training.
#' @param validate_every Validate every N epochs (default: 1). Set to NULL to use model default.
#' @param early_stopping_patience Stop training if validation loss doesn't improve for N epochs (default: 10). Set to NULL to disable early stopping.
#' @return Model ID (can be loaded with `from_pretrained(model_id)`).
#' @export
train_model <- function(dataset_id = NULL,
                        data_dir = NULL,
                        model_id = NULL,
                        model_variant = "nano",
                        resolution = NULL,
                        epochs = 10,
                        batch_size = NA,
                        grad_accum_steps = NA,
                        learning_rate = NULL,
                        device = "cuda",
                        use_amp = NULL,
                        amp_dtype = "bf16",
                        gradient_checkpointing = NULL,
                        num_workers = NULL,
                        time_hours = 4,
                        validate_every = 2L,
                        early_stopping_patience = NULL) {

  # Validate model variant
  valid_variants <- c("nano", "small", "medium", "large",
                      "seg_nano", "seg_small", "seg_medium", "seg_large",
                      "seg_xlarge", "seg_2xlarge", "seg_preview")
  model_variant <- match.arg(model_variant, valid_variants)
  device <- match.arg(device, c("cpu", "cuda", "mps"))

  # Resolution validation
  if (!is.null(resolution) && resolution %% 32 != 0) {
    cli::cli_abort("resolution must be divisible by 32")
  }

  # Validate AMP dtype
  amp_dtype <- match.arg(amp_dtype, c("bf16", "fp16"))

  # Set smart defaults for memory optimization based on device
  if (is.null(use_amp)) {
    use_amp <- (device == "cuda")  # Enable for CUDA, disable for CPU/MPS
  }
  if (is.null(gradient_checkpointing)) {
    gradient_checkpointing <- FALSE  # Disabled by default
  }

  # Batch size: "auto" lets rfdetr probe GPU, integer sets explicit size, NA
  # defaults to "auto". When batch is explicit, accumulate to reach an effective
  # batch of 16 (i.e. grad_accum = max(1, round(16 / batch))).
  if (is.na(batch_size) || identical(batch_size, "auto")) {
    batch_size <- "auto"
    if (is.na(grad_accum_steps)) grad_accum_steps <- 1L
    cli::cli_alert_info("Using auto batch size (rfdetr will probe GPU capacity)")
  } else {
    if (is.na(grad_accum_steps)) {
      grad_accum_steps <- max(1L, as.integer(round(16 / batch_size)))
      cli::cli_alert_info(
        "Auto-calculated grad_accum_steps = {grad_accum_steps} (effective batch size: {batch_size * grad_accum_steps})"
      )
    }
  }

  # Resolve dataset (must provide one or the other)
  if (!is.null(dataset_id) && !is.null(data_dir)) {
    cli::cli_abort("Provide either {.arg dataset_id} or {.arg data_dir}, not both")
  }
  if (is.null(dataset_id) && is.null(data_dir)) {
    cli::cli_abort("Must provide either {.arg dataset_id} or {.arg data_dir}")
  }

  # dataset_id flows into remote shell commands in the HPC path, so apply the
  # same safe-alphabet check we use for model_id. pin_dataset() also validates
  # at write time; this guards reads of pins that may have been created by
  # other means.
  if (!is.null(dataset_id) &&
      !grepl("^[A-Za-z0-9._-]{1,64}$", dataset_id)) {
    cli::cli_abort("Invalid dataset_id. Use only letters, numbers, ., _, - (max 64 chars).")
  }

  # Default model_id to dataset_id if not provided
  if (is.null(model_id)) {
    if (!is.null(dataset_id)) {
      model_id <- dataset_id
      cli::cli_alert_info("Using model_id = {.val {model_id}} (same as dataset_id)")
    } else {
      cli::cli_abort("Must provide {.arg model_id} when using {.arg data_dir}")
    }
  }

  resolved_dataset <- if (!is.null(dataset_id)) {
    # Use pinned dataset from persistent board
    list(
      id = dataset_id,
      path = get_dataset_path(dataset_id, board = "local"),
      board = .get_dataset_board(),
      is_temp = FALSE
    )
  } else {
    # Auto-pin to local board (persistent, reproducible). Marked temp = TRUE
    # in metadata so clean_temp_datasets() can garbage-collect later —
    # otherwise these tar.gz'd dataset copies accumulate indefinitely.
    temp_id <- paste0("_temp_", format(Sys.time(), "%Y%m%d_%H%M%S"))
    cli::cli_alert_info("Auto-pinning dataset as {.val {temp_id}} (temp; clean with {.fn clean_temp_datasets})")
    pin_dataset(
      data_dir,
      temp_id,
      board = .get_dataset_board(),
      metadata = list(temp = TRUE)
    )
    list(
      id = temp_id,
      path = get_dataset_path(temp_id, board = "local"),
      board = .get_dataset_board(),
      is_temp = TRUE
    )
  }

  # Capture dataset version for reproducibility
  dataset_meta <- pins::pin_meta(resolved_dataset$board, resolved_dataset$id)
  resolved_dataset$version <- dataset_meta$version

  config <- prepare_training_config(
    data_dir = resolved_dataset$path,
    dataset_id = resolved_dataset$id,
    dataset_version = resolved_dataset$version,
    model_id = model_id,
    model_variant = model_variant,
    resolution = resolution,
    epochs = epochs,
    grad_accum_steps = grad_accum_steps,
    learning_rate = learning_rate,
    device = device,
    batch_size = batch_size,
    use_amp = use_amp,
    amp_dtype = amp_dtype,
    gradient_checkpointing = gradient_checkpointing,
    num_workers = num_workers,
    time_hours = time_hours,
    validate_every = validate_every,
    early_stopping_patience = early_stopping_patience
  )

  cli::cli_h1("Model Training")
  cli::cli_h2("Training Configuration")
  cli::cli_dl(config$display$core)
  cli::cli_dl(config$display$extras)
  if (!is.null(config$display$mode)) cli::cli_dl(config$display$mode)


  start_time <- Sys.time()
  model_dir <- if (identical(config$mode, "local")) {
    run_local_training(config)
  } else {
    run_hpc_training(config)
  }
  duration_mins <- round(as.numeric(difftime(Sys.time(), start_time, units = "mins")), 1)
  cli::cli_alert_success("Training completed in {duration_mins} minute{?s}.")

  model_id <- finalize_trained_model(
    model_dir = model_dir,
    config = config,
    duration_mins = duration_mins
  )

  return(model_id)
}


#' Train model locally using available hardware
#' @keywords internal

prepare_training_config <- function(data_dir,
                                    dataset_id,
                                    dataset_version,
                                    model_id,
                                    model_variant,
                                    resolution,
                                    epochs,
                                    grad_accum_steps,
                                    learning_rate,
                                    device,
                                    batch_size,
                                    use_amp,
                                    amp_dtype,
                                    gradient_checkpointing,
                                    num_workers,
                                    time_hours,
                                    validate_every,
                                    early_stopping_patience) {

  # Check if hipergator has valid config to determine training mode
  hpg_cfg <- tryCatch(hipergator::hpg_config(), error = function(e) list(base_dir = NULL))
  training_mode <- if (!is.null(hpg_cfg$base_dir)) {
    # Verify hipergator is installed for HPC mode
    if (!requireNamespace("hipergator", quietly = TRUE)) {
      cli::cli_abort(c(
        "HPC training requires the {.pkg hipergator} package",
        "i" = "Install with: {.code remotes::install_github('flmnh-ai/hipergator')}"
      ))
    }
    "hpc"
  } else {
    "local"
  }

  # Default num_workers: 8 for HPC, 2 for local
  if (is.null(num_workers)) {
    num_workers <- if (training_mode == "hpc") 8L else 2L
  }

  data_dir <- fs::path_abs(fs::path_norm(data_dir))

  run_id <- format(Sys.time(), "%Y%m%d%H%M%S")

  # Skip reading class names from tar.gz (metadata only, not critical for training)
  class_names <- character()

  if (!grepl("^[A-Za-z0-9._-]{1,64}$", model_id)) {
    cli::cli_abort("Invalid model_id. Use only letters, numbers, ., _, - (max 64 chars).")
  }

  # Resolve batch size
  effective_batch_size <- resolve_batch_size(batch_size)

  # Normalize grad_accum_steps. train_model() does this upstream, but callers
  # that hit prepare_training_config() directly (e.g. the ops notebooks) can
  # still pass NA — in that case NA would propagate as the string "NA" onto
  # the Python argv and argparse would reject it. Formula matches train_model():
  # target effective batch = 16.
  if (is.na(grad_accum_steps)) {
    grad_accum_steps <- if (identical(effective_batch_size, "auto")) {
      1L
    } else {
      max(1L, as.integer(round(16 / effective_batch_size)))
    }
  }
  grad_accum_steps <- as.integer(grad_accum_steps)

  # use_amp / gradient_checkpointing defaults. Same story as grad_accum_steps:
  # train_model() defaults NULL, but a direct caller passing NULL would hit
  # `if (NULL)` in train_model_local() and get "argument is of length zero".
  if (is.null(use_amp)) {
    use_amp <- identical(device, "cuda")
  }
  if (is.null(gradient_checkpointing)) {
    gradient_checkpointing <- FALSE
  }

  # Check image sizes and provide batch size recommendations
  if (training_mode == "hpc" || (training_mode == "local" && device == "cuda")) {
    check_batch_size_for_images(data_dir, effective_batch_size, use_amp, gradient_checkpointing)
  }

  # Learning rate info (simple - just use what's provided or model default)
  lr_info <- list(
    lr = learning_rate %||% NA,  # RF-DETR uses model default if NULL
    method = if (is.null(learning_rate)) "Default" else "Manual"
  )

  # Create temp workspace for training
  workspace_dir <- fs::path_temp(paste0("training_", model_id, "_", run_id))
  fs::dir_create(workspace_dir)

  display <- build_training_display(
    model_id = model_id,
    training_mode = training_mode,
    data_dir = data_dir,
    workspace_dir = workspace_dir,
    model_variant = model_variant,
    device = device,
    epochs = epochs,
    grad_accum_steps = grad_accum_steps,
    lr_info = lr_info,
    batch_size = effective_batch_size
  )

  list(
    mode = training_mode,
    display = display,
    data_dir = data_dir,
    dataset_id = dataset_id,
    dataset_version = dataset_version,
    model_id = model_id,
    workspace_dir = workspace_dir,
    epochs = epochs,
    grad_accum_steps = grad_accum_steps,
    learning_rate = lr_info$lr,
    lr_method = lr_info$method,
    model_variant = model_variant,
    resolution = resolution,
    batch_size = effective_batch_size,
    device = device,
    run_id = run_id,
    use_amp = use_amp,
    amp_dtype = amp_dtype,
    gradient_checkpointing = gradient_checkpointing,
    num_workers = num_workers,
    time_hours = time_hours,
    validate_every = validate_every,
    early_stopping_patience = early_stopping_patience
  )
}

resolve_batch_size <- function(batch_size) {
  if (identical(batch_size, "auto")) return("auto")
  effective <- batch_size
  if (is.na(effective)) return("auto")
  if (!is.numeric(effective) || length(effective) != 1 || is.na(effective) || effective < 1) {
    cli::cli_abort("batch_size must be a positive integer, 'auto', or NA.")
  }
  as.integer(effective)
}

check_batch_size_for_images <- function(data_dir, batch_size, use_amp, gradient_checkpointing) {
  # "auto" lets rfdetr probe GPU capacity at runtime; we have no fixed number
  # to compare against, and string-vs-int coercion in `batch_size > N` below
  # would always trigger a spurious warning ("auto" > "4" is TRUE lexically).
  if (identical(batch_size, "auto")) return(invisible(NULL))

  # Sample a few images from training set to detect typical image size
  train_dir <- fs::path(data_dir, "train")

  # Skip if not extracted yet (HPC case)
  if (!fs::dir_exists(train_dir)) {
    return(invisible(NULL))
  }

  # Find image files
  image_files <- fs::dir_ls(train_dir, glob = "*.jpg", recurse = FALSE)
  if (length(image_files) == 0) {
    image_files <- fs::dir_ls(train_dir, glob = "*.png", recurse = FALSE)
  }

  if (length(image_files) == 0) {
    return(invisible(NULL))
  }

  # Sample up to 5 images to check size
  sample_files <- utils::head(image_files, 5)

  tryCatch({
    # Probe image dimensions via inst/python/visualize.py (OpenCV under the hood)
    # - avoids the magick dependency we dropped along with the annotation code.
    img_info <- visualize_py$image_dimensions(as.character(sample_files[1]))
    max_dim <- max(img_info$width, img_info$height)

    # Batch size recommendations based on resolution and memory optimizations
    # These are conservative estimates for RF-DETR nano/small models
    memory_multiplier <- 1.0
    if (use_amp) memory_multiplier <- memory_multiplier * 0.6  # ~40% reduction
    if (gradient_checkpointing) memory_multiplier <- memory_multiplier * 0.7  # ~30% reduction

    # Base recommendations for FP32 without checkpointing
    recommended_batch <- if (max_dim >= 1024) {
      ceiling(4 * memory_multiplier)
    } else if (max_dim >= 768) {
      ceiling(6 * memory_multiplier)
    } else if (max_dim >= 512) {
      ceiling(8 * memory_multiplier)
    } else {
      ceiling(16 * memory_multiplier)
    }

    if (batch_size > recommended_batch) {
      opts <- c("AMP", "gradient checkpointing")[c(use_amp, gradient_checkpointing)]
      opts_str <- if (length(opts) > 0) paste0(" (with ", paste(opts, collapse = " + "), ")") else ""

      cli::cli_alert_warning(
        "Batch size {batch_size} may be too large for {max_dim}px images{opts_str}"
      )
      cli::cli_alert_info(
        "Recommended: batch_size <= {recommended_batch} for {max_dim}px images"
      )

      if (!use_amp || !gradient_checkpointing) {
        missing <- c("AMP", "gradient checkpointing")[c(!use_amp, !gradient_checkpointing)]
        cli::cli_alert_info(
          "Consider enabling {paste(missing, collapse = ' and ')} to reduce memory usage"
        )
      }
    }
  }, error = function(e) {
    # Silently skip if image reading fails
    invisible(NULL)
  })
}

build_training_display <- function(model_id,
                                   training_mode,
                                   data_dir,
                                   workspace_dir,
                                   model_variant,
                                   device,
                                   epochs,
                                   grad_accum_steps,
                                   lr_info,
                                   batch_size) {

  core <- list(
    "Model" = model_id,
    "Data" = as.character(data_dir),
    "Variant" = model_variant,
    "Device" = device,
    "Batch size" = batch_size,
    "Epochs" = epochs,
    "Grad accum" = grad_accum_steps,
    "Learning rate" = if (!is.na(lr_info$lr)) sprintf("%g (%s)", signif(lr_info$lr, 3), lr_info$method) else "Default"
  )

  extras <- list(
    "Workspace" = as.character(workspace_dir)
  )

  mode_details <- NULL
  if (training_mode == "hpc") {
    # Get HPC config from hipergator
    hpg_cfg <- hipergator::hpg_config()
    mode_details <- list(
      "HPC host" = hpg_cfg$host
    )
    if (!is.null(hpg_cfg$user) && nzchar(hpg_cfg$user)) {
      mode_details[["User"]] <- hpg_cfg$user
    }
  }

  list(core = core, extras = extras, mode = mode_details)
}


finalize_trained_model <- function(model_dir, config, duration_mins) {
  # Read class names and num_classes from Python-generated metadata
  python_metadata_path <- fs::path(model_dir, "metadata.json")
  python_metadata <- if (fs::file_exists(python_metadata_path)) {
    jsonlite::read_json(python_metadata_path)
  } else {
    list()
  }

  # Build R metadata (pins versioning + training info)
  # Note: class_names and num_classes are in the Python metadata.json
  metadata <- list(
    dataset_id             = config$dataset_id,
    dataset_version        = config$dataset_version,
    data_dir               = as.character(config$data_dir),
    num_classes            = python_metadata$num_classes %||% NA,
    model_variant          = config$model_variant,
    epochs                 = config$epochs,
    batch_size             = config$batch_size,
    grad_accum_steps       = config$grad_accum_steps,
    learning_rate          = config$learning_rate,
    device                 = config$device,
    training_duration_mins = duration_mins,
    run_id                 = config$run_id,
    version                = config$run_id,
    created                = format(Sys.time(), "%Y-%m-%dT%H:%M:%SZ", tz = "UTC")
  )

  # Add HPC metadata if HPC mode
  if (identical(config$mode, "hpc")) {
    hpg_cfg <- hipergator::hpg_config()
    metadata$training_mode <- paste0("HPC (", hpg_cfg$host, ")")
    metadata$hpc_host <- hpg_cfg$host
  } else {
    metadata$training_mode <- "Local"
  }

  # Build the stable petrographer-owned training summary first. Other R code
  # should prefer this file over re-parsing upstream RF-DETR logs.
  csv_path     <- fs::path(model_dir, "metrics.csv")
  log_path     <- fs::path(model_dir, "log.txt")
  metrics_source <- if (fs::file_exists(csv_path)) csv_path
                   else if (fs::file_exists(log_path)) log_path
                   else NULL
  parsed_metrics <- list(
    training = tibble::tibble(),
    validation = tibble::tibble(),
    classwise = tibble::tibble()
  )
  if (!is.null(metrics_source)) {
    parsed_metrics <- tryCatch(
      parse_metrics(metrics_source),
      error = function(e) parsed_metrics
    )
    metrics <- list(
      validation = if (nrow(parsed_metrics$validation) > 0) as.list(parsed_metrics$validation[nrow(parsed_metrics$validation), , drop = FALSE]) else NULL,
      training = if (nrow(parsed_metrics$training) > 0) as.list(parsed_metrics$training[nrow(parsed_metrics$training), , drop = FALSE]) else NULL
    )
    if (!is.null(metrics)) {
      metadata$metrics <- metrics
    }
  }

  training_summary <- .write_training_summary(
    model_dir = model_dir,
    config = config,
    duration_mins = duration_mins,
    parsed_metrics = parsed_metrics,
    metrics_source = metrics_source,
    python_metadata = python_metadata
  )
  metadata$catalog_summary <- .build_catalog_summary(
    config = config,
    duration_mins = duration_mins,
    python_metadata = python_metadata,
    parsed_metrics = parsed_metrics,
    training_summary = training_summary
  )
  .write_model_manifest(
    model_dir = model_dir,
    config = config,
    duration_mins = duration_mins,
    python_metadata = python_metadata,
    training_summary = training_summary
  )

  # Pin the model to model board
  board <- .get_model_board()
  tryCatch({
    pin_model(
      model_dir = model_dir,
      model_id = config$model_id,
      board = board,
      metadata = metadata
    )
    cli::cli_alert_success("Model saved as {.val {config$model_id}}")
    cli::cli_alert_info("Load with: {.code from_pretrained(\"{config$model_id}\")}")
  }, error = function(e) {
    cli::cli_abort("Failed to pin model: {e$message}")
  })

  # Cleanup temp workspace
  if (fs::dir_exists(config$workspace_dir)) {
    fs::dir_delete(config$workspace_dir)
  }

  invisible(config$model_id)
}

run_local_training <- function(config) {
  train_model_local(
    data_dir                = config$data_dir,
    model_id                = config$model_id,
    model_variant           = config$model_variant,
    resolution              = config$resolution,
    epochs                  = config$epochs,
    batch_size              = config$batch_size,
    grad_accum_steps        = config$grad_accum_steps,
    learning_rate           = config$learning_rate,
    device                  = config$device,
    workspace_dir           = config$workspace_dir,
    use_amp                 = config$use_amp,
    amp_dtype               = config$amp_dtype,
    gradient_checkpointing  = config$gradient_checkpointing,
    num_workers             = config$num_workers,
    validate_every          = config$validate_every,
    early_stopping_patience = config$early_stopping_patience
  )
}

run_hpc_training <- function(config) {
  handle <- submit_hpc_training(config)
  handle <- wait_hpc_training(handle)
  collect_hpc_training(handle)
}

# ============================================================================
# Local and HPC Training Functions
# ============================================================================

#' Train model locally
#' @keywords internal
train_model_local <- function(data_dir, model_id, model_variant, resolution, epochs,
                              batch_size, grad_accum_steps, learning_rate,
                              device, workspace_dir, use_amp, amp_dtype,
                              gradient_checkpointing, num_workers, validate_every,
                              early_stopping_patience) {

  # Extract tar.gz dataset to workspace
  cli::cli_alert_info("Extracting dataset...")

  tar_basename <- fs::path_file(data_dir)
  dataset_id_from_tar <- sub("\\.tar\\.gz$", "", tar_basename)

  dataset_dir <- fs::path(workspace_dir, "dataset", dataset_id_from_tar)
  fs::dir_create(dataset_dir)

  untar_result <- utils::untar(
    tarfile = data_dir,
    exdir = dataset_dir,
    tar = "internal"
  )

  if (untar_result != 0) {
    cli::cli_abort("Failed to extract dataset from {.path {data_dir}}")
  }

  # Read class names from COCO annotations
  train_anno_path <- fs::path(dataset_dir, "train", "_annotations.coco.json")
  class_names <- extract_class_names_from_coco(train_anno_path)

  cli::cli_alert_info("Found {length(class_names)} classes: {paste(class_names, collapse = ', ')}")

  output_dir <- fs::path(workspace_dir, "output")
  fs::dir_create(output_dir)

  # Call Python training script
  python_exe <- reticulate::py_config()$python
  train_script <- system.file("python", "train.py", package = "petrographer")

  if (!fs::file_exists(train_script)) {
    cli::cli_abort("Training script not found. Package installation may be incomplete.")
  }

  args <- c(
    train_script,
    "--dataset-dir", dataset_dir,
    "--output-dir", output_dir,
    "--model-variant", model_variant,
    "--epochs", as.character(epochs),
    "--batch-size", as.character(batch_size),
    "--grad-accum-steps", as.character(grad_accum_steps),
    "--device", device
  )

  if (!is.null(learning_rate) && !is.na(learning_rate)) {
    args <- c(args, "--learning-rate", as.character(learning_rate))
  }

  if (!is.null(resolution)) {
    args <- c(args, "--resolution", as.character(resolution))
  }

  if (gradient_checkpointing) {
    args <- c(args, "--gradient-checkpointing")
  }

  if (use_amp) {
    args <- c(args, "--use-amp", "--amp-dtype", amp_dtype)
  }

  args <- c(args, "--num-workers", as.character(num_workers))

  eval_interval <- validate_every %||% 2L
  args <- c(args,
    "--eval-interval", as.character(eval_interval),
    "--checkpoint-interval", as.character(eval_interval),
    "--eval-max-dets", "500"
  )

  if (!is.null(early_stopping_patience)) {
    args <- c(args, "--early-stopping-patience", as.character(early_stopping_patience))
  }

  cli::cli_alert_info("Starting RF-DETR training...")
  res <- processx::run(python_exe, args = args, echo = TRUE, echo_cmd = FALSE, error_on_status = FALSE)

  if (!identical(res$status, 0L)) {
    cli::cli_abort("Training failed with exit code: {res$status}")
  }

  return(output_dir)
}

#' Train model on HPC
#' @keywords internal
train_model_hpc <- function(data_dir, dataset_id, model_id, run_id, model_variant,
                            resolution, epochs, batch_size, grad_accum_steps, learning_rate,
                            workspace_dir, use_amp, amp_dtype,
                            gradient_checkpointing, num_workers, time_hours, validate_every,
                            early_stopping_patience) {
  config <- list(
    data_dir = data_dir,
    dataset_id = dataset_id,
    model_id = model_id,
    run_id = run_id,
    model_variant = model_variant,
    resolution = resolution,
    epochs = epochs,
    batch_size = batch_size,
    grad_accum_steps = grad_accum_steps,
    learning_rate = learning_rate,
    workspace_dir = workspace_dir,
    use_amp = use_amp,
    amp_dtype = amp_dtype,
    gradient_checkpointing = gradient_checkpointing,
    num_workers = num_workers,
    time_hours = time_hours,
    validate_every = validate_every,
    early_stopping_patience = early_stopping_patience
  )
  handle <- submit_hpc_training(config)
  handle <- wait_hpc_training(handle)
  collect_hpc_training(handle)
}

# Internal: submit HPC training without waiting for completion
submit_hpc_training <- function(config) {
  setup <- hpc_training_setup(
    data_dir = config$data_dir,
    dataset_id = config$dataset_id,
    model_id = config$model_id,
    run_id = config$run_id,
    model_variant = config$model_variant,
    resolution = config$resolution,
    epochs = config$epochs,
    batch_size = config$batch_size,
    grad_accum_steps = config$grad_accum_steps,
    learning_rate = config$learning_rate,
    workspace_dir = config$workspace_dir,
    use_amp = config$use_amp,
    amp_dtype = config$amp_dtype,
    gradient_checkpointing = config$gradient_checkpointing,
    num_workers = config$num_workers,
    time_hours = config$time_hours,
    validate_every = config$validate_every,
    early_stopping_patience = config$early_stopping_patience
  )

  cli::cli_alert_info("Uploading artifacts")
  hpc_upload_artifacts(setup)

  handle <- list(
    config = config,
    setup = setup,
    job = hpc_submit_job(setup),
    submitted_at = Sys.time()
  )
  class(handle) <- "petrographer_hpc_training"
  handle
}

# Internal: wait for a submitted HPC training job
wait_hpc_training <- function(handle) {
  stopifnot(inherits(handle, "petrographer_hpc_training"))
  cli::cli_alert_info("Waiting for HPC job to complete...")
  hipergator::hpg_wait(handle$job)
  handle$completed_at <- Sys.time()
  handle
}

# Internal: download completed HPC training results
collect_hpc_training <- function(handle) {
  stopifnot(inherits(handle, "petrographer_hpc_training"))
  cli::cli_alert_info("Downloading results")
  hpc_download_results(handle$setup)
}

# Internal: build HPC training setup
hpc_training_setup <- function(data_dir, dataset_id, model_id, run_id, model_variant,
                               resolution, epochs, batch_size, grad_accum_steps, learning_rate,
                               workspace_dir, use_amp, amp_dtype,
                               gradient_checkpointing, num_workers, time_hours, validate_every,
                               early_stopping_patience) {

  # Use existing hipergator configuration
  config <- hipergator::hpg_config()
  if (is.null(config$base_dir)) {
    cli::cli_abort(c(
      "HPC base directory not configured",
      "i" = "Call {.code hipergator::hpg_configure(host = ..., base_dir = ...)} before {.code train_model()}"
    ))
  }

  target <- hipergator::hpg_authenticate(quiet = TRUE)

  # HPC resource defaults (single GPU only)
  cpus <- 16
  memory <- "128gb"

  # Convert time_hours to HH:MM:SS format
  hours <- floor(time_hours)
  minutes <- floor((time_hours - hours) * 60)
  seconds <- round(((time_hours - hours) * 60 - minutes) * 60)
  time_str <- sprintf("%02d:%02d:%02d", hours, minutes, seconds)

  gpu_spec <- hipergator::hpg_gpu(count = 1, type = "b200")

  conda_env_path <- "/blue/nicolas.gauthier/share/conda/envs/petrographer"
  resources <- hipergator::hpg_resources(
    cores = cpus,
    memory = memory,
    time = time_str,
    partition = "hpg-b200",
    gpu = gpu_spec,
    conda_env = conda_env_path,
    modules = c("conda")
  )

  # Shared HPC directory structure
  remote_dataset_dir <- fs::path("datasets", dataset_id)
  remote_dataset_tar <- fs::path("datasets", paste0(dataset_id, ".tar.gz"))
  remote_script <- "scripts/train.py"
  remote_output_dir <- fs::path("models", model_id, run_id, "output")

  # Build training arguments (rfdetr >= 1.6.0, PTL-based)
  training_args <- c(
    "--dataset-dir", remote_dataset_dir,
    "--output-dir", remote_output_dir,
    "--model-variant", model_variant,
    "--epochs", as.character(epochs),
    "--batch-size", as.character(batch_size),
    "--grad-accum-steps", as.character(grad_accum_steps)
  )

  if (!is.null(learning_rate) && !is.na(learning_rate)) {
    training_args <- c(training_args, "--learning-rate", as.character(learning_rate))
  }

  if (!is.null(resolution)) {
    training_args <- c(training_args, "--resolution", as.character(resolution))
  }

  if (gradient_checkpointing) {
    training_args <- c(training_args, "--gradient-checkpointing")
  }

  eval_interval <- validate_every %||% 5L
  training_args <- c(training_args,
    "--num-workers", as.character(num_workers),
    "--eval-interval", as.character(eval_interval),
    "--checkpoint-interval", as.character(eval_interval),
    "--eval-max-dets", "500"
  )

  if (!is.null(early_stopping_patience)) {
    training_args <- c(training_args, "--early-stopping-patience", as.character(early_stopping_patience))
  }

  # Extract tar.gz then stage to $SLURM_TMPDIR for fast local I/O
  # HiPerGator: /blue is parallel NFS (slow for small files), $SLURM_TMPDIR is local NVMe
  local_dataset_dir <- paste0("$SLURM_TMPDIR/", basename(remote_dataset_dir))
  extract_cmd <- sprintf(
    "mkdir -p %s && tar -xzf %s -C %s 2>/dev/null && cp -r %s %s && echo 'Dataset staged to local NVMe'",
    remote_dataset_dir,
    remote_dataset_tar,
    remote_dataset_dir,
    remote_dataset_dir,
    "$SLURM_TMPDIR/"
  )
  # Point training at the local copy
  training_args[which(training_args == remote_dataset_dir)] <- local_dataset_dir
  training_cmd <- paste("python", remote_script, paste(training_args, collapse = " "))
  command <- paste(extract_cmd, "&&", training_cmd)

  # Absolute paths for upload/download operations
  remote_base <- config$base_dir
  remote_dataset_tar_abs <- fs::path(remote_base, remote_dataset_tar)
  remote_dataset_dir_abs <- fs::path(remote_base, remote_dataset_dir)
  remote_scripts_abs <- fs::path(remote_base, "scripts")
  remote_output_abs <- fs::path(remote_base, "models", model_id, run_id, "output")

  python_src <- system.file("python", package = "petrographer")

  setup <- list(
    model_id = model_id,
    dataset_id = dataset_id,
    run_id = run_id,
    target = target,
    resources = resources,
    command = command,
    remote = list(
      base = remote_base,
      dataset_tar = remote_dataset_tar_abs,
      dataset_dir = remote_dataset_dir_abs,
      scripts = remote_scripts_abs,
      output = remote_output_abs
    ),
    local = list(
      workspace_dir = workspace_dir,
      download_dir = fs::path(workspace_dir, "hpc_output"),
      data_dir = data_dir
    ),
    python_src = python_src
  )

  setup
}

#' Upload artifacts to HPC
#' @keywords internal
hpc_upload_artifacts <- function(setup) {
  # Ensure remote directory structure exists
  remote_dirs <- c(
    setup$remote$base,
    fs::path(setup$remote$base, "datasets"),
    fs::path(setup$remote$base, "scripts"),
    fs::path_dir(setup$remote$output)
  )
  hipergator::hpg_mkdir(setup$target, remote_dirs, quiet = TRUE)

  # Upload dataset tar.gz
  hipergator::hpg_upload(setup$target, setup$local$data_dir, setup$remote$dataset_tar, quiet = TRUE)

  # Upload training script
  train_py_path <- fs::path(setup$python_src, "train.py")
  hipergator::hpg_upload(setup$target, train_py_path, fs::path(setup$remote$scripts, "train.py"), quiet = TRUE)
}

#' Submit HPC job
#' @keywords internal
hpc_submit_job <- function(setup) {
  cli::cli_alert_info("Submitting SLURM job")
  hipergator::hpg_submit(
    resources = setup$resources,
    command = setup$command,
    job_name = "petrographer_train",
    working_dir = setup$remote$base,
    ssh_target = setup$target,
    quiet = TRUE
  )
}

#' Download results from HPC
#' @keywords internal
hpc_download_results <- function(setup) {
  local_download_dir <- setup$local$download_dir
  fs::dir_create(local_download_dir)

  # Essential files — needed to build a usable pin afterwards.
  required_files <- c("checkpoint_best_total.pth", "metadata.json")
  # Optional files grouped by RF-DETR era - any one complete set is fine.
  # PTL set is produced by RF-DETR >= 1.6.0 (PyTorch Lightning CSVLogger).
  # Native set is produced by RF-DETR <  1.6.0 (native training loop).
  optional_ptl    <- c("metrics.csv", "hparams.yaml")
  optional_native <- c("log.txt", "metrics_plot.png", "results.json")
  optional_files  <- c(optional_ptl, optional_native)

  # Try everything in one pass so that if a run failed before writing the
  # required checkpoint, we still pull down log.txt / metrics.csv locally for
  # diagnosis. Previously we aborted on the first missing required file and
  # the user had to SSH in to see what actually went wrong.
  try_download <- function(file) {
    remote_file <- fs::path(setup$remote$output, file)
    local_file <- fs::path(local_download_dir, file)
    tryCatch({
      hipergator::hpg_download(setup$target, remote_file, local_file, quiet = TRUE)
      TRUE
    }, error = function(e) FALSE)
  }

  got_optional <- character(0)
  for (file in c(optional_files, required_files)) {
    ok <- try_download(file)
    if (ok && file %in% optional_files) got_optional <- c(got_optional, file)
  }

  # Report which metrics artifacts (if any) came back.
  got_ptl    <- intersect(optional_ptl, got_optional)
  got_native <- intersect(optional_native, got_optional)
  if (length(got_ptl) > 0L) {
    cli::cli_alert_info("Metrics artifacts (PTL): {.file {got_ptl}}")
  } else if (length(got_native) > 0L) {
    cli::cli_alert_info("Metrics artifacts (native loop): {.file {got_native}}")
  } else {
    cli::cli_alert_warning(
      "No metrics artifacts found in HPC output - training curves will be unavailable."
    )
  }

  # Only now fail hard on missing required files — and point at anything we
  # did pull so the user knows where to look.
  missing <- required_files[!fs::file_exists(fs::path(local_download_dir, required_files))]
  if (length(missing) > 0) {
    diagnostic_files <- intersect(got_optional, c("log.txt", "metrics.csv", "results.json"))
    cli::cli_abort(c(
      "Required training outputs missing from {.path {setup$remote$output}}: {.val {missing}}",
      "i" = if (length(diagnostic_files) > 0)
        "Downloaded for diagnosis: {.file {fs::path(local_download_dir, diagnostic_files)}}"
      else
        "No diagnostic artifacts were available either - the job may have failed before producing any output."
    ))
  }

  local_download_dir
}

#' Extract class names from COCO JSON
#' @keywords internal
extract_class_names_from_coco <- function(coco_json_path) {
  if (!fs::file_exists(coco_json_path)) {
    cli::cli_abort("COCO annotation file not found: {.path {coco_json_path}}")
  }

  anno <- jsonlite::read_json(coco_json_path)

  if (is.null(anno$categories) || length(anno$categories) == 0) {
    cli::cli_abort("No categories found in COCO annotation file")
  }

  # Extract category names in order of category ID
  categories <- anno$categories
  category_df <- do.call(rbind, lapply(categories, function(cat) {
    data.frame(
      id = cat$id,
      name = cat$name,
      stringsAsFactors = FALSE
    )
  }))

  # Sort by ID to maintain correct order
  category_df <- category_df[order(category_df$id), ]

  return(category_df$name)
}
