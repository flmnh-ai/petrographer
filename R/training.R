# ============================================================================
# Model Training Functions (YAGNI version)
# Stable defaults for dense, scale-diverse thin-section microscopy.
# R computes global batch, batch-scaled LR, workers, and calls a simplified
# Python trainer that sticks with Detectron2's SGD defaults plus a WarmupCosine schedule and FREEZE_AT=1 baseline.
# ============================================================================

# Utility function
`%||%` <- function(x, y) if (is.null(x)) y else x

#' Suggest learning rate based on batch size and freeze_at
#'
#' Returns head LR. Backbone automatically gets LR * 0.1 via BACKBONE_MULTIPLIER.
#'
#' @param batch_size Images per batch (global across all GPUs)
#' @param freeze_at Backbone freeze stage (0-5)
#' @return List with base_lr (for head) and note
#' @keywords internal
suggest_lr <- function(batch_size, freeze_at = 2) {
  # Base rates for batch_size = 8 (these are HEAD rates)
  # Backbone gets 0.1x these rates automatically
  # More frozen = higher LR (fewer params to update, can step faster)
  base_rates <- c(
    `0` = 0.00100,  # All trainable (backbone gets 0.0001, head gets 0.001)
    `1` = 0.00120,  # Stem frozen (default)
    `2` = 0.00150,  # stem + res2 frozen
    `3` = 0.00180,  # stem + res2 + res3 frozen
    `4` = 0.00200,  # stem + res2 + res3 + res4 frozen
    `5` = 0.00250   # Only head trainable (backbone frozen completely)
  )

  # Get base rate for freeze_at setting
  freeze_key <- as.character(pmin(freeze_at, 5))  # cap at 5
  base_lr <- base_rates[freeze_key]

  # Scale by batch size (linear scaling rule)
  scaled_lr <- base_lr * (batch_size / 8)

  # Return suggested LR (head rate; backbone gets 0.1x automatically)
  list(
    base_lr = scaled_lr,
    note = sprintf("Head LR (backbone gets 0.1x); freeze stages 1-%d", freeze_at)
  )
}

#' Train a new petrography detection model
#'
#' Orchestrates local or HPC training using Detectron2. R computes batch size,
#' workers, and a batch-scaled learning rate, then calls the Python trainer.
#' Models are automatically pinned to the local board (.petrographer/) for versioning.
#'
#' Training mode (local vs HPC) is auto-detected based on `hipergator` configuration.
#' For HPC training, call `hipergator::hpg_configure()` before `train_model()` to set
#' connection details (host, user, base_dir).
#'
#' @param dataset_id Name of pinned dataset to use for training (preferred).
#' @param data_dir Path to dataset directory (alternative to dataset_id; will be auto-pinned with temp ID).
#' @param model_id Name for the trained model (used for pins). Defaults to `dataset_id` if not provided.
#' @param num_classes Number of object classes in your dataset.
#' @param backbone Model backbone: "resnet50" (default), "resnet101", "resnext101", or a full Detectron2 model zoo key.
#' @param freeze_at Freeze backbone up to this stage: 0 (freeze nothing), 1 (freeze stem), 2 (freeze stem + res2; default).
#'   Lower values train more layers = slower but better domain adaptation.
#' @param max_iter Maximum training iterations. Default: 2000.
#' @param learning_rate Learning rate for the detection head. If NULL (default), uses smart
#'   auto-scaling based on `freeze_at` and batch size. Backbone automatically gets 0.1x this rate.
#'   If a number is provided, uses that exact value for the head (backbone still gets 0.1x).
#' @param device Device for local training: 'cpu', 'cuda', or 'mps' (default: 'cuda').
#' @param eval_period Validation evaluation frequency in iterations (default: 500).
#' @param checkpoint_period Checkpoint saving frequency (0 = final only; > 0 = every N iters).
#' @param ims_per_batch Total images per iteration across all GPUs. If NA (default), uses 2 images per GPU.
#' @param num_workers DataLoader workers per process (Detectron2). If NULL (default), set to images per GPU.
#' @param hpc_cpus_per_task Optional SLURM cpus-per-task hint for HPC training.
#' @param hpc_mem Optional SLURM memory hint for HPC training (e.g., "24gb", "96gb").
#' @param gpus Number of GPUs for HPC training (default: 1; ignored for local).
#' @return Model ID (can be loaded with `from_pretrained(model_id)`).
#' @export
train_model <- function(dataset_id = NULL,
                        data_dir = NULL,
                        model_id = NULL,
                        num_classes,
                        backbone = "resnet50",
                        freeze_at = 2,
                        max_iter = 2000,
                        learning_rate = NULL,
                        device = "cuda",
                        eval_period = 1000,
                        checkpoint_period = 0,
                        ims_per_batch = NA,
                        num_workers = NULL,
                        hpc_cpus_per_task = NULL,
                        hpc_mem = NULL,
                        gpus = 1) {

  # Resolve dataset (must provide one or the other)
  if (!is.null(dataset_id) && !is.null(data_dir)) {
    cli::cli_abort("Provide either {.arg dataset_id} or {.arg data_dir}, not both")
  }
  if (is.null(dataset_id) && is.null(data_dir)) {
    cli::cli_abort("Must provide either {.arg dataset_id} or {.arg data_dir}")
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
    # Auto-pin to local board (persistent, reproducible)
    temp_id <- paste0("_temp_", format(Sys.time(), "%Y%m%d_%H%M%S"))
    cli::cli_alert_info("Auto-pinning dataset as {.val {temp_id}} (will persist in .petrographer/datasets/)")
    pin_dataset(data_dir, temp_id, board = .get_dataset_board())
    list(
      id = temp_id,
      path = get_dataset_path(temp_id, board = "local"),
      board = .get_dataset_board(),
      is_temp = FALSE
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
    num_classes = num_classes,
    backbone = backbone,
    freeze_at = freeze_at,
    max_iter = max_iter,
    learning_rate = learning_rate,
    device = device,
    eval_period = eval_period,
    checkpoint_period = checkpoint_period,
    ims_per_batch = ims_per_batch,
    num_workers = num_workers,
    hpc_cpus_per_task = hpc_cpus_per_task,
    hpc_mem = hpc_mem,
    gpus = gpus
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
                                    num_classes,
                                    backbone,
                                    freeze_at,
                                    max_iter,
                                    learning_rate,
                                    device,
                                    eval_period,
                                    checkpoint_period,
                                    ims_per_batch,
                                    num_workers,
                                    hpc_cpus_per_task,
                                    hpc_mem,
                                    gpus) {

  # Check if hipergator has valid config to determine training mode
  hpg_cfg <- tryCatch(hipergator::hpg_config(), error = function(e) list(base_dir = NULL))
  training_mode <- if (!is.null(hpg_cfg$base_dir)) "hpc" else "local"

  data_dir <- fs::path_abs(fs::path_norm(data_dir))

  run_id <- format(Sys.time(), "%Y%m%d%H%M%S")

  # Skip reading class names from tar.gz (metadata only, not critical for training)
  class_names <- character()

  if (!grepl("^[A-Za-z0-9._-]{1,64}$", model_id)) {
    cli::cli_abort("Invalid model_id. Use only letters, numbers, ., _, - (max 64 chars).")
  }

  effective_ims <- resolve_batch_size(ims_per_batch, gpus)
  if (training_mode == "hpc" && (effective_ims %% max(1L, as.integer(gpus)) != 0)) {
    cli::cli_abort("ims_per_batch ({effective_ims}) must be divisible by gpus ({gpus}) for multi-GPU training.")
  }

  lr_info <- resolve_learning_rate(learning_rate, effective_ims, freeze_at)
  worker_count <- resolve_worker_count(num_workers, effective_ims, gpus)

  # Create temp workspace for training
  workspace_dir <- fs::path_temp(paste0("training_", model_id, "_", run_id))
  fs::dir_create(workspace_dir)

  display <- build_training_display(
    model_id = model_id,
    training_mode = training_mode,
    data_dir = data_dir,
    workspace_dir = workspace_dir,
    backbone = backbone,
    freeze_at = freeze_at,
    device = device,
    max_iter = max_iter,
    lr_info = lr_info,
    eval_period = eval_period,
    checkpoint_period = checkpoint_period,
    ims_per_batch = effective_ims,
    gpus = gpus
  )

  list(
    mode = training_mode,
    display = display,
    data_dir = data_dir,
    dataset_id = dataset_id,
    dataset_version = dataset_version,
    model_id = model_id,
    workspace_dir = workspace_dir,
    max_iter = max_iter,
    learning_rate = lr_info$lr,
    num_classes = num_classes,
    backbone = backbone,
    freeze_at = freeze_at,
    eval_period = eval_period,
    checkpoint_period = checkpoint_period,
    ims_per_batch = effective_ims,
    num_workers = worker_count,
    device = device,
    hpc_cpus_per_task = hpc_cpus_per_task,
    hpc_mem = hpc_mem,
    gpus = gpus,
    lr_method = lr_info$method,
    class_names = class_names,
    run_id = run_id
  )
}

resolve_batch_size <- function(ims_per_batch, gpus) {
  effective <- ims_per_batch
  if (is.na(effective)) {
    effective <- 2L * max(1L, as.integer(gpus))
  }
  if (!is.numeric(effective) || length(effective) != 1 || is.na(effective) || effective < 1) {
    cli::cli_abort("ims_per_batch must be a positive integer or NA for auto (2 per GPU).")
  }
  as.integer(effective)
}

resolve_learning_rate <- function(learning_rate, batch_size, freeze_at) {
  if (is.null(learning_rate)) {
    lr_suggestion <- suggest_lr(batch_size = batch_size, freeze_at = freeze_at)
    list(
      lr = lr_suggestion$base_lr,
      method = sprintf("Smart (freeze_at=%d, batch=%d)", freeze_at, batch_size)
    )
  } else {
    list(lr = learning_rate, method = "Manual")
  }
}

resolve_worker_count <- function(num_workers, ims_per_batch, gpus) {
  if (is.null(num_workers)) {
    per_gpu <- max(1L, as.integer(ims_per_batch / max(1L, as.integer(gpus))))
    num_workers <- per_gpu
  }
  if (!is.numeric(num_workers) || length(num_workers) != 1 || is.na(num_workers) || num_workers < 1) {
    cli::cli_abort("num_workers must be a positive integer or NULL.")
  }
  as.integer(num_workers)
}

build_training_display <- function(model_id,
                                   training_mode,
                                   data_dir,
                                   workspace_dir,
                                   backbone,
                                   freeze_at,
                                   device,
                                   max_iter,
                                   lr_info,
                                   eval_period,
                                   checkpoint_period,
                                   ims_per_batch,
                                   gpus) {

  core <- list(
    "Model" = model_id,
    "Data" = as.character(data_dir),
    "Backbone" = backbone,
    "Freeze at" = freeze_at,
    "Device" = device,
    "Images/batch" = ims_per_batch,
    "Max iter" = max_iter,
    "Head LR" = sprintf("%g (%s)", signif(lr_info$lr, 3), lr_info$method)
  )

  extras <- list(
    "Eval period" = eval_period
  )
  if (checkpoint_period > 0) extras[["Checkpoint"]] <- checkpoint_period
  extras[["Workspace"]] <- as.character(workspace_dir)

  mode_details <- NULL
  if (training_mode == "hpc") {
    # Get HPC config from hipergator
    hpg_cfg <- hipergator::hpg_config()
    mode_details <- list(
      "HPC host" = hpg_cfg$host,
      "GPUs" = gpus
    )
    if (!is.null(hpg_cfg$user) && nzchar(hpg_cfg$user)) {
      mode_details[["User"]] <- hpg_cfg$user
    }
  }

  list(core = core, extras = extras, mode = mode_details)
}


finalize_trained_model <- function(model_dir, config, duration_mins) {
  # Build metadata
  metadata <- list(
    dataset_id             = config$dataset_id,
    dataset_version        = config$dataset_version,
    data_dir               = as.character(config$data_dir),
    num_classes            = config$num_classes,
    class_names            = config$class_names,
    backbone               = config$backbone,
    freeze_at              = config$freeze_at,
    max_iter               = config$max_iter,
    learning_rate          = config$learning_rate,
    lr_method              = config$lr_method,
    ims_per_batch          = config$ims_per_batch,
    num_workers            = config$num_workers,
    device                 = config$device,
    eval_period            = config$eval_period,
    checkpoint_period      = config$checkpoint_period,
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
    metadata$gpus <- config$gpus
  } else {
    metadata$training_mode <- "Local"
  }

  # Add metrics if available
  metrics_path <- fs::path(model_dir, "metrics.json")
  if (fs::file_exists(metrics_path)) {
    metrics <- tryCatch({
      parsed <- parse_metrics(metrics_path)
      list(
        validation = if (nrow(parsed$validation) > 0) as.list(parsed$validation[nrow(parsed$validation), , drop = FALSE]) else NULL,
        training = if (nrow(parsed$training) > 0) as.list(parsed$training[nrow(parsed$training), , drop = FALSE]) else NULL
      )
    }, error = function(e) NULL)
    if (!is.null(metrics)) {
      metadata$metrics <- metrics
    }
  }

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

#' Parse detectron2 training log for progress
#'
#' Extracts iteration and loss information from detectron2 log output.
#'
#' @param log_text Character string containing log output
#' @return List with `iter` and `loss` fields, or NULL if parsing fails
#' @keywords internal
parse_detectron2_log <- function(log_text) {
  lines <- strsplit(log_text, "\n")[[1]]

  # Find lines with iteration info: "iter: XXXX"
  iter_lines <- lines[grepl("iter:\\s*\\d+", lines)]
  if (length(iter_lines) == 0) return(NULL)

  # Get latest iteration
  latest <- tail(iter_lines, 1)

  # Extract iteration number
  iter_match <- regexpr("iter:\\s*(\\d+)", latest, perl = TRUE)
  if (iter_match == -1) return(NULL)

  iter_str <- regmatches(latest, iter_match)
  iter <- as.integer(sub("iter:\\s*", "", iter_str))

  # Extract loss if available
  loss <- NA
  loss_match <- regexpr("total_loss:\\s*([0-9.]+)", latest, perl = TRUE)
  if (loss_match != -1) {
    loss_str <- regmatches(latest, loss_match)
    loss <- as.numeric(sub("total_loss:\\s*", "", loss_str))
  }

  list(iter = iter, loss = loss)
}

run_local_training <- function(config) {
  train_model_local(
    data_dir         = config$data_dir,
    model_id         = config$model_id,
    max_iter         = config$max_iter,
    learning_rate    = config$learning_rate,
    num_classes      = config$num_classes,
    backbone         = config$backbone,
    freeze_at        = config$freeze_at,
    device           = config$device,
    eval_period      = config$eval_period,
    checkpoint_period= config$checkpoint_period,
    ims_per_batch    = config$ims_per_batch,
    num_workers      = config$num_workers,
    workspace_dir    = config$workspace_dir
  )
}

run_hpc_training <- function(config) {
  train_model_hpc(
    data_dir         = config$data_dir,
    dataset_id       = config$dataset_id,
    model_id         = config$model_id,
    run_id           = config$run_id,
    max_iter         = config$max_iter,
    learning_rate    = config$learning_rate,
    num_classes      = config$num_classes,
    backbone         = config$backbone,
    freeze_at        = config$freeze_at,
    eval_period      = config$eval_period,
    checkpoint_period= config$checkpoint_period,
    ims_per_batch    = config$ims_per_batch,
    num_workers      = config$num_workers,
    hpc_cpus_per_task= config$hpc_cpus_per_task,
    hpc_mem          = config$hpc_mem,
    gpus             = config$gpus,
    workspace_dir    = config$workspace_dir
  )
}

train_model_local <- function(data_dir, model_id, max_iter, learning_rate, num_classes, backbone, freeze_at, device, eval_period,
                              checkpoint_period, ims_per_batch, num_workers, workspace_dir) {

  # Extract tar.gz dataset to workspace
  # Tar contains train/, valid/ at root (no parent directory)
  # Extract into directory named after dataset_id
  cli::cli_alert_info("Extracting dataset...")

  # Get dataset_id from tar filename (remove .tar.gz)
  tar_basename <- fs::path_file(data_dir)
  dataset_id_from_tar <- sub("\\.tar\\.gz$", "", tar_basename)

  # Create extraction directory
  dataset_dir <- fs::path(workspace_dir, "dataset", dataset_id_from_tar)
  fs::dir_create(dataset_dir)

  # Extract tar directly into dataset_dir
  untar_result <- untar(
    tarfile = data_dir,
    exdir = dataset_dir,
    tar = "internal"
  )

  if (untar_result != 0) {
    cli::cli_abort("Failed to extract dataset from {.path {data_dir}}")
  }

  output_dir <- fs::path(workspace_dir, "output")
  fs::dir_create(output_dir)

  python_exe <- reticulate::py_config()$python
  train_script <- system.file("python", "train.py", package = "petrographer")

  args <- c(
    train_script,
    "--dataset-name", paste0(model_id, "_train"),
    "--annotation-json", fs::path(dataset_dir, "train", "_annotations.coco.json"),
    "--image-root", fs::path(dataset_dir, "train"),
    "--val-annotation-json", fs::path(dataset_dir, "valid", "_annotations.coco.json"),
    "--val-image-root", fs::path(dataset_dir, "valid"),
    "--output-dir", output_dir,
    "--num-workers", as.character(num_workers),
    "--device", device,
    "--num-classes", as.character(num_classes),
    "--backbone", backbone,
    "--freeze-at", as.character(freeze_at),
    "--max-iter", as.character(max_iter),
    "--learning-rate", as.character(learning_rate),
    "--eval-period", as.character(eval_period),
    "--checkpoint-period", as.character(checkpoint_period),
    "--ims-per-batch", as.character(ims_per_batch)
  )

  res <- processx::run(python_exe, args = args, echo = TRUE, echo_cmd = FALSE, error_on_status = FALSE)
  if (!identical(res$status, 0L)) {
    cli::cli_abort("Training failed with exit code: {res$status}")
  }

  return(output_dir)
}

#' Train model on HPC using SLURM
#' @keywords internal
train_model_hpc <- function(data_dir, dataset_id, model_id, run_id, max_iter, learning_rate, num_classes, backbone, freeze_at, eval_period, checkpoint_period,
                            ims_per_batch, num_workers, hpc_cpus_per_task, hpc_mem, gpus, workspace_dir) {

  setup <- hpc_prepare_run(
    data_dir = data_dir,
    dataset_id = dataset_id,
    model_id = model_id,
    run_id = run_id,
    max_iter = max_iter,
    learning_rate = learning_rate,
    num_classes = num_classes,
    backbone = backbone,
    freeze_at = freeze_at,
    eval_period = eval_period,
    checkpoint_period = checkpoint_period,
    ims_per_batch = ims_per_batch,
    num_workers = num_workers,
    hpc_cpus_per_task = hpc_cpus_per_task,
    hpc_mem = hpc_mem,
    gpus = gpus,
    workspace_dir = workspace_dir
  )

  cli::cli_alert_info("Uploading artifacts")
  hpc_upload_artifacts(setup)

  job <- hpc_submit_job(setup)

  # Set up progress bar with log monitoring
  remote_log <- fs::path(setup$remote$output, "log.txt")
  pb_id <- NULL

  # Progress callback
  update_progress <- function(log_text) {
    parsed <- parse_detectron2_log(log_text)
    if (!is.null(parsed) && !is.null(pb_id)) {
      cli::cli_progress_update(
        id = pb_id,
        set = parsed$iter
      )
    }
  }

  # Create progress bar
  pb_id <- cli::cli_progress_bar(
    format = "{cli::pb_bar} {cli::pb_current}/{cli::pb_total} iter | ETA: {cli::pb_eta}",
    total = max_iter,
    clear = FALSE
  )

  # Wait with progress monitoring
  hipergator::hpg_wait(
    job,
    log_path = as.character(remote_log),
    progress_callback = update_progress
  )

  cli::cli_progress_done(id = pb_id)

  cli::cli_alert_info("Downloading results")
  artifact_dir <- hpc_download_results(setup)

  artifact_dir
}

hpc_prepare_run <- function(data_dir, dataset_id, model_id, run_id, max_iter, learning_rate, num_classes, backbone,
                            freeze_at, eval_period, checkpoint_period, ims_per_batch, num_workers,
                            hpc_cpus_per_task, hpc_mem, gpus, workspace_dir) {

  # Use existing hipergator configuration
  config <- hipergator::hpg_config()
  if (is.null(config$base_dir)) {
    cli::cli_abort(c(
      "HPC base directory not configured",
      "i" = "Call {.code hipergator::hpg_configure(host = ..., base_dir = ...)} before {.code train_model()}"
    ))
  }

  target <- hipergator::hpg_authenticate(quiet = TRUE)

  cpus <- if (!is.null(hpc_cpus_per_task)) {
    hpc_cpus_per_task
  } else {
    if (gpus > 1) gpus * 14 else 14
  }

  memory <- if (!is.null(hpc_mem)) {
    hpc_mem
  } else {
    if (gpus > 1) paste0(gpus * 24, "gb") else "24gb"
  }

  gpu_spec <- hipergator::hpg_gpu(count = gpus, type = "b200")

  conda_env_path <- "/blue/nicolas.gauthier/share/conda/envs/petrographer"
  resources <- hipergator::hpg_resources(
    cores = cpus,
    memory = memory,
    time = "02:00:00",
    partition = "hpg-b200",
    gpu = gpu_spec,
    conda_env = conda_env_path,
    modules = c("conda")
  )

  # New shared structure: base_dir/datasets/, base_dir/scripts/, base_dir/models/
  remote_dataset_dir <- fs::path("datasets", dataset_id)
  remote_dataset_tar <- fs::path("datasets", paste0(dataset_id, ".tar.gz"))
  remote_script <- "scripts/train.py"
  remote_output_dir <- fs::path("models", model_id, run_id, "output")

  training_args <- c(
    "--dataset-name", paste0(model_id, "_train"),
    "--annotation-json", fs::path(remote_dataset_dir, "train", "_annotations.coco.json"),
    "--image-root", fs::path(remote_dataset_dir, "train"),
    "--val-annotation-json", fs::path(remote_dataset_dir, "valid", "_annotations.coco.json"),
    "--val-image-root", fs::path(remote_dataset_dir, "valid"),
    "--output-dir", remote_output_dir,
    "--num-workers", as.character(num_workers),
    "--max-iter", as.character(max_iter),
    "--learning-rate", as.character(learning_rate),
    "--eval-period", as.character(eval_period),
    "--num-classes", as.character(num_classes),
    "--backbone", backbone,
    "--freeze-at", as.character(freeze_at),
    "--checkpoint-period", as.character(checkpoint_period),
    "--ims-per-batch", as.character(ims_per_batch),
    "--device", "cuda",
    "--num-gpus", as.character(gpus)
  )

  # Extract tar.gz before training
  # Tar contains train/, valid/ at root - extract into dataset_id directory
  # Suppress warnings with 2>/dev/null
  extract_cmd <- sprintf(
    "mkdir -p %s && tar -xzf %s -C %s 2>/dev/null",
    remote_dataset_dir,
    remote_dataset_tar,
    remote_dataset_dir
  )
  training_cmd <- paste("python", remote_script, paste(training_args, collapse = " "))
  command <- paste(extract_cmd, "&&", training_cmd)

  # Absolute paths for upload/download operations
  remote_base <- config$base_dir
  remote_dataset_tar_abs <- fs::path(remote_base, remote_dataset_tar)
  remote_dataset_dir_abs <- fs::path(remote_base, remote_dataset_dir)
  remote_scripts_abs <- fs::path(remote_base, "scripts")
  remote_output_abs <- fs::path(remote_base, "models", model_id, run_id, "output")

  python_src <- system.file("python", package = "petrographer")

  list(
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
}

hpc_upload_artifacts <- function(setup) {
  # Ensure remote directory structure exists
  remote_dirs <- c(
    setup$remote$base,
    fs::path(setup$remote$base, "datasets"),
    fs::path(setup$remote$base, "scripts"),
    fs::path_dir(setup$remote$output)  # models/{model_id}/{run_id}/
  )
  hipergator::hpg_mkdir(setup$target, remote_dirs, quiet = TRUE)

  # Upload dataset tar.gz to shared location (rsync skips if unchanged)
  hipergator::hpg_upload(setup$target, setup$local$data_dir, setup$remote$dataset_tar, quiet = TRUE)

  # Upload train.py to shared scripts location (rsync skips if unchanged)
  train_py_path <- fs::path(setup$python_src, "train.py")
  hipergator::hpg_upload(setup$target, train_py_path, fs::path(setup$remote$scripts, "train.py"), quiet = TRUE)
}

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

hpc_download_results <- function(setup) {
  local_download_dir <- setup$local$download_dir
  fs::dir_create(local_download_dir)

  # Download only essential files
  essential_files <- c("model_best.pth", "config.yaml", "metadata.json", "metrics.json", "log.txt")

  for (file in essential_files) {
    remote_file <- fs::path(setup$remote$output, file)
    local_file <- fs::path(local_download_dir, file)

    tryCatch({
      hipergator::hpg_download(setup$target, remote_file, local_file, quiet = TRUE)
    }, error = function(e) {
      # metadata.json, metrics.json and log.txt are optional
      if (!file %in% c("metadata.json", "metrics.json", "log.txt")) {
        cli::cli_abort("Failed to download required file {.path {file}}: {e$message}")
      }
    })
  }

  # Verify required files
  required_files <- c("model_best.pth", "config.yaml")
  missing <- required_files[!fs::file_exists(fs::path(local_download_dir, required_files))]
  if (length(missing) > 0) {
    cli::cli_abort("Required files missing after download: {paste(missing, collapse = ', ')}")
  }

  local_download_dir
}
