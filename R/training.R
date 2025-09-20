# ============================================================================
# Model Training Functions (YAGNI version)
# Stable defaults for dense, scale-diverse thin-section microscopy.
# R computes global batch, batch-scaled LR, workers, and calls a simplified
# Python trainer that is hard-set to AdamW + WarmupCosine + FREEZE_AT=1.
# ============================================================================

#' Train a new petrography detection model
#'
#' Orchestrates local or HPC training using Detectron2. R computes batch size,
#' workers, and a batch-scaled learning rate, then calls the Python trainer.
#' Optionally publishes the resulting model to a pins board.
#'
#' @param data_dir Directory containing `train/` and `valid/` subdirectories with COCO annotations.
#' @param output_name Name for the trained model (used for artifact directories and pin names).
#' @param max_iter Maximum training iterations. Default: 12000.
#' @param learning_rate Base learning rate before auto-scaling by batch (default: 5e-4).
#'   Effective LR = learning_rate * (ims_per_batch / 16) when `auto_scale_lr = TRUE`.
#' @param device Device for local training: 'cpu', 'cuda', or 'mps' (default: 'cuda').
#' @param eval_period Validation evaluation frequency in iterations (default: 500).
#' @param checkpoint_period Checkpoint saving frequency (0 = final only; > 0 = every N iters).
#' @param ims_per_batch Total images per iteration across all GPUs. If NA (default), uses 2 images per GPU.
#' @param auto_scale_lr If TRUE (default), scale LR linearly by global batch vs. reference batch 16.
#' @param num_workers DataLoader workers per process (Detectron2). If NULL (default), set to images per GPU.
#' @param hpc_env Character vector of SLURM script preamble lines (e.g., module loads). If NULL, none added.
#' @param hpc_cpus_per_task Optional SLURM cpus-per-task hint.
#' @param hpc_mem Optional SLURM memory hint.
#' @param gpus Number of GPUs for HPC training (default: 1; ignored for local).
#' @param hpc_host SSH hostname for HPC training (default: `PETROGRAPHER_HPC_HOST`; empty for local).
#' @param hpc_user Username for HPC (default: NULL).
#' @param hpc_base_dir Remote base directory on HPC (default: `PETROGRAPHER_HPC_BASE_DIR`).
#' @param local_output_dir Local directory to save trained model (default: `Detectron2_Models`).
#' @param rsync_mode Data sync mode: 'update' (default) or 'mirror' (adds --delete).
#' @param publish_after_train Whether to publish (pin) the trained model to a board.
#' @param model_board pins board for model storage (if NULL and `publish_after_train=TRUE`, uses [pg_board()]).
#' @param model_description Optional description to include with the published model.
#' @return Path to the trained model directory.
#' @export
train_model <- function(data_dir,
                        output_name,
                        num_classes,
                        max_iter = 12000,
                        learning_rate = 5e-4,
                        device = "cuda",
                        eval_period = 1000,
                        checkpoint_period = 0,
                        ims_per_batch = NA,
                        auto_scale_lr = TRUE,
                        num_workers = NULL,
                        hpc_env = NULL,
                        hpc_cpus_per_task = NULL,
                        hpc_mem = NULL,
                        gpus = 1,
                        hpc_host = Sys.getenv("PETROGRAPHER_HPC_HOST", ""),
                        hpc_user = NULL,
                        hpc_base_dir = Sys.getenv("PETROGRAPHER_HPC_BASE_DIR", ""),
                        local_output_dir = here::here("Detectron2_Models"),
                        rsync_mode = c("update", "mirror"),
                        publish_after_train = FALSE,
                        model_board = NULL,
                        model_description = NULL) {

  cli::cli_h1("Model Training")
  training_mode <- if (is.null(hpc_host) || hpc_host == "") "Local" else paste0("HPC (", hpc_host, ")")
  cli::cli_h2("Training Configuration")

  # ----------------------------
  # Resolve global batch (default: 2 per GPU)
  # ----------------------------
  effective_ims <- ims_per_batch
  if (is.na(effective_ims)) {
    effective_ims <- 2L * max(1L, as.integer(gpus))
  }
  if (!is.numeric(effective_ims) || length(effective_ims) != 1 || is.na(effective_ims) || effective_ims < 1) {
    cli::cli_abort("ims_per_batch must be a positive integer or NA for auto (2 per GPU).")
  }
  effective_ims <- as.integer(effective_ims)

  # ----------------------------
  # Batch-based LR scaling (reference batch = 16 images/iter)
  # ----------------------------
  ref_batch <- 16L
  eff_lr <- if (isTRUE(auto_scale_lr)) {
    learning_rate * (effective_ims / ref_batch)
  } else {
    learning_rate
  }

  # ----------------------------
  # Present configuration
  # ----------------------------
  details <- c(
    "Model name" = output_name,
    "Mode" = training_mode,
    "Data directory" = as.character(fs::path_abs(fs::path_norm(data_dir))),
    "Local output root" = as.character(fs::path_abs(fs::path_norm(local_output_dir))),
    "Device" = device,
    "Max iterations" = max_iter,
    "Learning rate (base)" = learning_rate,
    "LR auto-scale" = if (isTRUE(auto_scale_lr)) glue::glue("ON (by batch) -> {signif(eff_lr, 3)}") else "OFF",
    "Eval period" = eval_period,
    "Checkpoint period" = checkpoint_period,
    "Images per batch (global)" = effective_ims
  )
  if (!is.null(hpc_host) && hpc_host != "") {
    details <- c(details, "HPC host" = hpc_host)
    if (!is.null(hpc_user) && nzchar(hpc_user)) {
      details <- c(details, "HPC user" = hpc_user)
    }
    details <- c(details, "GPUs (HPC)" = gpus)
  }
  cli::cli_dl(details)

  start_time <- Sys.time()

  # ----------------------------
  # Validate inputs & paths
  # ----------------------------
  if (!fs::dir_exists(data_dir)) {
    cli::cli_abort("Data directory not found: {.path {data_dir}}")
  }
  data_dir <- fs::path_abs(fs::path_norm(data_dir))
  local_output_dir <- fs::path_abs(fs::path_norm(local_output_dir))

  if (!grepl("^[A-Za-z0-9._-]{1,64}$", output_name)) {
    cli::cli_abort("Invalid output_name. Use only letters, numbers, ., _, - (max 64 chars).")
  }

  train_dir <- fs::path(data_dir, "train")
  val_dir   <- fs::path(data_dir, "valid")
  if (!fs::dir_exists(train_dir) || !fs::dir_exists(val_dir)) {
    cli::cli_abort("Data directory must contain 'train' and 'valid' subdirectories.")
  }
  if (!fs::file_exists(fs::path(train_dir, "_annotations.coco.json"))) {
    cli::cli_abort("Missing COCO annotations in train directory.")
  }
  if (!fs::file_exists(fs::path(val_dir, "_annotations.coco.json"))) {
    cli::cli_abort("Missing COCO annotations in valid directory.")
  }

  # HPC-specific sanity
  if (!is.null(hpc_host) && hpc_host != "") {
    if (effective_ims %% gpus != 0) {
      cli::cli_abort("ims_per_batch ({effective_ims}) must be divisible by gpus ({gpus}) for multi-GPU training.")
    }
  }

  # ----------------------------
  # Resolve num_workers: default to images per GPU
  # ----------------------------
  if (is.null(num_workers)) {
    per_gpu <- max(1L, as.integer(effective_ims / max(1L, as.integer(gpus))))
    num_workers <- per_gpu
  }
  if (!is.numeric(num_workers) || length(num_workers) != 1 || is.na(num_workers) || num_workers < 1) {
    cli::cli_abort("num_workers must be a positive integer or NULL.")
  }
  num_workers <- as.integer(num_workers)

  # ----------------------------
  # Dispatch local vs HPC
  # ----------------------------
  if (is.null(hpc_host) || hpc_host == "") {
    result <- train_model_local(
      data_dir         = data_dir,
      output_name      = output_name,
      max_iter         = max_iter,
      learning_rate    = eff_lr,
      device           = device,
      eval_period      = eval_period,
      checkpoint_period= checkpoint_period,
      ims_per_batch    = effective_ims,
      num_workers      = num_workers,
      local_output_dir = local_output_dir
    )
  } else {
    result <- train_model_hpc(
      data_dir         = data_dir,
      output_name      = output_name,
      max_iter         = max_iter,
      learning_rate    = eff_lr,      # pass scaled LR
      num_classes      = num_classes,
      eval_period      = eval_period,
      checkpoint_period= checkpoint_period,
      ims_per_batch    = effective_ims,
      num_workers      = num_workers,
      hpc_env          = hpc_env,
      hpc_cpus_per_task= hpc_cpus_per_task,
      hpc_mem          = hpc_mem,
      gpus             = gpus,
      hpc_host         = hpc_host,
      hpc_user         = hpc_user,
      hpc_base_dir     = hpc_base_dir,
      local_output_dir = local_output_dir
    )
  }

  duration_mins <- round(as.numeric(difftime(Sys.time(), start_time, units = "mins")), 1)
  cli::cli_alert_success("Training completed in {duration_mins} minute{?s}.")
  cli::cli_alert_info("Model saved to: {.path {result}}")

  # ----------------------------
  # Optional: publish model
  # ----------------------------
  if (publish_after_train) {
    cli::cli_h2("Model Publishing")

    training_metadata <- list(
      data_dir                 = as.character(data_dir),
      num_classes              = num_classes,
      max_iter                 = max_iter,
      learning_rate_base       = learning_rate,
      auto_scale_lr            = auto_scale_lr,
      effective_learning_rate  = eff_lr,
      ims_per_batch            = effective_ims,
      device                   = device,
      training_duration_mins   = duration_mins,
      training_mode            = training_mode
    )
    if (!is.null(hpc_host) && hpc_host != "") {
      training_metadata$hpc_host <- hpc_host
      training_metadata$gpus     <- gpus
    }

    tryCatch({
      board_to_use <- if (!is.null(model_board)) model_board else pg_board()
      publish_model(
        model_dir = result,
        name      = output_name,
        board     = board_to_use,
        metadata  = training_metadata,
        include_metrics = TRUE
      )
      if (!is.null(model_description)) cli::cli_alert_info(model_description)
    }, error = function(e) {
      cli::cli_warn("Failed to publish model: {e$message}")
    })
  }

  return(result)
}

#' Train model locally using available hardware
#' @keywords internal
train_model_local <- function(data_dir, output_name, max_iter, learning_rate, num_classes, device, eval_period,
                              checkpoint_period, ims_per_batch, num_workers, local_output_dir) {

  output_dir <- fs::path(local_output_dir, output_name)
  fs::dir_create(output_dir)

  cli::cli_h2("Starting Local Training")
  cli::cli_dl(c(
    "Data directory" = data_dir,
    "Output directory" = output_dir,
    "Max iterations" = max_iter,
    "Device" = device
  ))

  python_exe <- reticulate::py_config()$python
  train_script <- system.file("python", "train.py", package = "petrographer")

  args <- c(
    train_script,
    "--dataset-name", paste0(output_name, "_train"),
    "--annotation-json", fs::path(data_dir, "train", "_annotations.coco.json"),
    "--image-root", fs::path(data_dir, "train"),
    "--val-annotation-json", fs::path(data_dir, "valid", "_annotations.coco.json"),
    "--val-image-root", fs::path(data_dir, "valid"),
    "--output-dir", output_dir,
    "--num-workers", as.character(num_workers),
    "--device", device,
    "--num-classes", as.character(num_classes),
    "--max-iter", as.character(max_iter),
    "--learning-rate", as.character(learning_rate),
    "--eval-period", as.character(eval_period),
    "--checkpoint-period", as.character(checkpoint_period),
    "--ims-per-batch", as.character(ims_per_batch)
  )

  display_cmd <- paste(
    shQuote(python_exe),
    paste(vapply(args, shQuote, character(1)), collapse = " ")
  )
  cli::cli_alert_info("Using Python: {.path {python_exe}}")
  cli::cli_alert_info("Running training command")
  cli::cli_code(display_cmd)

  res <- processx::run(python_exe, args = args, echo = TRUE, echo_cmd = FALSE, error_on_status = FALSE)
  if (!identical(res$status, 0L)) {
    cli::cli_abort("Training failed with exit code: {res$status}")
  }

  cli::cli_alert_success("Local training completed successfully!")
  cli::cli_alert_info("Model saved to: {.path {output_dir}}")
  return(output_dir)
}

#' Train model on HPC using SLURM
#' @keywords internal
train_model_hpc <- function(data_dir, output_name, max_iter, learning_rate, num_classes, eval_period, checkpoint_period,
                            ims_per_batch, num_workers, hpc_env, hpc_cpus_per_task, hpc_mem, gpus, hpc_host, hpc_user,
                            hpc_base_dir, local_output_dir) {

  if (is.null(hpc_base_dir) || hpc_base_dir == "") {
    cli::cli_abort("Missing `hpc_base_dir`: set the base path on your HPC system or PETROGRAPHER_HPC_BASE_DIR env var.")
  }

  # Build trainer CLI params for the remote run (relative to job working dir)
  training_params <- c(
    "--dataset-name", paste0(output_name, "_train"),
    "--annotation-json", "data/train/_annotations.coco.json",
    "--image-root", "data/train",
    "--val-annotation-json", "data/valid/_annotations.coco.json",
    "--val-image-root", "data/valid",
    "--output-dir", "output",
    "--num-workers", as.character(num_workers),
    "--max-iter", as.character(max_iter),
    "--learning-rate", as.character(learning_rate),   # already batch-scaled
    "--eval-period", as.character(eval_period),
    "--num-classes", as.character(num_classes),
    "--checkpoint-period", as.character(checkpoint_period),
    "--ims-per-batch", as.character(ims_per_batch),
    "--device", "cuda",
    "--num-gpus", as.character(gpus)
  )

  # Execute HPC workflow
  target <- hpc_authenticate(hpc_host, hpc_user)

  cli::cli_alert_info("Uploading data and submitting SLURM job...")
  job_info <- hpc_sync_and_submit(
    target,
    data_dir,
    hpc_base_dir,
    output_name,
    training_params,
    gpus,
    hpc_env,
    hpc_cpus_per_task,
    hpc_mem
  )


  hpc_monitor(target, job_info$job_id, job_info$remote_base)

  result <- hpc_download(target, job_info$remote_base, output_name, local_output_dir)

  cli::cli_alert_success("HPC training pipeline completed!")
  return(result)
}
