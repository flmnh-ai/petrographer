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

#' Resolve auto-versioned output name
#' @param base_name Base model name (e.g., "inclusions")
#' @param output_dir Directory where models are stored
#' @param remote_base_dir Optional remote HPC base directory to check for existing versions
#' @param ssh_target Optional SSH target (from hpg_authenticate) for remote checks
#' @return Versioned name (e.g., "inclusions_v2" if "inclusions" exists)
#' @keywords internal
resolve_version <- function(base_name, output_dir, remote_base_dir = NULL, ssh_target = NULL) {

  # Collect versions from local directory
  local_versions <- integer(0)

  if (fs::dir_exists(output_dir)) {
    all_dirs <- fs::dir_ls(output_dir, type = "directory", fail = FALSE)
    pattern <- paste0("^", base_name, "(_v(\\d+))?$")

    existing <- all_dirs %>%
      fs::path_file() %>%
      .[grepl(pattern, .)]

    local_versions <- vapply(existing, function(name) {
      if (name == base_name) return(1L)
      m <- regexec("_v(\\d+)$", name)
      matches <- regmatches(name, m)[[1]]
      if (length(matches) < 2) return(1L)
      as.integer(matches[2])
    }, integer(1))
  }

  # Collect versions from remote directory (HPC)
  remote_versions <- integer(0)

  if (!is.null(remote_base_dir) && !is.null(ssh_target)) {
    tryCatch({
      # List remote directories matching pattern
      pattern_glob <- paste0(base_name, "*")
      list_cmd <- sprintf("ls -d %s/%s 2>/dev/null || true",
                         shQuote(remote_base_dir),
                         shQuote(pattern_glob))

      result <- processx::run(
        "ssh", c(ssh_target, list_cmd),
        timeout = 10, error_on_status = FALSE
      )

      if (result$status == 0 && nzchar(result$stdout)) {
        remote_dirs <- strsplit(trimws(result$stdout), "\n")[[1]]
        remote_names <- basename(remote_dirs)
        pattern <- paste0("^", base_name, "(_v(\\d+))?$")
        matching <- remote_names[grepl(pattern, remote_names)]

        remote_versions <- vapply(matching, function(name) {
          if (name == base_name) return(1L)
          m <- regexec("_v(\\d+)$", name)
          matches <- regmatches(name, m)[[1]]
          if (length(matches) < 2) return(1L)
          as.integer(matches[2])
        }, integer(1))
      }
    }, error = function(e) {
      # Silently ignore remote check errors (SSH might not be available)
    })
  }

  # Combine local and remote versions
  all_versions <- c(local_versions, remote_versions)
  next_version <- if (length(all_versions) == 0) 1L else max(all_versions) + 1L

  if (next_version == 1L) {
    base_name  # First run
  } else {
    paste0(base_name, "_v", next_version)
  }
}

#' Train a new petrography detection model
#'
#' Orchestrates local or HPC training using Detectron2. R computes batch size,
#' workers, and a batch-scaled learning rate, then calls the Python trainer.
#' Optionally publishes the resulting model to a pins board.
#'
#' @param data_dir Directory containing `train/` and `valid/` subdirectories with COCO annotations.
#' @param output_name Name for the trained model (used for artifact directories and pin names).
#'   If `auto_version=TRUE` (default) and this name already exists, automatically appends `_v2`, `_v3`, etc.
#' @param num_classes Number of object classes in your dataset.
#' @param backbone Model backbone: "resnet50" (default), "resnet101", "resnext101", or a full Detectron2 model zoo key.
#' @param freeze_at Freeze backbone up to this stage: 0 (freeze nothing), 1 (freeze stem; default), 2 (freeze stem + res2).
#'   Lower values train more layers = slower but better domain adaptation.
#' @param max_iter Maximum training iterations. Default: 12000.
#' @param learning_rate Learning rate for the detection head. If NULL (default), uses smart
#'   auto-scaling based on `freeze_at` and batch size. Backbone automatically gets 0.1x this rate.
#'   If a number is provided, uses that exact value for the head (backbone still gets 0.1x).
#' @param device Device for local training: 'cpu', 'cuda', or 'mps' (default: 'cuda').
#' @param eval_period Validation evaluation frequency in iterations (default: 500).
#' @param checkpoint_period Checkpoint saving frequency (0 = final only; > 0 = every N iters).
#' @param ims_per_batch Total images per iteration across all GPUs. If NA (default), uses 2 images per GPU.
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
#' @param auto_version Auto-increment version suffix if output_name exists (default: TRUE).
#' @param publish_after_train Whether to publish (pin) the trained model to a board.
#' @param model_board pins board for model storage (if NULL and `publish_after_train=TRUE`, uses [pg_board()]).
#' @param model_description Optional description to include with the published model.
#' @return Path to the trained model directory.
#' @export
train_model <- function(data_dir,
                        output_name,
                        num_classes,
                        backbone = "resnet50",
                        freeze_at = 2,
                        max_iter = 12000,
                        learning_rate = NULL,
                        device = "cuda",
                        eval_period = 1000,
                        checkpoint_period = 0,
                        ims_per_batch = NA,
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
                        auto_version = TRUE,
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
  # Learning rate selection
  # ----------------------------
  if (is.null(learning_rate)) {
    # Smart auto-scaling based on freeze_at and batch size
    lr_suggestion <- suggest_lr(batch_size = effective_ims, freeze_at = freeze_at)
    eff_lr <- lr_suggestion$base_lr
    lr_method <- sprintf("Smart (freeze_at=%d, batch=%d)", freeze_at, effective_ims)
  } else {
    # User-specified LR
    eff_lr <- learning_rate
    lr_method <- "Manual"
  }

  # ----------------------------
  # Present configuration
  # ----------------------------
  details <- c(
    "Model name" = output_name,
    "Mode" = training_mode,
    "Data directory" = as.character(fs::path_abs(fs::path_norm(data_dir))),
    "Local output root" = as.character(fs::path_abs(fs::path_norm(local_output_dir))),
    "Backbone" = backbone,
    "Freeze at" = freeze_at,
    "Device" = device,
    "Max iterations" = max_iter,
    "Learning rate (head)" = sprintf("%g (%s)", signif(eff_lr, 3), lr_method),
    "Learning rate (backbone)" = sprintf("%g (auto: 0.1x head)", signif(eff_lr * 0.1, 3)),
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
  # Auto-versioning (if enabled)
  # ----------------------------
  # For HPC: need to check remote directory too, so defer to train_model_hpc()
  # For local: check now
  if (auto_version && (is.null(hpc_host) || hpc_host == "")) {
    versioned_name <- resolve_version(output_name, local_output_dir)
    if (versioned_name != output_name) {
      cli::cli_alert_info("Auto-versioning enabled: {.val {output_name}} -> {.val {versioned_name}}")
      output_name <- versioned_name
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
      num_classes      = num_classes,
      backbone         = backbone,
      freeze_at        = freeze_at,
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
      backbone         = backbone,
      freeze_at        = freeze_at,
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
      local_output_dir = local_output_dir,
      auto_version     = auto_version
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
      backbone                 = backbone,
      freeze_at                = freeze_at,
      max_iter                 = max_iter,
      learning_rate            = eff_lr,
      lr_method                = lr_method,
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
train_model_local <- function(data_dir, output_name, max_iter, learning_rate, num_classes, backbone, freeze_at, device, eval_period,
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
    "--backbone", backbone,
    "--freeze-at", as.character(freeze_at),
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
train_model_hpc <- function(data_dir, output_name, max_iter, learning_rate, num_classes, backbone, freeze_at, eval_period, checkpoint_period,
                            ims_per_batch, num_workers, hpc_env, hpc_cpus_per_task, hpc_mem, gpus, hpc_host, hpc_user,
                            hpc_base_dir, local_output_dir, auto_version = TRUE) {

  # Set up hipergator configuration if parameters provided
  if (!is.null(hpc_host) || !is.null(hpc_user) || !is.null(hpc_base_dir)) {
    hipergator::hpg_configure(
      host = hpc_host %||% "hpg",
      user = hpc_user,
      base_dir = hpc_base_dir
    )
  }

  # Check that base directory is configured
  config <- hipergator::hpg_config()
  if (is.null(config$base_dir)) {
    cli::cli_abort("Missing `hpc_base_dir`: set the base path on your HPC system or PETROGRAPHER_HPC_BASE_DIR env var.")
  }

  # Authenticate early to enable remote version checking
  target <- hipergator::hpg_authenticate()

  # Auto-versioning with remote directory check
  if (auto_version) {
    versioned_name <- resolve_version(
      base_name = output_name,
      output_dir = local_output_dir,
      remote_base_dir = config$base_dir,
      ssh_target = target
    )
    if (versioned_name != output_name) {
      cli::cli_alert_info("Auto-versioning enabled: {.val {output_name}} -> {.val {versioned_name}}")
      output_name <- versioned_name
    }
  }

  # Auto-scale resources for deep learning
  cpus <- if (!is.null(hpc_cpus_per_task)) {
    hpc_cpus_per_task
  } else {
    if (gpus > 1) gpus * 14 else 14  # B200 requires 14 cores per GPU
  }

  memory <- if (!is.null(hpc_mem)) {
    hpc_mem
  } else {
    if (gpus > 1) paste0(gpus * 24, "gb") else "24gb"  # Auto-scale for deep learning
  }

  # Create GPU and resource specifications
  gpu_spec <- hipergator::hpg_gpu(count = gpus, type = "b200")

  conda_env_path <- "/blue/nicolas.gauthier/share/conda/envs/petrographer"
  resources <- hipergator::hpg_resources(
    cores = cpus,
    memory = memory,
    time = "02:00:00",
    partition = "hpg-b200",
    gpu = gpu_spec,
    conda_env = if (is.null(hpc_env)) conda_env_path else NULL,
    modules = if (!is.null(hpc_env)) NULL else c("conda")
  )

  # Build training command
  training_args <- c(
    "--dataset-name", paste0(output_name, "_train"),
    "--annotation-json", "data/train/_annotations.coco.json",
    "--image-root", "data/train",
    "--val-annotation-json", "data/valid/_annotations.coco.json",
    "--val-image-root", "data/valid",
    "--output-dir", "output",
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

  command <- paste("python src/train.py", paste(training_args, collapse = " "))

  # Note: hpc_env is deprecated with the new hipergator API
  # Use modules parameter in hpg_resources instead
  if (!is.null(hpc_env)) {
    cli::cli_warn("hpc_env parameter is deprecated. Use modules configuration in hipergator package instead.")
  }

  # Set up remote paths (using versioned output_name from above)
  remote_base <- fs::path(config$base_dir, output_name)
  remote_data <- fs::path(remote_base, "data")
  remote_src <- fs::path(remote_base, "src")
  remote_output <- fs::path(remote_base, "output")

  # Check if remote output already exists (shouldn't happen with versioning)
  output_exists <- processx::run(
    "ssh", c(target, paste("test -d", shQuote(remote_output))),
    timeout = 10, error_on_status = FALSE
  )

  if (output_exists$status == 0) {
    cli::cli_warn(c(
      "Remote output directory already exists: {.path {remote_output}}",
      "i" = "This suggests a previous training run with the same version name.",
      "i" = "Versioning should prevent this. Consider manually removing the remote directory or using a different output_name."
    ))
  }

  # Upload data and code
  cli::cli_h2("Uploading to HPC")
  hipergator::hpg_upload(target, data_dir, remote_data)
  hipergator::hpg_upload(target, system.file("python", package = "petrographer"), remote_src)

  # Submit job
  cli::cli_h2("Submitting Job")
  job <- hipergator::hpg_submit(
    resources = resources,
    command = command,
    job_name = "petrographer_train",
    working_dir = remote_base,
    ssh_target = target
  )

  # Wait for completion
  cli::cli_h2("Monitoring Job")
  hipergator::hpg_wait(job)

  # Download results
  cli::cli_h2("Downloading Results")
  local_download_dir <- fs::path(local_output_dir, output_name)

  # Warn if local directory already exists (shouldn't happen with versioning)
  # Only clean if it looks safe (inside expected output directory)
  if (fs::dir_exists(local_download_dir)) {
    # Safety check: ensure it's in the expected location
    if (!fs::path_has_parent(local_download_dir, local_output_dir)) {
      cli::cli_abort(c(
        "Local download directory exists but is outside expected location:",
        "x" = "Expected parent: {.path {local_output_dir}}",
        "x" = "Actual path: {.path {local_download_dir}}",
        "i" = "Refusing to delete for safety. Please manually remove or use different output_name."
      ))
    }

    cli::cli_warn(c(
      "Local directory already exists: {.path {local_download_dir}}",
      "i" = "This may indicate an interrupted previous download.",
      "i" = "Removing to ensure clean download..."
    ))
    fs::dir_delete(local_download_dir)
  }

  hipergator::hpg_download(target, remote_output, local_download_dir)

  artifact_dir <- local_download_dir
  nested_output <- fs::path(local_download_dir, "output")
  if (!fs::file_exists(fs::path(artifact_dir, "model_final.pth")) &&
      fs::dir_exists(nested_output) &&
      fs::file_exists(fs::path(nested_output, "model_final.pth"))) {
    artifact_dir <- nested_output
  }

  required_files <- c("model_final.pth", "config.yaml")
  missing <- required_files[!fs::file_exists(fs::path(artifact_dir, required_files))]
  if (length(missing) > 0) {
    cli::cli_abort("Required files missing after download: {paste(missing, collapse = ', ')}")
  }

  cli::cli_alert_success("HPC training pipeline completed!")
  return(artifact_dir)
}
