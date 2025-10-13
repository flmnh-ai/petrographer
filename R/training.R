# ============================================================================
# Model Training Functions (YAGNI version)
# Stable defaults for dense, scale-diverse thin-section microscopy.
# R computes global batch, batch-scaled LR, workers, and calls a simplified
# Python trainer that sticks with Detectron2's SGD defaults plus a WarmupCosine schedule and FREEZE_AT=1 baseline.
# ============================================================================

# Utility function
`%||%` <- function(x, y) if (is.null(x)) y else x

# Minimal helpers for creating manifests after training ---------------------

create_training_manifest <- function(model_dir,
                                     model_id,
                                     version = NULL,
                                     metadata = list(),
                                     include_metrics = TRUE) {
  model_dir <- fs::path_abs(fs::path_norm(model_dir))
  if (!fs::dir_exists(model_dir)) {
    cli::cli_abort("Model directory not found: {.path {model_dir}}")
  }

  model_final <- fs::path(model_dir, "model_final.pth")
  config_path <- fs::path(model_dir, "config.yaml")
  best_path <- fs::path(model_dir, "model_best.pth")
  metrics_path <- fs::path(model_dir, "metrics.json")

  if (!fs::file_exists(model_final)) {
    cli::cli_abort("Missing file: {.path {model_final}}")
  }
  if (!fs::file_exists(config_path)) {
    cli::cli_abort("Missing file: {.path {config_path}}")
  }

  if (is.null(metadata$preview_image)) {
    preview <- fs::path(model_dir, "preview.png")
    if (fs::file_exists(preview)) metadata$preview_image <- fs::path_file(preview)
  } else if (fs::file_exists(metadata$preview_image)) {
    metadata$preview_image <- fs::path_file(metadata$preview_image)
  }

  manifest <- list(
    schema_version = 1L,
    model_id = model_id,
    version = pg_coalesce(version, metadata$version, pg_model_auto_version()),
    created = pg_model_iso_time(),
    artifacts = list(
      model_final = fs::path_file(model_final),
      config = fs::path_file(config_path)
    ),
    metadata = metadata,
    file_size_bytes = as.numeric(sum(fs::file_size(fs::dir_ls(model_dir, recurse = TRUE, type = "file")), na.rm = TRUE))
  )

  if (fs::file_exists(best_path)) {
    manifest$artifacts$model_best <- fs::path_file(best_path)
  }

  if (include_metrics && fs::file_exists(metrics_path)) {
    manifest$artifacts$metrics <- fs::path_file(metrics_path)
    manifest$metrics <- collect_training_metrics(metrics_path)
  }

  manifest
}

collect_training_metrics <- function(metrics_path) {
  parsed <- parse_metrics(metrics_path)
  val <- if (nrow(parsed$validation) > 0) {
    as.list(parsed$validation[nrow(parsed$validation), , drop = FALSE])
  } else {
    NULL
  }
  train <- if (nrow(parsed$training) > 0) {
    as.list(parsed$training[nrow(parsed$training), , drop = FALSE])
  } else {
    NULL
  }
  list(validation = val, training = train)
}

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
#' @param auto_version Auto-increment version suffix if output_name exists (default: TRUE).
#' @param publish_after_train Whether to publish (pin) the trained model to a board.
#' @param model_board pins board for model storage (if NULL and `publish_after_train=TRUE`, uses [board_user()]).
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
                        auto_version = TRUE,
                        publish_after_train = FALSE,
                        model_board = NULL,
                        model_description = NULL) {

  config <- prepare_training_config(
    data_dir = data_dir,
    output_name = output_name,
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
    hpc_env = hpc_env,
    hpc_cpus_per_task = hpc_cpus_per_task,
    hpc_mem = hpc_mem,
    gpus = gpus,
    hpc_host = hpc_host,
    hpc_user = hpc_user,
    hpc_base_dir = hpc_base_dir,
    local_output_dir = local_output_dir,
    auto_version = auto_version,
    publish_after_train = publish_after_train,
    model_board = model_board,
    model_description = model_description
  )

  cli::cli_h1("Model Training")
  cli::cli_h2("Training Configuration")
  cli::cli_dl(config$display$core)
  cli::cli_dl(config$display$extras)
  if (!is.null(config$display$mode)) cli::cli_dl(config$display$mode)


  start_time <- Sys.time()
  result <- if (identical(config$mode, "local")) {
    run_local_training(config)
  } else {
    run_hpc_training(config)
  }
  duration_mins <- round(as.numeric(difftime(Sys.time(), start_time, units = "mins")), 1)
  cli::cli_alert_success("Training completed in {duration_mins} minute{?s}.")
  cli::cli_alert_info("Model saved to: {.path {result}}")

  finalize_trained_model(
    model_dir = result,
    config = config,
    duration_mins = duration_mins
  )

  return(result)
}


#' Train model locally using available hardware
#' @keywords internal

prepare_training_config <- function(data_dir,
                                    output_name,
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
                                    hpc_env,
                                    hpc_cpus_per_task,
                                    hpc_mem,
                                    gpus,
                                    hpc_host,
                                    hpc_user,
                                    hpc_base_dir,
                                    local_output_dir,
                                    auto_version,
                                    publish_after_train,
                                    model_board,
                                    model_description) {

  training_mode <- if (is.null(hpc_host) || hpc_host == "") "local" else "hpc"

  data_dir <- fs::path_abs(fs::path_norm(data_dir))
  local_output_dir <- fs::path_abs(fs::path_norm(local_output_dir))

  validate_dataset(data_dir, quiet = TRUE)

  run_id <- format(Sys.time(), "%Y%m%d%H%M%S")

  train_categories <- tryCatch({
    anno <- jsonlite::read_json(fs::path(data_dir, "train", "_annotations.coco.json"))
    cats <- anno$categories
    if (is.null(cats)) list() else cats
  }, error = function(e) list())

  class_names <- if (length(train_categories) > 0) {
    vals <- purrr::map_chr(train_categories, function(cat) {
      nm <- cat$name
      if (is.null(nm)) NA_character_ else as.character(nm)
    })
    vals <- vals[!is.na(vals) & nzchar(vals)]
    unique(vals)
  } else {
    character()
  }

  if (!grepl("^[A-Za-z0-9._-]{1,64}$", output_name)) {
    cli::cli_abort("Invalid output_name. Use only letters, numbers, ., _, - (max 64 chars).")
  }

  effective_ims <- resolve_batch_size(ims_per_batch, gpus)
  if (training_mode == "hpc" && (effective_ims %% max(1L, as.integer(gpus)) != 0)) {
    cli::cli_abort("ims_per_batch ({effective_ims}) must be divisible by gpus ({gpus}) for multi-GPU training.")
  }

  lr_info <- resolve_learning_rate(learning_rate, effective_ims, freeze_at)
  worker_count <- resolve_worker_count(num_workers, effective_ims, gpus)

  fs::dir_create(local_output_dir)

  resolved <- resolve_training_output(
    output_name = output_name,
    auto_version = auto_version,
    local_output_dir = local_output_dir,
    training_mode = training_mode
  )

  display <- build_training_display(
    output_name = resolved$output_name,
    training_mode = training_mode,
    data_dir = data_dir,
    local_output_dir = local_output_dir,
    backbone = backbone,
    freeze_at = freeze_at,
    device = device,
    max_iter = max_iter,
    lr_info = lr_info,
    eval_period = eval_period,
    checkpoint_period = checkpoint_period,
    ims_per_batch = effective_ims,
    hpc_host = hpc_host,
    hpc_user = hpc_user,
    gpus = gpus
  )

  list(
    mode = training_mode,
    display = display,
    data_dir = data_dir,
    output_name = resolved$output_name,
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
    hpc_env = hpc_env,
    hpc_cpus_per_task = hpc_cpus_per_task,
    hpc_mem = hpc_mem,
    gpus = gpus,
    hpc_host = hpc_host,
    hpc_user = hpc_user,
    hpc_base_dir = hpc_base_dir,
    local_output_dir = local_output_dir,
    auto_version = auto_version,
    publish_after_train = publish_after_train,
    model_board = model_board,
    model_description = model_description,
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

resolve_training_output <- function(output_name, auto_version, local_output_dir, training_mode) {
  if (isTRUE(auto_version) && identical(training_mode, "local")) {
    versioned_name <- resolve_version(output_name, local_output_dir)
    if (versioned_name != output_name) {
      cli::cli_alert_info("Auto-versioning enabled: {.val {output_name}} -> {.val {versioned_name}}")
      output_name <- versioned_name
    }
  }
  list(output_name = output_name)
}

build_training_display <- function(output_name,
                                   training_mode,
                                   data_dir,
                                   local_output_dir,
                                   backbone,
                                   freeze_at,
                                   device,
                                   max_iter,
                                   lr_info,
                                   eval_period,
                                   checkpoint_period,
                                   ims_per_batch,
                                   hpc_host,
                                   hpc_user,
                                   gpus) {

  core <- list(
    "Model" = output_name,
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
  extras[["Output"]] <- as.character(local_output_dir)

  mode_details <- NULL
  if (training_mode == "hpc") {
    mode_details <- list(
      "HPC host" = hpc_host,
      "GPUs" = gpus
    )
    if (!is.null(hpc_user) && nzchar(hpc_user)) {
      mode_details[["User"]] <- hpc_user
    }
  }

  list(core = core, extras = extras, mode = mode_details)
}


finalize_trained_model <- function(model_dir, config, duration_mins) {
  cli::cli_h2("Model Artifacts")

  metadata <- list(
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
    training_mode          = if (identical(config$mode, "local")) "Local" else paste0("HPC (", config$hpc_host, ")"),
    run_id                 = config$run_id,
    prepared_at            = pg_model_iso_time()
  )

  if (!is.null(config$hpc_host) && nzchar(config$hpc_host)) {
    metadata$hpc_host <- config$hpc_host
    metadata$gpus <- config$gpus
  }
  if (!is.null(config$model_description)) {
    metadata$description <- config$model_description
  }

  manifest <- create_training_manifest(
    model_dir = model_dir,
    model_id = config$output_name,
    version = config$run_id,
    metadata = metadata,
    include_metrics = TRUE
  )
  metrics_path <- fs::path(model_dir, "metrics.json")
  if (fs::file_exists(metrics_path)) {
    manifest$metrics <- collect_training_metrics(metrics_path)
  }

  manifest_path <- fs::path(model_dir, "manifest.json")
  jsonlite::write_json(manifest, manifest_path, auto_unbox = TRUE, pretty = TRUE)
  cli::cli_alert_success("Manifest written: {.path {manifest_path}}")

  if (isTRUE(config$publish_after_train)) {
    cli::cli_h2("Model Publishing")
    board_to_use <- if (!is.null(config$model_board)) config$model_board else pg_board_user()
    tryCatch({
      publish_result <- pg_model_publish(
        model_dir = model_dir,
        model_id  = config$output_name,
        board     = board_to_use,
        include_metrics = TRUE,
        write_manifest = TRUE
      )
      if (!is.null(config$model_description)) cli::cli_alert_info(config$model_description)
      if (!inherits(board_to_use, "pins_board_url")) {
        cli::cli_alert_success("Board manifest updated.")
      }
    }, error = function(e) {
      cli::cli_warn("Failed to publish model: {e$message}")
    })
  }

  invisible(manifest)
}

run_local_training <- function(config) {
  train_model_local(
    data_dir         = config$data_dir,
    output_name      = config$output_name,
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
    local_output_dir = config$local_output_dir
  )
}

run_hpc_training <- function(config) {
  train_model_hpc(
    data_dir         = config$data_dir,
    output_name      = config$output_name,
    max_iter         = config$max_iter,
    learning_rate    = config$learning_rate,
    num_classes      = config$num_classes,
    backbone         = config$backbone,
    freeze_at        = config$freeze_at,
    eval_period      = config$eval_period,
    checkpoint_period= config$checkpoint_period,
    ims_per_batch    = config$ims_per_batch,
    num_workers      = config$num_workers,
    hpc_env          = config$hpc_env,
    hpc_cpus_per_task= config$hpc_cpus_per_task,
    hpc_mem          = config$hpc_mem,
    gpus             = config$gpus,
    hpc_host         = config$hpc_host,
    hpc_user         = config$hpc_user,
    hpc_base_dir     = config$hpc_base_dir,
    local_output_dir = config$local_output_dir,
    auto_version     = config$auto_version
  )
}

train_model_local <- function(data_dir, output_name, max_iter, learning_rate, num_classes, backbone, freeze_at, device, eval_period,
                              checkpoint_period, ims_per_batch, num_workers, local_output_dir) {

  output_dir <- fs::path(local_output_dir, output_name)
  fs::dir_create(output_dir)

  cli::cli_alert_info("Starting local training")
  cli::cli_alert_info("Data: {.path {data_dir}}")
  cli::cli_alert_info("Output: {.path {output_dir}}")

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

  setup <- hpc_prepare_run(
    data_dir = data_dir,
    output_name = output_name,
    max_iter = max_iter,
    learning_rate = learning_rate,
    num_classes = num_classes,
    backbone = backbone,
    freeze_at = freeze_at,
    eval_period = eval_period,
    checkpoint_period = checkpoint_period,
    ims_per_batch = ims_per_batch,
    num_workers = num_workers,
    hpc_env = hpc_env,
    hpc_cpus_per_task = hpc_cpus_per_task,
    hpc_mem = hpc_mem,
    gpus = gpus,
    hpc_host = hpc_host,
    hpc_user = hpc_user,
    hpc_base_dir = hpc_base_dir,
    local_output_dir = local_output_dir,
    auto_version = auto_version
  )

  cli::cli_alert_info("Remote base: {.path {setup$remote$base}}")
  cli::cli_alert_info("Training command")
  cli::cli_code(setup$command)

  cli::cli_alert_info("Uploading artifacts to HPC")
  hpc_upload_artifacts(setup, data_dir)

  job <- hpc_submit_job(setup)

  cli::cli_alert_info("Waiting for job completion")
  hipergator::hpg_wait(job)

  cli::cli_alert_info("Downloading results")
  artifact_dir <- hpc_download_results(setup)

  cli::cli_alert_success("HPC training pipeline completed!")
  artifact_dir
}

hpc_prepare_run <- function(data_dir, output_name, max_iter, learning_rate, num_classes, backbone,
                            freeze_at, eval_period, checkpoint_period, ims_per_batch, num_workers,
                            hpc_env, hpc_cpus_per_task, hpc_mem, gpus, hpc_host, hpc_user,
                            hpc_base_dir, local_output_dir, auto_version) {

  if (!is.null(hpc_host) || !is.null(hpc_user) || !is.null(hpc_base_dir)) {
    hipergator::hpg_configure(
      host = hpc_host %||% "hpg",
      user = hpc_user,
      base_dir = hpc_base_dir
    )
  }

  config <- hipergator::hpg_config()
  if (is.null(config$base_dir)) {
    cli::cli_abort("Missing `hpc_base_dir`: set the base path on your HPC system or PETROGRAPHER_HPC_BASE_DIR env var.")
  }

  target <- hipergator::hpg_authenticate()

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
    conda_env = if (is.null(hpc_env)) conda_env_path else NULL,
    modules = if (!is.null(hpc_env)) NULL else c("conda")
  )

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

  if (!is.null(hpc_env)) {
    cli::cli_warn("hpc_env parameter is deprecated. Use modules configuration in hipergator package instead.")
  }

  remote_base <- fs::path(config$base_dir, output_name)
  remote_data <- fs::path(remote_base, "data")
  remote_src <- fs::path(remote_base, "src")
  remote_output <- fs::path(remote_base, "output")

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

  python_src <- system.file("python", package = "petrographer")

  list(
    output_name = output_name,
    target = target,
    resources = resources,
    command = command,
    remote = list(base = remote_base, data = remote_data, src = remote_src, output = remote_output),
    local = list(output_root = local_output_dir, download_dir = fs::path(local_output_dir, output_name)),
    python_src = python_src
  )
}

hpc_upload_artifacts <- function(setup, data_dir) {
  hipergator::hpg_upload(setup$target, data_dir, setup$remote$data)
  hipergator::hpg_upload(setup$target, setup$python_src, setup$remote$src)
}

hpc_submit_job <- function(setup) {
  cli::cli_alert_info("Submitting SLURM job")
  hipergator::hpg_submit(
    resources = setup$resources,
    command = setup$command,
    job_name = "petrographer_train",
    working_dir = setup$remote$base,
    ssh_target = setup$target
  )
}

hpc_download_results <- function(setup) {
  local_download_dir <- setup$local$download_dir

  if (fs::dir_exists(local_download_dir)) {
    if (!fs::path_has_parent(local_download_dir, setup$local$output_root)) {
      cli::cli_abort(c(
        "Local download directory exists but is outside expected location:",
        "x" = "Expected parent: {.path {setup$local$output_root}}",
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

  hipergator::hpg_download(setup$target, setup$remote$output, local_download_dir)

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

  artifact_dir
}
