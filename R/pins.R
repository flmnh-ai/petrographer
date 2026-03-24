# ============================================================================
# Pins Integration for Model Hub
# ============================================================================

# The public model and dataset hub URLs (served via S3)
.hub_models_url <- "https://flmnh-ai.s3.us-east-1.amazonaws.com/.petrographer/models/"
.hub_datasets_url <- "https://flmnh-ai.s3.us-east-1.amazonaws.com/.petrographer/datasets/"

# Internal: Get dataset board
.get_dataset_board <- function() {
  path <- here::here(".petrographer/datasets")
  fs::dir_create(path, recurse = TRUE)
  pins::board_folder(path, versioned = TRUE)
}

# Internal: Get model board
.get_model_board <- function() {
  path <- here::here(".petrographer/models")
  fs::dir_create(path, recurse = TRUE)
  pins::board_folder(path, versioned = TRUE)
}

#' Load a pretrained RF-DETR model
#'
#' Loads RF-DETR models from local training board, hub, or custom board.
#' By default, checks local models first, then falls back to the public hub.
#'
#' @param model_id Model name (e.g., "shell_v3")
#' @param version Specific version (NULL for latest)
#' @param board Board to load from:
#'   - `NULL` (default): check local first (.petrographer/models/), then hub
#'   - `"local"`: only check locally trained models (.petrographer/models/)
#'   - Custom board object
#' @param device Device: "cpu", "cuda", or "mps"
#' @param confidence Detection threshold
#' @param resolution Input image resolution for RF-DETR (default: auto-detect from variant -
#'   nano=384, small=512, medium=576). Must be divisible by 32.
#' @return PetrographyModel object containing RF-DETR model
#' @export
#' @examples
#' \dontrun{
#' # Smart loading (checks local first, then hub)
#' model <- from_pretrained("my_model")
#'
#' # Force local only
#' model <- from_pretrained("my_model", board = "local")
#'
#' # Force hub only
#' hub_board <- pins::board_url("https://flmnh-ai.s3.us-east-1.amazonaws.com/.petrographer/models/")
#' model <- from_pretrained("public_model", board = hub_board)
#' }
from_pretrained <- function(model_id,
                            version = NULL,
                            board = NULL,
                            device = "cpu",
                            confidence = 0.5,
                            resolution = NULL) {

  # Resolve board
  if (is.null(board)) {
    # Smart default: check local first, then hub
    local_board <- .get_model_board()

    # Check if model exists locally
    local_pins <- tryCatch(
      pins::pin_list(local_board),
      error = function(e) character(0)
    )

    if (model_id %in% local_pins) {
      board <- local_board
      cli::cli_alert_info("Loading from local board")
    } else {
      board <- pins::board_url(Sys.getenv("PETROGRAPHER_HUB_URL", .hub_models_url))
      cli::cli_alert_info("Loading from hub")
    }
  } else if (identical(board, "local")) {
    board <- .get_model_board()
  }
  # else: use provided board object

  # Download model files
  files <- pins::pin_download(board, model_id, version = version)

  # Load metadata for class names
  metadata_path <- files[grepl("metadata\\.json$", files)][1]

  category_mapping <- NULL
  metadata_json <- NULL

  if (!is.na(metadata_path) && fs::file_exists(metadata_path)) {
    metadata_json <- jsonlite::read_json(metadata_path)

    if (!is.null(metadata_json$thing_classes)) {
      # Convert R list to Python dict: {0: "class1", 1: "class2", ...}
      class_names <- unlist(metadata_json$thing_classes)
      category_mapping <- as.list(setNames(class_names, seq_along(class_names) - 1))
      cli::cli_alert_info("Loaded {length(category_mapping)} class names from metadata")
    }
  }

  cli::cli_alert_success("Loading {.strong {model_id}}")

  # Load RF-DETR model
  sahi <- reticulate::import("sahi")
  rfdetr <- reticulate::import("rfdetr")

  # Find model weights
  model_path <- files[grepl("checkpoint_best_total\\.pth$", files)][1]
  if (is.na(model_path)) cli::cli_abort("No model weights found for {.val {model_id}}")

  # Get model variant from metadata
  model_variant <- metadata_json$model_variant %||% "nano"
  is_seg <- startsWith(model_variant, "seg")

  # Map variant to rfdetr class name
  variant_to_class <- c(
    nano = "RFDETRNano", small = "RFDETRSmall", medium = "RFDETRMedium",
    large = "RFDETRLarge",
    seg_preview = "RFDETRSegPreview",
    seg_nano = "RFDETRSegNano", seg_small = "RFDETRSegSmall",
    seg_medium = "RFDETRSegMedium", seg_large = "RFDETRSegLarge",
    seg_xlarge = "RFDETRSegXLarge", seg_2xlarge = "RFDETRSeg2XLarge"
  )

  model_class_name <- variant_to_class[model_variant]
  if (is.na(model_class_name) || !reticulate::py_has_attr(rfdetr, model_class_name)) {
    cli::cli_abort("Unknown or unavailable RF-DETR variant: {.val {model_variant}}")
  }

  model_class <- rfdetr[[model_class_name]]

  # Set resolution based on variant if not specified
  if (is.null(resolution)) {
    resolution <- switch(model_variant,
      nano = 384L, seg_nano = 384L,
      small = 512L, seg_small = 512L,
      medium = 576L, seg_medium = 576L,
      large = 640L, seg_large = 640L,
      seg_xlarge = 704L, seg_2xlarge = 880L,
      seg_preview = 432L,
      512L  # fallback
    )
    cli::cli_alert_info("Using resolution {resolution} for {model_variant} variant")
  }

  # Load model
  cli::cli_alert_info("Loading RF-DETR {model_variant} model...")

  if (is_seg) {
    # Segmentation models: load directly (SAHI doesn't support seg models yet)
    # See https://github.com/obss/sahi/pull/1315
    if (is.null(.petrographer_env$seg_warning_shown)) {
      cli::cli_alert_warning(
        "SAHI does not yet support RF-DETR segmentation models. Using direct inference (no slicing).
         Large images may miss small objects. See {.url https://github.com/obss/sahi/pull/1315}"
      )
      .petrographer_env$seg_warning_shown <- TRUE
    }
    direct_model <- model_class(
      pretrain_weights = as.character(model_path),
      device = device
    )

    # Wrap in PetrographyModel (no SAHI wrapper)
    model <- list(
      direct_model = direct_model,
      sahi_model = NULL,
      model_path = as.character(model_path),
      model_variant = model_variant,
      resolution = as.integer(resolution),
      confidence = confidence,
      device = device,
      is_segmentation = TRUE,
      manifest = metadata_json
    )
  } else {
    # Detection models: load via SAHI for sliced inference
    sahi_model <- sahi$AutoDetectionModel$from_pretrained(
      model_type = 'roboflow',
      model = model_class,
      model_path = as.character(model_path),
      image_size = as.integer(resolution),
      confidence_threshold = confidence,
      device = device,
      category_mapping = if (!is.null(category_mapping)) category_mapping else NULL
    )

    model <- list(
      direct_model = NULL,
      sahi_model = sahi_model,
      model_path = as.character(model_path),
      model_variant = model_variant,
      resolution = as.integer(resolution),
      confidence = confidence,
      device = device,
      is_segmentation = FALSE,
      manifest = metadata_json
    )
  }

  class(model) <- "PetrographyModel"
  return(model)
}

#' Pin a trained RF-DETR model to a board
#'
#' Uploads model files to a pins board for versioning and sharing.
#' Maintainers should call [pins::write_board_manifest()] after pinning
#' to update the board manifest for board_url() consumers.
#'
#' @param model_dir Directory with model files
#' @param model_id Name for the model
#' @param board Pins board to pin to
#' @param metadata Optional metadata list
#' @export
pin_model <- function(model_dir,
                      model_id,
                      board,
                      metadata = list()) {

  # RF-DETR required files
  required <- c("checkpoint_best_total.pth", "metadata.json")
  optional <- c("log.txt", "metrics.json", "metrics_plot.png", "results.json")

  files <- fs::path(model_dir, required)

  if (!all(fs::file_exists(files))) {
    missing <- required[!fs::file_exists(files)]
    cli::cli_abort("Missing required model files: {.val {missing}}")
  }

  # Add optional files if present
  opt_files <- fs::path(model_dir, optional)
  files <- c(files, opt_files[fs::file_exists(opt_files)])

  # Add timestamp
  metadata$pinned <- Sys.time()

  # Upload
  pins::pin_upload(board, files, name = model_id, metadata = metadata)

  invisible(model_id)
}

#' List available models
#'
#' @param board Pins board (NULL = public hub, "local" = local training board)
#' @export
list_models <- function(board = NULL) {
  if (is.null(board)) {
    board <- pins::board_url(Sys.getenv("PETROGRAPHER_HUB_URL", .hub_models_url))
  } else if (identical(board, "local")) {
    board <- .get_model_board()
  }
  pins::pin_list(board)
}

#' Get model info
#'
#' @param model_id Model name
#' @param board Pins board (NULL = public hub, "local" = local training board)
#' @export
model_info <- function(model_id, board = NULL) {
  if (is.null(board)) {
    board <- pins::board_url(Sys.getenv("PETROGRAPHER_HUB_URL", .hub_models_url))
  } else if (identical(board, "local")) {
    board <- .get_model_board()
  }

  meta <- pins::pin_meta(board, model_id)

  # Nice display with cli
  cli::cli_h2("{.strong {model_id}}")
  cli::cli_dl(c(
    "Created" = format(meta$created, "%Y-%m-%d %H:%M"),
    "Files" = paste(basename(meta$file), collapse = ", "),
    "Size" = paste0(round(sum(meta$file_size) / 1e6, 1), " MB")
  ))

  if (!is.null(meta$user)) {
    if (!is.null(meta$user$pinned)) {
      cli::cli_text("Pinned: {meta$user$pinned}")
    }
    if (!is.null(meta$user$notes)) {
      cli::cli_text("{meta$user$notes}")
    }
  }

  invisible(meta)
}
