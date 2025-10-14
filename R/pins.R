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

#' Load a pretrained model
#'
#' Loads models from local training board, hub, or custom board.
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
#' @return PetrographyModel object
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
                            confidence = 0.5) {

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

  # Find model weights
  model_path <- files[grepl("model_best\\.pth$", files)][1]
  config_path <- files[grepl("config\\.yaml$", files)][1]
  metadata_path <- files[grepl("metadata\\.json$", files)][1]

  if (is.na(model_path)) cli::cli_abort("No model weights found for {.val {model_id}}")
  if (is.na(config_path)) cli::cli_abort("No config found for {.val {model_id}}")

  # Load category mapping from metadata.json if available
  category_mapping <- NULL
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

  # Load with SAHI
  sahi <- reticulate::import("sahi")
  sahi_model <- sahi$AutoDetectionModel$from_pretrained(
    model_type = 'detectron2',
    model_path = as.character(model_path),
    config_path = as.character(config_path),
    confidence_threshold = confidence,
    device = device,
    category_mapping = if (!is.null(category_mapping)) category_mapping else NULL
  )

  # Wrap in PetrographyModel
  model <- list(
    sahi_model = sahi_model,
    model_path = as.character(model_path),
    config_path = as.character(config_path),
    confidence = confidence,
    device = device,
    manifest = NULL
  )
  class(model) <- "PetrographyModel"
  return(model)
}

#' Pin a trained model to a board
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

  # Required files
  required <- c("model_best.pth", "config.yaml")
  files <- fs::path(model_dir, required)

  if (!all(fs::file_exists(files))) {
    missing <- required[!fs::file_exists(files)]
    cli::cli_abort("Missing required files: {.val {missing}}")
  }

  # Add optional files if present
  optional <- c("metadata.json", "metrics.json", "log.txt")
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
