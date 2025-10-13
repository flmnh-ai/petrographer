# ============================================================================
# Minimal Pins Integration for Model Hub
# ============================================================================

# The public model hub URL
.hub_url <- "https://dl.dropboxusercontent.com/scl/fi/egsznhlwhzrjvf7ucazmr/_pins.yaml?rlkey=347x9dwz9h2rkhodeun7aoaom&dl=1"

#' Create a user board for publishing models
#'
#' @param path Optional path override (uses PETROGRAPHER_BOARD_PATH env var or default)
#' @return pins board object
#' @export
board_user <- function(path = Sys.getenv("PETROGRAPHER_BOARD_PATH", "")) {
  if (!nzchar(path)) {
    if (requireNamespace("here", quietly = TRUE)) {
      path <- here::here("petrographer-pins")
    } else {
      cli::cli_abort("Please install {.pkg here} or specify {.arg path}")
    }
  }
  fs::dir_create(path, recurse = TRUE)
  pins::board_folder(path, versioned = TRUE)
}

#' Load a pretrained model
#'
#' @param model_id Model name (e.g., "shell_v3")
#' @param version Specific version (NULL for latest)
#' @param board Pins board (NULL = public hub)
#' @param device Device: "cpu", "cuda", or "mps"
#' @param confidence Detection threshold
#' @return PetrographyModel object
#' @export
from_pretrained <- function(model_id,
                            version = NULL,
                            board = NULL,
                            device = "cpu",
                            confidence = 0.5) {

  # Default to public hub
  if (is.null(board)) {
    board <- pins::board_url(Sys.getenv("PETROGRAPHER_HUB_URL", .hub_url))
  }

  # Download model files
  files <- pins::pin_download(board, model_id, version = version)

  # Find model weights
  model_path <- files[grepl("model_best\\.pth$", files)][1]
  config_path <- files[grepl("config\\.yaml$", files)][1]

  if (is.na(model_path)) cli::cli_abort("No model weights found for {.val {model_id}}")
  if (is.na(config_path)) cli::cli_abort("No config found for {.val {model_id}}")

  cli::cli_alert_success("Loading {.strong {model_id}}")

  # Load with SAHI
  sahi <- reticulate::import("sahi")
  sahi_model <- sahi$AutoDetectionModel$from_pretrained(
    model_type = 'detectron2',
    model_path = as.character(model_path),
    config_path = as.character(config_path),
    confidence_threshold = confidence,
    device = device
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

#' Publish model to board
#'
#' @param model_dir Directory with model files
#' @param model_id Name for the model
#' @param board Pins board to publish to
#' @param metadata Optional metadata list
#' @param update_manifest Update _pins.yaml for board_url access
#' @export
publish_model <- function(model_dir,
                          model_id,
                          board,
                          metadata = list(),
                          update_manifest = TRUE) {

  # Required files
  required <- c("model_best.pth", "config.yaml")
  files <- fs::path(model_dir, required)

  if (!all(fs::file_exists(files))) {
    missing <- required[!fs::file_exists(files)]
    cli::cli_abort("Missing required files: {.val {missing}}")
  }

  # Add optional files if present
  optional <- c("metrics.json", "log.txt")
  opt_files <- fs::path(model_dir, optional)
  files <- c(files, opt_files[fs::file_exists(opt_files)])

  # Add timestamp
  metadata$published <- Sys.time()

  # Upload
  pins::pin_upload(board, files, name = model_id, metadata = metadata)

  if (update_manifest) {
    pins::write_board_manifest(board)
  }

  cli::cli_alert_success("Published {.strong {model_id}}")
}

#' List available models
#' @param board Pins board (NULL = public hub)
#' @export
list_models <- function(board = NULL) {
  if (is.null(board)) {
    board <- pins::board_url(Sys.getenv("PETROGRAPHER_HUB_URL", .hub_url))
  }
  pins::pin_list(board)
}

#' Get model info
#' @param model_id Model name
#' @param board Pins board (NULL = public hub)
#' @export
model_info <- function(model_id, board = NULL) {
  if (is.null(board)) {
    board <- pins::board_url(Sys.getenv("PETROGRAPHER_HUB_URL", .hub_url))
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
    if (!is.null(meta$user$published)) {
      cli::cli_text("Published: {meta$user$published}")
    }
    if (!is.null(meta$user$notes)) {
      cli::cli_text("{meta$user$notes}")
    }
  }

  invisible(meta)
}

# ============================================================================
# Internal helpers (for training.R compatibility)
# ============================================================================

# Null coalescing
pg_coalesce <- function(...) {
  for (val in list(...)) {
    if (!is.null(val)) return(val)
  }
  NULL
}

# ISO timestamp
pg_model_iso_time <- function(x = Sys.time()) {
  format(as.POSIXct(x, tz = "UTC"), "%Y-%m-%dT%H:%M:%SZ")
}

# Auto version string
pg_model_auto_version <- function(prefix = "v") {
  paste0(prefix, format(Sys.time(), "%Y%m%d%H%M%S"))
}

# For training.R compatibility
pg_board_user <- function(path = Sys.getenv("PETROGRAPHER_BOARD_PATH", "")) {
  board_user(path)
}

# For training.R compatibility
pg_model_publish <- function(model_dir,
                              model_id,
                              board,
                              metadata = list(),
                              include_metrics = TRUE,
                              write_manifest = TRUE) {
  publish_model(model_dir, model_id, board, metadata, write_manifest)
}
