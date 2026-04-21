# ============================================================================
# Model Management Utilities
# ============================================================================

#' List all locally trained models
#'
#' @return Character vector of pinned model names
#' @export
list_trained_models <- function() {
  list_models(board = "local")
}

#' Get the dataset used to train a model
#'
#' Retrieves the exact dataset version that was used to train a model.
#' This ensures reproducibility by downloading the specific version from pins.
#'
#' @param model_id Model name
#' @param board Pins board to load model from (NULL = local training board)
#' @return Path to the dataset tar.gz file
#' @export
#' @examples
#' \dontrun{
#' # Retrieve the exact dataset version used to train a model. Returns a
#' # .tar.gz path; `train_model()` and `validate_dataset()` accept this directly
#' # and extract transparently.
#' dataset_path <- get_training_dataset("my_model")
#'
#' # Retrain on the same data
#' train_model(data_dir = dataset_path, model_id = "my_model_v2", ...)
#'
#' # Or inspect without retraining
#' validate_dataset(dataset_path)
#' }
get_training_dataset <- function(model_id, board = NULL) {
  # Load model board
  if (is.null(board)) {
    board <- .get_model_board()
  } else if (identical(board, "local")) {
    board <- .get_model_board()
  }

  # Get model metadata
  model_meta <- tryCatch(
    pins::pin_meta(board, model_id),
    error = function(e) {
      cli::cli_abort("Model {.val {model_id}} not found: {e$message}")
    }
  )

  # Extract dataset info from metadata
  dataset_id <- model_meta$user$dataset_id
  dataset_version <- model_meta$user$dataset_version

  if (is.null(dataset_id)) {
    cli::cli_abort("Model {.val {model_id}} has no dataset_id in metadata")
  }
  if (is.null(dataset_version)) {
    cli::cli_warn("Model {.val {model_id}} has no dataset_version - using latest version")
    dataset_version <- NULL
  }

  cli::cli_alert_info("Model trained with dataset: {.val {dataset_id}} (version: {.val {dataset_version %||% 'latest'}})")

  # Get dataset path from dataset board
  dataset_path <- get_dataset_path(
    dataset_id = dataset_id,
    board = "local",
    version = dataset_version
  )

  return(dataset_path)
}
