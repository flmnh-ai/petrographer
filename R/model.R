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
