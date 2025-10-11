# Metrics parsing

#' Parse Detectron2 metrics.json into tibbles
#'
#' Reads a Detectron2 `metrics.json` (JSONL) and returns separate tibbles for
#' training rows, aggregate validation metrics (bbox + segm), and per-class AP
#' when available.
#'
#' @param metrics_file Path to `metrics.json`.
#' @return A list with elements `training`, `validation`, and `classwise` (tibbles).
#' @keywords internal
parse_metrics <- function(metrics_file) {
  if (!fs::file_exists(metrics_file)) return(list(training = tibble::tibble(), validation = tibble::tibble(), classwise = tibble::tibble()))

  raw_lines <- readLines(metrics_file, warn = FALSE)
  if (length(raw_lines) == 0) return(list(training = tibble::tibble(), validation = tibble::tibble(), classwise = tibble::tibble()))

  sanitized_lines <- stringr::str_replace_all(raw_lines, "(?<=[:\\s])NaN(?=[,}\\s])", "null")
  replaced_nan <- !identical(raw_lines, sanitized_lines)

  con <- textConnection(sanitized_lines); on.exit(close(con), add = TRUE)
  df <- tryCatch(jsonlite::stream_in(con, verbose = FALSE), error = function(e) NULL)
  if (is.null(df)) return(list(training = tibble::tibble(), validation = tibble::tibble(), classwise = tibble::tibble()))

  if (replaced_nan) {
    warning("metrics.json contained NaN values; replacing with NA for compatibility", call. = FALSE)
  }
  d <- tibble::as_tibble(df) |>
    clean_names() |>
    dplyr::mutate(.row_id = dplyr::row_number())

  validation_cols <- grep("(bbox|segm)", names(d), value = TRUE)
  if (length(validation_cols) > 0) {
    d <- d |>
      dplyr::mutate(.is_validation = dplyr::if_any(dplyr::all_of(validation_cols), ~ !is.na(.)))
  } else {
    d <- d |>
      dplyr::mutate(.is_validation = FALSE)
  }

  validation_rows <- d |>
    dplyr::filter(.is_validation)

  training <- d |>
    dplyr::filter(!.is_validation) |>
    dplyr::select(-.row_id, -.is_validation)

  validation <- validation_rows |>
    dplyr::select(-.row_id, -.is_validation) |>
    dplyr::select(iteration, dplyr::contains("bbox"), dplyr::contains("segm"))

  classwise_cols <- grep("^ap_", names(d), value = TRUE)
  classwise <- tibble::tibble()
  if (length(classwise_cols) > 0 && nrow(validation_rows) > 0) {
    classwise <- validation_rows |>
      dplyr::select(-.row_id, -.is_validation) |>
      dplyr::select(iteration, dplyr::all_of(classwise_cols))
  }
  list(training = training, validation = validation, classwise = classwise)
}
