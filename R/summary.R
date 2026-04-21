# ============================================================================
# Summary Functions
# ============================================================================

.mean_or_na <- function(x) {
  x <- x[!is.na(x)]
  if (!length(x)) NA_real_ else mean(x)
}

.median_or_na <- function(x) {
  x <- x[!is.na(x)]
  if (!length(x)) NA_real_ else stats::median(x)
}

.range_or_na <- function(x) {
  x <- x[!is.na(x)]
  if (!length(x)) c(NA_real_, NA_real_) else range(x)
}

#' Summarize detections by image
#' @param .data Data frame with detections
#' @return Summary tibble with per-image statistics
#' @export
summarize_by_image <- function(.data) {
  .data |>
    dplyr::group_by(image_name) |>
    dplyr::summarise(
      n_objects = dplyr::n(),
      total_area = sum(area, na.rm = TRUE),
      mean_area = .mean_or_na(area),
      median_area = .median_or_na(area),
      mean_circularity = .mean_or_na(circularity),
      mean_eccentricity = .mean_or_na(eccentricity),
      # Guard against divide-by-zero / non-finite mean (e.g. all-NA areas or a
      # single detection with area 0). Returns NA instead of Inf/NaN.
      area_cv = {
        .m <- .mean_or_na(area)
        if (is.finite(.m) && .m > 0) stats::sd(area, na.rm = TRUE) / .m else NA_real_
      },
      .groups = "drop"
    )
}

#' Get overall population statistics
#' @param .data Data frame with detections
#' @return Named list of population-level statistics
#' @export
get_population_stats <- function(.data) {
  if (nrow(.data) == 0) {
    return(list(
      total_objects = 0,
      unique_images = 0,
      mean_objects_per_image = 0
    ))
  }

  list(
    total_objects = nrow(.data),
    unique_images = length(unique(.data$image_name)),
    mean_objects_per_image = nrow(.data) / length(unique(.data$image_name)),
    total_area = sum(.data$area, na.rm = TRUE),
    mean_area = .mean_or_na(.data$area),
    median_area = .median_or_na(.data$area),
    area_range = .range_or_na(.data$area),
    mean_circularity = .mean_or_na(.data$circularity),
    mean_eccentricity = .mean_or_na(.data$eccentricity)
  )
}
