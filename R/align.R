# R/align.R

#' Align thin section image pair
#'
#' Aligns a PPL (plane-polarized light) image to an XPL (cross-polarized light)
#' reference image using SIFT feature matching. Designed for images of the same
#' slide taken after remounting (e.g., adding polarizing filters). Automatically
#' validates that scale and rotation are reasonable and falls back to
#' translation-only if needed.
#'
#' @param ppl_path Path to PPL image
#' @param xpl_path Path to XPL image (reference)
#' @param output_path Path to save aligned PPL image. If a directory, generates
#'   filename from input.
#' @param method Alignment method: "similarity" (rotation+scale+translation with
#'   automatic fallback to translation if checks fail) or "translation"
#'   (translation only, no rotation/scale)
#' @return List with n_matches and method_used (invisibly)
#' @export
align_images <- function(ppl_path, xpl_path, output_path, method = "similarity") {

  if (!fs::file_exists(ppl_path)) {
    cli::cli_abort("PPL image not found: {.path {ppl_path}}")
  }
  if (!fs::file_exists(xpl_path)) {
    cli::cli_abort("XPL image not found: {.path {xpl_path}}")
  }
  if (!method %in% c("similarity", "translation")) {
    cli::cli_abort("{.arg method} must be 'similarity' or 'translation'")
  }

  # If output_path is a directory, create filename
  if (fs::is_dir(output_path)) {
    base_name <- fs::path_ext_remove(fs::path_file(ppl_path))
    output_path <- fs::path(output_path, paste0(base_name, "_aligned.jpg"))
  }

  fs::dir_create(fs::path_dir(output_path))

  result <- align_py$align_images(
    as.character(ppl_path),
    as.character(xpl_path),
    as.character(output_path),
    method
  )

  cli::cli_alert_success("Aligned using {result$n_matches} feature matches")
  if (result$method_used == "translation_fallback") {
    cli::cli_alert_warning("Used translation fallback (similarity transform failed sanity checks)")
  } else {
    cli::cli_alert_info("Method: {result$method_used}")
  }
  cli::cli_alert_info("Saved: {.path {output_path}}")

  invisible(result)
}
