#' Plot COCO annotations on an image
#'
#' Draws ground-truth bounding boxes (and segmentation masks when present) from
#' a COCO-style annotation file on top of an image. Rendering is delegated to
#' Roboflow's `supervision` library via reticulate, so GT overlays stay visually
#' consistent with prediction overlays produced by [predict_image()].
#'
#' @param image_path Path to the image file.
#' @param annotation_json Path to the COCO annotations JSON containing the image.
#' @param categories Optional vector of category names or ids to keep.
#'   `NULL` keeps all.
#' @param output Destination PNG path. Defaults to a tempfile.
#' @param display Whether to render the annotated image on the current graphics
#'   device. Defaults to `TRUE` in interactive sessions and inside knitr
#'   (so the image appears under the chunk).
#' @param draw_labels Whether to draw category labels. Default `TRUE`.
#' @param mask_opacity Opacity for mask fills (segmentation datasets). Default `0.4`.
#' @return The path to the written PNG, invisibly.
#' @export
pg_plot_annotations <- function(image_path,
                                annotation_json,
                                categories = NULL,
                                output = NULL,
                                display = NULL,
                                draw_labels = TRUE,
                                mask_opacity = 0.4) {

  image_path <- fs::path_abs(image_path)
  annotation_json <- fs::path_abs(annotation_json)

  if (!fs::file_exists(image_path)) {
    cli::cli_abort("Image not found: {.path {image_path}}")
  }
  if (!fs::file_exists(annotation_json)) {
    cli::cli_abort("Annotation json not found: {.path {annotation_json}}")
  }

  if (is.null(output)) {
    output <- fs::file_temp(ext = "png")
  }
  fs::dir_create(fs::path_dir(output))

  categories_py <- if (is.null(categories)) {
    NULL
  } else if (is.numeric(categories)) {
    as.list(as.integer(categories))
  } else {
    as.list(as.character(categories))
  }

  visualize_py$render_coco_overlay(
    image_path       = as.character(image_path),
    annotations_json = as.character(annotation_json),
    output_path      = as.character(output),
    categories_keep  = categories_py,
    draw_labels      = isTRUE(draw_labels),
    mask_opacity     = as.numeric(mask_opacity)
  )

  if (is.null(display)) {
    display <- interactive() || isTRUE(getOption("knitr.in.progress"))
  }
  if (isTRUE(display)) {
    .display_png(output)
  }

  invisible(as.character(output))
}

#' Plot a sample image from a dataset directory
#'
#' @param dataset_dir Directory containing dataset splits (train, valid, or optionally test).
#' @param split Which split to draw from (`"train"`, `"valid"`, or `"test"`).
#' @param image_name Optional specific image file name; otherwise a random one.
#' @param annotation_json Optional explicit path to annotation JSON.
#' @param ... Additional arguments passed to [pg_plot_annotations()].
#' @return The path to the written PNG, invisibly.
#' @export
pg_plot_dataset_image <- function(dataset_dir,
                                  split = c("train", "valid", "test"),
                                  image_name = NULL,
                                  annotation_json = NULL,
                                  ...) {
  split <- match.arg(split)
  split_dir <- fs::path(dataset_dir, split)
  if (!fs::dir_exists(split_dir)) {
    cli::cli_abort("Split directory not found: {.path {split_dir}}")
  }
  annotation_json <- annotation_json %||% fs::path(split_dir, "_annotations.coco.json")
  ann <- jsonlite::read_json(annotation_json)

  images <- ann$images %||% list()
  if (!length(images)) {
    cli::cli_abort("No images listed in {annotation_json}.")
  }

  if (is.null(image_name)) {
    selected <- images[[sample(length(images), 1)]]
  } else {
    idx <- which(vapply(images, function(im) identical(im$file_name, image_name), logical(1)))
    if (!length(idx)) {
      cli::cli_abort("Image '{image_name}' not found in annotations.")
    }
    selected <- images[[idx]]
  }

  image_path <- fs::path(split_dir, selected$file_name)
  if (!fs::file_exists(image_path)) {
    cli::cli_abort("Image file not found: {.path {image_path}}")
  }

  pg_plot_annotations(image_path, annotation_json, ...)
}

#' Save a preview image for a dataset
#'
#' Convenience helper that writes a `preview.png` alongside the dataset so the
#' pkgdown catalog can embed a thumbnail.
#'
#' @param dataset_dir Dataset directory (containing split subfolders).
#' @param split Which split to sample (default `"valid"`).
#' @param output Path for the preview image (default `<dataset_dir>/preview.png`).
#' @param overwrite Whether to overwrite an existing preview.
#' @param ... Additional arguments passed to [pg_plot_dataset_image()].
#' @return The path to the written preview image (invisibly).
#' @export
pg_save_dataset_preview <- function(dataset_dir,
                                    split = "valid",
                                    output = fs::path(dataset_dir, "preview.png"),
                                    overwrite = TRUE,
                                    ...) {
  if (!overwrite && fs::file_exists(output)) {
    return(invisible(output))
  }
  pg_plot_dataset_image(
    dataset_dir = dataset_dir,
    split       = split,
    output      = output,
    display     = FALSE,
    ...
  )
  invisible(output)
}

# ----------------------------------------------------------------------------
# Internal helpers
# ----------------------------------------------------------------------------

#' Render a PNG on the current graphics device.
#'
#' Uses `png` + `grid` so the output is captured by knitr inside a chunk, shows
#' in the RStudio viewer when interactive, and otherwise writes to whatever
#' device is active. No magick dependency.
#'
#' @noRd
.display_png <- function(path) {
  if (!requireNamespace("png", quietly = TRUE)) {
    cli::cli_warn(
      "Install the {.pkg png} package to display annotated images inline."
    )
    return(invisible(NULL))
  }
  if (!requireNamespace("grid", quietly = TRUE)) {
    # `grid` is base R, but be defensive.
    return(invisible(NULL))
  }
  img <- png::readPNG(path)
  grid::grid.newpage()
  grid::grid.raster(img)
  invisible(NULL)
}
