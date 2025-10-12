#' Plot COCO annotations on an image
#'
#' Draws bounding boxes (and optional labels) from a COCO-style annotation
#' file on top of an image. Returns a `magick-image` object that can be printed
#' or written to disk.
#'
#' @param image_path Path to the image file.
#' @param annotation_json Path to the COCO annotations JSON containing the image.
#' @param categories Optional vector of category names or ids to keep. NULL keeps all.
#' @param class_colors Optional named vector of colours keyed by category name.
#' @param outline_width Line width for bounding boxes.
#' @param label Whether to draw category labels.
#' @param label_cex Text expansion factor for labels.
#' @param label_alpha Alpha for label background fill.
#' @param label_colour Label text colour.
#' @return A `magick-image` with annotations drawn.
#' @export
#' @importFrom graphics rect text strwidth strheight par
#' @importFrom grDevices adjustcolor hcl.colors
pg_plot_annotations <- function(image_path,
                                annotation_json,
                                categories = NULL,
                                class_colors = NULL,
                                outline_width = 4,
                                label = TRUE,
                                label_cex = 0.8,
                                label_alpha = 0.6,
                                label_colour = "white") {
  if (!requireNamespace("magick", quietly = TRUE)) {
    cli::cli_abort("The 'magick' package is required. Install with install.packages('magick').")
  }

  image_path <- fs::path_abs(image_path)
  annotation_json <- fs::path_abs(annotation_json)

  if (!fs::file_exists(image_path)) {
    cli::cli_abort("Image not found: {.path {image_path}}")
  }
  if (!fs::file_exists(annotation_json)) {
    cli::cli_abort("Annotation json not found: {.path {annotation_json}}")
  }

  ann <- jsonlite::read_json(annotation_json)
  image_name <- fs::path_file(image_path)
  images <- ann$images %||% list()
  idx <- which(vapply(images, function(im) identical(im$file_name, image_name), logical(1)))
  if (!length(idx)) {
    cli::cli_abort("Image '{image_name}' is not listed in {annotation_json}.")
  }
  image_id <- images[[idx]]$id

  annotations <- ann$annotations %||% list()
  annotations <- Filter(function(a) identical(a$image_id, image_id), annotations)
  if (!length(annotations)) {
    cli::cli_warn("No annotations for image {image_name} in {annotation_json}.")
  }

  category_table <- ann$categories %||% list()
  category_map <- setNames(
    vapply(category_table, function(cat) cat$name %||% as.character(cat$id), character(1)),
    vapply(category_table, function(cat) cat$id, numeric(1))
  )

  if (!is.null(categories) && length(annotations)) {
    keep_ids <- if (is.numeric(categories)) {
      categories
    } else {
      names(category_map)[match(categories, category_map, nomatch = 0)] |> as.numeric()
    }
    annotations <- Filter(function(a) a$category_id %in% keep_ids, annotations)
  }

  if (!length(annotations)) {
    return(magick::image_read(image_path))
  }

  cat_ids <- unique(vapply(annotations, function(a) a$category_id, numeric(1)))
  cat_names <- category_map[as.character(cat_ids)]
  if (is.null(class_colors)) {
    palette <- grDevices::hcl.colors(length(cat_ids), palette = "Dark3")
    class_colors <- setNames(palette, cat_names)
  } else {
    missing <- setdiff(cat_names, names(class_colors))
    if (length(missing)) {
      cli::cli_warn("No colour provided for categories: {missing}. Using defaults.")
      palette <- grDevices::hcl.colors(length(missing), palette = "Dark3")
      class_colors <- c(class_colors, setNames(palette, missing))
    }
  }

  img <- magick::image_read(image_path)
  info <- magick::image_info(img)
  width <- info$width
  height <- info$height

  draw <- magick::image_draw(img)
  graphics::par(mar = c(0, 0, 0, 0))

  for (a in annotations) {
    bbox <- a$bbox
    x1 <- bbox[[1]]
    y_top <- bbox[[2]]
    w <- bbox[[3]]
    h <- bbox[[4]]
    x2 <- x1 + w
    y_bottom <- y_top + h

    colour <- class_colors[[ category_map[[as.character(a$category_id)]] ]] %||% "#ff9800"

    graphics::rect(
      x1,
      height - y_top,
      x2,
      height - y_bottom,
      border = colour,
      lwd = outline_width
    )

    if (isTRUE(label)) {
      label_text <- category_map[[as.character(a$category_id)]] %||% as.character(a$category_id)
      str_w <- graphics::strwidth(label_text, cex = label_cex)
      str_h <- graphics::strheight(label_text, cex = label_cex)
      pad <- 4
      x_lab <- x1 + pad
      y_lab_top <- height - y_top - pad
      graphics::rect(
        x1,
        y_lab_top - str_h - pad,
        x1 + str_w + 2 * pad,
        y_lab_top + pad,
        col = grDevices::adjustcolor(colour, alpha.f = label_alpha),
        border = NA
      )
      graphics::text(
        x_lab,
        y_lab_top,
        labels = label_text,
        adj = c(0, 1),
        cex = label_cex,
        col = label_colour
      )
    }
  }

  grDevices::dev.off()
  draw
}

#' Plot a sample image from a dataset directory
#'
#' @param dataset_dir Directory containing dataset splits (train/valid[/test]).
#' @param split Which split to draw from (`"train"`, `"valid"`, or `"test"`).
#' @param image_name Optional specific image file name; otherwise a random one.
#' @param annotation_json Optional explicit path to annotation JSON.
#' @param ... Additional arguments passed to [pg_plot_annotations()].
#' @return A `magick-image` with annotations drawn.
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
  img <- pg_plot_dataset_image(dataset_dir, split = split, ...)
  magick::image_write(img, path = output)
  invisible(output)
}
