# Morphology helpers: SAHI result -> tibble with properties

#' Calculate morphological properties from SAHI result
#'
#' Internal helper converting SAHI object predictions into a tibble of
#' morphological properties using scikit-image via reticulate.
#'
#' @param result SAHI prediction result object.
#' @param image_path Original image path (used to populate `image_name`).
#' @return A tibble with morphological properties per object.
#' @keywords internal
calculate_morphology_from_result <- function(result, image_path) {
  predictions <- result$object_prediction_list
  if (length(predictions) == 0) return(tibble::tibble())

  morphology_list <- vector("list", length(predictions))
  for (i in seq_along(predictions)) {
    pred <- predictions[[i]]
    mask <- pred$mask$bool_mask

    # Skip if mask is None (detection model without segmentation)
    if (is.null(mask)) {
      # For detection models, calculate basic properties from bounding box
      bbox <- as.numeric(pred$bbox$to_xyxy())
      x1 <- bbox[1]; y1 <- bbox[2]; x2 <- bbox[3]; y2 <- bbox[4]
      width <- x2 - x1
      height <- y2 - y1
      area <- width * height

      morphology_list[[i]] <- list(
        class_id = pred$category$id,
        class_name = pred$category$name,
        confidence = pred$score$value,
        area = area,
        perimeter = 2 * (width + height),
        centroid_x = (x1 + x2) / 2,
        centroid_y = (y1 + y2) / 2,
        eccentricity = NA_real_,
        orientation = NA_real_,
        major_axis_length = max(width, height),
        minor_axis_length = min(width, height),
        circularity = NA_real_,
        aspect_ratio = max(width, height) / min(width, height),
        solidity = NA_real_,
        extent = NA_real_
      )
      next
    }

    # Calculate morphology from mask
    labeled_mask <- skimage$measure$label(mask)
    storage.mode(labeled_mask) <- 'integer'
    props <- skimage$measure$regionprops(labeled_mask)
    if (length(props) == 0) stop("scikit-image could not extract region properties from mask")
    prop <- props[[1]]
    morphology_list[[i]] <- list(
      class_id = pred$category$id,
      class_name = pred$category$name,
      confidence = pred$score$value,
      area = prop$area,
      perimeter = prop$perimeter,
      centroid_x = prop$centroid[[1]],
      centroid_y = prop$centroid[[2]],
      eccentricity = prop$eccentricity,
      orientation = prop$orientation,
      major_axis_length = prop$major_axis_length,
      minor_axis_length = prop$minor_axis_length,
      circularity = (4 * pi * prop$area) / (prop$perimeter^2),
      aspect_ratio = prop$major_axis_length / prop$minor_axis_length,
      solidity = prop$solidity,
      extent = prop$extent
    )
  }

  morphology_list |>
    purrr::map_dfr(tibble::as_tibble) |>
    dplyr::mutate(image_name = basename(image_path))
}

#' Calculate morphological properties from supervision Detections
#'
#' For segmentation models that return supervision Detections with masks.
#'
#' @param detections supervision.Detections object with masks.
#' @param image_path Original image path.
#' @param class_names_map Optional mapping of class id -> class name.
#' @return A tibble with morphological properties per object.
#' @keywords internal
calculate_morphology_from_detections <- function(detections,
                                                 image_path,
                                                 class_names_map = NULL) {
  masks <- detections$mask
  n_det <- nrow(detections$xyxy)
  if (n_det == 0) return(tibble::tibble())

  class_ids <- as.integer(detections$class_id)
  confidences <- as.numeric(detections$confidence)

  resolve_class_name <- function(class_id) {
    if (!is.null(class_names_map)) {
      name <- class_names_map[[as.character(class_id)]]
      if (is.null(name)) {
        name <- class_names_map[[class_id]]
      }
      if (!is.null(name)) {
        return(as.character(name))
      }
    }
    as.character(class_id)
  }

  morphology_list <- vector("list", n_det)
  for (i in seq_len(n_det)) {
    mask <- tryCatch(masks[i, , ], error = function(e) NULL)
    if (is.null(mask) || !any(mask)) {
      # Skip empty/invalid masks
      next
    }

    labeled_mask <- tryCatch(skimage$measure$label(mask), error = function(e) NULL)
    if (is.null(labeled_mask)) next
    storage.mode(labeled_mask) <- 'integer'
    props <- skimage$measure$regionprops(labeled_mask)

    if (length(props) == 0) {
      # Fallback to bbox
      bbox <- as.numeric(detections$xyxy[i, ])
      width <- bbox[3] - bbox[1]
      height <- bbox[4] - bbox[2]
      morphology_list[[i]] <- list(
        class_id = class_ids[i], class_name = resolve_class_name(class_ids[i]),
        confidence = confidences[i],
        area = width * height, perimeter = 2 * (width + height),
        centroid_x = (bbox[1] + bbox[3]) / 2, centroid_y = (bbox[2] + bbox[4]) / 2,
        eccentricity = NA_real_, orientation = NA_real_,
        major_axis_length = max(width, height),
        minor_axis_length = min(width, height),
        circularity = NA_real_,
        aspect_ratio = max(width, height) / min(width, height),
        solidity = NA_real_, extent = NA_real_
      )
      next
    }

    prop <- props[[1]]
    perim <- as.numeric(prop$perimeter)
    area <- as.numeric(prop$area)

    morphology_list[[i]] <- list(
      class_id = class_ids[i],
      class_name = resolve_class_name(class_ids[i]),
      confidence = confidences[i],
      area = area,
      perimeter = perim,
      centroid_x = as.numeric(prop$centroid[[2]]),
      centroid_y = as.numeric(prop$centroid[[1]]),
      eccentricity = as.numeric(prop$eccentricity),
      orientation = as.numeric(prop$orientation),
      major_axis_length = as.numeric(prop$major_axis_length),
      minor_axis_length = as.numeric(prop$minor_axis_length),
      circularity = if (perim > 0) min((4 * pi * area) / (perim^2), 1.0) else NA_real_,
      aspect_ratio = if (prop$minor_axis_length > 0) as.numeric(prop$major_axis_length) / as.numeric(prop$minor_axis_length) else NA_real_,
      solidity = as.numeric(prop$solidity),
      extent = as.numeric(prop$extent)
    )
  }

  morphology_list |>
    purrr::map_dfr(tibble::as_tibble) |>
    dplyr::mutate(image_name = basename(image_path))
}
