utils::globalVariables(c(
  "sahi", "skimage", "sv", "align_py", "visualize_py",
  ".data", "image_id", "category_id", "count", "file_name", "full_path",
  "image_name", "metric", "area", "orientation", "circularity",
  "eccentricity"
))

# Package-level environment for session state (e.g., one-time warnings)
.petrographer_env <- new.env(parent = emptyenv())

.onLoad <- function(libname, pkgname) {
  # Declare Python requirements (Reticulate >= 1.41) without initializing Python
  if (utils::packageVersion("reticulate") >= "1.41") {
    reticulate::py_require(c(
      "sahi", "rfdetr", "supervision",
      "opencv-python", "scikit-image",
      # pycocotools: used directly by evaluate_model_sahi() and by the COCO
      # segmentation mask decoder in inst/python/visualize.py. It used to come
      # in transitively via the now-removed `inference` package, so declare it
      # explicitly rather than depend on a sibling's dependency graph.
      "pycocotools"
    ))
  }

  # Delay-load Python modules (keeps package load fast + CRAN-safe)
  sahi <<- reticulate::import("sahi", delay_load = TRUE)
  skimage <<- reticulate::import("skimage", delay_load = TRUE)
  sv <<- reticulate::import("supervision", delay_load = TRUE)
  align_py <<- reticulate::import_from_path(
    "align",
    path = system.file("python", package = "petrographer"),
    delay_load = TRUE
  )
  visualize_py <<- reticulate::import_from_path(
    "visualize",
    path = system.file("python", package = "petrographer"),
    delay_load = TRUE
  )
}
