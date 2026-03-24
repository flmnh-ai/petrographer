utils::globalVariables(c("sahi", "skimage", "align_py"))

# Package-level environment for session state (e.g., one-time warnings)
.petrographer_env <- new.env(parent = emptyenv())

.onLoad <- function(libname, pkgname) {
  # Declare Python requirements (Reticulate >= 1.41) without initializing Python
  if (utils::packageVersion("reticulate") >= "1.41") {
    reticulate::py_require(c(
      "sahi", 'rfdetr', "inference", "opencv-python", "scikit-image"
    ))
  }

  # Delay-load Python modules (keeps package load fast + CRAN-safe)
  sahi <<- reticulate::import("sahi", delay_load = TRUE)
  skimage <<- reticulate::import("skimage", delay_load = TRUE)
  align_py <<- reticulate::import_from_path(
    "align",
    path = system.file("python", package = "petrographer"),
    delay_load = TRUE
  )
}
