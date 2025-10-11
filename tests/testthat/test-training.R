test_that("train_model passes num_classes to local trainer", {
  data_dir <- withr::local_tempdir()
  train_dir <- file.path(data_dir, "train")
  valid_dir <- file.path(data_dir, "valid")
  dir.create(train_dir, recursive = TRUE)
  dir.create(valid_dir, recursive = TRUE)

  writeLines("{}", file.path(train_dir, "_annotations.coco.json"))
  writeLines("{}", file.path(valid_dir, "_annotations.coco.json"))

  local_output_dir <- file.path(data_dir, "models")
  captured <- NULL

  result <- testthat::with_mocked_bindings(
    petrographer::train_model(
      data_dir = data_dir,
      output_name = "test_model",
      num_classes = 7L,
      max_iter = 10L,
      device = "cpu",
      local_output_dir = local_output_dir,
      publish_after_train = FALSE,
      hpc_host = ""
    ),
    train_model_local = function(...) {
      args <- list(...)
      if (!"num_classes" %in% names(args)) {
        stop("num_classes argument was not supplied")
      }
      captured <<- args
      "mocked_result"
    },
    .package = "petrographer"
  )

  expect_equal(result, "mocked_result")
  expect_equal(captured$num_classes, 7L)
  expect_equal(captured$output_name, "test_model")
})
