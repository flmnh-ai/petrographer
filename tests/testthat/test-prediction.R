test_that("metadata helpers preserve model names and original COCO ids", {
  manifest <- list(
    categories = list(
      list(model_id = 0L, coco_id = 5L, name = "matrix"),
      list(model_id = 1L, coco_id = 9L, name = "inclusion")
    )
  )
  model <- structure(list(
    sahi_model = NULL,
    manifest = manifest,
    is_segmentation = TRUE
  ), class = "PetrographyModel")

  name_map <- petrographer:::.category_map(model)
  expect_equal(name_map[["0"]], "matrix")
  expect_equal(name_map[["1"]], "inclusion")
  expect_equal(petrographer:::.prediction_category_id(model, 0L), 5L)
  expect_equal(petrographer:::.prediction_category_id(model, 1L), 9L)
})

test_that("category name fallback uses thing_classes for segmentation models", {
  manifest <- list(
    thing_classes = list("matrix", "inclusion")
  )
  model <- structure(list(
    sahi_model = NULL,
    manifest = manifest,
    is_segmentation = TRUE
  ), class = "PetrographyModel")

  name_map <- petrographer:::.category_map(model)
  expect_equal(name_map[["0"]], "matrix")
  expect_equal(name_map[["1"]], "inclusion")
})

test_that("manifest validation requires the new schema", {
  manifest <- list(
    manifest_version = 1L,
    model = list(
      backend = "rfdetr",
      variant = "seg_small",
      task = "segmentation",
      weights = "checkpoint_best_total.pth"
    ),
    categories = list(
      list(model_id = 0L, coco_id = 5L, name = "matrix")
    ),
    training = list(dataset_id = "rocks", dataset_version = "123"),
    artifacts = list(
      weights = "checkpoint_best_total.pth",
      training_summary = "training_summary.json"
    ),
    software_versions = list(rfdetr = "1.6.2")
  )

  expect_silent(petrographer:::.validate_model_manifest(manifest))
  bad_manifest <- manifest
  bad_manifest$categories <- NULL
  expect_error(
    petrographer:::.validate_model_manifest(bad_manifest),
    "missing required fields"
  )
})

test_that("evaluate_model_sahi rejects segmentation models before Python eval", {
  ann_path <- withr::local_tempfile(fileext = ".json")
  writeLines("{}", ann_path)

  seg_model <- structure(list(
    sahi_model = NULL,
    manifest = list(thing_classes = list("matrix")),
    is_segmentation = TRUE
  ), class = "PetrographyModel")

  expect_error(
    petrographer::evaluate_model_sahi(seg_model, ann_path),
    "currently supports detection models only"
  )
})

test_that("analyze_segmentation_dir writes summary artifacts", {
  input_dir <- withr::local_tempdir()
  output_dir <- withr::local_tempdir()

  file.create(file.path(input_dir, "img1.png"))

  fake_detections <- tibble::tibble(
    image_name = "img1",
    area = c(10, 20),
    circularity = c(0.8, 0.7),
    eccentricity = c(0.2, 0.4)
  )

  seg_model <- structure(list(
    sahi_model = NULL,
    manifest = list(
      categories = list(list(model_id = 0L, coco_id = 5L, name = "matrix"))
    ),
    is_segmentation = TRUE
  ), class = "PetrographyModel")

  result <- testthat::with_mocked_bindings(
    petrographer:::analyze_segmentation_dir(
      input_dir = input_dir,
      model = seg_model,
      output_dir = output_dir
    ),
    predict_images = function(...) fake_detections,
    .package = "petrographer"
  )

  expect_equal(nrow(result$detections), 2)
  expect_true(fs::file_exists(fs::path(output_dir, "measurements.csv")))
  expect_true(fs::file_exists(fs::path(output_dir, "image_summary.csv")))
  expect_true(fs::file_exists(fs::path(output_dir, "population_stats.json")))
})
