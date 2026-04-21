test_that("train_model passes RF-DETR args through to local trainer", {
  captured <- NULL
  fake_model_dir <- withr::local_tempdir()

  result <- testthat::with_mocked_bindings(
    testthat::with_mocked_bindings(
      petrographer::train_model(
        dataset_id    = "test_dataset",
        model_id      = "test_model",
        model_variant = "small",
        epochs        = 10L,
        batch_size    = 4L,
        device        = "cpu"
      ),
      pin_meta = function(...) list(version = "20260416T000000Z"),
      .package = "pins"
    ),
    train_model_local = function(...) {
      captured <<- list(...)
      fake_model_dir
    },
    finalize_trained_model = function(model_dir, config, duration_mins) {
      # Skip pin/manifest creation in tests — just return model_id
      config$model_id
    },
    get_dataset_path = function(...) "/tmp/fake_dataset.tar.gz",
    .get_dataset_board = function() structure(list(), class = "board_fake"),
    .package = "petrographer"
  )

  expect_equal(result, "test_model")
  expect_equal(captured$model_id, "test_model")
  expect_equal(captured$model_variant, "small")
  expect_equal(captured$epochs, 10L)
  expect_equal(captured$batch_size, 4L)
  expect_equal(captured$device, "cpu")
})

test_that("train_model_local forwards device and AMP flags to Python trainer", {
  workspace_dir <- withr::local_tempdir()
  captured <- NULL
  tar_path <- tempfile(fileext = ".tar.gz")

  testthat::with_mocked_bindings(
    testthat::with_mocked_bindings(
      testthat::with_mocked_bindings(
        testthat::with_mocked_bindings(
          petrographer:::train_model_local(
            data_dir = tar_path,
            model_id = "test_model",
            model_variant = "small",
            resolution = 512L,
            epochs = 5L,
            batch_size = 2L,
            grad_accum_steps = 4L,
            learning_rate = 1e-4,
            device = "cpu",
            workspace_dir = workspace_dir,
            use_amp = TRUE,
            amp_dtype = "fp16",
            gradient_checkpointing = TRUE,
            num_workers = 3L,
            validate_every = 2L,
            early_stopping_patience = 7L
          ),
          extract_class_names_from_coco = function(...) "matrix",
          .package = "petrographer"
        ),
        untar = function(tarfile, exdir, tar) {
          fs::dir_create(file.path(exdir, "train"))
          fs::dir_create(file.path(exdir, "valid"))
          0L
        },
        .package = "utils"
      ),
      py_config = function() list(python = "python3"),
      .package = "reticulate"
    ),
    run = function(command, args, ...) {
      captured <<- list(command = command, args = args)
      list(status = 0L)
    },
    .package = "processx"
  )

  expect_equal(captured$command, "python3")
  expect_true("--device" %in% captured$args)
  expect_equal(captured$args[match("--device", captured$args) + 1], "cpu")
  expect_true("--use-amp" %in% captured$args)
  expect_true("--amp-dtype" %in% captured$args)
  expect_equal(captured$args[match("--amp-dtype", captured$args) + 1], "fp16")
  expect_true("--gradient-checkpointing" %in% captured$args)
})

test_that("finalize_trained_model writes manifest and training summary", {
  model_dir <- withr::local_tempdir()
  workspace_dir <- withr::local_tempdir()
  captured_metadata <- NULL

  jsonlite::write_json(list(
    thing_classes = list("matrix", "inclusion"),
    category_ids = list(5L, 9L),
    categories = list(
      list(model_id = 0L, coco_id = 5L, name = "matrix"),
      list(model_id = 1L, coco_id = 9L, name = "inclusion")
    ),
    num_classes = 2L,
    model_variant = "seg_small",
    rfdetr_version = "1.6.2",
    max_objects = 100L
  ), file.path(model_dir, "metadata.json"), auto_unbox = TRUE, pretty = TRUE)
  file.create(file.path(model_dir, "checkpoint_best_total.pth"))
  readr::write_csv(
    tibble::tibble(
      epoch = c(0, 0),
      `train/loss` = c(1, NA),
      `val/mAP_50_95` = c(NA, 0.5)
    ),
    file.path(model_dir, "metrics.csv")
  )

  config <- list(
    dataset_id = "rocks",
    dataset_version = "20260417",
    data_dir = "/tmp/dataset.tar.gz",
    model_id = "seg_model",
    model_variant = "seg_small",
    epochs = 5L,
    batch_size = 2L,
    grad_accum_steps = 4L,
    learning_rate = NA_real_,
    device = "cuda",
    run_id = "run123",
    mode = "local",
    resolution = 384L,
    workspace_dir = workspace_dir
  )

  model_id <- testthat::with_mocked_bindings(
    petrographer:::finalize_trained_model(model_dir, config, duration_mins = 12.5),
    pin_model = function(model_dir, model_id, board, metadata) {
      captured_metadata <<- metadata
      NULL
    },
    .get_model_board = function() structure(list(), class = "board_fake"),
    .package = "petrographer"
  )

  manifest <- jsonlite::read_json(file.path(model_dir, "manifest.json"))
  summary <- jsonlite::read_json(file.path(model_dir, "training_summary.json"))

  expect_equal(model_id, "seg_model")
  expect_equal(manifest$manifest_version, 1L)
  expect_equal(manifest$model$variant, "seg_small")
  expect_equal(manifest$artifacts$training_summary, "training_summary.json")
  expect_equal(summary$training_summary_version, 1L)
  expect_equal(summary$model$id, "seg_model")
  expect_equal(captured_metadata$catalog_summary$model_variant, "seg_small")
  expect_equal(captured_metadata$catalog_summary$dataset_id, "rocks")
  expect_equal(captured_metadata$catalog_summary$final_metrics$ap_50_95, 0.5)
})

test_that("submit_hpc_training returns a staged handle", {
  config <- list(
    data_dir = "/tmp/dataset.tar.gz",
    dataset_id = "rocks",
    model_id = "rocks_small",
    run_id = "run123",
    model_variant = "small",
    resolution = NULL,
    epochs = 10L,
    batch_size = "auto",
    grad_accum_steps = 1L,
    learning_rate = NA_real_,
    workspace_dir = withr::local_tempdir(),
    use_amp = TRUE,
    amp_dtype = "bf16",
    gradient_checkpointing = FALSE,
    num_workers = 2L,
    time_hours = 4,
    validate_every = 2L,
    early_stopping_patience = NULL
  )

  uploaded <- FALSE
  submitted <- FALSE

  handle <- testthat::with_mocked_bindings(
    petrographer:::submit_hpc_training(config),
    hpc_training_setup = function(...) list(example = TRUE),
    hpc_upload_artifacts = function(setup) {
      uploaded <<- isTRUE(setup$example)
      invisible(NULL)
    },
    hpc_submit_job = function(setup) {
      submitted <<- isTRUE(setup$example)
      "job-123"
    },
    .package = "petrographer"
  )

  expect_s3_class(handle, "petrographer_hpc_training")
  expect_true(uploaded)
  expect_true(submitted)
  expect_equal(handle$job, "job-123")
  expect_identical(handle$config$model_id, "rocks_small")
})

test_that("train_model_hpc uses staged submit wait collect flow", {
  called <- character()

  result <- testthat::with_mocked_bindings(
    petrographer:::train_model_hpc(
      data_dir = "/tmp/dataset.tar.gz",
      dataset_id = "rocks",
      model_id = "rocks_small",
      run_id = "run123",
      model_variant = "small",
      resolution = NULL,
      epochs = 10L,
      batch_size = "auto",
      grad_accum_steps = 1L,
      learning_rate = NA_real_,
      workspace_dir = withr::local_tempdir(),
      use_amp = TRUE,
      amp_dtype = "bf16",
      gradient_checkpointing = FALSE,
      num_workers = 2L,
      time_hours = 4,
      validate_every = 2L,
      early_stopping_patience = NULL
    ),
    submit_hpc_training = function(config) {
      called <<- c(called, "submit")
      structure(list(config = config, setup = list(), job = "job"), class = "petrographer_hpc_training")
    },
    wait_hpc_training = function(handle) {
      called <<- c(called, "wait")
      handle
    },
    collect_hpc_training = function(handle) {
      called <<- c(called, "collect")
      "/tmp/hpc_output"
    },
    .package = "petrographer"
  )

  expect_equal(called, c("submit", "wait", "collect"))
  expect_equal(result, "/tmp/hpc_output")
})
