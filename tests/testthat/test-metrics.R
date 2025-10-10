test_that("parse_metrics separates training and validation rows", {
  metrics_path <- withr::local_tempfile(fileext = ".json")
  lines <- c(
    jsonlite::toJSON(list(iteration = 10L, total_loss = 0.5), auto_unbox = TRUE),
    jsonlite::toJSON(list(iteration = 20L, total_loss = 0.4), auto_unbox = TRUE),
    jsonlite::toJSON(list(iteration = 20L, bbox_ap = 0.3), auto_unbox = TRUE),
    jsonlite::toJSON(list(iteration = 30L, total_loss = 0.3), auto_unbox = TRUE)
  )
  writeLines(lines, metrics_path)

  parsed <- petrographer:::parse_metrics(metrics_path)

  expect_equal(parsed$training$iteration, c(10L, 20L, 30L))
  expect_equal(nrow(parsed$validation), 1)
  expect_equal(parsed$validation$iteration, 20L)
  expect_false(".row_id" %in% names(parsed$training))
  expect_false(".row_id" %in% names(parsed$validation))
  expect_true(tibble::is_tibble(parsed$classwise))
})
