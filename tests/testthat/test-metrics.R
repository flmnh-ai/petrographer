# ---------------------------------------------------------------------------
# RF-DETR >= 1.6.0 — PyTorch Lightning CSVLogger (metrics.csv)
# ---------------------------------------------------------------------------

test_that("parse_metrics reads PTL metrics.csv and collapses per-epoch rows", {
  csv_path <- withr::local_tempfile(fileext = ".csv")
  # PTL writes train and val on separate rows with NaN in the other columns.
  # Two epochs, classwise columns for two classes, plus an EMA AP column.
  csv <- paste(
    "epoch,step,train/loss,train/loss_ce,train/loss_bbox,train/loss_giou,val/loss,val/mAP_50_95,val/mAP_50,val/F1,val/ema_mAP_50_95,val/AP/inclusion,val/AP/matrix,val/ema_AP/inclusion,lr-AdamW",
    # Epoch 0 train row
    "0,100,11.5,0.78,0.30,1.16,,,,,,,,,0.0001",
    # Epoch 0 val row
    "0,100,,,,,9.55,0.19,0.37,0.45,0.21,0.19,0.12,0.20,",
    # Epoch 1 train row
    "1,200,9.2,0.70,0.25,1.05,,,,,,,,,0.0001",
    # Epoch 1 val row
    "1,200,,,,,8.6,0.25,0.45,0.52,0.27,0.25,0.18,0.26,",
    sep = "\n"
  )
  writeLines(csv, csv_path)

  parsed <- petrographer:::parse_metrics(csv_path)

  # Training: one row per epoch, train/* prefix stripped
  expect_equal(parsed$training$epoch, c(0L, 1L))
  expect_true("loss" %in% names(parsed$training))
  expect_true("loss_bbox" %in% names(parsed$training))
  expect_equal(parsed$training$loss, c(11.5, 9.2))
  # lr-AdamW normalized to `lr` since it's the only lr-* column
  expect_true("lr" %in% names(parsed$training))
  expect_equal(parsed$training$lr, c(1e-4, 1e-4))

  # Validation: one row per epoch, val/ prefix stripped, aliases present
  expect_equal(nrow(parsed$validation), 2)
  expect_equal(parsed$validation$epoch, c(0L, 1L))
  expect_true(all(c("mAP_50_95", "mAP_50", "F1", "ema_mAP_50_95", "loss")
                  %in% names(parsed$validation)))
  # Compat aliases: ap -> mAP_50_95, ap50 -> mAP_50, map -> mAP_50_95
  expect_true(all(c("ap", "ap50", "map") %in% names(parsed$validation)))
  expect_equal(parsed$validation$ap,  c(0.19, 0.25))
  expect_equal(parsed$validation$ap50, c(0.37, 0.45))
  expect_equal(parsed$validation$map, c(0.19, 0.25))
  # Classwise AP columns should be pulled out of scalar validation
  expect_false("AP/inclusion" %in% names(parsed$validation))
  expect_false("AP/matrix" %in% names(parsed$validation))

  # Classwise: long format, one row per (epoch, class_name)
  expect_equal(nrow(parsed$classwise), 4)
  expect_setequal(parsed$classwise$class_name, c("inclusion", "matrix"))
  expect_true(all(c("map_50_95", "map_50_95_ema") %in% names(parsed$classwise)))
  incl_e0 <- parsed$classwise[parsed$classwise$epoch == 0 &
                              parsed$classwise$class_name == "inclusion", ]
  expect_equal(incl_e0$map_50_95, 0.19)
  expect_equal(incl_e0$map_50_95_ema, 0.20)
})

test_that("parse_metrics tolerates PTL csv without EMA or classwise columns", {
  csv_path <- withr::local_tempfile(fileext = ".csv")
  csv <- paste(
    "epoch,step,train/loss,val/mAP_50_95",
    "0,100,2.5,",
    "0,100,,0.1",
    "1,200,1.8,",
    "1,200,,0.2",
    sep = "\n"
  )
  writeLines(csv, csv_path)

  parsed <- petrographer:::parse_metrics(csv_path)
  expect_equal(parsed$training$loss, c(2.5, 1.8))
  expect_equal(parsed$validation$mAP_50_95, c(0.1, 0.2))
  # No classwise columns present in the input -> empty classwise tibble
  expect_equal(nrow(parsed$classwise), 0)
})

# ---------------------------------------------------------------------------
# RF-DETR < 1.6.0 — native training loop (log.txt JSONL)
# ---------------------------------------------------------------------------

test_that("parse_metrics splits train/test rows and unpacks COCO eval (log.txt)", {
  log_path <- withr::local_tempfile(fileext = ".txt")
  epoch0 <- list(
    train_lr = 1e-4,
    train_class_error = 0,
    train_loss = 11.5,
    train_loss_ce = 0.78,
    train_loss_bbox = 0.30,
    train_loss_giou = 1.16,
    train_epoch_time = "0:01:49",  # non-numeric — parser should drop
    test_class_error = 0,
    test_loss = 9.55,
    test_loss_ce = 0.77,
    test_loss_bbox = 0.15,
    test_loss_giou = 0.95,
    test_coco_eval_bbox = c(0.19, 0.37, 0.18, 0.18, 0.28, 0.15,
                            0.01, 0.05, 0.29, 0.27, 0.46, 0.20),
    test_results_json = list(
      class_map = list(list(class = "inclusion",
                            `map@50:95` = 0.19,
                            `map@50` = 0.37,
                            precision = 0.62,
                            recall = 0.49)),
      map = 0.37,
      precision = 0.62,
      recall = 0.49
    ),
    epoch = 0L,
    n_parameters = 33363638L
  )
  epoch1 <- epoch0
  epoch1$epoch <- 1L
  epoch1$train_loss <- 9.2
  epoch1$test_loss <- 8.6

  writeLines(c(jsonlite::toJSON(epoch0, auto_unbox = TRUE),
               jsonlite::toJSON(epoch1, auto_unbox = TRUE)),
             log_path)

  parsed <- petrographer:::parse_metrics(log_path)

  # Training
  expect_equal(parsed$training$epoch, c(0L, 1L))
  expect_true("loss" %in% names(parsed$training))
  expect_true("loss_bbox" %in% names(parsed$training))
  expect_equal(parsed$training$loss, c(11.5, 9.2))
  expect_false("epoch_time" %in% names(parsed$training))

  # Validation: keyed on epoch, COCO array unpacked
  expect_equal(nrow(parsed$validation), 2)
  expect_equal(parsed$validation$epoch, c(0L, 1L))
  expect_true(all(c("ap", "ap50", "ap75", "map", "precision", "recall")
                  %in% names(parsed$validation)))
  expect_equal(parsed$validation$ap[1], 0.19)
  expect_equal(parsed$validation$ap50[1], 0.37)
  expect_equal(parsed$validation$map[1], 0.37)
  expect_equal(parsed$validation$loss, c(9.55, 8.6))

  # Classwise: one row per (epoch, class)
  expect_equal(nrow(parsed$classwise), 2)
  expect_equal(parsed$classwise$class_name, rep("inclusion", 2))
  expect_equal(parsed$classwise$map_50_95, c(0.19, 0.19))
})

test_that("parse_metrics handles missing / empty log gracefully", {
  expect_equal(nrow(petrographer:::parse_metrics("/nonexistent/log.txt")$training), 0)
  expect_equal(nrow(petrographer:::parse_metrics("/nonexistent/metrics.csv")$training), 0)

  empty_path <- withr::local_tempfile(fileext = ".txt")
  writeLines(character(0), empty_path)
  expect_equal(nrow(petrographer:::parse_metrics(empty_path)$training), 0)

  empty_csv <- withr::local_tempfile(fileext = ".csv")
  writeLines(character(0), empty_csv)
  expect_equal(nrow(petrographer:::parse_metrics(empty_csv)$training), 0)
})

test_that("parse_metrics falls back to row order when epoch key is missing (log.txt)", {
  log_path <- withr::local_tempfile(fileext = ".txt")
  lines <- c(
    jsonlite::toJSON(list(train_loss = 1.2), auto_unbox = TRUE),
    jsonlite::toJSON(list(train_loss = 0.8), auto_unbox = TRUE)
  )
  writeLines(lines, log_path)

  parsed <- petrographer:::parse_metrics(log_path)
  expect_equal(parsed$training$epoch, c(0L, 1L))
  expect_equal(parsed$training$loss, c(1.2, 0.8))
})
