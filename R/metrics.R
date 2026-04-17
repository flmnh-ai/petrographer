# Metrics parsing
#
# RF-DETR has two very different log formats depending on version:
#
#   * >= 1.6.0 (PyTorch Lightning migration): writes `metrics.csv` via
#     `CSVLogger(save_dir=output_dir, name="", version="")`, which lands at
#     `{output_dir}/metrics.csv` directly (not under `lightning_logs/...`).
#     Wide CSV, one row per logging event, with NaN/empty cells where a
#     metric wasn't updated on that event. Typical columns:
#       `epoch`, `step`,
#       `train/loss`, `train/loss_ce`, `train/loss_bbox`, `train/loss_giou`,
#       `val/loss`, `val/mAP_50_95`, `val/mAP_50`, `val/F1`,
#       `val/ema_mAP_50_95`, `val/ema_mAP_50`, `val/ema_F1`,
#       `val/AP/<class>`, `val/ema_AP/<class>`,
#       `lr-<optimizer>` / `lr/<group>`.
#
#   * < 1.6.0 (native training loop): writes `log.txt` as JSONL with
#     explicit `epoch`, `train_*`, `test_*` keys plus a 12-element
#     `test_coco_eval_bbox` array and nested `test_results_json`.
#
# `parse_metrics()` dispatches on filename, but either path returns the same
# tibble triple (`training`, `validation`, `classwise`) so downstream callers
# (`evaluate_training`, pin metadata, notebooks) don't need to care.

#' Parse RF-DETR training metrics into tibbles
#'
#' Reads RF-DETR training artifacts and returns tibbles for training losses,
#' validation metrics, and per-class AP when available.
#'
#' Dispatches on filename:
#'   * `metrics.csv` -> PyTorch Lightning CSVLogger (RF-DETR >= 1.6.0).
#'   * `log.txt` -> native-loop JSONL (RF-DETR < 1.6.0).
#'
#' Both paths produce the same output shape so callers can stay agnostic.
#'
#' Returned tibbles use a common, stable schema (prefix-stripped column names):
#'   * `training`  - keyed on `epoch`, with numeric columns for loss
#'                   components, `lr`, etc.
#'   * `validation`- keyed on `epoch`, with validation metrics. Where possible
#'                   we expose a canonical `ap` alias (maps to `mAP_50_95` on
#'                   PTL and to COCO `AP` on legacy) plus `ap50`, `map`,
#'                   `precision`, `recall` when available.
#'   * `classwise` - long format, one row per (`epoch`, `class_name`) with
#'                   `map_50_95`, `map_50`, `precision`, `recall` when
#'                   provided. On PTL runs, we also populate `map_50_95_ema`
#'                   and `map_50_ema` when EMA AP columns are present.
#'
#' @param log_file Path to `metrics.csv` (PTL) or `log.txt` (native-loop JSONL).
#' @return list(training, validation, classwise)
#' @keywords internal
parse_metrics <- function(log_file) {
  empty <- list(
    training = tibble::tibble(),
    validation = tibble::tibble(),
    classwise = tibble::tibble()
  )
  if (!fs::file_exists(log_file)) return(empty)

  fname <- tolower(fs::path_file(log_file))
  if (endsWith(fname, ".csv")) {
    .parse_metrics_csv(log_file)
  } else {
    .parse_metrics_jsonl(log_file)
  }
}

# ---------------------------------------------------------------------------
# PTL CSVLogger (RF-DETR >= 1.6.0)
# ---------------------------------------------------------------------------

.parse_metrics_csv <- function(csv_file) {
  empty <- list(
    training = tibble::tibble(),
    validation = tibble::tibble(),
    classwise = tibble::tibble()
  )

  df <- tryCatch(
    readr::read_csv(csv_file, show_col_types = FALSE, progress = FALSE,
                    na = c("", "NA", "NaN", "nan")),
    error = function(e) NULL
  )
  if (is.null(df) || nrow(df) == 0L) return(empty)

  # epoch may be missing or named differently; fall back to step or row index
  if (!"epoch" %in% names(df)) {
    if ("step" %in% names(df)) {
      df$epoch <- df$step
    } else {
      df$epoch <- seq_len(nrow(df)) - 1L
    }
  }
  df$epoch <- suppressWarnings(as.integer(df$epoch))

  train_cols <- grep("^train[/_]", names(df), value = TRUE)
  val_cols <- grep("^val[/_]", names(df), value = TRUE)
  # Learning-rate columns land outside train/* — e.g. `lr-AdamW`, `lr/group0`.
  lr_cols <- grep("^lr[-/]", names(df), value = TRUE)

  # Classwise columns look like `val/AP/<class>` and `val/ema_AP/<class>`.
  cls_ap_cols <- grep("^val/AP/", val_cols, value = TRUE)
  cls_ema_cols <- grep("^val/ema_AP/", val_cols, value = TRUE)
  cls_map50_cols <- grep("^val/AP_50/", val_cols, value = TRUE)
  cls_map50_ema_cols <- grep("^val/ema_AP_50/", val_cols, value = TRUE)
  class_cols <- c(cls_ap_cols, cls_ema_cols, cls_map50_cols, cls_map50_ema_cols)

  val_scalar_cols <- setdiff(val_cols, class_cols)

  training <- .collapse_by_epoch(df, c(train_cols, lr_cols))
  if (nrow(training) > 0L) {
    nm <- names(training)
    nm <- sub("^train/", "", nm)
    nm <- sub("^train_", "", nm)
    # Normalize `lr-Something` / `lr/group0` to a single `lr` when unambiguous
    lr_idx <- grepl("^lr[-/]", nm)
    if (sum(lr_idx) == 1L) nm[lr_idx] <- "lr"
    names(training) <- nm
  }

  validation <- .collapse_by_epoch(df, val_scalar_cols)
  if (nrow(validation) > 0L) {
    names(validation) <- sub("^val/", "", names(validation))
    # Canonical aliases so downstream code (evaluate_training, notebooks)
    # can look up `ap` / `ap50` / `map` without caring about the version.
    validation <- .add_val_aliases(validation)
  }

  classwise <- .pivot_classwise_csv(df, cls_ap_cols, cls_ema_cols,
                                    cls_map50_cols, cls_map50_ema_cols)

  list(training = training, validation = validation, classwise = classwise)
}

# Group a data.frame by epoch and collapse each column to its last non-NA
# value. PTL logs train and val on separate rows with NaN in the other
# columns; this flattens to one row per epoch without losing data.
.collapse_by_epoch <- function(df, cols) {
  cols <- intersect(cols, names(df))
  if (length(cols) == 0L) return(tibble::tibble())

  sub <- df[, c("epoch", cols), drop = FALSE]
  # Drop rows that are all-NA across the relevant columns (nothing to keep).
  any_non_na <- rowSums(!is.na(sub[, cols, drop = FALSE])) > 0L
  sub <- sub[any_non_na, , drop = FALSE]
  if (nrow(sub) == 0L) return(tibble::tibble())

  # Within each epoch, take the last non-NA value per column.
  ep_order <- order(sub$epoch)
  sub <- sub[ep_order, , drop = FALSE]

  out <- dplyr::group_by(sub, .data$epoch)
  out <- dplyr::summarise(out,
    dplyr::across(dplyr::all_of(cols), function(x) {
      x_non_na <- x[!is.na(x)]
      if (length(x_non_na) == 0L) NA_real_ else x_non_na[length(x_non_na)]
    }),
    .groups = "drop"
  )
  tibble::as_tibble(out)
}

# Add compatibility aliases so downstream code can look up the usual names
# regardless of RF-DETR version.
.add_val_aliases <- function(validation) {
  alias <- function(target, sources) {
    if (target %in% names(validation)) return(invisible())
    for (s in sources) {
      if (s %in% names(validation)) {
        validation[[target]] <<- validation[[s]]
        return(invisible())
      }
    }
  }
  alias("ap",    c("mAP_50_95", "mAP", "ema_mAP_50_95"))
  alias("ap50",  c("mAP_50", "ema_mAP_50"))
  alias("map",   c("mAP_50_95", "ema_mAP_50_95", "mAP"))
  validation
}

# PTL classwise reshape: one row per (epoch, class_name).
.pivot_classwise_csv <- function(df, ap_cols, ema_cols, ap50_cols, ema50_cols) {
  ap_cols    <- intersect(ap_cols,    names(df))
  ema_cols   <- intersect(ema_cols,   names(df))
  ap50_cols  <- intersect(ap50_cols,  names(df))
  ema50_cols <- intersect(ema50_cols, names(df))

  all_cols <- c(ap_cols, ema_cols, ap50_cols, ema50_cols)
  if (length(all_cols) == 0L) return(tibble::tibble())

  long <- dplyr::select(df, "epoch", dplyr::all_of(all_cols))
  long <- tidyr::pivot_longer(long,
                              cols = -"epoch",
                              names_to = "key",
                              values_to = "value",
                              values_drop_na = TRUE)
  if (nrow(long) == 0L) return(tibble::tibble())

  long$metric <- dplyr::case_when(
    grepl("^val/ema_AP_50/", long$key) ~ "map_50_ema",
    grepl("^val/AP_50/",     long$key) ~ "map_50",
    grepl("^val/ema_AP/",    long$key) ~ "map_50_95_ema",
    grepl("^val/AP/",        long$key) ~ "map_50_95",
    TRUE ~ NA_character_
  )
  long$class_name <- sub("^val/(?:ema_)?AP(?:_50)?/", "", long$key)
  long$key <- NULL

  long <- tidyr::pivot_wider(long,
                             id_cols = c("epoch", "class_name"),
                             names_from = "metric",
                             values_from = "value")

  # Ensure canonical column order / presence for downstream consumers.
  for (cn in c("map_50_95", "map_50", "map_50_95_ema", "map_50_ema")) {
    if (!cn %in% names(long)) long[[cn]] <- NA_real_
  }
  dplyr::select(long, "epoch", "class_name",
                "map_50_95", "map_50", "map_50_95_ema", "map_50_ema")
}

# ---------------------------------------------------------------------------
# Legacy native-loop JSONL (RF-DETR < 1.6.0)
# ---------------------------------------------------------------------------

.parse_metrics_jsonl <- function(log_file) {
  empty <- list(
    training = tibble::tibble(),
    validation = tibble::tibble(),
    classwise = tibble::tibble()
  )

  raw_lines <- readLines(log_file, warn = FALSE)
  raw_lines <- raw_lines[nzchar(trimws(raw_lines))]
  if (length(raw_lines) == 0) return(empty)

  # RF-DETR occasionally writes NaN in loss fields (e.g., on divergence); JSON
  # spec disallows it, so swap to null before parsing.
  sanitized_lines <- stringr::str_replace_all(raw_lines, "(?<=[:\\s])NaN(?=[,}\\s])", "null")
  replaced_nan <- !identical(raw_lines, sanitized_lines)

  records <- lapply(sanitized_lines, function(line) {
    tryCatch(jsonlite::fromJSON(line, simplifyVector = FALSE),
             error = function(e) NULL)
  })
  records <- records[!vapply(records, is.null, logical(1))]
  if (length(records) == 0) return(empty)

  if (replaced_nan) {
    warning("log.txt contained NaN values; replacing with NA for compatibility",
            call. = FALSE)
  }

  epochs <- vapply(seq_along(records), function(i) {
    ep <- records[[i]][["epoch"]]
    if (is.null(ep) || !is.numeric(ep)) i - 1L else as.integer(ep)
  }, integer(1))

  # Training tibble: train_* keys, prefix stripped, plus epoch
  training_rows <- lapply(seq_along(records), function(i) {
    rec <- records[[i]]
    train_keys <- grep("^train_", names(rec), value = TRUE)
    if (length(train_keys) == 0) return(NULL)
    vals <- rec[train_keys]
    vals <- vals[vapply(vals, function(v) is.numeric(v) && length(v) == 1L, logical(1))]
    if (length(vals) == 0) return(NULL)
    names(vals) <- sub("^train_", "", names(vals))
    c(list(epoch = epochs[i]), lapply(vals, function(v) if (is.null(v)) NA_real_ else v))
  })
  training_rows <- training_rows[!vapply(training_rows, is.null, logical(1))]
  training <- if (length(training_rows) > 0) {
    dplyr::bind_rows(lapply(training_rows, tibble::as_tibble))
  } else tibble::tibble()

  # Validation tibble
  coco_names <- c(
    "ap", "ap50", "ap75", "ap_small", "ap_medium", "ap_large",
    "ar1", "ar10", "ar100", "ar_small", "ar_medium", "ar_large"
  )

  validation_rows <- lapply(seq_along(records), function(i) {
    rec <- records[[i]]
    test_keys <- grep("^test_", names(rec), value = TRUE)
    scalar_keys <- setdiff(test_keys, c("test_coco_eval_bbox", "test_results_json"))
    scalar_vals <- rec[scalar_keys]
    scalar_vals <- scalar_vals[vapply(scalar_vals,
                                      function(v) is.numeric(v) && length(v) == 1L,
                                      logical(1))]
    if (length(scalar_vals) > 0) {
      names(scalar_vals) <- sub("^test_", "", names(scalar_vals))
    }

    coco_vec <- rec[["test_coco_eval_bbox"]]
    coco_out <- stats::setNames(rep(NA_real_, length(coco_names)), coco_names)
    if (!is.null(coco_vec) && length(coco_vec) == length(coco_names)) {
      coco_out[] <- vapply(coco_vec,
                           function(v) if (is.null(v)) NA_real_ else as.numeric(v),
                           numeric(1))
    }

    results <- rec[["test_results_json"]]
    summary_map <- if (!is.null(results$map)) as.numeric(results$map) else NA_real_
    summary_prec <- if (!is.null(results$precision)) as.numeric(results$precision) else NA_real_
    summary_recall <- if (!is.null(results$recall)) as.numeric(results$recall) else NA_real_

    if (length(scalar_vals) == 0 &&
        all(is.na(coco_out)) &&
        is.na(summary_map)) return(NULL)

    c(
      list(epoch = epochs[i]),
      lapply(scalar_vals, function(v) if (is.null(v)) NA_real_ else v),
      as.list(coco_out),
      list(map = summary_map, precision = summary_prec, recall = summary_recall)
    )
  })
  validation_rows <- validation_rows[!vapply(validation_rows, is.null, logical(1))]
  validation <- if (length(validation_rows) > 0) {
    dplyr::bind_rows(lapply(validation_rows, tibble::as_tibble))
  } else tibble::tibble()

  # Classwise tibble
  classwise_rows <- lapply(seq_along(records), function(i) {
    rec <- records[[i]]
    cm <- rec[["test_results_json"]][["class_map"]]
    if (is.null(cm) || length(cm) == 0) return(NULL)
    rows <- lapply(cm, function(entry) {
      tibble::tibble(
        epoch = epochs[i],
        class_name = entry[["class"]] %||% NA_character_,
        map_50_95 = as.numeric(entry[["map@50:95"]] %||% NA_real_),
        map_50    = as.numeric(entry[["map@50"]]    %||% NA_real_),
        precision = as.numeric(entry[["precision"]] %||% NA_real_),
        recall    = as.numeric(entry[["recall"]]    %||% NA_real_)
      )
    })
    dplyr::bind_rows(rows)
  })
  classwise_rows <- classwise_rows[!vapply(classwise_rows, is.null, logical(1))]
  classwise <- if (length(classwise_rows) > 0) {
    dplyr::bind_rows(classwise_rows)
  } else tibble::tibble()

  list(training = training, validation = validation, classwise = classwise)
}
