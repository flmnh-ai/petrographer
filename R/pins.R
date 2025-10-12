# Model publishing and retrieval helpers built on pins --------------------------------

#' Local user board
#'
#' This board lives in the user's filesystem (e.g. Dropbox). It is writable and
#' versioned. Set `PETRO_PINS_PATH` to override the default path.
#' @export
pg_board_user <- function(path = Sys.getenv("PETRO_PINS_PATH", "")) {
  if (!requireNamespace("pins", quietly = TRUE)) {
    cli::cli_abort("pins is required. Install with install.packages('pins')")
  }

  if (!nzchar(path)) {
    path <- fs::path_expand("~/Dropbox/petrographer-pins")
  }
  fs::dir_create(path, recurse = TRUE)
  pins::board_folder(path, versioned = TRUE)
}

#' @export
pg_board_pkgdown <- function(path = fs::path("pkgdown", "assets", "pins")) {
  if (!requireNamespace("pins", quietly = TRUE)) {
    cli::cli_abort("pins is required. Install with install.packages('pins')")
  }
  path <- fs::path_abs(path)
  fs::dir_create(path, recurse = TRUE)
  pins::board_folder(path, versioned = TRUE)
}

#' Read-only hub board served from URL
#'
#' Uses `board_url()` backed by a manifest stored at the given URL. Requires
#' `PETRO_PINS_URL` to be set. Downloads cache into the petrographer model cache.
#' @export
pg_board_hub <- function(url = Sys.getenv("PETRO_PINS_URL", "")) {
  if (!nzchar(url)) {
    cli::cli_abort("PETRO_PINS_URL must be set to use the hosted pretrained models.")
  }
  if (!requireNamespace("pins", quietly = TRUE)) {
    cli::cli_abort("pins is required. Install with install.packages('pins')")
  }
  cache_dir <- pg_model_cache_root()
  fs::dir_create(cache_dir, recurse = TRUE)
  pins::board_url(url = url, cache = cache_dir, versioned = TRUE)
}


pg_coalesce <- function(...) {
  for (val in list(...)) {
    if (!is.null(val)) return(val)
  }
  NULL
}

pg_model_cache_root <- function() {
  fs::path(tools::R_user_dir("petrographer", which = "cache"), "models")
}

pg_model_cache_path <- function(model_id, version) {
  fs::path(pg_model_cache_root(), model_id, pg_coalesce(version, "latest"))
}

pg_model_manifest_path <- function(base_dir) {
  fs::path(base_dir, "manifest.json")
}

pg_model_iso_time <- function(x = Sys.time()) {
  format(as.POSIXct(x, tz = "UTC"), "%Y-%m-%dT%H:%M:%SZ")
}

pg_model_auto_version <- function(prefix = "v") {
  paste0(prefix, format(Sys.time(), "%Y%m%d%H%M%S"))
}

pg_model_collect_metrics <- function(metrics_path) {
  if (!fs::file_exists(metrics_path)) return(NULL)
  parsed <- parse_metrics(metrics_path)
  val <- if (nrow(parsed$validation) > 0) {
    as.list(parsed$validation[nrow(parsed$validation), , drop = FALSE])
  } else {
    NULL
  }
  train <- if (nrow(parsed$training) > 0) {
    as.list(parsed$training[nrow(parsed$training), , drop = FALSE])
  } else {
    NULL
  }
  list(validation = val, training = train)
}

pg_model_safe_copy <- function(files, dest_dir) {
  files <- files[fs::file_exists(files)]
  if (length(files) == 0) return(invisible(character(0)))
  fs::dir_create(dest_dir)
  purrr::walk(files, function(path) {
    fs::file_copy(path, fs::path(dest_dir, fs::path_file(path)), overwrite = TRUE)
  })
  invisible(files)
}

pg_model_prepare_manifest <- function(model_dir,
                                      model_id,
                                      version = NULL,
                                      metadata = list(),
                                      include_metrics = TRUE) {
  model_dir <- fs::path_abs(fs::path_norm(model_dir))
  if (!fs::dir_exists(model_dir)) {
    cli::cli_abort("Model directory not found: {.path {model_dir}}")
  }

  model_file <- fs::path(model_dir, "model_final.pth")
  config_file <- fs::path(model_dir, "config.yaml")
  best_file <- fs::path(model_dir, "model_best.pth")
  metrics_file <- fs::path(model_dir, "metrics.json")

  if (!fs::file_exists(model_file)) {
    cli::cli_abort("Missing file: {.path {model_file}}")
  }
  if (!fs::file_exists(config_file)) {
    cli::cli_abort("Missing file: {.path {config_file}}")
  }

  if (is.null(metadata$preview_image)) {
    default_preview <- fs::path(model_dir, "preview.png")
    if (fs::file_exists(default_preview)) {
      metadata$preview_image <- fs::path_file(default_preview)
    }
  } else if (fs::file_exists(metadata$preview_image)) {
    metadata$preview_image <- fs::path_file(metadata$preview_image)
  }

  manifest <- list(
    schema_version = 1L,
    model_id = model_id,
    version = pg_coalesce(version, metadata$version, pg_model_auto_version()),
    created = pg_model_iso_time(),
    artifacts = list(
      model_final = fs::path_file(model_file),
      config = fs::path_file(config_file)
    ),
    metadata = metadata,
    file_size_bytes = as.numeric(sum(fs::file_size(fs::dir_ls(model_dir, recurse = TRUE, type = "file")), na.rm = TRUE))
  )

  if (fs::file_exists(best_file)) {
    manifest$artifacts$model_best <- fs::path_file(best_file)
  }

  if (include_metrics && fs::file_exists(metrics_file)) {
    manifest$artifacts$metrics <- fs::path_file(metrics_file)
    manifest$metrics <- pg_model_collect_metrics(metrics_file)
  }

  manifest
}

pg_model_write_manifest <- function(model_dir, manifest) {
  fs::dir_create(model_dir)
  jsonlite::write_json(manifest, pg_model_manifest_path(model_dir), auto_unbox = TRUE, pretty = TRUE)
  invisible(manifest)
}

pg_model_bundle <- function(model_dir,
                            manifest,
                            include_metrics = TRUE,
                            additional_files = NULL) {
  bundle_dir <- fs::file_temp("pg-model-bundle")
  fs::dir_create(bundle_dir)

  files_core <- c(
    fs::path(model_dir, manifest$artifacts$model_final),
    fs::path(model_dir, manifest$artifacts$config)
  )
  if (!is.null(manifest$artifacts$model_best)) {
    files_core <- c(files_core, fs::path(model_dir, manifest$artifacts$model_best))
  }
  if (include_metrics && !is.null(manifest$artifacts$metrics)) {
    files_core <- c(files_core, fs::path(model_dir, manifest$artifacts$metrics))
  }
  files_core <- unique(c(files_core, additional_files))
  pg_model_safe_copy(files_core, bundle_dir)

  jsonlite::write_json(manifest, fs::path(bundle_dir, "manifest.json"), auto_unbox = TRUE, pretty = TRUE)

  safe_id <- gsub("[^A-Za-z0-9._-]", "-", manifest$model_id)
  bundle_path <- fs::file_temp(paste0(safe_id, "-", manifest$version, ".tar.gz"))
  old_wd <- getwd()
  on.exit(setwd(old_wd), add = TRUE)
  setwd(bundle_dir)
  archive_files <- fs::dir_ls(bundle_dir, all = TRUE, type = "any") |> fs::path_file()
  utils::tar(bundle_path, files = archive_files, compression = "gzip", tar = "internal")
  bundle_path
}

#' Publish a trained model to a pins board
#'
#' Builds a self-describing artifact bundle (weights, config, metrics, manifest)
#' and uploads it to the configured board. The manifest is also written back to
#' the source directory for local use.
#'
#' @param model_dir Directory containing Detectron2 artifacts.
#' @param model_id Pin/model identifier (e.g. "petrography/inclusions").
#' @param board Pins board (defaults to [pg_board_user()]).
#' @param metadata Optional metadata list merged into the manifest.
#' @param include_metrics Whether to include `metrics.json` when present.
#' @param manifest Optional pre-built manifest. When `NULL`, a manifest is
#'   derived automatically.
#' @return A list containing `manifest`, `pin_meta`, and `bundle_path` (invisible).
#' @export
pg_model_publish <- function(model_dir,
                             model_id,
                             board = pg_board_user(),
                             metadata = list(),
                             include_metrics = TRUE,
                             manifest = NULL,
                             write_manifest = TRUE) {
  if (!requireNamespace("pins", quietly = TRUE)) {
    cli::cli_abort("pins is required. Install with install.packages('pins')")
  }

  if (is.null(manifest)) {
    manifest <- pg_model_prepare_manifest(model_dir, model_id, metadata = metadata, include_metrics = include_metrics)
  } else {
    manifest$model_id <- pg_coalesce(manifest$model_id, model_id)
    manifest$metadata <- modifyList(pg_coalesce(manifest$metadata, list()), metadata)
  }

  pg_model_write_manifest(model_dir, manifest)
  bundle_path <- pg_model_bundle(model_dir, manifest, include_metrics = include_metrics)
  on.exit(fs::file_delete(bundle_path), add = TRUE)

  safe_pin <- gsub('/+', '--', model_id)
  pins::pin_upload(board, bundle_path, name = safe_pin, metadata = list(manifest = manifest))
  pin_meta <- pins::pin_meta(board, safe_pin)
  manifest$pin_version <- pin_meta$version

  if (isTRUE(write_manifest)) {
    pg_write_board_manifest(board)
  }

  invisible(list(manifest = manifest, pin_meta = pin_meta, bundle_path = bundle_path))
}

#' @export
publish_model <- function(model_dir,
                          name,
                          board = pg_board_user(),
                          metadata = list(),
                          include_metrics = TRUE,
                          write_manifest = TRUE) {
  pg_model_publish(model_dir,
                   model_id = name,
                   board = board,
                   metadata = metadata,
                   include_metrics = include_metrics,
                   write_manifest = write_manifest)
}

pg_write_board_manifest <- function(board) {
  if (!requireNamespace("pins", quietly = TRUE)) {
    cli::cli_abort("pins is required. Install with install.packages('pins')")
  }
  if (inherits(board, "pins_board_url")) {
    return(invisible(board))
  }
  tryCatch({
    pins::write_board_manifest(board)
  }, error = function(e) {
    cli::cli_warn("Failed to write board manifest: {e$message}")
  })
  invisible(board)
}


# Dataset publishing utilities -------------------------------------------------

pg_dataset_cache_root <- function() {
  fs::path(tools::R_user_dir("petrographer", which = "cache"), "datasets")
}

pg_dataset_cache_path <- function(dataset_id, version) {
  fs::path(pg_dataset_cache_root(), dataset_id, pg_coalesce(version, "latest"))
}

pg_dataset_resolve_manifest <- function(dir) {
  manifest_path <- fs::path(dir, "manifest.json")
  if (fs::file_exists(manifest_path)) {
    jsonlite::read_json(manifest_path, simplifyVector = FALSE)
  } else {
    NULL
  }
}

pg_dataset_collect_metadata <- function(dataset_dir) {
  validated <- validate_dataset(dataset_dir, quiet = TRUE)
  splits <- lapply(validated$splits, function(split) {
    list(
      name = split$name,
      images = split$images,
      has_annotations = split$has_annotations,
      annotation = if (!is.null(split$annotation_path)) fs::path_file(split$annotation_path) else NULL
    )
  })

  annotation_paths <- vapply(validated$splits, function(split) split$annotation_path %||% NA_character_, character(1))
  annotation_paths <- annotation_paths[!is.na(annotation_paths)]
  class_names <- character()
  if (length(annotation_paths) > 0) {
    cats <- tryCatch({
      ann <- jsonlite::read_json(annotation_paths[[1]])
      ann$categories
    }, error = function(e) NULL)
    if (!is.null(cats)) {
      class_names <- unique(vapply(cats, function(cat) cat$name %||% NA_character_, character(1)))
      class_names <- class_names[!is.na(class_names) & nzchar(class_names)]
    }
  }

  warnings <- unique(unlist(lapply(validated$diagnostics, function(diag) {
    if (is.null(diag$warnings)) character() else unlist(diag$warnings, use.names = FALSE)
  })))

  list(
    splits = splits,
    class_names = class_names,
    size_mb = validated$size_mb,
    warnings = warnings
  )
}

#' @noRd
pg_dataset_prepare_manifest <- function(dataset_dir,
                                        dataset_id,
                                        version = NULL,
                                        metadata = list()) {
  dataset_dir <- fs::path_abs(fs::path_norm(dataset_dir))
  if (!fs::dir_exists(dataset_dir)) {
    cli::cli_abort("Dataset directory not found: {.path {dataset_dir}}")
  }

  metadata$dataset_id <- metadata$dataset_id %||% dataset_id
  metadata$data_dir <- metadata$data_dir %||% dataset_dir
  if (is.null(metadata$preview_image)) {
    default_preview <- fs::path(dataset_dir, "preview.png")
    if (fs::file_exists(default_preview)) {
      metadata$preview_image <- fs::path_file(default_preview)
    }
  } else if (fs::file_exists(metadata$preview_image)) {
    metadata$preview_image <- fs::path_file(metadata$preview_image)
  }
  stats <- pg_dataset_collect_metadata(dataset_dir)
  manifest <- list(
    schema_version = 1L,
    dataset_id = dataset_id,
    version = pg_coalesce(version, metadata$version, pg_model_auto_version()),
    created = pg_model_iso_time(),
    artifacts = list(dataset_dir = fs::path_file(dataset_dir)),
    metadata = utils::modifyList(stats, metadata)
  )
  manifest
}
#' @noRd
pg_dataset_write_manifest <- function(dataset_dir, manifest) {
  jsonlite::write_json(
    manifest,
    fs::path(dataset_dir, "dataset_manifest.json"),
    auto_unbox = TRUE,
    pretty = TRUE
  )
  invisible(manifest)
}

#' @noRd
pg_dataset_bundle <- function(dataset_dir, manifest) {
  bundle_dir <- fs::file_temp("pg-dataset-bundle")
  fs::dir_create(bundle_dir)
  dataset_name <- manifest$artifacts$dataset_dir %||% fs::path_file(dataset_dir)
  dataset_name <- gsub("[/\\]", "_", dataset_name)
  dest_dir <- fs::path(bundle_dir, dataset_name)
  fs::dir_copy(dataset_dir, dest_dir, overwrite = TRUE)
  jsonlite::write_json(manifest, fs::path(bundle_dir, "manifest.json"), auto_unbox = TRUE, pretty = TRUE)

  safe_id <- gsub("[^A-Za-z0-9._-]", "-", manifest$dataset_id)
  bundle_path <- fs::file_temp(paste0(safe_id, "-", manifest$version, ".tar.gz"))
  old_wd <- getwd()
  on.exit(setwd(old_wd), add = TRUE)
  setwd(bundle_dir)
  archive_files <- fs::dir_ls(bundle_dir, all = TRUE, type = "any") |> fs::path_file()
  utils::tar(bundle_path, files = archive_files, compression = "gzip", tar = "internal")
  bundle_path
}

#' Publish a dataset bundle as a pin
#'
#' @param dataset_dir Directory containing the dataset splits.
#' @param dataset_id Pin name to publish under.
#' @param board Pins board to upload to (defaults to [pg_board_user()]).
#' @param metadata Optional metadata merged into the manifest.
#' @param write_manifest When `TRUE`, rewrites the board `_pins.yaml` manifest.
#' @return Invisibly returns a list with manifest, pin metadata, and bundle path.
#' @export
pg_dataset_publish <- function(dataset_dir,
                               dataset_id,
                               board = pg_board_user(),
                               metadata = list(),
                               write_manifest = TRUE) {
  pin_name <- gsub("/+$", "", dataset_id)
  safe_pin <- gsub("/", "--", pin_name)
  manifest <- pg_dataset_prepare_manifest(dataset_dir, dataset_id, metadata = metadata)
  pg_dataset_write_manifest(dataset_dir, manifest)
  bundle_path <- pg_dataset_bundle(dataset_dir, manifest)
  on.exit(fs::file_delete(bundle_path), add = TRUE)
  pins::pin_upload(board, bundle_path, name = safe_pin, metadata = list(manifest = manifest))
  pin_meta <- pins::pin_meta(board, safe_pin)
  manifest$pin_version <- pin_meta$version
  if (isTRUE(write_manifest)) {
    pg_write_board_manifest(board)
  }
  invisible(list(manifest = manifest, pin_meta = pin_meta, bundle_path = bundle_path))
}

#' @noRd
pg_dataset_download_bundle <- function(dataset_id,
                                       version = NULL,
                                       board) {
  safe_id <- gsub("/+", "--", dataset_id)
  files <- if (is.null(version)) {
    pins::pin_download(board, safe_id)
  } else {
    pins::pin_download(board, safe_id, version = version)
  }
  bundle <- files[grepl("\\.(tar\\.gz|tar\\.bz2|tar\\.xz|zip)$", files, ignore.case = TRUE)]
  list(bundle = bundle, raw_files = files)
}

#' @noRd
pg_dataset_unpack_bundle <- function(bundle_path, cache_dir) {
  fs::dir_create(cache_dir, recurse = TRUE)
  if (grepl("\\.(zip)$", bundle_path, ignore.case = TRUE)) {
    utils::unzip(bundle_path, exdir = cache_dir)
  } else {
    utils::untar(bundle_path, exdir = cache_dir)
  }
  manifest <- pg_dataset_resolve_manifest(cache_dir)
  if (is.null(manifest)) {
    cli::cli_abort("Downloaded dataset bundle missing manifest.json")
  }
  manifest
}

#' Install (or resolve) a dataset bundle from pins
#'
#' @param dataset_id Dataset pin identifier.
#' @param version Optional version hash (defaults to latest).
#' @param board Pins board to read from (defaults to hosted hub or user board).
#' @return List with dataset directory, manifest, cache dir, and pin metadata.
#' @export
pg_dataset_from_pretrained <- function(dataset_id,
                                       version = NULL,
                                       board = NULL) {
  if (is.null(board)) {
    board <- tryCatch(pg_board_hub(), error = function(e) pg_board_user())
  }
  safe_id <- gsub("/+", "--", dataset_id)
  pin_meta <- pins::pin_meta(board, safe_id, version = version)
  manifest_meta <- pg_coalesce(pin_meta$user$manifest, pin_meta$metadata$manifest)
  cache_version <- pg_coalesce(pin_meta$version, manifest_meta$version, "latest")
  cache_id <- gsub("/+", "--", dataset_id)
  cache_dir <- pg_dataset_cache_path(cache_id, cache_version)

  manifest <- pg_dataset_resolve_manifest(cache_dir)
  if (is.null(manifest)) {
    download <- pg_dataset_download_bundle(dataset_id, version = version, board = board)
    bundle <- download$bundle
    if (length(bundle) == 0) {
      fs::dir_create(cache_dir, recurse = TRUE)
      purrr::walk(download$raw_files, function(path) {
        fs::file_copy(path, fs::path(cache_dir, fs::path_file(path)), overwrite = TRUE)
      })
      manifest <- pg_dataset_resolve_manifest(cache_dir)
      if (is.null(manifest)) {
        cli::cli_abort("Downloaded dataset bundle missing manifest.json")
      }
    } else {
      manifest <- pg_dataset_unpack_bundle(bundle[[1]], cache_dir)
    }
  }

  manifest$pin_version <- pin_meta$version
  manifest$dataset_id <- pg_coalesce(manifest$dataset_id, dataset_id)
  dataset_root <- fs::path(cache_dir, pg_coalesce(manifest$artifacts$dataset_dir, manifest$dataset_id))

  list(
    dataset_dir = dataset_root,
    manifest = manifest,
    cache_dir = cache_dir,
    pin_meta = pin_meta
  )
}

#' Cache a published dataset bundle locally
#'
#' @inheritParams pg_dataset_from_pretrained
#' @return Invisibly returns the resolved manifest and cache paths.
#' @export
pg_install_dataset <- function(dataset_id,
                               version = NULL,
                               board = NULL) {
  resolved <- pg_dataset_from_pretrained(dataset_id, version = version, board = board)
  invisible(resolved)
}


#' Build a catalog tibble for published models
#'
#' @param board Pins board to read from (defaults to pkgdown board).
#' @return Tibble compatible with `bs4cards::bs4cards()`.
#' @export
pg_model_catalog <- function(board = pg_board_pkgdown()) {
  pin_names <- tryCatch(pins::pin_list(board), error = function(e) character())
  if (!length(pin_names)) return(tibble::tibble())

  purrr::map_dfr(pin_names, function(name) {
    meta <- tryCatch(pins::pin_meta(board, name), error = function(e) NULL)
    if (is.null(meta)) return(NULL)
    manifest <- pg_coalesce(meta$user$manifest, meta$metadata$manifest)
    if (is.null(manifest) || is.null(manifest$model_id)) return(NULL)

    version <- pg_coalesce(manifest$pin_version, meta$version, "unknown")
    created <- manifest$created %||% ""
    size_mb <- if (!is.null(manifest$file_size_bytes)) round(manifest$file_size_bytes / (1024^2), 1) else NA_real_

    md <- manifest$metadata %||% list()
    metrics <- manifest$metrics$validation %||% list()
    segm_ap <- metrics$segm_AP %||% NA_real_
    bbox_ap <- metrics$bbox_AP %||% NA_real_
    dataset_path <- md$dataset_id %||% md$data_dir %||% ""
    class_names <- md$class_names %||% character()

    preview_data <- NA_character_
    preview_rel <- md$preview_image %||% NA_character_
    if (!is.na(preview_rel) && !is.null(board$path)) {
      preview_abs <- fs::path(board$path, name, version, preview_rel)
      if (fs::file_exists(preview_abs)) {
        preview_data <- base64enc::dataURI(file = preview_abs)
      }
    }

    badge_vals <- c(
      Backbone = md$backbone %||% "",
      Freeze = if (!is.null(md$freeze_at)) as.character(md$freeze_at) else "",
      `Imgs/Batch` = if (!is.null(md$ims_per_batch)) as.character(md$ims_per_batch) else ""
    )
    badge_vals <- badge_vals[nzchar(badge_vals)]

    stats_lines <- c(
      if (!is.na(segm_ap)) glue::glue("**Segm AP**: {round(segm_ap, 3)}") else NULL,
      if (!is.na(bbox_ap)) glue::glue("**BBox AP**: {round(bbox_ap, 3)}") else NULL,
      if (nzchar(dataset_path)) glue::glue("**Dataset**: {dataset_path}") else NULL,
      if (length(class_names)) glue::glue("**Classes**: {paste(class_names, collapse = ', ')}") else NULL
    )
    body <- paste(stats_lines, collapse = '<br>')
    if (!nzchar(body)) body <- '<em>No metrics recorded.</em>'
    body <- paste0(body, '<br><br><code>pg_install_pretrained("', manifest$model_id %||% name, '")</code>')

    download_btn <- list()
    if (!is.null(board$path)) {
      version_dir <- fs::path(board$path, name, version)
      bundle_file <- tryCatch(fs::dir_ls(version_dir, type = "file", glob = "*.tar*"), error = function(e) character())
      if (length(bundle_file)) {
        rel_path <- fs::path_rel(bundle_file[[1]], board$path)
        download_btn <- list(list(
          text = "Download",
          href = as.character(fs::path("..", rel_path)),
          icon = "fa-solid fa-download",
          class = "btn-primary"
        ))
      }
    }

    subtitle_parts <- c(paste0("Version ", version))
    if (!is.na(size_mb)) subtitle_parts <- c(subtitle_parts, glue::glue("{size_mb} MB"))
    created_display <- if (nzchar(created)) substr(created, 1, 10) else ""
    if (nzchar(created_display)) subtitle_parts <- c(subtitle_parts, created_display)

    tibble::tibble(
      title = manifest$model_id %||% name,
      subtitle = paste(subtitle_parts, collapse = " · "),
      body = body,
      badges = list(badge_vals),
      buttons = list(download_btn),
      image_data = preview_data
    )
  })
}

#' Build a catalog tibble for published datasets
#'
#' @inheritParams pg_model_catalog
#' @return Tibble with dataset summary information.
#' @export
pg_dataset_catalog <- function(board = pg_board_pkgdown()) {
  pin_names <- tryCatch(pins::pin_list(board), error = function(e) character())
  if (!length(pin_names)) return(tibble::tibble())

  purrr::map_dfr(pin_names, function(name) {
    meta <- tryCatch(pins::pin_meta(board, name), error = function(e) NULL)
    if (is.null(meta)) return(NULL)
    manifest <- pg_coalesce(meta$user$manifest, meta$metadata$manifest)
    if (is.null(manifest) || is.null(manifest$dataset_id)) return(NULL)

    version <- pg_coalesce(manifest$pin_version, meta$version, "unknown")
    size_mb <- manifest$metadata$size_mb %||% NA_real_
    class_names <- manifest$metadata$class_names %||% character()

    splits <- manifest$metadata$splits %||% list()
    split_lines <- vapply(splits, function(split) {
      glue::glue("{stringr::str_to_title(split$name)}: {split$images} images")
    }, character(1))

    warning_lines <- manifest$metadata$warnings %||% character()

    preview_data <- NA_character_
    preview_rel <- manifest$metadata$preview_image %||% NA_character_
    if (!is.na(preview_rel) && !is.null(board$path)) {
      dataset_dir_name <- manifest$artifacts$dataset_dir %||% manifest$dataset_id
      preview_abs <- fs::path(board$path, name, version, dataset_dir_name, preview_rel)
      if (fs::file_exists(preview_abs)) {
        preview_data <- base64enc::dataURI(file = preview_abs)
      }
    }

  badge_vals <- c(
      `Total images` = as.character(sum(vapply(splits, function(split) split$images %||% 0L, integer(1)))),
      Classes = as.character(length(class_names))
    )
  badge_vals <- badge_vals[nzchar(badge_vals)]

    body_lines <- c(
      split_lines,
      if (length(class_names)) glue::glue("Classes: {paste(class_names, collapse = ', ')}") else NULL,
      if (length(warning_lines)) glue::glue("Warnings: {paste(warning_lines, collapse = '; ')}") else NULL
    )
    if (length(body_lines)) {
      body <- paste(body_lines, collapse = '<br>')
    } else {
      body <- '<em>No split summary available.</em>'
    }
    body <- paste0(body, '<br><br><code>pg_install_dataset("', manifest$dataset_id %||% name, '")</code>')

    download_btn <- list()
    if (!is.null(board$path)) {
      version_dir <- fs::path(board$path, name, version)
      bundle_file <- tryCatch(fs::dir_ls(version_dir, type = "file", glob = "*.tar*"), error = function(e) character())
      if (length(bundle_file)) {
        rel_path <- fs::path_rel(bundle_file[[1]], board$path)
        download_btn <- list(list(
          text = "Download",
          href = as.character(fs::path("..", rel_path)),
          icon = "fa-solid fa-download",
          class = "btn-primary"
        ))
      }
    }

    subtitle_parts <- c(paste0("Version ", version))
    if (!is.na(size_mb)) subtitle_parts <- c(subtitle_parts, glue::glue("{round(size_mb, 1)} MB"))

    tibble::tibble(
      title = manifest$dataset_id %||% name,
      subtitle = paste(subtitle_parts, collapse = " · "),
      body = body,
      badges = list(badge_vals),
      buttons = list(download_btn),
      image_data = preview_data
    )
  })
}

#' Save model/dataset catalog to disk for pkgdown use
#'
#' @param path Output path for the serialized catalog (RDS).
#' @param board Pins board to read from.
#' @return Invisibly returns the output path.
#' @export
pg_write_model_catalog <- function(path = fs::path("pkgdown", "assets", "model_catalog.rds"),
                                   board = pg_board_pkgdown()) {
  fs::dir_create(fs::path_dir(path), recurse = TRUE)
  catalog <- list(
    models = pg_model_catalog(board),
    datasets = pg_dataset_catalog(board)
  )
  saveRDS(catalog, path)
  cli::cli_alert_success("Model catalog saved to {.path {path}}")
  invisible(path)
}




#' @noRd
pg_model_resolve_manifest <- function(dir) {
  manifest_path <- pg_model_manifest_path(dir)
  if (fs::file_exists(manifest_path)) {
    jsonlite::read_json(manifest_path, simplifyVector = FALSE)
  } else {
    NULL
  }
}

#' @noRd
pg_model_download_bundle <- function(model_id,
                                     version = NULL,
                                     board = NULL) {
  if (is.null(board)) {
    board <- tryCatch(pg_board_hub(), error = function(e) pg_board_user())
  }
  safe_id <- gsub("/+", "--", model_id)
  files <- pins::pin_download(board, safe_id, version = version)
  bundle <- files[grepl("\\.(tar\\.gz|tar\\.bz2|tar\\.xz|zip)$", files)]
  list(bundle = bundle, raw_files = files)
}

#' @noRd
pg_model_unpack_bundle <- function(bundle_path, cache_dir) {
  fs::dir_create(cache_dir, recurse = TRUE)
  if (grepl("\\.zip$", bundle_path, ignore.case = TRUE)) {
    utils::unzip(bundle_path, exdir = cache_dir)
  } else {
    utils::untar(bundle_path, exdir = cache_dir)
  }
  pg_model_resolve_manifest(cache_dir)
}

#' Load a model bundle from pins into the local cache
#'
#' Downloads (if needed) and unpacks the requested model version into the
#' petrographer cache. Returns resolved file paths and manifest metadata.
#'
#' @param model_id Model identifier / pin name.
#' @param version Optional board version.
#' @param board Pins board override; when NULL, the hosted hub is tried first,
#'   then the user board.
#' @param use_best When TRUE, prefer `model_best.pth` if present.
#' @return List with `model_path`, `config_path`, `manifest`, `cache_dir`, and `pin_meta`.
#' @export
pg_model_from_pretrained <- function(model_id,
                                     version = NULL,
                                     board = NULL,
                                     use_best = TRUE) {
  if (!requireNamespace("pins", quietly = TRUE)) {
    cli::cli_abort("pins is required. Install with install.packages('pins')")
  }

  if (is.null(board)) {
    board <- tryCatch(pg_board_hub(), error = function(e) pg_board_user())
  }

  safe_id <- gsub("/+", "--", model_id)

  pin_meta <- pins::pin_meta(board, safe_id, version = version)
  manifest_meta <- pg_coalesce(pin_meta$user$manifest, pin_meta$metadata$manifest)

  cache_version <- pg_coalesce(pin_meta$version, manifest_meta$version, "latest")
  cache_id <- gsub("/+", "--", model_id)
  cache_dir <- pg_model_cache_path(cache_id, cache_version)

  manifest <- pg_model_resolve_manifest(cache_dir)
  if (is.null(manifest)) {
    download <- pg_model_download_bundle(model_id, version = version, board = board)
    bundle <- download$bundle

    if (length(bundle) == 0) {
      # Legacy pins with raw files
      fs::dir_create(cache_dir, recurse = TRUE)
      purrr::walk(download$raw_files, function(path) {
        fs::file_copy(path, fs::path(cache_dir, fs::path_file(path)), overwrite = TRUE)
      })
      manifest <- pg_model_resolve_manifest(cache_dir)
      if (is.null(manifest)) {
        manifest <- pg_model_prepare_manifest(cache_dir, model_id, metadata = list(source = "legacy"), include_metrics = fs::file_exists(fs::path(cache_dir, "metrics.json")))
        pg_model_write_manifest(cache_dir, manifest)
      }
    } else {
      manifest <- pg_model_unpack_bundle(bundle[[1]], cache_dir)
    }
  }

  manifest$pin_version <- pin_meta$version
  manifest$model_id <- pg_coalesce(manifest$model_id, model_id)

  model_final <- fs::path(cache_dir, pg_coalesce(manifest$artifacts$model_final, "model_final.pth"))
  config_path <- fs::path(cache_dir, pg_coalesce(manifest$artifacts$config, "config.yaml"))
  best_path <- NULL
  if (use_best && !is.null(manifest$artifacts$model_best)) {
    candidate <- fs::path(cache_dir, manifest$artifacts$model_best)
    if (fs::file_exists(candidate)) best_path <- candidate
  }
  model_path <- pg_coalesce(best_path, model_final)

  list(
    model_path = model_path,
    config_path = config_path,
    manifest = manifest,
    cache_dir = cache_dir,
    pin_meta = pin_meta
  )
}

#' Retrieve a model pin and return resolved file paths (legacy helper)
#'
#' @inheritParams pg_model_from_pretrained
#' @return List with `model_path`, `config_path`, `manifest`, and `pin_meta`.
#' @export
get_model <- function(name, version = NULL, board = NULL) {
  pg_model_from_pretrained(model_id = name, version = version, board = board)
}

#' Install (cache) a pretrained model from the hosted hub or specified board
#'
#' Downloads the requested model bundle into the local cache and returns the
#' resolved manifest (invisibly). Defaults to the hosted board at
#' `PETRO_PINS_URL` when available, falling back to the user board otherwise.
#'
#' @inheritParams pg_model_from_pretrained
#' @param model_id Pin/model identifier to install.
#' @export
pg_install_pretrained <- function(model_id,
                                  version = NULL,
                                  board = NULL,
                                  use_best = TRUE) {
  resolved <- pg_model_from_pretrained(
    model_id = model_id,
    version = version,
    board = board,
    use_best = use_best
  )
  invisible(resolved)
}

#' Refresh the manifest for the pkgdown-hosted board
#'
#' Writes `_pins.yaml` into the pkgdown assets directory so the site can be
#' consumed via `board_url()`.
#'
#' @param path Directory containing the pkgdown pins board (defaults to
#'   `pkgdown/assets/pins`).
#' @return Path to the manifest file (invisibly).
#' @export
pg_publish_pkgdown_manifest <- function(path = fs::path("pkgdown", "assets", "pins")) {
  board <- pg_board_pkgdown(path)
  pg_write_board_manifest(board)
  manifest_path <- fs::path_abs(fs::path(path, "_pins.yaml"))
  cli::cli_alert_success("Pkgdown manifest updated: {.path {manifest_path}}")
  invisible(manifest_path)
}

#' Inspect manifest metadata without loading weights
#'
#' @inheritParams pg_model_from_pretrained
#' @return Manifest list (may be downloaded if not cached locally).
#' @export
pg_model_info <- function(model_id, version = NULL, board = NULL) {
  if (is.null(board)) {
    board <- tryCatch(pg_board_hub(), error = function(e) pg_board_user())
  }
  meta <- pins::pin_meta(board, model_id, version = version)
  manifest <- pg_coalesce(meta$user$manifest, meta$metadata$manifest)
  if (!is.null(manifest)) return(manifest)
  resolved <- pg_model_from_pretrained(model_id, version = version, board = board)
  resolved$manifest
}
