#' Refresh pkgdown manifests and catalog
#'
#' @param board Pins board to update (defaults to pkgdown board).
#' @param catalog_path Output path for the catalog RDS.
#' @return Invisibly returns the catalog path.
#' @export
pg_pkgdown_refresh <- function(board = pg_board_pkgdown(),
                               catalog_path = fs::path("pkgdown", "assets", "model_catalog.rds")) {
  cli::cli_h2("Refreshing pkgdown manifests")
  pg_write_board_manifest(board)
  pg_write_model_catalog(path = catalog_path, board = board)
  cli::cli_alert_success("Pkgdown assets refreshed.")
  invisible(catalog_path)
}
