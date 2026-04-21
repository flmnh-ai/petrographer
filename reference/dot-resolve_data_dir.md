# Resolve a dataset source (directory or .tar.gz) to an extracted directory

Accepts either a dataset directory or a `.tar.gz` / `.tgz` archive and
returns a path to an extracted dataset directory. Archives are extracted
once per session into a cache dir keyed on path + mtime, so subsequent
calls on the same archive reuse the extraction instead of re-untarring.

## Usage

``` r
.resolve_data_dir(data_source)
```

## Arguments

- data_source:

  Path to a dataset directory or archive

## Value

Absolute path to a dataset directory
