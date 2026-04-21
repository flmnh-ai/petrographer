# Calculate morphological properties from SAHI result

Internal helper converting SAHI object predictions into a tibble of
morphological properties using scikit-image via reticulate.

## Usage

``` r
calculate_morphology_from_result(result, image_path)
```

## Arguments

- result:

  SAHI prediction result object.

- image_path:

  Original image path (used to populate `image_name`).

## Value

A tibble with morphological properties per object.
