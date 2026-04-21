# Calculate morphological properties from supervision Detections

For segmentation models that return supervision Detections with masks.

## Usage

``` r
calculate_morphology_from_detections(
  detections,
  image_path,
  class_names_map = NULL
)
```

## Arguments

- detections:

  supervision.Detections object with masks.

- image_path:

  Original image path.

- class_names_map:

  Optional mapping of class id -\> class name.

## Value

A tibble with morphological properties per object.
