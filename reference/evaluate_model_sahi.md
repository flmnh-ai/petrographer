# Evaluate detections with SAHI and COCO metrics

Runs SAHI inference across the validation set described by a COCO-style
annotation file and computes COCO metrics using `pycocotools`. The
function returns a tidy summary of the standard 12 bbox metrics
alongside the raw prediction table for further analysis.

## Usage

``` r
evaluate_model_sahi(
  model,
  annotation_json,
  image_dir = NULL,
  use_slicing = TRUE,
  slice_size = NULL,
  overlap = 0.2,
  max_images = NULL,
  save_predictions = NULL,
  iou_type = "bbox",
  max_dets = 100
)
```

## Arguments

- model:

  A `PetrographyModel` from
  [`from_pretrained()`](https://flmnh-ai.github.io/petrographer/reference/from_pretrained.md).

- annotation_json:

  Path to COCO annotation JSON (e.g. `valid/_annotations.coco.json`).

- image_dir:

  Directory containing the images referenced in the annotation file. If
  `NULL`, image paths are resolved relative to the annotation file.

- use_slicing:

  Whether to use SAHI sliced inference (default `TRUE`).

- slice_size:

  Slice size for SAHI inference (pixels, default: use model's
  resolution)

- overlap:

  Overlap ratio between slices (default 0.2).

- max_images:

  Optional maximum number of images to evaluate (useful for smoke
  tests).

- save_predictions:

  Optional path to write COCO-format predictions JSON.

- iou_type:

  IoU type to evaluate (`"bbox"` by default).

- max_dets:

  Maximum detections per image for evaluation. For dense detection (100+
  objects), set to 300 or higher (default: 100).

## Value

A list with elements `summary` (tibble of COCO metrics), `predictions`
(tibble of detections), and `coco_eval` (pycocotools object).
