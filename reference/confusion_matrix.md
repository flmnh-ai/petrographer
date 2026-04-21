# Compute a confusion matrix from SAHI predictions + COCO ground truth

For each image, greedily match predicted boxes to ground-truth boxes by
IoU (highest-scoring prediction matches first). Predictions that match a
GT box with IoU \>= `iou_threshold` contribute a (gt_class, pred_class)
cell. Unmatched predictions land in the "background" *row* (false
positives); unmatched GT boxes land in the "background" *column* (false
negatives).

## Usage

``` r
confusion_matrix(
  predictions,
  annotation_json,
  iou_threshold = 0.5,
  score_threshold = 0
)
```

## Arguments

- predictions:

  Tibble of predictions with `image_id`, `category_id`, `score`, `xmin`,
  `ymin`, `width`, `height` (the `predictions` element of
  [`evaluate_model_sahi()`](https://flmnh-ai.github.io/petrographer/reference/evaluate_model_sahi.md)
  output works directly).

- annotation_json:

  Path to the COCO annotation JSON the predictions were produced
  against.

- iou_threshold:

  Minimum IoU for a prediction to count as matching a GT box (default
  0.5).

- score_threshold:

  Filter predictions below this score before matching (default 0; keep
  all).

## Value

A list with:

- `long`: long-format tibble `(gt_label, pred_label, count)`

- `matrix`: wide tibble, one row per GT label

- `class_names`: character vector of class labels (includes
  "background")

- `iou_threshold`: the threshold used

## Details

Detection doesn't have a clean confusion-matrix definition (unmatched
predictions and unmatched GT don't map onto classification confusion
cleanly), so treat this as a diagnostic heatmap rather than a calibrated
metric.
