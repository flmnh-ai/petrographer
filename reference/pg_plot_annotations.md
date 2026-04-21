# Plot COCO annotations on an image

Draws ground-truth bounding boxes (and segmentation masks when present)
from a COCO-style annotation file on top of an image. Rendering is
delegated to Roboflow's `supervision` library via reticulate, so GT
overlays stay visually consistent with prediction overlays produced by
[`predict_image()`](https://flmnh-ai.github.io/petrographer/reference/predict_image.md).

## Usage

``` r
pg_plot_annotations(
  image_path,
  annotation_json,
  categories = NULL,
  output = NULL,
  display = NULL,
  draw_labels = TRUE,
  mask_opacity = 0.4
)
```

## Arguments

- image_path:

  Path to the image file.

- annotation_json:

  Path to the COCO annotations JSON containing the image.

- categories:

  Optional vector of category names or ids to keep. `NULL` keeps all.

- output:

  Destination PNG path. Defaults to a tempfile.

- display:

  Whether to render the annotated image on the current graphics device.
  Defaults to `TRUE` in interactive sessions and inside knitr (so the
  image appears under the chunk).

- draw_labels:

  Whether to draw category labels. Default `TRUE`.

- mask_opacity:

  Opacity for mask fills (segmentation datasets). Default `0.4`.

## Value

The path to the written PNG, invisibly.
