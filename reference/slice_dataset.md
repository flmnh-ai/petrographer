# Slice COCO dataset for varying image sizes

Uses SAHI to slice images and annotations into tiles. Images smaller
than `slice_size` are treated as single slices (no fragmentation).
Larger images are split into overlapping tiles, which increases training
samples and improves detection of small objects in large images.

## Usage

``` r
slice_dataset(
  input_dir,
  output_dir,
  slice_size = 1024,
  overlap = 0.2,
  min_area_ratio = 0.1,
  output_format = ".jpg"
)
```

## Arguments

- input_dir:

  Input directory with `train/` and `valid/` subdirectories containing
  COCO annotations. If a `test/` directory exists, it will also be
  sliced.

- output_dir:

  Output directory for sliced dataset (will be created)

- slice_size:

  Slice size in pixels (default: 1024). Images smaller than this will
  not be fragmented.

- overlap:

  Overlap ratio between adjacent slices (default: 0.2, or 20%)

- min_area_ratio:

  Minimum area ratio to keep object fragments (default: 0.1). Objects
  cut by slice boundaries with less than 10% visible area are dropped.

- output_format:

  Output image format: ".jpg" or ".png" (default: ".jpg"). JPG minimizes
  storage (~10x smaller) with minimal quality loss. Use PNG for lossless
  slicing.

## Value

Path to sliced dataset directory (invisibly)

## Details

This is particularly useful for dense detection datasets with varying
image sizes. For the inclusions dataset, slicing with
`slice_size = 1024` will:

- Keep small images (\<1024px) intact as single slices

- Split large images (\>1024px) into 2-4 overlapping tiles

- Result: ~2x more training images with better small object coverage

## Examples

``` r
if (FALSE) { # \dontrun{
# Slice dataset for training
sliced_dir <- slice_dataset(
  input_dir = "data/processed/inclusions",
  output_dir = "data/processed/inclusions_sliced",
  slice_size = 1024,
  overlap = 0.2
)

# Train on sliced dataset
train_model(data_dir = sliced_dir, ...)
} # }
```
