# Align thin section image pair

Aligns a PPL (plane-polarized light) image to an XPL (cross-polarized
light) reference image using SIFT feature matching. Designed for images
of the same slide taken after remounting (e.g., adding polarizing
filters). Automatically validates that scale and rotation are reasonable
and falls back to translation-only if needed.

## Usage

``` r
align_images(ppl_path, xpl_path, output_path, method = "similarity")
```

## Arguments

- ppl_path:

  Path to PPL image

- xpl_path:

  Path to XPL image (reference)

- output_path:

  Path to save aligned PPL image. If a directory, generates filename
  from input.

- method:

  Alignment method: "similarity" (rotation+scale+translation with
  automatic fallback to translation if checks fail) or "translation"
  (translation only, no rotation/scale)

## Value

List with n_matches and method_used (invisibly)
