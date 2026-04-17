# inst/python/visualize.py
"""Annotation and prediction visualization via Roboflow's `supervision`.

Two entry points:

- `render_coco_overlay(image_path, annotations_json, output_path, ...)`
    Draws ground-truth COCO annotations (bboxes + masks when present +
    labels) over an image and writes the result to disk.

- `render_detections_overlay(image_path, xyxy, class_ids, scores,
    class_names, masks=None, output_path=..., ...)`
    Draws arbitrary detections (predictions) over an image using the same
    annotator stack so GT and predictions look identical.

All rendering goes through supervision's `Detections` + annotators; callers
never touch magick/ImageMagick.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional, Sequence

import cv2
import numpy as np
import supervision as sv


# ---------------------------------------------------------------------------
# COCO ground-truth rendering
# ---------------------------------------------------------------------------

def _decode_coco_segmentation(seg, height: int, width: int) -> Optional[np.ndarray]:
    """Return a bool mask from a COCO `segmentation` field, or None if absent."""
    if not seg:
        return None
    try:
        from pycocotools import mask as mask_utils
    except ImportError:
        return None

    try:
        if isinstance(seg, list) and len(seg) > 0:
            # Polygon(s)
            rles = mask_utils.frPyObjects(seg, height, width)
            rle = mask_utils.merge(rles)
            return mask_utils.decode(rle).astype(bool)
        if isinstance(seg, dict):
            # Uncompressed (counts = list) or compressed RLE
            counts = seg.get("counts")
            if isinstance(counts, list):
                rle = mask_utils.frPyObjects(seg, height, width)
            else:
                rle = seg
            return mask_utils.decode(rle).astype(bool)
    except Exception:
        return None
    return None


def _build_coco_detections(ann_data: dict, image_id, width: int, height: int):
    """Build an (sv.Detections, category_map) pair for a single COCO image.

    Returns (None, {}) when the image has no annotations.
    """
    annotations = [
        a for a in ann_data.get("annotations", [])
        if a.get("image_id") == image_id
    ]
    categories = {
        c["id"]: c.get("name", str(c["id"]))
        for c in ann_data.get("categories", [])
    }
    if not annotations:
        return None, categories

    xyxy_list = []
    class_ids = []
    per_mask = []
    has_any_mask = False

    for ann in annotations:
        bbox = ann.get("bbox")
        if not bbox or len(bbox) < 4:
            continue
        x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
        xyxy_list.append([x, y, x + w, y + h])
        class_ids.append(int(ann["category_id"]))

        mask = _decode_coco_segmentation(ann.get("segmentation"), height, width)
        if mask is not None:
            has_any_mask = True
        per_mask.append(mask)

    if not xyxy_list:
        return None, categories

    xyxy = np.asarray(xyxy_list, dtype=np.float32)
    class_id = np.asarray(class_ids, dtype=int)

    mask_array = None
    if has_any_mask:
        filled = []
        for m in per_mask:
            if m is None:
                filled.append(np.zeros((height, width), dtype=bool))
            else:
                # Ensure expected shape (some polygons can return off-by-one)
                if m.shape != (height, width):
                    resized = np.zeros((height, width), dtype=bool)
                    hh = min(height, m.shape[0])
                    ww = min(width, m.shape[1])
                    resized[:hh, :ww] = m[:hh, :ww]
                    m = resized
                filled.append(m)
        mask_array = np.stack(filled, axis=0)

    detections = sv.Detections(
        xyxy=xyxy,
        mask=mask_array,
        class_id=class_id,
    )
    return detections, categories


def render_coco_overlay(
    image_path: str,
    annotations_json: str,
    output_path: str,
    categories_keep: Optional[Sequence] = None,
    draw_labels: bool = True,
    mask_opacity: float = 0.4,
) -> str:
    """Render COCO ground-truth annotations over `image_path` to `output_path`.

    Parameters
    ----------
    image_path : str
        Path to an image present in `annotations_json`.
    annotations_json : str
        Path to a COCO-format annotations file (`_annotations.coco.json`).
    output_path : str
        Destination PNG path.
    categories_keep : Sequence of str or int, optional
        Filter to these category names or ids.
    draw_labels : bool
        Whether to draw class-name labels.
    mask_opacity : float
        Opacity for polygon/mask fills (segmentation datasets).

    Returns
    -------
    str
        `output_path`.
    """
    image_path = str(image_path)
    annotations_json = str(annotations_json)
    output_path = str(output_path)

    with open(annotations_json, "r", encoding="utf-8") as fh:
        ann_data = json.load(fh)

    image_name = Path(image_path).name
    images = ann_data.get("images") or []
    matches = [im for im in images if im.get("file_name") == image_name]
    if not matches:
        raise ValueError(
            f"Image '{image_name}' is not listed in {annotations_json}."
        )
    image_info = matches[0]
    image_id = image_info["id"]

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    height = int(image_info.get("height") or image.shape[0])
    width = int(image_info.get("width") or image.shape[1])

    detections, categories = _build_coco_detections(
        ann_data, image_id, width=width, height=height
    )

    if detections is None or len(detections) == 0:
        cv2.imwrite(output_path, image)
        return output_path

    # Optional category filter
    if categories_keep:
        keep_ids = set()
        for c in categories_keep:
            if isinstance(c, (int, np.integer)):
                keep_ids.add(int(c))
            else:
                for cid, cname in categories.items():
                    if cname == c:
                        keep_ids.add(cid)
        mask = np.array(
            [int(cid) in keep_ids for cid in detections.class_id],
            dtype=bool,
        )
        detections = detections[mask]
        if len(detections) == 0:
            cv2.imwrite(output_path, image)
            return output_path

    labels = [categories.get(int(c), str(c)) for c in detections.class_id]

    annotated = image.copy()
    if detections.mask is not None:
        annotated = sv.MaskAnnotator(opacity=mask_opacity).annotate(annotated, detections)
    annotated = sv.BoxAnnotator().annotate(annotated, detections)
    if draw_labels:
        annotated = sv.LabelAnnotator().annotate(annotated, detections, labels=labels)

    cv2.imwrite(output_path, annotated)
    return output_path


# ---------------------------------------------------------------------------
# Prediction rendering
# ---------------------------------------------------------------------------

def _annotate_image(image, detections, labels, draw_labels: bool, mask_opacity: float):
    """Apply the shared annotator stack to a BGR numpy image."""
    annotated = image.copy()
    if getattr(detections, "mask", None) is not None:
        annotated = sv.MaskAnnotator(opacity=mask_opacity).annotate(annotated, detections)
    annotated = sv.BoxAnnotator().annotate(annotated, detections)
    if draw_labels:
        annotated = sv.LabelAnnotator().annotate(annotated, detections, labels=labels)
    return annotated


def _resolve_class_name(class_id: int, sahi_name, class_names_map) -> str:
    """Prefer the caller-supplied names map over SAHI's internal name resolution.

    SAHI's `obj.category.name` is hydrated from its `category_mapping`, which
    sometimes mis-maps (int vs. string keys, or an entirely empty mapping when
    SAHI can't infer one for custom RF-DETR models). If the caller passes a
    class_names_map we trust that; otherwise fall back to SAHI's name, then to
    the raw class id.
    """
    if class_names_map is not None:
        name = class_names_map.get(class_id)
        if name is None:
            name = class_names_map.get(str(class_id))
        if name is not None:
            return str(name)
    if sahi_name is not None and str(sahi_name) != str(class_id):
        return str(sahi_name)
    return str(class_id)


def render_sahi_overlay(
    image_path: str,
    sahi_result,
    output_path: str,
    class_names_map: Optional[dict] = None,
    draw_labels: bool = True,
    mask_opacity: float = 0.4,
    show_confidence: bool = False,
) -> str:
    """Render a SAHI `PredictionResult` over `image_path`.

    Pulls bboxes, scores, class ids, and (when present) masks off the
    `object_prediction_list`, then delegates to the shared annotator stack so
    prediction overlays match ground-truth overlays pixel-for-pixel.

    Label resolution prefers `class_names_map` (caller-supplied, typically
    from `metadata.json$thing_classes`) over SAHI's internal name lookup —
    that internal lookup is unreliable for RF-DETR models where SAHI's
    `category_mapping` can end up with the wrong key type or be empty.
    """
    image_path = str(image_path)
    output_path = str(output_path)

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    preds = list(getattr(sahi_result, "object_prediction_list", []) or [])
    if not preds:
        cv2.imwrite(output_path, image)
        return output_path

    xyxy_list = []
    class_ids = []
    scores = []
    class_names = []
    per_mask = []
    has_any_mask = False

    for obj in preds:
        bb = obj.bbox
        xyxy_list.append([float(bb.minx), float(bb.miny), float(bb.maxx), float(bb.maxy)])
        cid = int(obj.category.id)
        class_ids.append(cid)
        scores.append(float(obj.score.value))
        class_names.append(
            _resolve_class_name(cid, getattr(obj.category, "name", None), class_names_map)
        )
        mask = None
        if getattr(obj, "mask", None) is not None and getattr(obj.mask, "bool_mask", None) is not None:
            mask = np.asarray(obj.mask.bool_mask, dtype=bool)
            has_any_mask = True
        per_mask.append(mask)

    xyxy = np.asarray(xyxy_list, dtype=np.float32)
    class_id = np.asarray(class_ids, dtype=int)
    confidence = np.asarray(scores, dtype=np.float32)

    mask_array = None
    if has_any_mask:
        h, w = image.shape[:2]
        filled = []
        for m in per_mask:
            if m is None:
                filled.append(np.zeros((h, w), dtype=bool))
            else:
                filled.append(m)
        mask_array = np.stack(filled, axis=0)

    detections = sv.Detections(
        xyxy=xyxy,
        mask=mask_array,
        class_id=class_id,
        confidence=confidence,
    )

    if show_confidence:
        labels = [f"{n} {s:.2f}" for n, s in zip(class_names, scores)]
    else:
        labels = class_names

    annotated = _annotate_image(image, detections, labels, draw_labels, mask_opacity)
    cv2.imwrite(output_path, annotated)
    return output_path


def render_sv_overlay(
    image_path: str,
    detections,
    class_names_map: dict,
    output_path: str,
    draw_labels: bool = True,
    mask_opacity: float = 0.4,
    show_confidence: bool = False,
) -> str:
    """Render a pre-built `sv.Detections` over `image_path`.

    Used by the segmentation path in `predict_image()` — the RF-DETR
    segmentation model returns `sv.Detections` directly, so we skip the
    SAHI layer and hand the object straight to the annotator stack.
    """
    image_path = str(image_path)
    output_path = str(output_path)

    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    if len(detections) == 0:
        cv2.imwrite(output_path, image)
        return output_path

    class_ids = [int(c) for c in detections.class_id]
    # class_names_map may key on ints (Python) or strings (COCO/JSON) — try both.
    names = []
    for c in class_ids:
        name = None
        if class_names_map is not None:
            name = class_names_map.get(c)
            if name is None:
                name = class_names_map.get(str(c))
        names.append(name if name is not None else str(c))

    if show_confidence and detections.confidence is not None:
        scores = [float(s) for s in detections.confidence]
        labels = [f"{n} {s:.2f}" for n, s in zip(names, scores)]
    else:
        labels = names

    annotated = _annotate_image(image, detections, labels, draw_labels, mask_opacity)
    cv2.imwrite(output_path, annotated)
    return output_path


def render_detections_overlay(
    image_path: str,
    xyxy: Iterable[Iterable[float]],
    class_ids: Iterable[int],
    scores: Iterable[float],
    class_names: Iterable[str],
    masks: Optional[Iterable[np.ndarray]] = None,
    output_path: Optional[str] = None,
    draw_labels: bool = True,
    mask_opacity: float = 0.4,
    show_confidence: bool = False,
) -> Optional[str]:
    """Render arbitrary detections over `image_path` using supervision.

    Parameters
    ----------
    image_path : str
    xyxy : iterable of [x1, y1, x2, y2]
    class_ids : iterable of int
    scores : iterable of float
    class_names : iterable of str
        One label per detection (already resolved from class_id).
    masks : iterable of bool ndarray, optional
        Per-detection masks for segmentation models.
    output_path : str, optional
        If None, the annotated image is not written to disk.
    draw_labels : bool
    mask_opacity : float
    show_confidence : bool
        If True, label text becomes "<name> <score>".

    Returns
    -------
    Optional[str]
        `output_path` if provided, else None.
    """
    image_path = str(image_path)
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    xyxy_arr = np.asarray(list(xyxy), dtype=np.float32)
    if xyxy_arr.size == 0:
        xyxy_arr = xyxy_arr.reshape((0, 4))
    elif xyxy_arr.ndim == 1:
        # Single detection passed as flat [x1,y1,x2,y2]
        xyxy_arr = xyxy_arr.reshape((1, 4))

    class_id = np.asarray(list(class_ids), dtype=int)
    confidence = np.asarray(list(scores), dtype=np.float32)
    class_names_list = list(class_names)
    scores_list = confidence.tolist()

    mask_array = None
    if masks is not None:
        masks_list = list(masks)
        if len(masks_list):
            h, w = image.shape[:2]
            filled = []
            for m in masks_list:
                if m is None:
                    filled.append(np.zeros((h, w), dtype=bool))
                else:
                    filled.append(np.asarray(m, dtype=bool))
            mask_array = np.stack(filled, axis=0)

    detections = sv.Detections(
        xyxy=xyxy_arr,
        mask=mask_array,
        class_id=class_id,
        confidence=confidence,
    )

    if show_confidence:
        labels = [
            f"{name} {score:.2f}"
            for name, score in zip(class_names_list, scores_list)
        ]
    else:
        labels = class_names_list

    annotated = image.copy()
    if mask_array is not None:
        annotated = sv.MaskAnnotator(opacity=mask_opacity).annotate(annotated, detections)
    annotated = sv.BoxAnnotator().annotate(annotated, detections)
    if draw_labels:
        annotated = sv.LabelAnnotator().annotate(annotated, detections, labels=labels)

    if output_path is not None:
        cv2.imwrite(str(output_path), annotated)
        return str(output_path)
    return None


# ---------------------------------------------------------------------------
# Utility — image dimensions (replaces magick::image_info in training.R)
# ---------------------------------------------------------------------------

def image_dimensions(image_path: str) -> dict:
    """Return a dict with width/height/channels for an image file."""
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    h, w = image.shape[:2]
    c = 1 if image.ndim == 2 else image.shape[2]
    return {"width": int(w), "height": int(h), "channels": int(c)}
