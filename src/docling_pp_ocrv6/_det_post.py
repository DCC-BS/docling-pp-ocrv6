"""RapidOCR's DB detection post-processing (``DBPostProcess``) with less Python and numpy overhead.

The text detector's probability map becomes boxes in a per-contour Python loop. This module runs
the same OpenCV, shapely and pyclipper calls on the same values and leaves out what only costs
time:

* The binary map goes to ``cv2.dilate`` and ``cv2.findContours`` as a uint8 view of the boolean
  threshold, without the copies to 0/255 (``findContours`` treats every non-zero pixel as 1).
* Box corners are handled as Python floats instead of one-element numpy operations. They come from
  ``cv2.boxPoints`` as float32, so every value, comparison and rounding is the same.
* Polygon area and perimeter for the unclip distance come from shapely's vectorised functions on
  all boxes at once, the same GEOS computations as ``Polygon(box).area`` / ``.length``.
* The final scaling to the input image runs on all boxes at once, element by element in float32
  as RapidOCR does it per box.

The steps that are not plain arithmetic stay RapidOCR's (``filter_det_res``). :func:`postprocess`
falls back to RapidOCR's own ``DBPostProcess.__call__`` for any configuration it does not
reproduce (the ``slow`` score mode, a subclass).
"""

from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np
import pyclipper
import shapely


def _mini_box(contour: np.ndarray) -> tuple[list[list[float]], float]:
    """``DBPostProcess.get_mini_boxes`` with the corners as Python floats (exact float32 values)."""
    rect = cv2.minAreaRect(contour)
    pts = sorted(cv2.boxPoints(rect).tolist(), key=lambda p: p[0])
    i1, i4 = (0, 1) if pts[1][1] > pts[0][1] else (1, 0)
    i2, i3 = (2, 3) if pts[3][1] > pts[2][1] else (3, 2)
    return [pts[i1], pts[i2], pts[i3], pts[i4]], min(rect[1])


def _box_score_fast(bitmap: np.ndarray, box: list[list[float]]) -> float:
    """``DBPostProcess.box_score_fast``: mean of the map inside the box.

    ``x - xmin`` of a float32 ``x`` and an integer ``xmin <= x`` (or ``xmin = 0``) is exact in float32
    and in double, so truncating it gives the int32 corner RapidOCR draws.
    """
    h, w = bitmap.shape[:2]
    xs = [p[0] for p in box]
    ys = [p[1] for p in box]
    xmin = min(max(math.floor(min(xs)), 0), w - 1)
    xmax = min(max(math.ceil(max(xs)), 0), w - 1)
    ymin = min(max(math.floor(min(ys)), 0), h - 1)
    ymax = min(max(math.ceil(max(ys)), 0), h - 1)
    mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
    corners = np.array([[[int(x - xmin), int(y - ymin)] for x, y in box]], dtype=np.int32)
    cv2.fillPoly(mask, corners, 1)  # ty: ignore[no-matching-overload]
    return cv2.mean(bitmap[ymin : ymax + 1, xmin : xmax + 1], mask)[0]


def _boxes_from_bitmap(
    op: Any,  # noqa: ANN401
    pred: np.ndarray,
    bitmap: np.ndarray,
    dest_width: int,
    dest_height: int,
) -> tuple[np.ndarray, list[float]]:
    """``DBPostProcess.boxes_from_bitmap`` for the ``fast`` score mode."""
    height, width = bitmap.shape
    outs = cv2.findContours(bitmap, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = outs[1] if len(outs) == 3 else outs[0]  # noqa: PLR2004
    candidates, scores = [], []
    for contour in contours[: min(len(contours), op.max_candidates)]:
        box, sside = _mini_box(contour)
        if sside < op.min_size:
            continue
        score = _box_score_fast(pred, box)
        if op.box_thresh > score:
            continue
        candidates.append(box)
        scores.append(score)
    if not candidates:
        return np.array([], dtype=np.int32), []

    points = np.array(candidates, dtype=np.float32)  # what get_mini_boxes returns, per box
    polygons = shapely.polygons(points)
    distances = shapely.area(polygons) * op.unclip_ratio / shapely.length(polygons)
    kept, kept_scores = [], []
    for box, distance, score in zip(points, distances.tolist(), scores, strict=True):
        offset = pyclipper.PyclipperOffset()  # ty: ignore[unresolved-attribute]
        offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)  # ty: ignore[unresolved-attribute]
        expanded = np.array(offset.Execute(distance)).reshape((-1, 1, 2))
        mini, sside = _mini_box(expanded)
        if sside < op.min_size + 2:
            continue
        kept.append(mini)
        kept_scores.append(score)
    if not kept:
        return np.array([], dtype=np.int32), []
    boxes = np.array(kept, dtype=np.float32)
    boxes[:, :, 0] = np.clip(np.round(boxes[:, :, 0] / width * dest_width), 0, dest_width)
    boxes[:, :, 1] = np.clip(np.round(boxes[:, :, 1] / height * dest_height), 0, dest_height)
    return boxes.astype(np.int32), kept_scores


def supported(op: Any) -> bool:  # noqa: ANN401
    """Whether ``op`` is a plain RapidOCR ``DBPostProcess`` in the ``fast`` score mode."""
    try:
        from rapidocr.ch_ppocr_det.utils import DBPostProcess  # noqa: PLC0415
    except ImportError:
        return False
    return type(op) is DBPostProcess and op.score_mode == "fast"


def postprocess(op: Any, pred: np.ndarray, ori_shape: tuple[int, int]) -> tuple[np.ndarray, list[float]]:  # noqa: ANN401
    """``op(pred, ori_shape)`` for a RapidOCR ``DBPostProcess``: the detected boxes and their scores."""
    if not supported(op):
        return op(pred, ori_shape)
    src_h, src_w = ori_shape
    prob = pred[0, 0]
    bitmap = (prob > op.thresh).view(np.uint8)
    if op.dilation_kernel is not None:
        bitmap = cv2.dilate(bitmap, op.dilation_kernel)
    boxes, scores = _boxes_from_bitmap(op, prob, bitmap, src_w, src_h)
    return op.filter_det_res(boxes, scores, src_h, src_w)
