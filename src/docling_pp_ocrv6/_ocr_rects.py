"""docling's OCR rectangle deduplication, computed on the rectangles instead of a page-sized raster.

docling's ``BaseOcrModel._deduplicate_rects`` draws every rectangle into a binary image of the page
(one pixel per point), dilates it with OpenCV, labels the connected components with SciPy and takes
their bounding boxes. On a photo page (a 4000 x 3000 px JPEG at 72 dpi is a 4000 x 3000 pt page)
that is several full-size raster passes per page for a handful of rectangles.

The same result follows from the rectangles alone, since every step maps rectangles to rectangles:

* ``ImageDraw.rectangle`` fills the rounded, inclusive pixel range, clipped to the image.
* ``cv2.dilate`` with a ``k x k`` kernel anchored at ``k // 2`` grows a rectangle by ``k - 1 - k // 2``
  pixels to the left/top and ``k // 2`` to the right/bottom, clipped to the image.
* ``scipy.ndimage.label`` (4-connectivity) joins two rectangles when they overlap on one axis and
  overlap or touch on the other; a component's bounding box spans its rectangles.
* Labels are numbered in raster order of each component's first pixel: its top row, then the
  leftmost rectangle starting on that row.
* The covered area is the area of the union of the rectangles.

:func:`deduplicate_rects` returns ``None`` whenever it cannot vouch for that equivalence (another
docling version's code, no OpenCV, an empty page raster, a rectangle with negative extent or
non-finite coordinates); the caller then runs docling's own implementation.
"""

from __future__ import annotations

import functools
import hashlib
import inspect
import math
from typing import TYPE_CHECKING, Any

import numpy as np
from docling_core.types.doc import BoundingBox, CoordOrigin

if TYPE_CHECKING:
    from collections.abc import Iterable

    from docling.datamodel.base_models import Size


#: sha256 prefixes of the docling functions this module reproduces (docling 2.142)
_DOCLING_DEDUPLICATE_RECTS = "cbc6bebea835f3b0"


@functools.cache
def source_digest(func: Any) -> str | None:  # noqa: ANN401
    """Short sha256 of a function's source, None when the source is not available."""
    try:
        return hashlib.sha256(inspect.getsource(func).encode()).hexdigest()[:16]
    except (OSError, TypeError):
        return None


def _union_area(boxes: np.ndarray) -> int:
    """Pixel count of the union of inclusive integer rectangles (n, 4: x0, y0, x1, y1)."""
    xs = np.unique(np.concatenate([boxes[:, 0], boxes[:, 2] + 1]))
    ys = np.unique(np.concatenate([boxes[:, 1], boxes[:, 3] + 1]))
    grid = np.zeros((len(ys) - 1, len(xs) - 1), dtype=bool)
    lo_x, hi_x = np.searchsorted(xs, boxes[:, 0]), np.searchsorted(xs, boxes[:, 2] + 1)
    lo_y, hi_y = np.searchsorted(ys, boxes[:, 1]), np.searchsorted(ys, boxes[:, 3] + 1)
    for x0, x1, y0, y1 in zip(lo_x.tolist(), hi_x.tolist(), lo_y.tolist(), hi_y.tolist(), strict=True):
        grid[y0:y1, x0:x1] = True
    cell = np.outer(np.diff(ys), np.diff(xs))
    return int(cell[grid].sum())


def _components(boxes: np.ndarray) -> list[list[int]]:
    """Groups of 4-connected rectangles (overlap on one axis, overlap or touch on the other)."""
    x0, y0, x1, y1 = (boxes[:, k] for k in range(4))
    x_overlap = (x0[:, None] <= x1[None, :]) & (x0[None, :] <= x1[:, None])
    y_overlap = (y0[:, None] <= y1[None, :]) & (y0[None, :] <= y1[:, None])
    x_touch = (x0[:, None] <= x1[None, :] + 1) & (x0[None, :] <= x1[:, None] + 1)
    y_touch = (y0[:, None] <= y1[None, :] + 1) & (y0[None, :] <= y1[:, None] + 1)
    adjacent = (x_overlap & y_touch) | (y_overlap & x_touch)
    seen = np.zeros(len(boxes), dtype=bool)
    groups = []
    for start in range(len(boxes)):
        if seen[start]:
            continue
        seen[start] = True
        group, todo = [start], [start]
        while todo:
            for j in np.flatnonzero(adjacent[todo.pop()] & ~seen).tolist():
                seen[j] = True
                group.append(j)
                todo.append(j)
        groups.append(group)
    return groups


def deduplicate_rects(
    size: Size, rects: Iterable[BoundingBox], dilation_size: int = 0
) -> tuple[np.float64, list[BoundingBox]] | None:
    """Docling's ``_deduplicate_rects`` before ``_drop_contained_rects``: coverage and component boxes.

    Returns None when the caller must run docling's raster implementation instead.
    """
    from docling.models import base_ocr_model  # noqa: PLC0415

    same_docling = source_digest(base_ocr_model.BaseOcrModel._deduplicate_rects) == _DOCLING_DEDUPLICATE_RECTS  # noqa: SLF001
    if not same_docling or (dilation_size > 0 and not getattr(base_ocr_model, "CV2_INSTALLED", False)):
        return None  # another docling's code, or docling dilates with scipy, whose anchor differs
    width, height = round(size.width), round(size.height)
    if width <= 0 or height <= 0:
        return None  # an empty raster: docling's code decides what happens
    grow_lo = dilation_size - 1 - dilation_size // 2 if dilation_size > 0 else 0
    grow_hi = dilation_size // 2 if dilation_size > 0 else 0
    drawn = []
    for rect in rects:
        coords = rect.as_tuple()
        if not all(math.isfinite(v) for v in coords):
            return None
        x0, y0, x1, y1 = (round(v) for v in coords)
        if x1 < x0 or y1 < y0:
            return None  # ImageDraw raises for these; let docling's code do it
        x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, width - 1), min(y1, height - 1)
        if x0 > x1 or y0 > y1:
            continue  # entirely off the page: nothing drawn
        drawn.append((
            max(x0 - grow_lo, 0),
            max(y0 - grow_lo, 0),
            min(x1 + grow_hi, width - 1),
            min(y1 + grow_hi, height - 1),
        ))
    if not drawn:
        return np.int64(0) / (size.width * size.height), []
    boxes = np.array(drawn, dtype=np.int64)
    found = []
    for group in _components(boxes):
        members = boxes[group]
        top = int(members[:, 1].min())
        first_x = int(members[members[:, 1] == top, 0].min())
        found.append((
            (top, first_x),
            BoundingBox(
                l=int(members[:, 0].min()),
                t=top,
                r=int(members[:, 2].max()),
                b=int(members[:, 3].max()),
                coord_origin=CoordOrigin.TOPLEFT,
            ),
        ))
    found.sort(key=lambda item: item[0])
    return np.int64(_union_area(boxes)) / (size.width * size.height), [box for _, box in found]
