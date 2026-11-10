"""docling's RapidOCR engine with a faster page loop and the same text (engine kind ``rapidocr-fast``).

docling's ``RapidOcrModel`` renders the whole page once per OCR rectangle, calls RapidOCR once per
rectangle and lets RapidOCR prepare every image on the CPU. On a GPU worker with few CPU cores the
OCR stage is bound by that CPU work, not by the models. This engine keeps docling's model, options
and post-processing and changes only how a page batch is run:

* Each page is rendered once at the OCR scale; every rectangle is cropped from that render with
  docling-parse's own crop box, which gives the pixels ``get_page_image(cropbox=...)`` gives,
  without the PIL copies in between.
* The OCR rectangles are docling's; their deduplication is computed on the rectangles instead of a
  page-sized raster (:mod:`~docling_pp_ocrv6._ocr_rects`, same boxes in the same order).
* With RapidOCR's PyTorch engine (``backend: torch``), all rectangles of a page batch go through
  :class:`~docling_pp_ocrv6._rapidocr_driver.FastRapidOcr`: preprocessing on the GPU, the det/cls/rec
  forwards in RapidOCR's own batches, RapidOCR's own pre- and post-processing (the DB detection
  post-processing in a leaner, equal form). Other backends call RapidOCR as docling does.

Select it with an OCR preset of kind ``rapidocr-fast``; it takes every option of docling's
``rapidocr`` kind, e.g. ``{"kind": "rapidocr-fast", "backend": "torch"}``.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Self

import cv2
import numpy as np
from docling.datamodel.pipeline_options import RapidOcrOptions
from docling.datamodel.settings import settings
from docling.models.stages.ocr.rapid_ocr_model import RapidOcrModel
from docling.utils.profiling import TimeRecorder
from docling_core.types.doc import BoundingBox, CoordOrigin
from docling_core.types.doc.page import BoundingRectangle, TextCell
from PIL import Image

from docling_pp_ocrv6._ocr_rects import deduplicate_rects

if TYPE_CHECKING:
    from collections.abc import Iterable

    from docling.datamodel.base_models import Page, Size
    from docling.datamodel.document import ConversionResult
    from docling.datamodel.pipeline_options import OcrOptions

    from docling_pp_ocrv6._rapidocr_driver import FastRapidOcr

logger = logging.getLogger(__name__)


class _PerPageTime:
    """docling's ``TimeRecorder`` for a page batch: one sample per page, the batch time split evenly.

    docling counts OCR time per page; one sample for the whole batch would report the pages of a
    batch as one page taking their summed time.
    """

    def __init__(self, conv_res: ConversionResult, key: str, pages: int) -> None:
        self._recorder = TimeRecorder(conv_res, key)
        self._conv_res, self._key, self._pages = conv_res, key, pages

    def __enter__(self) -> Self:
        self._recorder.__enter__()
        return self

    def __exit__(self, *exc: object) -> None:
        self._recorder.__exit__(*exc)
        item = self._conv_res.timings.get(self._key) if settings.debug.profile_pipeline_timings else None
        if item is None or not item.times or self._pages <= 1:
            return
        share = item.times.pop() / self._pages
        start = item.start_timestamps[-1]
        item.times.extend([share] * self._pages)
        item.start_timestamps.extend([start] * (self._pages - 1))
        item.count += self._pages - 1


def _is_docling_parse(backend: object) -> bool:
    """Whether ``backend`` is docling-parse's page backend, whose rendered page this engine crops itself."""
    try:
        from docling.backend.docling_parse_backend import ThreadedDoclingParsePageBackend  # noqa: PLC0415
    except ImportError:
        return False
    return isinstance(backend, ThreadedDoclingParsePageBackend)


class _Canvas:
    """Stands in for a PIL image in docling-parse's ``_crop_image``: records the crop box it computes."""

    def __init__(self, height: int, width: int) -> None:
        self.height, self.width = height, width

    def crop(self, box: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
        return box


def _render_rgba(result: Any, scale: float) -> np.ndarray | None:  # noqa: ANN401
    """docling-parse's page render at ``scale`` as a (H, W, 4) array, without its PIL copies.

    ``get_image(scale)`` renders into a byte buffer, copies it into a PIL image and, through
    ``get_page_image``, crops, converts to RGB and copies again for numpy. The render is the same
    call; the copies are left out. None when docling-parse would not render here (the scale of its
    default image, a missing private API): the caller then takes the PIL path.
    """
    try:
        config = result._rendering_config()  # noqa: SLF001
        if math.isclose(scale, config.scale, rel_tol=0.0, abs_tol=result._scale_abs_tolerance()):  # noqa: SLF001
            return None  # get_image returns its cached default image here
        decoder = result._require_page_decoder()  # noqa: SLF001
    except (AttributeError, RuntimeError):
        return None
    config.scale = scale
    config.canvas_width = -1
    config.canvas_height = -1
    raw, shape = decoder.render_image(config)
    height, width, _ = shape
    if not raw or len(raw) < height * width * 4:
        return None  # let docling-parse raise its own error
    return np.frombuffer(raw, dtype=np.uint8, count=height * width * 4).reshape(height, width, 4)


def _crop_rgb(result: Any, rgba: np.ndarray, rect: BoundingBox) -> np.ndarray:  # noqa: ANN401
    """``get_page_image(cropbox=rect)`` of a rendered page: docling-parse's crop box, alpha dropped.

    PIL's RGBA to RGB conversion drops the alpha channel without compositing, as OpenCV's
    ``COLOR_RGBA2RGB`` does (which is many times faster than PIL or a numpy copy).
    """
    box = result._crop_image(_Canvas(*rgba.shape[:2]), rect)  # noqa: SLF001
    if isinstance(box, _Canvas):
        crop = rgba
    else:
        left, top, right, bottom = box
        if right < left or bottom < top:  # PIL refuses these crops: raise its error
            Image.new("RGBA", (rgba.shape[1], rgba.shape[0])).crop(box)
        crop = rgba[top:bottom, left:right]
    if crop.size == 0:
        return np.ascontiguousarray(crop[:, :, :3])
    return cv2.cvtColor(crop, cv2.COLOR_RGBA2RGB)


class FastRapidOcrOptions(RapidOcrOptions):
    """docling's RapidOCR options under the engine kind ``rapidocr-fast``."""

    kind: ClassVar[Literal["rapidocr-fast"]] = "rapidocr-fast"  # ty: ignore[invalid-attribute-override]


class FastRapidOcrModel(RapidOcrModel):
    """docling's ``RapidOcrModel`` with one render per page and batched GPU inference for a page batch."""

    _driver: FastRapidOcr | Literal[False] | None = None

    @classmethod
    def get_options_type(cls) -> type[OcrOptions]:
        """Return the options class for this OCR engine."""
        return FastRapidOcrOptions

    def __call__(self, conv_res: ConversionResult, page_batch: Iterable[Page]) -> Iterable[Page]:
        """OCR a page batch: every rectangle of every valid page in one pass."""
        if not self.enabled:
            yield from page_batch
            return
        pages = list(page_batch)
        valid = [p for p in pages if p._backend is not None and p._backend.is_valid()]  # noqa: SLF001
        if valid:
            with _PerPageTime(conv_res, "ocr", len(valid)):
                self._ocr_pages(conv_res, valid)
        yield from pages

    def _fast_driver(self) -> FastRapidOcr | None:
        """The batched GPU driver, or None to call the reader per image as docling does."""
        if self._driver is None:
            try:
                from docling_pp_ocrv6._rapidocr_driver import FastRapidOcr, supported  # noqa: PLC0415
            except ImportError:  # no torch: RapidOCR runs another backend anyway
                self._driver = False
                return None
            options_allow = self.options.use_det is not False and self.options.use_cls is not False
            if options_allow and self.options.use_rec is not False and supported(self.reader):
                self._driver = FastRapidOcr(self.reader)
            else:
                logger.info("rapidocr-fast: batched driver needs the torch backend with det, cls and rec; per image")
                self._driver = False
        return None if self._driver is False else self._driver

    def _deduplicate_rects(
        self, size: Size, rects: Iterable[BoundingBox], dilation_size: int = 0
    ) -> tuple[float, list[BoundingBox]]:
        """Docling's OCR rectangle deduplication, computed on the rectangles (same result, no page raster)."""
        rects = list(rects)
        fast = deduplicate_rects(size, rects, dilation_size)
        if fast is None:
            return super()._deduplicate_rects(size, rects, dilation_size=dilation_size)
        coverage, boxes = fast
        return coverage, self._drop_contained_rects(boxes)

    def _page_images(self, page: Page, rects: list[BoundingBox]) -> list[np.ndarray]:
        """The OCR input image of every rectangle, with the pixels docling's own loop renders."""
        backend = page._backend  # noqa: SLF001
        assert backend is not None  # noqa: S101
        if rects and _is_docling_parse(backend):
            result = getattr(backend, "_result", None)
            if getattr(backend, "_rendered", False) and hasattr(result, "_crop_image"):
                # get_page_image(scale, cropbox) renders the whole page, crops and converts to RGB:
                # render once and crop every rectangle with the same function.
                rgba = _render_rgba(result, self.scale)
                if rgba is not None:
                    return [_crop_rgb(result, rgba, rect) for rect in rects]
                full = result.get_image(scale=self.scale)
                return [
                    np.asarray(result._crop_image(full, rect).convert("RGB"))  # noqa: SLF001
                    for rect in rects
                ]
        return [np.asarray(backend.get_page_image(scale=self.scale, cropbox=rect)) for rect in rects]

    def _cells(self, result: object, ocr_rect: BoundingBox) -> list[TextCell]:
        """Docling's text cells of one RapidOCR result, in page coordinates."""
        boxes = getattr(result, "boxes", None)
        if result is None or boxes is None:
            logger.warning("RapidOCR returned empty result!")
            return []
        lines = list(zip(boxes.tolist(), result.txts, result.scores, strict=False))  # ty: ignore[unresolved-attribute]
        return [
            TextCell(
                index=ix,
                text=text,
                orig=text,
                confidence=score,
                from_ocr=True,
                rect=BoundingRectangle.from_bounding_box(
                    BoundingBox.from_tuple(
                        coord=(
                            (box[0][0] / self.scale) + ocr_rect.l,
                            (box[0][1] / self.scale) + ocr_rect.t,
                            (box[2][0] / self.scale) + ocr_rect.l,
                            (box[2][1] / self.scale) + ocr_rect.t,
                        ),
                        origin=CoordOrigin.TOPLEFT,
                    )
                ),
            )
            for ix, (box, text, score) in enumerate(lines)
        ]

    def _ocr_pages(self, conv_res: ConversionResult, pages: list[Page]) -> None:
        work = []
        for page in pages:
            all_rects = self.get_ocr_rects(page)
            rects = [r for r in all_rects if r.area() != 0]
            work.append((page, all_rects, rects, self._page_images(page, rects)))
        images = [im for _, _, _, ims in work for im in ims]
        driver = self._fast_driver()
        if driver is not None:
            results = driver(images)
        else:
            results = [
                self.reader(
                    im, use_det=self.options.use_det, use_cls=self.options.use_cls, use_rec=self.options.use_rec
                )
                for im in images
            ]
        pos = 0
        for page, all_rects, rects, _ in work:
            cells: list[TextCell] = []
            for rect in rects:
                cells.extend(self._cells(results[pos], rect))
                pos += 1
            self.post_process_cells(cells, page, conv_res)
            if settings.debug.visualize_ocr:
                self.draw_ocr_rects_and_cells(conv_res, page, all_rects)
