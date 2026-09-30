"""PP-OCRv6 OCR engine for the docling standard pipeline.

Runs PaddlePaddle PP-OCRv6 detection and recognition ONNX models locally via
RapidOCR (onnxruntime) and returns the recognised text as ``TextCell`` objects
that docling merges with its standard-pipeline output.

The detection and recognition ONNX models are downloaded from HuggingFace on
first use and cached under docling's model cache. The recognition character
dictionary is extracted from the recognition model's ``inference.yml``. Angle
classification uses RapidOCR's bundled cls model unless an explicit path is
given.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import yaml
from docling.datamodel.accelerator_options import AcceleratorDevice
from docling.datamodel.settings import settings
from docling.models.base_ocr_model import BaseOcrModel
from docling.utils.accelerator_utils import decide_device
from docling.utils.profiling import TimeRecorder
from docling_core.types.doc import BoundingBox, CoordOrigin
from docling_core.types.doc.page import BoundingRectangle, TextCell

from docling_pp_ocrv6.options import PPOCRv6Options

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from docling.datamodel.accelerator_options import AcceleratorOptions
    from docling.datamodel.base_models import Page
    from docling.datamodel.document import ConversionResult
    from docling.datamodel.pipeline_options import OcrOptions

logger = logging.getLogger(__name__)

_ONNX_FILE = "inference.onnx"
_CONFIG_FILE = "inference.yml"
_REC_KEYS_FILE = "ppocrv6_keys.txt"


#: What RapidOCR gives per line when asked for word boxes: the word, how sure
#: it is, and the four corners it was read in, or None when it could not place it.
type WordResults = Sequence[Sequence[tuple[str, float, Sequence[Sequence[float]] | None]]] | None


def _word_cells(
    word_results: WordResults,
    ocr_rect: BoundingBox,
    scale: float,
    first_index: int,
) -> list[TextCell]:
    """The words of every line the recogniser read, in the page's own points.

    Each word keeps the quadrilateral it was read in, so a word of a line
    running at an angle is not widened into an upright box around it.
    """
    cells: list[TextCell] = []
    for line in word_results or ():
        for text, confidence, polygon in line:
            if polygon is None or not str(text).strip():
                continue
            (x0, y0), (x1, y1), (x2, y2), (x3, y3) = (
                ((x / scale) + ocr_rect.l, (y / scale) + ocr_rect.t) for x, y in polygon
            )
            cells.append(
                TextCell(
                    index=first_index + len(cells),
                    text=text,
                    orig=text,
                    confidence=confidence,
                    from_ocr=True,
                    rect=BoundingRectangle(
                        r_x0=x0,
                        r_y0=y0,
                        r_x1=x1,
                        r_y1=y1,
                        r_x2=x2,
                        r_y2=y2,
                        r_x3=x3,
                        r_y3=y3,
                        coord_origin=CoordOrigin.TOPLEFT,
                    ),
                )
            )
    return cells


def _post_process(
    model: BaseOcrModel,
    cells: list[TextCell],
    page: Page,
    conv_res: ConversionResult,
) -> None:
    """Hand the read cells to docling, whichever docling this is.

    Newer docling takes the conversion result as well, to record the OCR
    confidence of the page; older docling does not know the argument.
    """
    try:
        model.post_process_cells(cells, page, conv_res)
    except TypeError:
        model.post_process_cells(cells, page)


class PPOCRv6Model(BaseOcrModel):
    """OCR engine running PP-OCRv6 ONNX models through RapidOCR."""

    _model_repo_folder = "PPOCRv6"

    def __init__(
        self,
        enabled: bool,  # noqa: FBT001
        artifacts_path: Path | None,
        options: PPOCRv6Options,
        accelerator_options: AcceleratorOptions,
    ) -> None:
        """Initialise the OCR engine, downloading models on first use when enabled."""
        super().__init__(
            enabled=enabled,
            artifacts_path=artifacts_path,
            options=options,
            accelerator_options=accelerator_options,
        )
        self.options: PPOCRv6Options
        self.scale = 3  # multiplier for 72 dpi == 216 dpi.

        if not self.enabled:
            return

        try:
            from rapidocr import EngineType, RapidOCR  # noqa: PLC0415
        except ImportError as err:
            msg = (
                "RapidOCR is not installed. Install it via "
                "`pip install rapidocr onnxruntime` (or `onnxruntime-gpu` for CUDA) "
                "to use the PP-OCRv6 OCR engine."
            )
            raise ImportError(msg) from err

        device = decide_device(accelerator_options.device)
        use_cuda = str(AcceleratorDevice.CUDA.value).lower() in device
        use_dml = accelerator_options.device == AcceleratorDevice.AUTO
        gpu_id = int(device.split(":")[1]) if (use_cuda and ":" in device) else 0

        det_path, rec_path, rec_keys_path, cls_path = self._resolve_models()
        logger.info(
            "Loading PP-OCRv6 (device=%s, cuda=%s): det=%s rec=%s",
            device,
            use_cuda,
            det_path,
            rec_path,
        )

        params: dict = {
            "Global.text_score": self.options.text_score,
            "EngineConfig.onnxruntime.intra_op_num_threads": accelerator_options.num_threads,
            "Det.model_path": str(det_path),
            "Det.engine_type": EngineType.ONNXRUNTIME,
            "Det.use_cuda": use_cuda,
            "Det.use_dml": use_dml,
            "Rec.model_path": str(rec_path),
            "Rec.rec_keys_path": str(rec_keys_path),
            "Rec.engine_type": EngineType.ONNXRUNTIME,
            "Rec.use_cuda": use_cuda,
            "Rec.use_dml": use_dml,
            "Cls.engine_type": EngineType.ONNXRUNTIME,
            "Cls.use_cuda": use_cuda,
            "Cls.use_dml": use_dml,
            "EngineConfig.onnxruntime.use_cuda": use_cuda,
            "EngineConfig.onnxruntime.cuda_ep_cfg.device_id": gpu_id,
        }
        # When no explicit cls model is given, let RapidOCR use its bundled cls model.
        if cls_path is not None:
            params["Cls.model_path"] = str(cls_path)

        if self.options.rapidocr_params:
            params.update(self.options.rapidocr_params)

        self.reader = RapidOCR(params=params)

    def _resolve_models(self) -> tuple[Path, Path, Path, Path | None]:
        """Resolve detection, recognition, rec-keys and (optional) cls model paths.

        Explicit option paths win; otherwise models are downloaded from
        HuggingFace and cached. The recognition character dictionary is
        extracted from the recognition model's ``inference.yml`` when not
        provided explicitly.
        """
        local_dir = self.download_models(
            det_repo=self.options.det_repo,
            rec_repo=self.options.rec_repo,
        )

        det_path = Path(self.options.det_model_path) if self.options.det_model_path else local_dir / "det" / _ONNX_FILE
        rec_path = Path(self.options.rec_model_path) if self.options.rec_model_path else local_dir / "rec" / _ONNX_FILE

        if self.options.rec_keys_path:
            rec_keys_path = Path(self.options.rec_keys_path)
        else:
            rec_keys_path = self._ensure_rec_keys(local_dir / "rec")

        cls_path = Path(self.options.cls_model_path) if self.options.cls_model_path else None

        for path in (det_path, rec_path, rec_keys_path):
            if not path.exists():
                logger.warning("PP-OCRv6 model path does not exist: %s", path)

        return det_path, rec_path, rec_keys_path, cls_path

    @staticmethod
    def _ensure_rec_keys(rec_dir: Path) -> Path:
        """Extract the recognition character dictionary into a RapidOCR keys file.

        PaddleX exports embed the dictionary as ``PostProcess.character_dict``
        inside ``inference.yml``; RapidOCR expects a plain text file with one
        character per line.
        """
        keys_path = rec_dir / _REC_KEYS_FILE
        if keys_path.exists():
            return keys_path

        config = yaml.safe_load((rec_dir / _CONFIG_FILE).read_text(encoding="utf-8"))
        chars = config.get("PostProcess", {}).get("character_dict")
        if not chars:
            msg = f"No 'PostProcess.character_dict' found in {rec_dir / _CONFIG_FILE}"
            raise ValueError(msg)

        keys_path.write_text("\n".join(chars) + "\n", encoding="utf-8")
        logger.info("Wrote PP-OCRv6 recognition dictionary (%d entries) to %s", len(chars), keys_path)
        return keys_path

    @staticmethod
    def download_models(
        det_repo: str = "PaddlePaddle/PP-OCRv6_medium_det_onnx",
        rec_repo: str = "PaddlePaddle/PP-OCRv6_medium_rec_onnx",
        local_dir: Path | None = None,
        force: bool = False,  # noqa: FBT001, FBT002
    ) -> Path:
        """Download the PP-OCRv6 detection and recognition ONNX models from HuggingFace.

        Returns the local directory containing ``det/`` and ``rec/`` sub-folders.
        Pre-fetching at image-build time avoids blocking the first request on the
        download. Pass ``force=True`` to re-download even when a cached copy exists
        (e.g. to repair a corrupted file).
        """
        from huggingface_hub import hf_hub_download  # noqa: PLC0415

        if local_dir is None:
            local_dir = settings.cache_dir / "models" / PPOCRv6Model._model_repo_folder

        det_dir = local_dir / "det"
        rec_dir = local_dir / "rec"
        det_dir.mkdir(parents=True, exist_ok=True)
        rec_dir.mkdir(parents=True, exist_ok=True)

        hf_hub_download(det_repo, _ONNX_FILE, local_dir=det_dir, force_download=force)
        hf_hub_download(rec_repo, _ONNX_FILE, local_dir=rec_dir, force_download=force)
        hf_hub_download(rec_repo, _CONFIG_FILE, local_dir=rec_dir, force_download=force)

        return local_dir

    def get_ocr_rects(self, page: Page) -> list[BoundingBox]:
        """What to read on this page: the whole of it, or docling's own choice.

        Docling crops the page into the boxes its layout model found. Those
        are drawn around what a region means, and a line of text that crosses
        one is handed to the recogniser cut in two.
        """
        if not self.options.whole_page or page.size is None:
            return super().get_ocr_rects(page)
        return [
            BoundingBox(
                l=0,
                t=0,
                r=page.size.width,
                b=page.size.height,
                coord_origin=CoordOrigin.TOPLEFT,
            )
        ]

    def __call__(self, conv_res: ConversionResult, page_batch: Iterable[Page]) -> Iterable[Page]:
        """Run OCR on each page crop and yield pages with recognised text cells."""
        if not self.enabled:
            yield from page_batch
            return

        for page in page_batch:
            if page._backend is None or not page._backend.is_valid():  # noqa: SLF001
                yield page
                continue

            with TimeRecorder(conv_res, "ocr"):
                ocr_rects = self.get_ocr_rects(page)
                all_ocr_cells: list[TextCell] = []
                all_ocr_words: list[TextCell] = []
                cell_idx = 0

                for ocr_rect in ocr_rects:
                    if ocr_rect.area() == 0:
                        continue
                    high_res_image = page._backend.get_page_image(scale=self.scale, cropbox=ocr_rect)  # noqa: SLF001
                    im = np.array(high_res_image)
                    # cast: RapidOCR's return type is a union of stage-specific outputs
                    result = cast(
                        "Any",
                        self.reader(
                            im,
                            use_det=self.options.use_det,
                            use_cls=self.options.use_cls,
                            use_rec=self.options.use_rec,
                            return_word_box=self.options.return_word_box,
                        ),
                    )
                    # a disabled stage means the matching attribute is absent
                    all_ocr_words.extend(
                        _word_cells(
                            getattr(result, "word_results", None),
                            ocr_rect,
                            self.scale,
                            len(all_ocr_words),
                        )
                    )
                    boxes = getattr(result, "boxes", None) if result is not None else None
                    txts = getattr(result, "txts", None)
                    scores = getattr(result, "scores", None)
                    if boxes is None or txts is None or scores is None:
                        continue

                    for box, text, score in zip(boxes.tolist(), txts, scores, strict=False):
                        all_ocr_cells.append(
                            TextCell(
                                index=cell_idx,
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
                        )
                        cell_idx += 1

                _post_process(self, all_ocr_cells, page, conv_res)
                if all_ocr_words and page.parsed_page is not None:
                    # post-processing writes the lines and leaves the words of
                    # the text layer alone, so the read words are added after it.
                    page.parsed_page.word_cells = [*page.parsed_page.word_cells, *all_ocr_words]
                    page.parsed_page.has_words = True

            if settings.debug.visualize_ocr:
                self.draw_ocr_rects_and_cells(conv_res, page, ocr_rects)

            yield page

    @classmethod
    def get_options_type(cls) -> type[OcrOptions]:
        """Return the options class for this OCR engine."""
        return PPOCRv6Options
