"""Configuration model for the PP-OCRv6 OCR engine."""

from __future__ import annotations

import os
from typing import ClassVar, Literal

from docling.datamodel.pipeline_options import OcrOptions
from pydantic import ConfigDict, Field

_DEFAULT_DET_REPO = "PaddlePaddle/PP-OCRv6_medium_det_onnx"
_DEFAULT_REC_REPO = "PaddlePaddle/PP-OCRv6_medium_rec_onnx"

# PP-OCRv6 medium is a single multilingual model covering Latin-script
# European languages (and many more); ``lang`` is only a passthrough hint to
# docling and does not switch models. The default leads with German and
# includes the common European languages.
_DEFAULT_LANGS = "de,en,fr,it,es,nl,pt,pl,sv,da,fi,nb,cs,ro,hu"


def _env_bool(name: str, default: bool) -> bool:  # noqa: FBT001
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


class PPOCRv6Options(OcrOptions):
    """Options for the PP-OCRv6 OCR engine.

    The engine runs PaddlePaddle's PP-OCRv6 detection and recognition ONNX
    models locally through RapidOCR (onnxruntime). Models are downloaded from
    HuggingFace on first use and cached. GPU is used automatically when the
    docling accelerator device resolves to CUDA and ``onnxruntime-gpu`` is
    installed.

    All options fall back to environment variables when not set explicitly,
    allowing configuration without code changes (e.g. in Docker / Compose
    deployments).

    Attributes:
        lang: Language hint list (passed to docling). PP-OCRv6 medium is a
            single multilingual model, so this does not switch models; it
            recognises its supported languages automatically. Defaults to a
            German-led set of common European languages. Falls back to the
            ``PPOCRV6_LANG`` env var as a comma-separated string.
        text_score: Minimum recognition confidence; lower-scoring cells are
            dropped by RapidOCR. Falls back to ``PPOCRV6_TEXT_SCORE``.
        use_det: Run the text-detection stage. Falls back to ``PPOCRV6_USE_DET``.
        use_cls: Run the angle-classification stage (RapidOCR's bundled cls
            model). Falls back to ``PPOCRV6_USE_CLS``.
        use_rec: Run the text-recognition stage. Falls back to
            ``PPOCRV6_USE_REC``.
        det_repo: HuggingFace repo id for the detection ONNX model.
            Falls back to ``PPOCRV6_DET_REPO``.
        rec_repo: HuggingFace repo id for the recognition ONNX model.
            Falls back to ``PPOCRV6_REC_REPO``.
        det_model_path: Explicit local path to a detection ONNX file. When set,
            no detection model is downloaded. Falls back to ``PPOCRV6_DET_MODEL_PATH``.
        rec_model_path: Explicit local path to a recognition ONNX file. When set,
            no recognition model is downloaded. Falls back to ``PPOCRV6_REC_MODEL_PATH``.
        rec_keys_path: Explicit local path to the recognition character
            dictionary (one character per line). When unset it is extracted
            from the recognition model's ``inference.yml``. Falls back to
            ``PPOCRV6_REC_KEYS_PATH``.
        cls_model_path: Explicit local path to an angle-classification ONNX
            file. When unset, RapidOCR's bundled cls model is used. Falls back
            to ``PPOCRV6_CLS_MODEL_PATH``.
        rapidocr_params: Extra RapidOCR ``params`` overrides merged on top of
            the engine defaults.
    """

    kind: ClassVar[Literal["pp-ocrv6"]] = "pp-ocrv6"

    lang: list[str] = Field(default_factory=lambda: os.environ.get("PPOCRV6_LANG", _DEFAULT_LANGS).split(","))
    text_score: float = Field(default_factory=lambda: float(os.environ.get("PPOCRV6_TEXT_SCORE", "0.5")))
    use_det: bool = Field(default_factory=lambda: _env_bool("PPOCRV6_USE_DET", default=True))
    use_cls: bool = Field(default_factory=lambda: _env_bool("PPOCRV6_USE_CLS", default=True))
    use_rec: bool = Field(default_factory=lambda: _env_bool("PPOCRV6_USE_REC", default=True))

    det_repo: str = Field(default_factory=lambda: os.environ.get("PPOCRV6_DET_REPO", _DEFAULT_DET_REPO))
    rec_repo: str = Field(default_factory=lambda: os.environ.get("PPOCRV6_REC_REPO", _DEFAULT_REC_REPO))

    det_model_path: str | None = Field(default_factory=lambda: os.environ.get("PPOCRV6_DET_MODEL_PATH") or None)
    rec_model_path: str | None = Field(default_factory=lambda: os.environ.get("PPOCRV6_REC_MODEL_PATH") or None)
    rec_keys_path: str | None = Field(default_factory=lambda: os.environ.get("PPOCRV6_REC_KEYS_PATH") or None)
    cls_model_path: str | None = Field(default_factory=lambda: os.environ.get("PPOCRV6_CLS_MODEL_PATH") or None)

    rapidocr_params: dict = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")
