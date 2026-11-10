"""Docling plugin entry point registering the PP-OCRv6 and fast RapidOCR engines."""

from docling.models.base_ocr_model import BaseOcrModel

from docling_pp_ocrv6.model import PPOCRv6Model
from docling_pp_ocrv6.rapidocr_fast import FastRapidOcrModel


def ocr_engines() -> dict[str, list[type[BaseOcrModel]]]:
    """Return the OCR engine classes provided by this plugin."""
    return {"ocr_engines": [PPOCRv6Model, FastRapidOcrModel]}
