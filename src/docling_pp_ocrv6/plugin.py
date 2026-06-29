"""Docling plugin entry point registering the PP-OCRv6 OCR engine."""

from docling_pp_ocrv6.model import PPOCRv6Model


def ocr_engines() -> dict[str, list[type[PPOCRv6Model]]]:
    """Return the OCR engine classes provided by this plugin."""
    return {"ocr_engines": [PPOCRv6Model]}
