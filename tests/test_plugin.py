"""Tests for the docling plugin entry point."""

from __future__ import annotations

from docling.models.base_ocr_model import BaseOcrModel

from docling_pp_ocrv6.plugin import ocr_engines


def test_ocr_engines_returns_dict():
    result = ocr_engines()
    assert isinstance(result, dict)
    assert "ocr_engines" in result


def test_ocr_engines_contains_model_class():
    classes = ocr_engines()["ocr_engines"]
    assert len(classes) == 2
    assert all(issubclass(c, BaseOcrModel) for c in classes)


def test_registered_model_name():
    names = [cls.__name__ for cls in ocr_engines()["ocr_engines"]]
    assert names == ["PPOCRv6Model", "FastRapidOcrModel"]
