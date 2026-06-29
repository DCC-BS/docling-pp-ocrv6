"""Pytest configuration and fixtures for docling_pp_ocrv6 tests."""

from __future__ import annotations

import os
import sys
from types import SimpleNamespace
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

if TYPE_CHECKING:
    from pathlib import Path


def pytest_collection_modifyitems(config, items):
    """Auto-skip e2e tests unless ``PPOCRV6_E2E`` is set.

    The e2e tests download the PP-OCRv6 ONNX models and run real inference.
    """
    if os.environ.get("PPOCRV6_E2E"):
        return
    skip_e2e = pytest.mark.skip(reason="PPOCRV6_E2E not set")
    for item in items:
        if "e2e" in item.keywords:
            item.add_marker(skip_e2e)


@pytest.fixture
def fake_rapidocr(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Install a fake ``rapidocr`` module so the model can be built without ONNX.

    Returns the ``RapidOCR`` mock class; ``RapidOCR.return_value`` is the reader
    instance the model stores as ``self.reader``.
    """
    module = SimpleNamespace()
    module.EngineType = SimpleNamespace(ONNXRUNTIME="onnxruntime")
    module.RapidOCR = MagicMock(name="RapidOCR")
    monkeypatch.setitem(sys.modules, "rapidocr", module)
    return module.RapidOCR


@pytest.fixture
def stub_models(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Stub model resolution so no network / ONNX files are required."""
    from docling_pp_ocrv6.model import PPOCRv6Model

    det = tmp_path / "det.onnx"
    rec = tmp_path / "rec.onnx"
    keys = tmp_path / "keys.txt"
    for f in (det, rec, keys):
        f.write_text("stub")

    monkeypatch.setattr(
        PPOCRv6Model,
        "_resolve_models",
        lambda self: (det, rec, keys, None),
    )
    return tmp_path


@pytest.fixture
def mock_model(fake_rapidocr: MagicMock, stub_models: Path):
    """Build a PPOCRv6Model with fake RapidOCR and stubbed model paths."""
    from docling.datamodel.accelerator_options import AcceleratorOptions

    from docling_pp_ocrv6.model import PPOCRv6Model
    from docling_pp_ocrv6.options import PPOCRv6Options

    model = PPOCRv6Model(
        enabled=True,
        artifacts_path=None,
        options=PPOCRv6Options(),
        accelerator_options=AcceleratorOptions(),
    )
    return model, fake_rapidocr
