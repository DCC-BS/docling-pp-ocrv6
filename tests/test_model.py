"""Tests for PPOCRv6Model."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
from docling.datamodel.pipeline_options import OcrOptions
from docling_core.types.doc import BoundingBox, CoordOrigin

from docling_pp_ocrv6.model import PPOCRv6Model
from docling_pp_ocrv6.options import PPOCRv6Options


def test_get_options_type():
    assert PPOCRv6Model.get_options_type() is PPOCRv6Options


def test_disabled_model_passes_pages_through():
    from docling.datamodel.accelerator_options import AcceleratorOptions

    model = PPOCRv6Model(
        enabled=False,
        artifacts_path=None,
        options=PPOCRv6Options(),
        accelerator_options=AcceleratorOptions(),
    )
    pages = [object(), object()]
    assert list(model(MagicMock(), iter(pages))) == pages


def test_build_passes_expected_params(mock_model):
    _model, rapidocr_cls = mock_model
    rapidocr_cls.assert_called_once()
    params = rapidocr_cls.call_args.kwargs["params"]
    assert params["Det.engine_type"] == "onnxruntime"
    assert params["Rec.engine_type"] == "onnxruntime"
    assert params["Det.model_path"].endswith("det.onnx")
    assert params["Rec.model_path"].endswith("rec.onnx")
    assert params["Rec.rec_keys_path"].endswith("keys.txt")
    # No explicit cls model -> RapidOCR's bundled cls model is used.
    assert "Cls.model_path" not in params


def test_rapidocr_params_override(fake_rapidocr, stub_models):
    from docling.datamodel.accelerator_options import AcceleratorOptions

    opts = PPOCRv6Options(rapidocr_params={"Global.text_score": 0.99})
    PPOCRv6Model(
        enabled=True,
        artifacts_path=None,
        options=opts,
        accelerator_options=AcceleratorOptions(),
    )
    params = fake_rapidocr.call_args.kwargs["params"]
    assert params["Global.text_score"] == 0.99


def test_ensure_rec_keys_extracts_dict(tmp_path):
    (tmp_path / "inference.yml").write_text(
        "PostProcess:\n  character_dict:\n    - a\n    - b\n    - c\n",
        encoding="utf-8",
    )
    keys = PPOCRv6Model._ensure_rec_keys(tmp_path)
    assert keys.read_text(encoding="utf-8") == "a\nb\nc\n"


def test_ensure_rec_keys_missing_dict_raises(tmp_path):
    (tmp_path / "inference.yml").write_text("PostProcess: {}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="character_dict"):
        PPOCRv6Model._ensure_rec_keys(tmp_path)


def test_call_converts_reader_result_to_cells(mock_model, monkeypatch):
    model, _ = mock_model

    # One OCR region covering the page.
    rect = BoundingBox(l=0, t=0, r=100, b=50, coord_origin=CoordOrigin.TOPLEFT)
    monkeypatch.setattr(model, "get_ocr_rects", lambda page: [rect])
    monkeypatch.setattr(model, "post_process_cells", lambda cells, page: cells)

    # Reader returns one detected line: a quad box, text, score.
    model.reader.return_value = SimpleNamespace(
        boxes=np.array([[[0, 0], [30, 0], [30, 10], [0, 10]]]),
        txts=["hello"],
        scores=[0.97],
    )

    backend = MagicMock()
    backend.is_valid.return_value = True
    backend.get_page_image.return_value = MagicMock()
    page = SimpleNamespace(_backend=backend)

    captured: list = []
    monkeypatch.setattr(model, "post_process_cells", lambda cells, page: captured.extend(cells))

    out = list(model(MagicMock(errors=[]), iter([page])))
    assert out == [page]
    assert len(captured) == 1
    cell = captured[0]
    assert cell.text == "hello"
    assert cell.from_ocr is True
    assert cell.confidence == pytest.approx(0.97)


def test_call_assigns_global_index_across_rects(mock_model, monkeypatch):
    model, _ = mock_model

    rect_a = BoundingBox(l=0, t=0, r=50, b=20, coord_origin=CoordOrigin.TOPLEFT)
    rect_b = BoundingBox(l=0, t=30, r=50, b=50, coord_origin=CoordOrigin.TOPLEFT)
    monkeypatch.setattr(model, "get_ocr_rects", lambda page: [rect_a, rect_b])

    # Reader returns one detected line per rect.
    model.reader.return_value = SimpleNamespace(
        boxes=np.array([[[0, 0], [30, 0], [30, 10], [0, 10]]]),
        txts=["word"],
        scores=[0.9],
    )

    captured: list = []
    monkeypatch.setattr(model, "post_process_cells", lambda cells, page: captured.extend(cells))

    backend = MagicMock()
    backend.is_valid.return_value = True
    backend.get_page_image.return_value = MagicMock()
    page = SimpleNamespace(_backend=backend)

    list(model(MagicMock(), iter([page])))
    # Two rects -> two cells with distinct running indices, not duplicated 0,0.
    assert [c.index for c in captured] == [0, 1]


def test_call_skips_when_boxes_missing(mock_model, monkeypatch):
    model, _ = mock_model
    rect = BoundingBox(l=0, t=0, r=50, b=20, coord_origin=CoordOrigin.TOPLEFT)
    monkeypatch.setattr(model, "get_ocr_rects", lambda page: [rect])

    # Recognition-only style output without a `boxes` attribute (use_det=False).
    model.reader.return_value = SimpleNamespace(txts=["x"], scores=[0.9])

    captured: list = []
    monkeypatch.setattr(model, "post_process_cells", lambda cells, page: captured.extend(cells))

    backend = MagicMock()
    backend.is_valid.return_value = True
    backend.get_page_image.return_value = MagicMock()
    page = SimpleNamespace(_backend=backend)

    out = list(model(MagicMock(), iter([page])))
    assert out == [page]
    assert captured == []


def test_call_skips_invalid_backend(mock_model):
    model, _ = mock_model
    backend = MagicMock()
    backend.is_valid.return_value = False
    page = SimpleNamespace(_backend=backend)
    assert list(model(MagicMock(), iter([page]))) == [page]


def test_options_type_is_ocr_options():
    assert issubclass(PPOCRv6Model.get_options_type(), OcrOptions)


def test_missing_rapidocr_raises(monkeypatch, stub_models):
    import builtins

    from docling.datamodel.accelerator_options import AcceleratorOptions

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "rapidocr":
            msg = "No module named 'rapidocr'"
            raise ImportError(msg)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match="RapidOCR is not installed"):
        PPOCRv6Model(
            enabled=True,
            artifacts_path=None,
            options=PPOCRv6Options(),
            accelerator_options=AcceleratorOptions(),
        )


def test_download_models_calls_hf(monkeypatch, tmp_path):
    calls = []

    def fake_download(repo, filename, local_dir, force_download=False):
        calls.append((repo, filename))
        return str(local_dir)

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    out = PPOCRv6Model.download_models(
        det_repo="org/det",
        rec_repo="org/rec",
        local_dir=tmp_path,
    )
    assert out == tmp_path
    assert (tmp_path / "det").is_dir()
    assert (tmp_path / "rec").is_dir()
    assert ("org/det", "inference.onnx") in calls
    assert ("org/rec", "inference.onnx") in calls
    assert ("org/rec", "inference.yml") in calls


def test_download_models_force_passed_through(monkeypatch, tmp_path):
    seen = []

    def fake_download(repo, filename, local_dir, force_download):
        seen.append(force_download)
        return str(local_dir)

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    PPOCRv6Model.download_models(local_dir=tmp_path, force=True)
    assert seen == [True, True, True]


def test_resolve_models_explicit_paths(monkeypatch, tmp_path):
    det = tmp_path / "d.onnx"
    rec = tmp_path / "r.onnx"
    keys = tmp_path / "k.txt"
    for f in (det, rec, keys):
        f.write_text("x")

    monkeypatch.setattr(PPOCRv6Model, "download_models", staticmethod(lambda **_: tmp_path))

    model = PPOCRv6Model.__new__(PPOCRv6Model)
    model.options = PPOCRv6Options(
        det_model_path=str(det),
        rec_model_path=str(rec),
        rec_keys_path=str(keys),
    )
    det_p, rec_p, keys_p, cls_p = model._resolve_models()
    assert det_p == det
    assert rec_p == rec
    assert keys_p == keys
    assert cls_p is None


def test_resolve_models_downloads_and_extracts_keys(monkeypatch, tmp_path):
    (tmp_path / "rec").mkdir()
    (tmp_path / "det").mkdir()
    (tmp_path / "det" / "inference.onnx").write_text("x")
    (tmp_path / "rec" / "inference.onnx").write_text("x")
    (tmp_path / "rec" / "inference.yml").write_text(
        "PostProcess:\n  character_dict:\n    - a\n    - b\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(PPOCRv6Model, "download_models", staticmethod(lambda **_: tmp_path))

    model = PPOCRv6Model.__new__(PPOCRv6Model)
    model.options = PPOCRv6Options()
    det_p, rec_p, keys_p, cls_p = model._resolve_models()
    assert det_p.name == "inference.onnx"
    assert rec_p.name == "inference.onnx"
    assert keys_p.read_text(encoding="utf-8") == "a\nb\n"
    assert cls_p is None
