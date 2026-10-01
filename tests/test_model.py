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


_REC_CONFIG = "PostProcess:\n  character_dict:\n    - a\n    - b\n"


def _fake_hub(calls: list):
    """hf_hub_download, writing a stand-in for each file it is asked for."""

    def fake_download(repo, filename, local_dir, *, force_download=False):
        calls.append((repo, filename, force_download))
        (local_dir / filename).write_text(_REC_CONFIG if filename == "inference.yml" else "x", encoding="utf-8")
        return str(local_dir / filename)

    return fake_download


def test_download_models_calls_hf(monkeypatch, tmp_path):
    import huggingface_hub

    calls: list = []
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", _fake_hub(calls))
    out = PPOCRv6Model.download_models(det_repo="org/det", rec_repo="org/rec", local_dir=tmp_path)
    assert out == tmp_path
    assert [(repo, name) for repo, name, _ in calls] == [
        ("org/det", "inference.onnx"),
        ("org/rec", "inference.onnx"),
        ("org/rec", "inference.yml"),
    ]
    # the keys are written at download time, so a baked image needs no write later
    assert (tmp_path / "rec" / "ppocrv6_keys.txt").read_text(encoding="utf-8") == "a\nb\n"


def test_download_models_force_passed_through(monkeypatch, tmp_path):
    import huggingface_hub

    calls: list = []
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", _fake_hub(calls))
    PPOCRv6Model.download_models(local_dir=tmp_path, force=True)
    assert [force for _, _, force in calls] == [True, True, True]


def _unbuilt_model(artifacts_path, **options):
    model = PPOCRv6Model.__new__(PPOCRv6Model)
    model.artifacts_path = artifacts_path
    model.options = PPOCRv6Options(**options)
    return model


def _no_download(**_):
    msg = "must not download"
    raise AssertionError(msg)


def test_resolve_models_explicit_paths(monkeypatch, tmp_path):
    det, rec, keys = tmp_path / "d.onnx", tmp_path / "r.onnx", tmp_path / "k.txt"
    for f in (det, rec, keys):
        f.write_text("x")
    monkeypatch.setattr(PPOCRv6Model, "download_models", staticmethod(_no_download))

    model = _unbuilt_model(tmp_path, det_model_path=str(det), rec_model_path=str(rec), rec_keys_path=str(keys))
    assert model._resolve_models() == (det, rec, keys, None)


def test_resolve_models_uses_models_in_artifacts_path(monkeypatch, tmp_path):
    # what an image bakes: the files under <artifacts_path>/PPOCRv6
    local = tmp_path / "PPOCRv6"
    (local / "det").mkdir(parents=True)
    (local / "rec").mkdir()
    (local / "det" / "inference.onnx").write_text("x")
    (local / "rec" / "inference.onnx").write_text("x")
    (local / "rec" / "ppocrv6_keys.txt").write_text("a\n")
    monkeypatch.setattr(PPOCRv6Model, "download_models", staticmethod(_no_download))

    det_p, rec_p, keys_p, cls_p = _unbuilt_model(tmp_path)._resolve_models()
    assert (det_p, rec_p, keys_p) == (
        local / "det" / "inference.onnx",
        local / "rec" / "inference.onnx",
        local / "rec" / "ppocrv6_keys.txt",
    )
    assert cls_p is None


def test_resolve_models_downloads_what_is_missing(monkeypatch, tmp_path):
    import huggingface_hub

    calls: list = []
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", _fake_hub(calls))

    det_p, rec_p, keys_p, _ = _unbuilt_model(tmp_path)._resolve_models()
    assert len(calls) == 3
    assert det_p.exists()
    assert rec_p.exists()
    assert keys_p.read_text(encoding="utf-8") == "a\nb\n"


# --- words -----------------------------------------------------------------

_ANNA = ("Anna", 0.9, [[0, 0], [30, 0], [30, 15], [0, 15]])
_MUSTER = ("Muster", 0.9, [[36, 0], [90, 0], [90, 15], [36, 15]])
#: What RapidOCR answers where it read nothing, asked for words or not.
_READ_NOTHING = (("", 1.0, None),)


def _rect(left, top, right, bottom):
    from docling_core.types.doc.page import BoundingRectangle

    return BoundingRectangle.from_bounding_box(
        BoundingBox(l=left, t=top, r=right, b=bottom, coord_origin=CoordOrigin.TOPLEFT)
    )


def _text_layer_word(text, box, *, invisible=False):
    from docling_core.types.doc.page import PdfCellRenderingMode, PdfTextCell

    return PdfTextCell(
        index=0,
        text=text,
        orig=text,
        rect=_rect(*box),
        from_ocr=False,
        rendering_mode=PdfCellRenderingMode.INVISIBLE if invisible else PdfCellRenderingMode.FILL_TEXT,
        text_direction="left_to_right",
        font_key="f",
        font_name="f",
        widget=False,
    )


def _page(*words):
    backend = MagicMock()
    backend.is_valid.return_value = True
    backend.get_page_image.return_value = MagicMock()
    return SimpleNamespace(
        _backend=backend,
        size=SimpleNamespace(width=595, height=842),
        parsed_page=SimpleNamespace(word_cells=list(words), has_words=bool(words)),
    )


def _run(model, monkeypatch, page, word_results, rect=None):
    rect = rect or BoundingBox(l=0, t=0, r=100, b=50, coord_origin=CoordOrigin.TOPLEFT)
    monkeypatch.setattr(model, "get_ocr_rects", lambda page: [rect])
    lines: list = []
    monkeypatch.setattr(model, "post_process_cells", lambda cells, page: lines.extend(cells))
    model.reader.return_value = SimpleNamespace(
        boxes=np.array([[[0, 0], [90, 0], [90, 15], [0, 15]]]),
        txts=["Anna Muster"],
        scores=[0.9],
        word_results=word_results,
    )
    list(model(MagicMock(), iter([page])))
    return lines


def test_whole_page_reads_the_page_as_one_rect(mock_model):
    model, _ = mock_model
    model.options.whole_page = True
    rects = model.get_ocr_rects(SimpleNamespace(size=SimpleNamespace(width=595, height=842)))
    assert [(r.l, r.t, r.r, r.b) for r in rects] == [(0, 0, 595, 842)]


def test_without_whole_page_docling_chooses_the_rects(mock_model, monkeypatch):
    from docling.models.base_ocr_model import BaseOcrModel

    model, _ = mock_model
    layout_rect = BoundingBox(l=10, t=10, r=20, b=20, coord_origin=CoordOrigin.TOPLEFT)
    monkeypatch.setattr(BaseOcrModel, "get_ocr_rects", lambda self, page: [layout_rect])
    assert model.get_ocr_rects(SimpleNamespace(size=SimpleNamespace(width=595, height=842))) == [layout_rect]


def test_word_cells_are_placed_in_page_points():
    from docling_pp_ocrv6.model import _word_cells

    rect = BoundingBox(l=100, t=200, r=400, b=300, coord_origin=CoordOrigin.TOPLEFT)
    cells = _word_cells([[_ANNA, _MUSTER]], rect, scale=3)
    assert [c.text for c in cells] == ["Anna", "Muster"]
    assert all(c.from_ocr for c in cells)
    box = cells[1].rect
    assert (box.r_x0, box.r_y0, box.r_x2, box.r_y2) == (112, 200, 130, 205)


def test_word_cells_skip_placeholders_and_unplaced_words():
    from docling_pp_ocrv6.model import _word_cells

    rect = BoundingBox(l=0, t=0, r=10, b=10, coord_origin=CoordOrigin.TOPLEFT)
    assert _word_cells(_READ_NOTHING, rect, scale=3) == []
    assert _word_cells([[("lost", 0.7, None), (" ", 0.7, _ANNA[2])]], rect, scale=3) == []
    assert _word_cells(None, rect, scale=3) == []


@pytest.mark.parametrize("return_word_box", [True, False])
def test_a_region_read_empty_does_not_fail(mock_model, monkeypatch, return_word_box):
    model, _ = mock_model
    model.options.return_word_box = return_word_box
    page = _page()
    _run(model, monkeypatch, page, _READ_NOTHING)
    assert page.parsed_page.word_cells == []


def test_return_word_box_adds_read_words_after_the_text_layer(mock_model, monkeypatch):
    model, _ = mock_model
    model.options.return_word_box = True
    # a word drawn elsewhere on the page
    drawn = _text_layer_word("Seite", (500, 800, 520, 810))
    page = _page(drawn)

    lines = _run(model, monkeypatch, page, [[_ANNA, _MUSTER]])

    assert model.reader.call_args.kwargs["return_word_box"] is True
    assert [c.text for c in lines] == ["Anna Muster"]
    words = page.parsed_page.word_cells
    assert words[0] is drawn
    assert [(w.text, w.index) for w in words[1:]] == [("Anna", 1), ("Muster", 2)]


def test_words_are_not_added_unless_asked(mock_model, monkeypatch):
    model, _ = mock_model
    page = _page()
    _run(model, monkeypatch, page, [[_ANNA, _MUSTER]])
    assert model.reader.call_args.kwargs["return_word_box"] is False
    assert page.parsed_page.word_cells == []
    assert page.parsed_page.has_words is False


def test_a_word_the_text_layer_draws_is_not_doubled(mock_model, monkeypatch):
    model, _ = mock_model
    model.options.return_word_box = True
    # "Anna" is drawn where OCR reads it (at 1/3 scale: 0,0-10,5); "Muster" only shows in the pixels
    drawn = _text_layer_word("Anna", (1, 1, 9, 4))
    page = _page(drawn)
    _run(model, monkeypatch, page, [[_ANNA, _MUSTER]])
    assert [w.text for w in page.parsed_page.word_cells] == ["Anna", "Muster"]
    assert page.parsed_page.word_cells[0] is drawn


def test_a_word_over_invisible_text_is_kept(mock_model, monkeypatch):
    model, _ = mock_model
    model.options.return_word_box = True
    hidden = _text_layer_word("Anna", (1, 1, 9, 4), invisible=True)
    page = _page(hidden)
    _run(model, monkeypatch, page, [[_ANNA]])
    assert [(w.text, w.from_ocr) for w in page.parsed_page.word_cells] == [("Anna", False), ("Anna", True)]
