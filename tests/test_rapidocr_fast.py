"""Tests for the ``rapidocr-fast`` engine: docling's RapidOCR with one render per page and a batched GPU driver."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
from docling.datamodel.pipeline_options import RapidOcrOptions
from docling.models.base_ocr_model import BaseOcrModel
from docling_core.types.doc import BoundingBox, Size
from PIL import Image, ImageDraw, ImageFont

from docling_pp_ocrv6.rapidocr_fast import FastRapidOcrModel, FastRapidOcrOptions


def test_options_are_rapidocr_options_under_their_own_kind():
    opts = FastRapidOcrOptions(backend="torch")
    assert FastRapidOcrOptions.kind == "rapidocr-fast"
    assert RapidOcrOptions.kind == "rapidocr"
    assert isinstance(opts, RapidOcrOptions)
    assert opts.backend == "torch"
    assert FastRapidOcrModel.get_options_type() is FastRapidOcrOptions


def _bare_model() -> FastRapidOcrModel:
    model = object.__new__(FastRapidOcrModel)
    model.enabled = True
    model.options = FastRapidOcrOptions()
    model.scale = 3
    model.reader = MagicMock(name="RapidOCR")
    model._driver = False  # the per-image path: the reader mock is called as docling calls it
    return model


def test_no_rects_skip_rasterising():
    model = _bare_model()
    assert model._deduplicate_rects(Size(width=600, height=800), iter([])) == (0.0, [])


def test_rects_merge_as_docling_does():
    model = _bare_model()
    size = Size(width=600, height=800)
    rects = [BoundingBox(l=10, t=10, r=100, b=50), BoundingBox(l=90, t=20, r=200, b=60)]
    assert model._deduplicate_rects(size, iter(rects), dilation_size=2) == BaseOcrModel._deduplicate_rects(
        model, size, iter(rects), dilation_size=2
    )


def _page(rects: list[BoundingBox]) -> SimpleNamespace:
    backend = MagicMock(name="backend")
    backend.is_valid.return_value = True
    backend.get_page_image.side_effect = lambda scale, cropbox: Image.new("RGB", (30, 10), "white")
    return SimpleNamespace(_backend=backend, rects=rects)


def test_call_runs_every_rect_of_the_batch_and_maps_cells_to_the_page(monkeypatch):
    model = _bare_model()
    rects = [BoundingBox(l=10, t=20, r=110, b=60), BoundingBox(l=0, t=0, r=0, b=0)]
    pages = [_page(rects), _page(rects[:1])]
    invalid = SimpleNamespace(_backend=MagicMock(is_valid=MagicMock(return_value=False)))
    monkeypatch.setattr(model, "get_ocr_rects", lambda page: page.rects)
    collected = {}
    monkeypatch.setattr(
        model, "post_process_cells", lambda cells, page, conv_res: collected.setdefault(id(page), cells)
    )
    model.reader.return_value = SimpleNamespace(
        boxes=np.array([[[3, 6], [30, 6], [30, 15], [3, 15]]], dtype=float), txts=("Rechnung",), scores=(0.9,)
    )
    conv_res = SimpleNamespace(timings={})

    out = list(model(conv_res, [pages[0], invalid, pages[1]]))

    assert out == [pages[0], invalid, pages[1]]
    assert model.reader.call_count == 2  # the zero-area rect is skipped
    cell = collected[id(pages[0])][0]
    assert cell.text == "Rechnung"
    assert cell.from_ocr
    box = cell.rect.to_bounding_box()
    assert (box.l, box.t, box.r, box.b) == pytest.approx((11, 22, 20, 25))


def test_disabled_model_passes_pages_through():
    model = _bare_model()
    model.enabled = False
    pages = [object(), object()]
    assert list(model(SimpleNamespace(), pages)) == pages
    model.reader.assert_not_called()


def _text_image(lines: list[str], *, flip: bool = False) -> np.ndarray:
    font = ImageFont.load_default(size=28)
    img = Image.new("RGB", (900, 60 * len(lines) + 20), "white")
    draw = ImageDraw.Draw(img)
    for i, line in enumerate(lines):
        draw.text((20, 15 + 60 * i), line, fill="black", font=font)
    if flip:
        img = img.rotate(180)
    return np.asarray(img)


@pytest.mark.e2e
def test_driver_gives_rapidocrs_own_results():
    """The batched driver returns what RapidOCR returns per image, bit for bit (needs RapidOCR's torch models)."""
    torch = pytest.importorskip("torch")
    from rapidocr import EngineType, RapidOCR

    from docling_pp_ocrv6._rapidocr_driver import FastRapidOcr, supported

    params = {f"{part}.engine_type": EngineType.TORCH for part in ("Det", "Cls", "Rec")}
    if torch.cuda.is_available():
        params |= {"EngineConfig.torch.use_cuda": True, "EngineConfig.torch.cuda_ep_cfg.device_id": 0}
    reader = RapidOCR(params=params)
    assert supported(reader)
    images = [
        _text_image(["Rechnung Nr. 2026-114", "Total CHF 1234.56"]),
        _text_image(["Crème brûlée café", "kurz", "Eine deutlich längere Zeile mit vielen Wörtern darin"]),
        _text_image(["Auf dem Kopf stehender Text"], flip=True),
        np.full((200, 300, 3), 255, dtype=np.uint8),
    ]
    expected = [reader(im) for im in images]
    for batch in (False, True):
        got = FastRapidOcr(reader, batch=batch)(images)
        for want, have in zip(expected, got, strict=True):
            assert have.txts == want.txts
            assert have.scores == want.scores
            if want.boxes is None:
                assert have.boxes is None
            else:
                np.testing.assert_array_equal(have.boxes, want.boxes)
    assert any(r.txts for r in expected)


class _Pre:
    """RapidOCR's DetPreProcess, reduced to the formula and an identity resize."""

    mean = np.array([0.5, 0.5, 0.5], dtype=np.float32)
    std = np.array([0.5, 0.5, 0.5], dtype=np.float32)

    def normalize(self, img: np.ndarray) -> np.ndarray:
        return (img.astype(np.float32) / 255.0 - self.mean) / self.std

    def resize(self, img: np.ndarray) -> np.ndarray:
        return img


def _fake_reader(torch):
    """A RapidOCR reader whose nets are simple functions of their (normalised, padded) input."""
    dev = torch.device("cpu")

    def det_net(x):
        return x.mean(dim=1, keepdim=True)

    def det_post(pred, shape):
        h, w = shape
        boxes = np.array(
            [[[2, y], [w - 2, y], [w - 2, y + h // 4], [2, y + h // 4]] for y in range(0, h - h // 4, h // 4)],
            dtype=np.float32,
        )
        return boxes, [float(pred.mean())] * len(boxes)

    def crop(img, boxes):
        return [img[int(b[0][1]) : int(b[2][1]), : 20 + 37 * k % img.shape[1]] for k, b in enumerate(boxes)]

    def cls_net(x):
        filled = (x != 0).float().mean(dim=(1, 2, 3))
        return torch.stack([1 - filled, filled], dim=1)

    def cls_post(prob):
        return [("180", float(p[1])) if p[1] > 0.6 else ("0", float(p[0])) for p in prob]

    def rec_net(x):
        n, _, _, w = x.shape
        cols = x.sum(dim=(1, 2)).reshape(n, w // 8, 8)
        return torch.softmax(cols / 50.0, dim=2)

    def decode(idx, prob, _wb, *, remove_duplicate):
        assert remove_duplicate
        return [("".join(map(str, row)), float(p.mean())) for row, p in zip(idx, prob, strict=True)], None

    def model(net, **extra):
        return SimpleNamespace(session=SimpleNamespace(device=dev, predictor=net), **extra)

    det = model(det_net, postprocess_op=det_post, sorted_boxes=lambda b: b, get_preprocess=_Pre)
    cls = model(cls_net, cls_image_shape=[3, 48, 192], cls_batch_num=6, cls_thresh=0.9, postprocess_op=cls_post)
    rec = model(rec_net, rec_image_shape=[3, 48, 320], rec_batch_num=3, postprocess_op=SimpleNamespace(decode=decode))
    return SimpleNamespace(
        _load_det_model=lambda: det,
        _load_cls_model=lambda: cls,
        _load_rec_model=lambda: rec,
        load_img=lambda im: im,
        preprocess_img=lambda im: (im, {"preprocess": {"ratio_h": 1.0, "ratio_w": 1.0}}),
        cfg=SimpleNamespace(Global=SimpleNamespace(use_vertical_padding=False)),
        crop_text_regions=crop,
        text_score=0.0,
    )


def test_driver_batch_gives_the_per_image_results():
    """All rectangles of a batch in one pass give what one image at a time gives; inputs normalised as on the CPU."""
    torch = pytest.importorskip("torch")
    from docling_pp_ocrv6._rapidocr_driver import FastRapidOcr

    rng = np.random.default_rng(7)
    images = [rng.integers(0, 256, (h, w, 3), dtype=np.uint8) for h, w in ((96, 400), (160, 230), (64, 900))]
    reader = _fake_reader(torch)
    one = FastRapidOcr(reader, batch=False)
    many = FastRapidOcr(reader, batch=True)

    np.testing.assert_array_equal(
        one._det_input(images[0])[0].permute(1, 2, 0).numpy(), _Pre().normalize(images[0]).astype(np.float32)
    )
    u8 = images[1][:48, :100][None]
    padded = one._rc_input(np.pad(u8, ((0, 0), (0, 0), (0, 20), (0, 0))), np.array([100]))
    want = (u8.astype(np.float32) / 255 - 0.5) / 0.5
    np.testing.assert_array_equal(padded[0, :, :, :100].permute(1, 2, 0).numpy(), want[0])
    assert not padded[0, :, :, 100:].any()

    want_all, got_all = one(images), many(images)
    assert len(got_all) == len(images)
    for want_res, got_res in zip(want_all, got_all, strict=True):
        assert got_res.txts == want_res.txts
        assert got_res.scores == want_res.scores
        np.testing.assert_array_equal(got_res.boxes, want_res.boxes)
    assert all(r.txts for r in got_all)
