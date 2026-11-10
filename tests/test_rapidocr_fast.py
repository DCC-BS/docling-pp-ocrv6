"""Tests for the ``rapidocr-fast`` engine: docling's RapidOCR with one render per page and a batched GPU driver."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest
from docling.datamodel.pipeline_options import RapidOcrOptions
from docling_core.types.doc import BoundingBox, CoordOrigin
from PIL import Image, ImageDraw, ImageFont

from docling_pp_ocrv6.rapidocr_fast import FastRapidOcrModel, FastRapidOcrOptions, _render_rgba


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


def _many_text_images(n: int) -> list[np.ndarray]:
    """Text images with lines of many widths, some upside down: rec batches of the same padded width in many images."""
    words = [
        "Rechnung",
        "Total CHF 1234.56",
        "Kunde IBAN Zürich 31452",
        "kurz",
        "Crème brûlée café",
        "Nr. 2026-114",
        "Eine deutlich längere Zeile mit vielen Wörtern darin",
        "Basel",
        "MwSt 8.1 %",
        "Seite 3 von 12",
    ]
    images = []
    for i in range(n):
        lines = [words[(i * 3 + k) % len(words)] for k in range(1 + i % 5)]
        images.append(_text_image(lines, flip=i % 7 == 3))
    return images


@pytest.mark.e2e
def test_driver_gives_rapidocrs_own_results():
    """The driver returns what RapidOCR returns per image, bit for bit (needs RapidOCR's torch models).

    Many images with lines of the same widths: on the GPU (TF32 convolutions) other batch sizes than
    RapidOCR's would change scores and text here.
    """
    torch = pytest.importorskip("torch")
    from rapidocr import EngineType, RapidOCR

    from docling_pp_ocrv6._rapidocr_driver import FastRapidOcr, supported

    params = {f"{part}.engine_type": EngineType.TORCH for part in ("Det", "Cls", "Rec")}
    if torch.cuda.is_available():
        params |= {"EngineConfig.torch.use_cuda": True, "EngineConfig.torch.cuda_ep_cfg.device_id": 0}
    reader = RapidOCR(params=params)
    assert supported(reader)
    images = [*_many_text_images(40), np.full((200, 300, 3), 255, dtype=np.uint8)]
    expected = [reader(im) for im in images]
    got = FastRapidOcr(reader)(images)
    for want, have in zip(expected, got, strict=True):
        assert have.txts == want.txts
        assert have.scores == want.scores
        if want.boxes is None:
            assert have.boxes is None
        else:
            np.testing.assert_array_equal(have.boxes, want.boxes)
    assert sum(len(r.txts or ()) for r in expected) > 80


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


def test_driver_gives_the_per_image_results():
    """All images in one call give what one image at a time gives; inputs normalised as on the CPU."""
    torch = pytest.importorskip("torch")
    from docling_pp_ocrv6._rapidocr_driver import FastRapidOcr

    rng = np.random.default_rng(7)
    images = [rng.integers(0, 256, (h, w, 3), dtype=np.uint8) for h, w in ((96, 400), (160, 230), (64, 900))]
    reader = _fake_reader(torch)
    one = many = FastRapidOcr(reader)

    np.testing.assert_array_equal(
        one._det_input(images[0])[0].permute(1, 2, 0).numpy(), _Pre().normalize(images[0]).astype(np.float32)
    )
    u8 = images[1][:48, :100][None]
    padded = one._rc_input(np.pad(u8, ((0, 0), (0, 0), (0, 20), (0, 0))), np.array([100]))
    want = (u8.astype(np.float32) / 255 - 0.5) / 0.5
    np.testing.assert_array_equal(padded[0, :, :, :100].permute(1, 2, 0).numpy(), want[0])
    assert not padded[0, :, :, 100:].any()

    want_all, got_all = [one([im])[0] for im in images], many(images)
    assert len(got_all) == len(images)
    for want_res, got_res in zip(want_all, got_all, strict=True):
        assert got_res.txts == want_res.txts
        assert got_res.scores == want_res.scores
        np.testing.assert_array_equal(got_res.boxes, want_res.boxes)
    assert all(r.txts for r in got_all)


def test_ocr_time_is_recorded_per_page(monkeypatch):
    from docling.datamodel.settings import settings

    model = _bare_model()
    monkeypatch.setattr(settings.debug, "profile_pipeline_timings", True)
    monkeypatch.setattr(model, "get_ocr_rects", lambda page: page.rects)
    monkeypatch.setattr(model, "post_process_cells", lambda *_args: None)
    model.reader.return_value = SimpleNamespace(boxes=None, txts=None, scores=None)
    conv_res = SimpleNamespace(timings={})
    rect = BoundingBox(l=10, t=20, r=110, b=60)
    list(model(conv_res, [_page([rect]) for _ in range(3)]))
    item = conv_res.timings["ocr"]
    assert item.count == 3
    assert len(item.times) == len(item.start_timestamps) == 3


def test_page_images_are_docling_parses_crops(tmp_path):
    """The single render cropped per rectangle gives the pixels of docling-parse's get_page_image(cropbox=...)."""
    from docling.backend.docling_parse_backend import ThreadedDoclingParseDocumentBackend
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.document import InputDocument

    image = Image.new("RGB", (612, 792), "white")
    draw = ImageDraw.Draw(image)
    for y in range(40, 760, 37):
        draw.text((50, y), f"Line {y} with some text", fill=(y % 256, 40, 200 - y % 200))
    image.paste(Image.effect_noise((120, 90), 60).convert("RGB"), (400, 300))
    image.save(tmp_path / "page.pdf", resolution=72)
    doc = InputDocument(
        path_or_stream=tmp_path / "page.pdf", format=InputFormat.PDF, backend=ThreadedDoclingParseDocumentBackend
    )
    backend = next(iter(doc._backend.iter_pages()))
    size = backend.get_size()
    rects = [
        BoundingBox(l=10.3, t=20.7, r=300.2, b=400.9, coord_origin=CoordOrigin.TOPLEFT),
        BoundingBox(l=0, t=0, r=size.width, b=size.height, coord_origin=CoordOrigin.TOPLEFT),
        BoundingBox(l=100.5, t=700.25, r=611.9, b=791.99, coord_origin=CoordOrigin.TOPLEFT),
        BoundingBox(l=10.0, t=20, r=10.1, b=50, coord_origin=CoordOrigin.TOPLEFT),  # an empty crop
        BoundingBox(l=-20, t=-5, r=50, b=900, coord_origin=CoordOrigin.TOPLEFT),
    ]
    model = _bare_model()
    page = SimpleNamespace(_backend=backend)
    assert _render_rgba(backend._result, model.scale) is not None  # the copy-free path runs here
    got = model._page_images(page, rects)
    for rect, array in zip(rects, got, strict=True):
        want = np.asarray(backend.get_page_image(scale=model.scale, cropbox=rect))
        assert array.shape == want.shape
        assert np.array_equal(array, want)
        (single,) = model._page_images(page, [rect])
        assert np.array_equal(single, want)
    off_page = BoundingBox(l=700, t=10, r=720, b=30, coord_origin=CoordOrigin.TOPLEFT)
    with pytest.raises(ValueError, match="right"):
        backend.get_page_image(scale=model.scale, cropbox=off_page)
    with pytest.raises(ValueError, match="right"):
        model._page_images(page, [off_page])


def test_page_images_fall_back_for_another_docling_parse(tmp_path, monkeypatch):
    from docling.backend.docling_parse_backend import ThreadedDoclingParseDocumentBackend
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.document import InputDocument

    from docling_pp_ocrv6 import rapidocr_fast

    Image.new("RGB", (300, 200), (200, 30, 90)).save(tmp_path / "page.pdf", resolution=72)
    doc = InputDocument(
        path_or_stream=tmp_path / "page.pdf", format=InputFormat.PDF, backend=ThreadedDoclingParseDocumentBackend
    )
    backend = next(iter(doc._backend.iter_pages()))
    monkeypatch.setattr(rapidocr_fast, "_DOCLING_PARSE_RENDER", {"get_image": "another docling-parse"})
    model = _bare_model()
    assert _render_rgba(backend._result, model.scale) is None
    rect = BoundingBox(l=10, t=20, r=150, b=120, coord_origin=CoordOrigin.TOPLEFT)
    (got,) = model._page_images(SimpleNamespace(_backend=backend), [rect])
    np.testing.assert_array_equal(got, np.asarray(backend.get_page_image(scale=model.scale, cropbox=rect)))


def _random_rects(rng: np.random.Generator, width: float, height: float) -> list[BoundingBox]:
    rects = []
    for _ in range(int(rng.choice([0, 1, 2, 3, 5, 12, 30]))):
        step = rng.choice([1.0, 0.5, 0.25, 0.0])
        x, y = rng.uniform(-40, width + 20), rng.uniform(-40, height + 20)
        w = abs(rng.uniform(0, rng.choice([5, 40, width + 1])))
        h = abs(rng.uniform(0, rng.choice([5, 40, height + 1])))
        if step:
            x, y, w, h = (round(v / step) * step for v in (x, y, w, h))
        rects.append(BoundingBox(l=x, t=y, r=x + w, b=y + h, coord_origin=CoordOrigin.TOPLEFT))
    return rects


def test_rect_deduplication_equals_doclings_raster():
    """The rectangle-based deduplication gives docling's boxes, order and coverage, case by case."""
    from docling.datamodel.base_models import Size
    from docling.models.base_ocr_model import BaseOcrModel

    model = _bare_model()
    rng = np.random.default_rng(11)
    for _ in range(3000):
        width, height = (
            float(rng.choice([rng.uniform(1, 60), rng.uniform(50, 300), rng.uniform(300, 900)])) for _ in "xy"
        )
        size = Size(width=width, height=height)
        rects = _random_rects(rng, width, height)
        dilation = int(rng.choice([0, 1, 2, 3, 7, 20]))
        want_cov, want = BaseOcrModel._deduplicate_rects(model, size, rects, dilation_size=dilation)
        got_cov, got = model._deduplicate_rects(size, iter(rects), dilation_size=dilation)
        assert [b.as_tuple() for b in got] == [b.as_tuple() for b in want]
        assert got_cov == want_cov
        assert type(got_cov) is type(want_cov)


def test_rect_deduplication_falls_back_to_docling(monkeypatch):
    from docling.datamodel.base_models import Size

    from docling_pp_ocrv6 import _ocr_rects

    rect = BoundingBox(l=10, t=10, r=20, b=40, coord_origin=CoordOrigin.TOPLEFT)
    assert _ocr_rects.deduplicate_rects(Size(width=100, height=100), [rect], 20) is not None  # this docling's code
    monkeypatch.setattr(_ocr_rects, "_DOCLING_DEDUPLICATE_RECTS", "another docling")
    assert _ocr_rects.deduplicate_rects(Size(width=100, height=100), [rect], 20) is None
    monkeypatch.undo()

    model = _bare_model()
    flipped = BoundingBox(l=30, t=10, r=20, b=40, coord_origin=CoordOrigin.TOPLEFT)
    with pytest.raises(ValueError, match="x1"):
        model._deduplicate_rects(Size(width=100, height=100), [flipped], dilation_size=20)


def _det_map(rng: np.random.Generator, h: int, w: int) -> np.ndarray:
    """A detector-like probability map: noisy text-line blobs, some rotated, some touching."""
    pred = rng.random((h, w), dtype=np.float32) * 0.15
    for _ in range(int(rng.integers(0, 25))):
        cx, cy = rng.uniform(-10, w + 10), rng.uniform(-10, h + 10)
        size = (float(rng.uniform(2, w / 2)), float(rng.uniform(1, 30)))
        rect = ((cx, cy), size, float(rng.choice([0.0, rng.uniform(-30, 30), 90.0])))
        pts = cv2.boxPoints(rect).astype(np.int32)
        cv2.fillPoly(pred, [pts], float(rng.uniform(0.4, 1.0)))
    return np.clip(pred + rng.normal(0, 0.03, pred.shape).astype(np.float32), 0, 1)[None, None]


@pytest.mark.parametrize(
    "config",
    [{}, {"use_dilation": False}, {"thresh": 0.2, "box_thresh": 0.3, "unclip_ratio": 2.0}],
)
def test_det_postprocess_equals_rapidocrs(config):
    """The leaner DB post-processing returns RapidOCR's boxes and scores, bit for bit."""
    from rapidocr.ch_ppocr_det.utils import DBPostProcess

    from docling_pp_ocrv6._det_post import postprocess

    op = DBPostProcess(**({"box_thresh": 0.5, "unclip_ratio": 1.6, "use_dilation": True} | config))
    rng = np.random.default_rng(5)
    found = 0
    for _ in range(80):
        h, w = (int(v) for v in rng.integers(1, 20, 2) * 32)
        pred = _det_map(rng, h, w)
        shape = (int(rng.integers(10, 2500)), int(rng.integers(10, 2500)))
        want_boxes, want_scores = op(pred, shape)
        got_boxes, got_scores = postprocess(op, pred, shape)
        assert got_scores == want_scores
        assert got_boxes.dtype == want_boxes.dtype
        np.testing.assert_array_equal(got_boxes, want_boxes)
        found += len(want_scores)
    assert found > 250


def test_det_postprocess_falls_back_for_the_slow_score_mode():
    from rapidocr.ch_ppocr_det.utils import DBPostProcess

    from docling_pp_ocrv6._det_post import postprocess, supported

    op = DBPostProcess(score_mode="slow")
    assert not supported(op)
    pred = _det_map(np.random.default_rng(1), 96, 160)
    want, got = op(pred, (300, 500)), postprocess(op, pred, (300, 500))
    assert got[1] == want[1]
    np.testing.assert_array_equal(got[0], want[0])


def test_pinned_staging_uploads_and_reuses_its_buffers():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    from docling_pp_ocrv6._rapidocr_driver import _PinnedStaging

    dev = torch.device("cuda")
    staging = _PinnedStaging(chunk=1 << 12)
    rng = np.random.default_rng(3)
    for _ in range(3):
        staging.reset()
        arrays = [
            rng.integers(0, 256, (5, 7, 3), dtype=np.uint8),
            rng.integers(-9, 9, 11, dtype=np.int64),
            rng.integers(0, 256, (70, 90, 3), dtype=np.uint8),  # larger than a chunk
            rng.integers(0, 256, (2, 3), dtype=np.uint8),
        ]
        uploaded = [staging.upload(a, dev) for a in arrays]
        for a, t in zip(arrays, uploaded, strict=True):
            assert t.device.type == "cuda"
            np.testing.assert_array_equal(t.cpu().numpy(), a)
    assert len(staging._bufs) <= 3


def test_cell_merge_equals_doclings():
    from docling.models.base_ocr_model import BaseOcrModel
    from docling_core.types.doc.page import BoundingRectangle, TextCell

    rng = np.random.default_rng(2)

    def cells(n: int) -> list[TextCell]:
        out = []
        for i in range(n):
            x, y = rng.uniform(0, 500, 2)
            box = BoundingBox(
                l=x, t=y, r=x + rng.uniform(1, 80), b=y + rng.uniform(1, 20), coord_origin=CoordOrigin.TOPLEFT
            )
            rect = BoundingRectangle.from_bounding_box(box)
            out.append(TextCell(index=i, text="t", orig="t", rect=rect, from_ocr=False))
        return out

    model = _bare_model()
    for n_first, n_second in [(0, 0), (0, 5), (7, 0), (3, 9), (12, 4), (40, 40)]:
        first, second = cells(n_first), cells(n_second)
        want = BaseOcrModel._merge_cells_by_priority(model, first, second)
        got = model._merge_cells_by_priority(first, second)
        assert [id(c) for c in got] == [id(c) for c in want]
        assert got is not first
        assert got is not second


def test_fx_inline_gives_the_eager_results():
    torch = pytest.importorskip("torch")
    from torch import nn

    from docling_pp_ocrv6._fx import inline, same_outputs

    class Block(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = nn.Conv2d(3, 8, 3, padding=1)
            self.bn = nn.BatchNorm2d(8)
            self.act = nn.Hardswish()
            self.head = nn.Linear(8, 5)

        def forward(self, x):
            x = self.act(self.bn(self.conv(x)))
            assert x.shape[2] >= 3, "too small"  # shape checks are allowed in a trace
            x = x.mean(dim=2).permute(0, 2, 1)  # (N, W, C): width-dependent shape
            return {"out": self.head(x), "w": [x.shape[1]]}

    class Branchy(Block):
        def forward(self, x):
            if x.sum() > 0:  # a real data-dependent branch: not traceable
                return x
            return -x

    torch.manual_seed(0)
    net = Block()
    with torch.no_grad():
        net.bn.running_mean.uniform_(-1, 1)
        net.bn.running_var.uniform_(0.5, 2)
    net.eval()
    fx_net = inline(net)
    assert fx_net is not net
    called = [fx_net.get_submodule(n.target) for n in fx_net.graph.nodes if n.op == "call_module"]
    assert not called  # every module inlined, batch norm included
    for shape in [(1, 3, 48, 40), (6, 3, 48, 123), (2, 3, 16, 7)]:
        x = torch.randn(shape)
        assert same_outputs(net(x), fx_net(x))
    branchy = Branchy().eval()
    assert inline(branchy) is branchy
    training = Block()
    assert inline(training) is training
