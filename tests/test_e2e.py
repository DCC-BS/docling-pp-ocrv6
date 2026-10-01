"""End-to-end tests running the real PP-OCRv6 ONNX models.

These download the detection and recognition models from HuggingFace (~85 MB)
and run real ONNX inference, so they are skipped unless ``PPOCRV6_E2E`` is set::

    PPOCRV6_E2E=1 pytest -m e2e
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

from docling_pp_ocrv6.model import PPOCRv6Model
from docling_pp_ocrv6.options import PPOCRv6Options

pytestmark = pytest.mark.e2e


@pytest.fixture(scope="module")
def e2e_model() -> PPOCRv6Model:
    """Build a PPOCRv6Model with the real RapidOCR reader and downloaded models."""
    from docling.datamodel.accelerator_options import AcceleratorOptions

    return PPOCRv6Model(
        enabled=True,
        artifacts_path=None,
        options=PPOCRv6Options(),
        accelerator_options=AcceleratorOptions(),
    )


def _font(size: int) -> ImageFont.ImageFont:
    """Return a legible scalable font, falling back to the bundled default."""
    for name in ("DejaVuSans.ttf", "Arial.ttf", "LiberationSans-Regular.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def _make_text_image(text: str, size: tuple[int, int] = (700, 140)) -> Image.Image:
    """Render *text* in large black type on a white image for OCR."""
    img = Image.new("RGB", size, color="white")
    draw = ImageDraw.Draw(img)
    draw.text((20, 40), text, fill="black", font=_font(56))
    return img


def _make_ocr_rect(right: int, bottom: int):
    rect = MagicMock()
    rect.area.return_value = right * bottom
    rect.l, rect.t, rect.r, rect.b = 0, 0, right, bottom
    return rect


def _run_call(model: PPOCRv6Model, image: Image.Image):
    """Drive the full ``__call__`` path with a mocked page backend; return cells."""
    page = MagicMock()
    page._backend.is_valid.return_value = True
    page._backend.get_page_image.return_value = image
    rect = _make_ocr_rect(image.width, image.height)

    with (
        patch.object(model, "get_ocr_rects", return_value=[rect]),
        patch.object(model, "post_process_cells") as mock_post,
        patch("docling_pp_ocrv6.model.TimeRecorder"),
    ):
        pages = list(model(MagicMock(), [page]))

    assert len(pages) == 1
    mock_post.assert_called_once()
    return mock_post.call_args[0][0]


def _joined(cells) -> str:
    return " ".join(c.text for c in cells).lower()


class TestRealOcr:
    def test_recognises_german_word(self, e2e_model):
        cells = _run_call(e2e_model, _make_text_image("Rechnung"))
        assert len(cells) >= 1
        assert all(c.from_ocr for c in cells)
        assert "rechnung" in _joined(cells)

    def test_recognises_european_phrase(self, e2e_model):
        cells = _run_call(e2e_model, _make_text_image("Crème brûlée café"))
        text = _joined(cells)
        # Accents may degrade; require the recognisable Latin stems.
        assert "caf" in text
        assert "br" in text

    def test_recognises_digits(self, e2e_model):
        cells = _run_call(e2e_model, _make_text_image("Total 1234.56"))
        digits = "".join(ch for ch in _joined(cells) if ch.isdigit())
        assert "123456" in digits

    def test_blank_image_yields_no_cells(self, e2e_model):
        cells = _run_call(e2e_model, Image.new("RGB", (300, 120), color="white"))
        assert cells == []

    @pytest.mark.parametrize("return_word_box", [True, False])
    def test_a_region_without_text_does_not_fail(self, e2e_model, monkeypatch, return_word_box):
        # RapidOCR answers a placeholder word here, asked for words or not
        monkeypatch.setattr(e2e_model.options, "return_word_box", return_word_box)
        # a colour gradient: a picture, with nothing a letter could be read in
        ramp = np.linspace(40, 220, 300, dtype=np.uint8)
        picture = Image.fromarray(
            np.stack(
                [np.tile(ramp, (300, 1)), np.tile(ramp[:, None], (1, 300)), np.full((300, 300), 120, np.uint8)], axis=-1
            )
        )
        assert _run_call(e2e_model, picture) == []

    def test_reads_each_word_with_its_box(self, e2e_model):
        from docling_pp_ocrv6.model import _word_cells

        image = _make_text_image("Total 1234.56")
        result = e2e_model.reader(np.array(image), return_word_box=True)
        words = _word_cells(result.word_results, _make_ocr_rect(image.width, image.height), scale=1)
        assert [w.text for w in words] == ["Total", "1234.56"]
        first, second = (w.rect.to_bounding_box() for w in words)
        assert first.r <= second.l
