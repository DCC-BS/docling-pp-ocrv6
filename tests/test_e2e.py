"""End-to-end tests running the real PP-OCRv6 ONNX models.

These download the detection and recognition models from HuggingFace (~85 MB)
and run real ONNX inference, so they are skipped unless ``PPOCRV6_E2E`` is set::

    PPOCRV6_E2E=1 pytest -m e2e
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

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
