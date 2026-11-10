# docling-pp-ocrv6

A [Docling](https://github.com/docling-project/docling) OCR plugin for PaddlePaddle's **PP-OCRv6** models.

This plugin seamlessly integrates with Docling's standard pipeline to provide local OCR capabilities using PaddlePaddle's **PP-OCRv6** models. It runs the PP-OCRv6 detection and recognition **ONNX** checkpoints locally through [RapidOCR](https://github.com/RapidAI/RapidOCR) (onnxruntime), so OCR happens inside the docling worker — no external service required.

---

<p align="center">
  <a href="https://github.com/DCC-BS/docling-pp-ocrv6">GitHub</a>
  &nbsp;|&nbsp;
  <a href="https://pypi.org/project/docling-pp-ocrv6/">PyPI</a>
</p>

---

[![PyPI version](https://img.shields.io/pypi/v/docling-pp-ocrv6.svg)](https://pypi.org/project/docling-pp-ocrv6/)
[![Python versions](https://img.shields.io/pypi/pyversions/docling-pp-ocrv6.svg)](https://pypi.org/project/docling-pp-ocrv6/)
[![License](https://img.shields.io/github/license/DCC-BS/docling-pp-ocrv6)](https://github.com/DCC-BS/docling-pp-ocrv6/blob/main/LICENSE)
[![CI](https://github.com/DCC-BS/docling-pp-ocrv6/actions/workflows/main.yml/badge.svg)](https://github.com/DCC-BS/docling-pp-ocrv6/actions/workflows/main.yml)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Coverage](https://codecov.io/gh/DCC-BS/docling-pp-ocrv6/graph/badge.svg)](https://codecov.io/gh/DCC-BS/docling-pp-ocrv6)


GPU acceleration is automatic when the docling accelerator device resolves to CUDA and `onnxruntime-gpu` is installed.

## Installation

Pick exactly one onnxruntime extra — installing both the CPU and GPU wheels at
once is unsupported and prevents the CUDA provider from registering:

```bash
pip install "docling-pp-ocrv6[cpu]"     # CPU (onnxruntime)
pip install "docling-pp-ocrv6[gpu]"     # CUDA (onnxruntime-gpu)
```

The detection and recognition ONNX models are downloaded from HuggingFace on
first use and cached under docling's model cache. To pre-fetch them (e.g. at
container build time):

```python
from docling_pp_ocrv6 import PPOCRv6Model

PPOCRv6Model.download_models()
```

## Usage

```python
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling_pp_ocrv6 import PPOCRv6Options

pipeline_options = PdfPipelineOptions(do_ocr=True)
pipeline_options.ocr_options = PPOCRv6Options()

converter = DocumentConverter(format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)})
result = converter.convert("scanned.pdf")
print(result.document.export_to_markdown())
```

With **docling-serve**, request the engine by its `kind`:

```json
{ "options": { "ocr": true, "ocr_engine": "pp-ocrv6" } }
```

(`DOCLING_SERVE_ALLOW_EXTERNAL_PLUGINS=true` must be set for the plugin to load.)

## Configuration

All options are settable via `PPOCRv6Options(...)` or environment variables:

| Option | Env var | Default |
| --- | --- | --- |
| `lang` | `PPOCRV6_LANG` | `de,en,fr,it,es,nl,pt,...` (German-led European set) |
| `text_score` | `PPOCRV6_TEXT_SCORE` | `0.5` |
| `use_det` / `use_cls` / `use_rec` | `PPOCRV6_USE_DET` / `_CLS` / `_REC` | `true` |
| `det_repo` | `PPOCRV6_DET_REPO` | `PaddlePaddle/PP-OCRv6_medium_det_onnx` |
| `rec_repo` | `PPOCRV6_REC_REPO` | `PaddlePaddle/PP-OCRv6_medium_rec_onnx` |
| `det_model_path` / `rec_model_path` / `rec_keys_path` / `cls_model_path` | `PPOCRV6_*_MODEL_PATH` / `_KEYS_PATH` | auto |

The recognition character dictionary is extracted automatically from the
recognition model's `inference.yml`. Angle classification uses RapidOCR's
bundled cls model unless `cls_model_path` is set.

## Fast RapidOCR engine (`rapidocr-fast`)

The plugin also registers `rapidocr-fast`: docling's own RapidOCR engine with the same text and a
faster page loop. It takes every option of docling's `rapidocr` kind and changes only how a page
batch is run:

- each page is rendered once at the OCR scale and every OCR rectangle is cropped from it
  (docling renders the whole page once per rectangle);
- with RapidOCR's PyTorch engine (`backend: torch`), all rectangles of a page batch go through one
  pass: image normalisation, padding and the CTC argmax on the GPU, and the detection, angle and
  recognition forwards of all rectangles queued back to back before their results are read. The
  batches stay RapidOCR's own: on the GPU a convolution's result can depend on the batch size, so
  other batches would change the text. RapidOCR's own pre- and post-processing does the rest.
  Other backends, right-to-left languages, word boxes and rapidocr releases other than 3.10 call
  RapidOCR per image as docling does.

For docling-serve, as an OCR preset:

```bash
DOCLING_SERVE_CUSTOM_OCR_PRESETS='{"rapidocr": {"kind": "rapidocr-fast", "backend": "torch"}}'
```

Written against rapidocr 3.10 (checked at runtime) and docling-parse 7.x.

## Development

```bash
make install   # uv sync + pre-commit
make check     # ruff lint + format check + ty type check
make test      # pytest with coverage
```

## License

MIT
