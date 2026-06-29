"""Tests for PPOCRv6Options."""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from docling_pp_ocrv6.options import PPOCRv6Options


class TestKind:
    def test_kind_value(self):
        assert PPOCRv6Options.kind == "pp-ocrv6"

    def test_kind_is_class_var(self):
        opts = PPOCRv6Options()
        assert "kind" not in opts.model_fields


class TestDefaults:
    def test_default_lang(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("PPOCRV6_LANG", None)
            lang = PPOCRv6Options().lang
            # German-led common European languages.
            assert lang[0] == "de"
            for code in ("en", "fr", "it", "es", "nl", "pt"):
                assert code in lang

    def test_default_text_score(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("PPOCRV6_TEXT_SCORE", None)
            assert PPOCRv6Options().text_score == 0.5

    def test_default_stages_enabled(self):
        with patch.dict(os.environ, {}, clear=False):
            for var in ("PPOCRV6_USE_DET", "PPOCRV6_USE_CLS", "PPOCRV6_USE_REC"):
                os.environ.pop(var, None)
            opts = PPOCRv6Options()
            assert opts.use_det is True
            assert opts.use_cls is True
            assert opts.use_rec is True

    def test_default_repos(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("PPOCRV6_DET_REPO", None)
            os.environ.pop("PPOCRV6_REC_REPO", None)
            opts = PPOCRv6Options()
            assert "det" in opts.det_repo
            assert "rec" in opts.rec_repo

    def test_default_paths_none(self):
        with patch.dict(os.environ, {}, clear=False):
            for var in (
                "PPOCRV6_DET_MODEL_PATH",
                "PPOCRV6_REC_MODEL_PATH",
                "PPOCRV6_REC_KEYS_PATH",
                "PPOCRV6_CLS_MODEL_PATH",
            ):
                os.environ.pop(var, None)
            opts = PPOCRv6Options()
            assert opts.det_model_path is None
            assert opts.rec_model_path is None
            assert opts.rec_keys_path is None
            assert opts.cls_model_path is None


class TestEnvVars:
    def test_env_lang_multiple(self):
        with patch.dict(os.environ, {"PPOCRV6_LANG": "en,de,fr"}):
            assert PPOCRv6Options().lang == ["en", "de", "fr"]

    def test_env_text_score(self):
        with patch.dict(os.environ, {"PPOCRV6_TEXT_SCORE": "0.8"}):
            assert PPOCRv6Options().text_score == 0.8

    def test_env_use_cls_false(self):
        with patch.dict(os.environ, {"PPOCRV6_USE_CLS": "false"}):
            assert PPOCRv6Options().use_cls is False

    def test_env_det_repo(self):
        with patch.dict(os.environ, {"PPOCRV6_DET_REPO": "my-org/det"}):
            assert PPOCRv6Options().det_repo == "my-org/det"

    def test_env_det_model_path(self):
        with patch.dict(os.environ, {"PPOCRV6_DET_MODEL_PATH": "/models/det.onnx"}):
            assert PPOCRv6Options().det_model_path == "/models/det.onnx"

    def test_explicit_arg_overrides_env(self):
        with patch.dict(os.environ, {"PPOCRV6_TEXT_SCORE": "0.9"}):
            assert PPOCRv6Options(text_score=0.3).text_score == 0.3


class TestValidation:
    def test_extra_fields_forbidden(self):
        with pytest.raises(ValidationError):
            PPOCRv6Options(unknown_field="value")


class TestInheritance:
    def test_is_ocr_options_subclass(self):
        from docling.datamodel.pipeline_options import OcrOptions

        assert issubclass(PPOCRv6Options, OcrOptions)
