"""Unit tests for CrossEncoderReranker model wrapper."""

from unittest.mock import patch
import pytest

from backend.app.config import DEFAULT_RERANKER_MODEL
from backend.app.reranking.cross_encoder import (
    CrossEncoderModelError,
    CrossEncoderReranker,
)


def test_cross_encoder_init_default():
    """Verify CrossEncoderReranker loads default model and exposes properties."""
    reranker = CrossEncoderReranker()
    assert reranker.model_name == DEFAULT_RERANKER_MODEL
    assert reranker.is_loaded is True
    assert reranker.underlying_model is not None


def test_cross_encoder_init_invalid_name():
    """Verify CrossEncoderReranker rejects empty or invalid model name."""
    with pytest.raises(ValueError, match="model_name must be a non-empty string"):
        CrossEncoderReranker(model_name="")

    with pytest.raises(ValueError, match="model_name must be a non-empty string"):
        CrossEncoderReranker(model_name="   ")


def test_cross_encoder_load_failure():
    """Verify CrossEncoderModelError is raised when model loading fails."""
    with patch("backend.app.reranking.cross_encoder.CrossEncoder", side_effect=RuntimeError("Model not found")):
        with pytest.raises(CrossEncoderModelError, match="Failed to load CrossEncoder model"):
            CrossEncoderReranker(model_name="nonexistent/fake-model")
