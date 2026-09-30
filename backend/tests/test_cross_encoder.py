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


def test_cross_encoder_score_pairs():
    """Verify score_pairs calculates relevance scores and preserves order."""
    reranker = CrossEncoderReranker()
    query = "data protection board"
    docs = [
        "The Data Protection Board of India is established under section 18.",
        "A recipe for baking chocolate cake with cocoa and sugar.",
    ]
    scores = reranker.score_pairs(query, docs)

    assert isinstance(scores, list)
    assert len(scores) == 2
    assert all(isinstance(s, float) for s in scores)
    # The relevant document must score higher than chocolate cake
    assert scores[0] > scores[1]


def test_cross_encoder_score_pairs_empty():
    """Verify score_pairs handles empty documents list safely."""
    reranker = CrossEncoderReranker()
    assert reranker.score_pairs("valid query", []) == []


def test_cross_encoder_score_pairs_validation():
    """Verify input validation on score_pairs arguments."""
    reranker = CrossEncoderReranker()

    with pytest.raises(TypeError, match="query must be a string"):
        reranker.score_pairs(123, ["doc"])  # type: ignore

    with pytest.raises(ValueError, match="query cannot be empty"):
        reranker.score_pairs("", ["doc"])

    with pytest.raises(ValueError, match="query cannot be empty"):
        reranker.score_pairs("   ", ["doc"])

    with pytest.raises(TypeError, match="documents must be a sequence"):
        reranker.score_pairs("query", "not a list")  # type: ignore

    with pytest.raises(TypeError, match="must be a string"):
        reranker.score_pairs("query", [123])  # type: ignore

    with pytest.raises(ValueError, match="batch_size must be a positive integer"):
        reranker.score_pairs("query", ["doc"], batch_size=0)
