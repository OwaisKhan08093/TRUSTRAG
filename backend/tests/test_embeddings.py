"""Unit tests for the embedding model foundation."""

import pytest
from backend.app.embeddings.encoder import EmbeddingEncoder, EmbeddingModelError
from backend.app.config import DEFAULT_EMBEDDING_MODEL


def test_encoder_initializes_and_loads():
    """Test 1: EmbeddingEncoder can initialize and confirms successful loading."""
    encoder = EmbeddingEncoder()
    assert encoder.is_loaded is True


def test_default_model_name():
    """Test 2: Default model name matches configuration."""
    encoder = EmbeddingEncoder()
    assert encoder.model_name == DEFAULT_EMBEDDING_MODEL
    assert encoder.model_name == "sentence-transformers/all-MiniLM-L6-v2"


def test_embedding_dimension():
    """Test 3: Model exposes the expected embedding dimensionality (384 for all-MiniLM-L6-v2)."""
    encoder = EmbeddingEncoder()
    assert isinstance(encoder.embedding_dimension, int)
    assert encoder.embedding_dimension == 384


def test_invalid_model_name_raises_error():
    """Test 4: Invalid model identifier produces a clear EmbeddingModelError."""
    with pytest.raises(EmbeddingModelError) as exc_info:
        EmbeddingEncoder(model_name="non_existent_invalid_model_12345_xyz")
    assert "Failed to load embedding model" in str(exc_info.value)
