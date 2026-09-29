"""Unit tests for the embedding model foundation and document encoding."""

import json
from pathlib import Path
import numpy as np
import pytest

from backend.app.config import DEFAULT_EMBEDDING_MODEL
from backend.app.embeddings.encoder import EmbeddingEncoder, EmbeddingModelError
from scripts.embed_chunks import generate_embeddings


@pytest.fixture(scope="module")
def encoder() -> EmbeddingEncoder:
    """Shared EmbeddingEncoder fixture for fast test execution."""
    return EmbeddingEncoder()


def test_encoder_initializes_and_loads(encoder: EmbeddingEncoder):
    """Foundation Test 1: EmbeddingEncoder can initialize and confirms successful loading."""
    assert encoder.is_loaded is True


def test_default_model_name(encoder: EmbeddingEncoder):
    """Foundation Test 2: Default model name matches configuration."""
    assert encoder.model_name == DEFAULT_EMBEDDING_MODEL
    assert encoder.model_name == "sentence-transformers/all-MiniLM-L6-v2"


def test_embedding_dimension(encoder: EmbeddingEncoder):
    """Foundation Test 3: Model exposes the expected embedding dimensionality (384 for all-MiniLM-L6-v2)."""
    assert isinstance(encoder.embedding_dimension, int)
    assert encoder.embedding_dimension == 384


def test_invalid_model_name_raises_error():
    """Foundation Test 4: Invalid model identifier produces a clear EmbeddingModelError."""
    with pytest.raises(EmbeddingModelError) as exc_info:
        EmbeddingEncoder(model_name="non_existent_invalid_model_12345_xyz")
    assert "Failed to load embedding model" in str(exc_info.value)


def test_multiple_texts_produce_multiple_embeddings(encoder: EmbeddingEncoder):
    """Step 2.2 Test 1: Multiple input texts produce multiple embeddings."""
    texts = [
        "First document chunk regarding data protection.",
        "Second document chunk regarding data fiduciaries.",
        "Third document chunk regarding penalty provisions.",
    ]
    embeddings = encoder.encode_documents(texts)
    assert isinstance(embeddings, np.ndarray)
    assert len(embeddings) == len(texts)


def test_output_shape(encoder: EmbeddingEncoder):
    """Step 2.2 Test 2: Output shape matches (number_of_texts, 384)."""
    texts = ["Sample clause A", "Sample clause B", "Sample clause C", "Sample clause D"]
    embeddings = encoder.encode_documents(texts)
    assert embeddings.shape == (4, 384)


def test_input_ordering_preserved(encoder: EmbeddingEncoder):
    """Step 2.2 Test 3: Input ordering is strictly preserved."""
    t1 = "Unique phrase alpha for testing ordering."
    t2 = "Unique phrase beta for testing ordering."
    
    batch_embeddings = encoder.encode_documents([t1, t2])
    indiv_emb1 = encoder.encode_documents([t1])
    indiv_emb2 = encoder.encode_documents([t2])

    assert np.allclose(batch_embeddings[0], indiv_emb1[0], atol=1e-5)
    assert np.allclose(batch_embeddings[1], indiv_emb2[0], atol=1e-5)


def test_empty_input_handled(encoder: EmbeddingEncoder):
    """Step 2.2 Test 4: Empty input sequence produces an empty 2D array of shape (0, 384)."""
    embeddings = encoder.encode_documents([])
    assert isinstance(embeddings, np.ndarray)
    assert embeddings.shape == (0, 384)


def test_embeddings_contain_finite_numeric_values(encoder: EmbeddingEncoder):
    """Step 2.2 Test 5: All embedding values are finite real numbers (no NaN or Inf)."""
    texts = ["Legal text chunk 1", "Legal text chunk 2"]
    embeddings = encoder.encode_documents(texts)
    assert np.isfinite(embeddings).all()


def test_normalized_embeddings_have_unit_norm(encoder: EmbeddingEncoder):
    """Step 2.2 Test 6: L2 normalized embeddings have approximately unit norm (||v|| ≈ 1.0)."""
    texts = [
        "An Act to provide for digital personal data protection.",
        "Obligations of Data Fiduciary under the Act.",
    ]
    normalized_embeddings = encoder.encode_documents(texts, normalize_embeddings=True)
    norms = np.linalg.norm(normalized_embeddings, axis=1)
    assert np.isclose(norms, 1.0, atol=1e-5).all()


def test_saved_metadata_mapping_matches_embeddings(tmp_path: Path):
    """Step 2.2 Test 7: The saved metadata mapping has exactly the same number of entries as embeddings."""
    chunks_file = tmp_path / "chunks.json"
    embeddings_file = tmp_path / "embeddings.npy"
    metadata_file = tmp_path / "embedding_metadata.json"

    sample_chunks = [
        {"chunk_id": "doc1_page1_chunk1", "text": "Sample text chunk 1"},
        {"chunk_id": "doc1_page2_chunk2", "text": "Sample text chunk 2"},
        {"chunk_id": "doc1_page3_chunk3", "text": "Sample text chunk 3"},
    ]

    with open(chunks_file, "w", encoding="utf-8") as f:
        json.dump(sample_chunks, f)

    exit_code = generate_embeddings(
        chunks_file=chunks_file,
        embeddings_file=embeddings_file,
        metadata_file=metadata_file,
    )
    assert exit_code == 0
    assert embeddings_file.exists()
    assert metadata_file.exists()

    loaded_embeddings = np.load(str(embeddings_file))
    with open(metadata_file, "r", encoding="utf-8") as f:
        loaded_metadata = json.load(f)

    assert len(loaded_metadata) == len(loaded_embeddings) == len(sample_chunks)
    assert loaded_metadata[0]["chunk_id"] == "doc1_page1_chunk1"
    assert loaded_metadata[0]["embedding_index"] == 0
    assert loaded_metadata[2]["chunk_id"] == "doc1_page3_chunk3"
    assert loaded_metadata[2]["embedding_index"] == 2
