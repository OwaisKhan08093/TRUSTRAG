"""Unit tests for the embedding model foundation and document encoding."""

import json
from pathlib import Path
import numpy as np
import pytest

from backend.app.config import DEFAULT_EMBEDDING_MODEL
from backend.app.embeddings.encoder import EmbeddingEncoder, EmbeddingModelError
from backend.app.embeddings.validator import (
    DIFFERENT_PAIR,
    SIMILAR_PAIR,
    cosine_similarity,
    run_semantic_sanity_check,
    validate_embeddings,
)
from scripts.embed_chunks import generate_embeddings
from scripts.validate_embeddings import validate_saved_embeddings


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


# =========================================================================
# Step 2.3: Embedding Validation Suite Tests
# =========================================================================

def test_valid_embedding_matrix_passes(encoder: EmbeddingEncoder):
    """Step 2.3 Test 1: Valid 2D normalized embedding matrix passes all validation checks."""
    texts = ["Valid document chunk 1", "Valid document chunk 2"]
    embeddings = encoder.encode_documents(texts, normalize_embeddings=True)

    stats = validate_embeddings(embeddings, expected_dim=384, check_normalized=True)
    assert stats["num_vectors"] == 2
    assert stats["dimension"] == 384
    assert stats["nan_count"] == 0
    assert stats["inf_count"] == 0
    assert stats["zero_vector_count"] == 0
    assert stats["is_normalized"] is True
    assert np.isclose(stats["mean_norm"], 1.0, atol=1e-4)


def test_wrong_dimension_fails():
    """Step 2.3 Test 2: Embedding matrix with unexpected dimension raises ValueError."""
    invalid_dim_matrix = np.ones((4, 128), dtype=np.float32)
    with pytest.raises(ValueError) as exc_info:
        validate_embeddings(invalid_dim_matrix, expected_dim=384)
    assert "Expected embedding dimension of 384, got 128" in str(exc_info.value)


def test_one_dimensional_input_fails():
    """Step 2.3 Test 3: 1-dimensional array raises ValueError instead of silently broadcasting."""
    one_dim_vector = np.ones((384,), dtype=np.float32)
    with pytest.raises(ValueError) as exc_info:
        validate_embeddings(one_dim_vector)
    assert "Expected 2-dimensional embedding array" in str(exc_info.value)


def test_nan_values_fail():
    """Step 2.3 Test 4: Embedding matrix containing NaN values is rejected."""
    matrix_with_nan = np.ones((3, 384), dtype=np.float32)
    matrix_with_nan[1, 50] = np.nan
    with pytest.raises(ValueError) as exc_info:
        validate_embeddings(matrix_with_nan)
    assert "NaN values" in str(exc_info.value)


def test_infinity_values_fail():
    """Step 2.3 Test 5: Embedding matrix containing Inf or -Inf values is rejected."""
    matrix_with_inf = np.ones((3, 384), dtype=np.float32)
    matrix_with_inf[0, 10] = np.inf
    with pytest.raises(ValueError) as exc_info:
        validate_embeddings(matrix_with_inf)
    assert "infinite values" in str(exc_info.value)

    matrix_with_neginf = np.ones((3, 384), dtype=np.float32)
    matrix_with_neginf[2, 20] = -np.inf
    with pytest.raises(ValueError) as exc_info:
        validate_embeddings(matrix_with_neginf)
    assert "infinite values" in str(exc_info.value)


def test_zero_vector_fails():
    """Step 2.3 Test 6: Embedding matrix containing zero vectors is rejected."""
    matrix_with_zero = np.ones((3, 384), dtype=np.float32)
    matrix_with_zero[1, :] = 0.0
    with pytest.raises(ValueError) as exc_info:
        validate_embeddings(matrix_with_zero)
    assert "zero vector" in str(exc_info.value)


def test_non_normalized_vectors_detected_when_expected():
    """Step 2.3 Test 7: Non-normalized vectors fail when normalization is expected, pass when optional."""
    unnormalized_matrix = np.ones((3, 384), dtype=np.float32) * 2.5
    with pytest.raises(ValueError) as exc_info:
        validate_embeddings(unnormalized_matrix, check_normalized=True)
    assert "not unit L2 normalized" in str(exc_info.value)

    stats = validate_embeddings(unnormalized_matrix, check_normalized=False)
    assert stats["is_normalized"] is False
    assert stats["num_vectors"] == 3


def test_cosine_similarity_works(encoder: EmbeddingEncoder):
    """Step 2.3 Test 8: Cosine similarity accurately compares identical, distinct, and opposite vectors."""
    v1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    v2 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    v_ortho = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    v_opp = np.array([-1.0, 0.0, 0.0], dtype=np.float32)

    assert np.isclose(cosine_similarity(v1, v2), 1.0)
    assert np.isclose(cosine_similarity(v1, v_ortho), 0.0)
    assert np.isclose(cosine_similarity(v1, v_opp), -1.0)

    # Verify sanity check behavior with real embeddings
    sanity_res = run_semantic_sanity_check(encoder)
    assert sanity_res["passed"] is True
    assert sanity_res["similar_score"] > sanity_res["different_score"]


def test_cosine_similarity_dimension_mismatch_and_invalid_rejected():
    """Step 2.3 Test 9: Cosine similarity rejects dimension mismatches, NaNs, and zero vectors."""
    v_3d = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    v_2d = np.array([1.0, 0.0], dtype=np.float32)

    with pytest.raises(ValueError) as exc_info:
        cosine_similarity(v_3d, v_2d)
    assert "Dimension mismatch" in str(exc_info.value)

    v_zero = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    with pytest.raises(ValueError) as exc_info:
        cosine_similarity(v_3d, v_zero)
    assert "zero vector" in str(exc_info.value)

    v_nan = np.array([np.nan, 0.0, 0.0], dtype=np.float32)
    with pytest.raises(ValueError) as exc_info:
        cosine_similarity(v_3d, v_nan)
    assert "finite numbers" in str(exc_info.value)


def test_metadata_count_mismatch_detected(tmp_path: Path):
    """Step 2.3 Test 10: Metadata count mismatch with embedding count is detected and flagged."""
    embeddings_file = tmp_path / "embeddings.npy"
    metadata_file = tmp_path / "embedding_metadata.json"

    # Create 3 valid normalized embeddings
    raw_vecs = np.random.randn(3, 384).astype(np.float32)
    norms = np.linalg.norm(raw_vecs, axis=1, keepdims=True)
    normed_vecs = raw_vecs / norms
    np.save(str(embeddings_file), normed_vecs)

    # Create metadata with only 2 items (mismatch)
    mismatched_metadata = [
        {"embedding_index": 0, "chunk_id": "chunk_0"},
        {"embedding_index": 1, "chunk_id": "chunk_1"},
    ]
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(mismatched_metadata, f)

    # Validation should fail (exit code 1)
    status_mismatch = validate_saved_embeddings(
        embeddings_file=embeddings_file,
        metadata_file=metadata_file,
        run_sanity_check=False,
    )
    assert status_mismatch == 1

    # Fix metadata so counts match (3 items)
    matched_metadata = mismatched_metadata + [{"embedding_index": 2, "chunk_id": "chunk_2"}]
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(matched_metadata, f)

    status_match = validate_saved_embeddings(
        embeddings_file=embeddings_file,
        metadata_file=metadata_file,
        run_sanity_check=False,
    )
    assert status_match == 0


# =========================================================================
# Milestone 4: Query Embedding Tests
# =========================================================================

def test_encode_query_valid(encoder: EmbeddingEncoder):
    """Milestone 4 Test 1: encode_query returns a 1D (384,) array with unit norm."""
    query = "What is the penalty for data breach?"
    vector = encoder.encode_query(query, normalize_embeddings=True)

    assert isinstance(vector, np.ndarray)
    assert vector.ndim == 1
    assert vector.shape == (384,)
    assert vector.dtype == np.float32
    assert np.isfinite(vector).all()
    norm = np.linalg.norm(vector)
    assert np.isclose(norm, 1.0, atol=1e-5)


def test_encode_query_empty_or_whitespace_raises(encoder: EmbeddingEncoder):
    """Milestone 4 Test 2: encode_query raises ValueError for empty or whitespace query."""
    with pytest.raises(ValueError) as exc_info:
        encoder.encode_query("")
    assert "cannot be empty" in str(exc_info.value)

    with pytest.raises(ValueError) as exc_info:
        encoder.encode_query("   \n\t  ")
    assert "cannot be empty" in str(exc_info.value)


def test_encode_query_non_string_raises(encoder: EmbeddingEncoder):
    """Milestone 4 Test 3: encode_query raises TypeError for non-string input."""
    with pytest.raises(TypeError) as exc_info:
        encoder.encode_query(12345)  # type: ignore
    assert "Query must be a string" in str(exc_info.value)

    with pytest.raises(TypeError):
        encoder.encode_query(None)  # type: ignore


def test_encode_query_deterministic(encoder: EmbeddingEncoder):
    """Milestone 4 Test 4: encode_query produces deterministic embeddings for identical queries."""
    q = "What are the duties of a Data Protection Officer?"
    v1 = encoder.encode_query(q)
    v2 = encoder.encode_query(q)
    assert np.allclose(v1, v2, atol=1e-6)


def test_encode_query_matches_encode_documents(encoder: EmbeddingEncoder):
    """Milestone 4 Test 5: encode_query matches the output of encode_documents for the same query."""
    q = "General obligations of data fiduciaries."
    v_query = encoder.encode_query(q, normalize_embeddings=True)
    v_docs = encoder.encode_documents([q], normalize_embeddings=True)[0]
    assert np.allclose(v_query, v_docs, atol=1e-5)

