"""Unit tests for FAISS vector index foundation (Milestone 1)."""

import numpy as np
import pytest

from backend.app.retrieval.faiss_index import FaissIndexError, FaissVectorIndex


def _generate_normalized_vectors(num_vectors: int, dimension: int = 384) -> np.ndarray:
    """Helper to generate synthetic L2-normalized vectors."""
    raw = np.random.randn(num_vectors, dimension).astype(np.float32)
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    return raw / norms


def test_faiss_index_initialization():
    """Verify FAISS index initializes properly with expected dimension and 0 vectors."""
    index = FaissVectorIndex(dimension=384)
    assert index.dimension == 384
    assert index.total_vectors == 0
    assert index.underlying_index is not None
    assert index.underlying_index.ntotal == 0


def test_faiss_index_invalid_dimension_rejected():
    """Verify non-positive dimension raises ValueError."""
    with pytest.raises(ValueError) as exc_info:
        FaissVectorIndex(dimension=0)
    assert "positive integer" in str(exc_info.value)

    with pytest.raises(ValueError):
        FaissVectorIndex(dimension=-5)


def test_faiss_add_vectors():
    """Verify valid normalized vectors can be added and increment total_vectors."""
    index = FaissVectorIndex(dimension=384)
    vectors = _generate_normalized_vectors(5, 384)

    total = index.add_embeddings(vectors)
    assert total == 5
    assert index.total_vectors == 5

    # Add 3 more vectors
    more_vectors = _generate_normalized_vectors(3, 384)
    total_after = index.add_embeddings(more_vectors)
    assert total_after == 8
    assert index.total_vectors == 8


def test_faiss_add_wrong_dimension_rejected():
    """Verify adding vectors with wrong dimension raises ValueError."""
    index = FaissVectorIndex(dimension=384)
    wrong_dim_vectors = _generate_normalized_vectors(4, 128)

    with pytest.raises(ValueError) as exc_info:
        index.add_embeddings(wrong_dim_vectors)
    assert "Expected embedding dimension of 384, got 128" in str(exc_info.value)
    assert index.total_vectors == 0


def test_faiss_add_nan_rejected():
    """Verify adding vectors containing NaN raises ValueError."""
    index = FaissVectorIndex(dimension=384)
    vectors = _generate_normalized_vectors(3, 384)
    vectors[1, 10] = np.nan

    with pytest.raises(ValueError) as exc_info:
        index.add_embeddings(vectors)
    assert "NaN values" in str(exc_info.value)
    assert index.total_vectors == 0


def test_faiss_add_infinity_rejected():
    """Verify adding vectors containing Infinity raises ValueError."""
    index = FaissVectorIndex(dimension=384)
    vectors = _generate_normalized_vectors(3, 384)
    vectors[0, 5] = np.inf

    with pytest.raises(ValueError) as exc_info:
        index.add_embeddings(vectors)
    assert "infinite values" in str(exc_info.value)
    assert index.total_vectors == 0


def test_faiss_add_zero_vector_rejected():
    """Verify adding zero vectors raises ValueError."""
    index = FaissVectorIndex(dimension=384)
    vectors = _generate_normalized_vectors(3, 384)
    vectors[2, :] = 0.0

    with pytest.raises(ValueError) as exc_info:
        index.add_embeddings(vectors)
    assert "zero vector" in str(exc_info.value)
    assert index.total_vectors == 0


def test_faiss_add_invalid_shape_rejected():
    """Verify 1-dimensional array or 3-dimensional array is rejected."""
    index = FaissVectorIndex(dimension=384)
    one_dim = np.ones((384,), dtype=np.float32)

    with pytest.raises(ValueError) as exc_info:
        index.add_embeddings(one_dim)
    assert "Expected 2-dimensional" in str(exc_info.value)

    three_dim = np.ones((2, 2, 384), dtype=np.float32)
    with pytest.raises(ValueError):
        index.add_embeddings(three_dim)


# =========================================================================
# Milestone 2: FAISS Persistence Tests
# =========================================================================

def test_faiss_save_and_load(tmp_path):
    """Milestone 2 Test 1: Create index -> add vectors -> save -> load -> check equality."""
    index = FaissVectorIndex(dimension=384)
    vectors = _generate_normalized_vectors(7, 384)
    index.add_embeddings(vectors)

    save_path = tmp_path / "index.faiss"
    index.save(save_path)
    assert save_path.exists()

    loaded_index = FaissVectorIndex.load(save_path)
    assert loaded_index.dimension == 384
    assert loaded_index.total_vectors == 7
    assert loaded_index.underlying_index.ntotal == 7


def test_faiss_load_non_existent_file_raises():
    """Milestone 2 Test 2: Loading from a non-existent path raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError) as exc_info:
        FaissVectorIndex.load("non_existent_path_to_index.faiss")
    assert "not found" in str(exc_info.value)


# =========================================================================
# Milestone 3: FAISS Index Builder Tests
# =========================================================================

def test_build_faiss_index_script(tmp_path):
    """Milestone 3 Test 1: build_index loads embeddings, validates, and creates valid .faiss file."""
    from scripts.build_faiss_index import build_index

    emb_file = tmp_path / "embeddings.npy"
    index_file = tmp_path / "index.faiss"

    vectors = _generate_normalized_vectors(4, 384)
    np.save(str(emb_file), vectors)

    exit_code = build_index(
        embeddings_file=emb_file,
        index_file=index_file,
        expected_dim=384,
    )
    assert exit_code == 0
    assert index_file.exists()

    loaded = FaissVectorIndex.load(index_file)
    assert loaded.dimension == 384
    assert loaded.total_vectors == 4


def test_build_faiss_index_missing_embeddings_fails(tmp_path):
    """Milestone 3 Test 2: build_index exits with non-zero when embeddings are missing."""
    from scripts.build_faiss_index import build_index

    non_existent = tmp_path / "missing_embeddings.npy"
    index_file = tmp_path / "index.faiss"

    exit_code = build_index(
        embeddings_file=non_existent,
        index_file=index_file,
    )
    assert exit_code == 1
    assert not index_file.exists()
