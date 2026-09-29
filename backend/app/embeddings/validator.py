"""Embedding validation utilities and mathematical metrics for TrustRAG."""

from typing import Any, Dict, Optional, Sequence, Tuple
import numpy as np

from backend.app.config import DEFAULT_EMBEDDING_MODEL
from backend.app.embeddings.encoder import EmbeddingEncoder


SIMILAR_PAIR: Tuple[str, str] = (
    "What is personal data?",
    "What information is considered personal data?",
)

DIFFERENT_PAIR: Tuple[str, str] = (
    "What is personal data?",
    "What are the duties of a Data Fiduciary?",
)


class EmbeddingValidationError(ValueError):
    """Raised when an embedding array or vector fails validation checks."""
    pass


def validate_embeddings(
    embeddings: np.ndarray,
    expected_dim: int = 384,
    check_normalized: bool = True,
    tolerance: float = 1e-4,
) -> Dict[str, Any]:
    """Validate that an embedding matrix satisfies all structural and mathematical invariants.

    Checks performed:
    - Must be a NumPy array.
    - Must be 2-dimensional (rows, dimensions).
    - Must have at least 1 row (non-empty).
    - Dimension must match expected_dim (384 by default).
    - Dtype must be a floating-point numeric type.
    - All values must be finite (no NaN, no Inf).
    - No vector may have zero norm (zero vectors are rejected).
    - When check_normalized is True, all vectors must have approximately unit L2 norm (||v|| ≈ 1.0).

    Args:
        embeddings: NumPy 2D array of vector embeddings.
        expected_dim: Expected dimensionality per vector (default: 384).
        check_normalized: Whether to require unit L2 normalization (default: True).
        tolerance: Absolute tolerance for unit norm check (default: 1e-4).

    Returns:
        A dictionary containing validation statistics and metrics.

    Raises:
        TypeError: If input is not a numpy array or has invalid dtype.
        ValueError: If shape, dimension, finite values, zero vector, or normalization checks fail.
    """
    if not isinstance(embeddings, np.ndarray):
        raise TypeError(f"Embeddings must be a numpy.ndarray, got {type(embeddings).__name__}.")

    if not np.issubdtype(embeddings.dtype, np.floating):
        raise TypeError(f"Embeddings must have a floating-point dtype, got {embeddings.dtype}.")

    if embeddings.ndim != 2:
        raise ValueError(
            f"Expected 2-dimensional embedding array, got {embeddings.ndim}-dimensional array with shape {embeddings.shape}."
        )

    num_vectors, dim = embeddings.shape

    if num_vectors == 0:
        raise ValueError("Embedding array must contain at least one vector (rows > 0).")

    if dim != expected_dim:
        raise ValueError(
            f"Expected embedding dimension of {expected_dim}, got {dim}."
        )

    # Check for NaN values
    nan_count = int(np.isnan(embeddings).sum())
    if nan_count > 0:
        raise ValueError(f"Embedding array contains {nan_count} NaN values.")

    # Check for Infinite values
    inf_count = int(np.isinf(embeddings).sum())
    if inf_count > 0:
        raise ValueError(f"Embedding array contains {inf_count} infinite values.")

    # Calculate L2 norms across all vectors
    norms = np.linalg.norm(embeddings, axis=1)

    # Check for zero vectors
    zero_vector_count = int(np.sum(np.isclose(norms, 0.0, atol=1e-7)))
    if zero_vector_count > 0:
        raise ValueError(f"Embedding array contains {zero_vector_count} zero vector(s).")

    # Check normalization if required
    is_normalized = bool(np.allclose(norms, 1.0, atol=tolerance))
    if check_normalized and not is_normalized:
        raise ValueError(
            f"Embeddings are not unit L2 normalized within tolerance {tolerance}. "
            f"Min norm: {float(np.min(norms)):.6f}, Max norm: {float(np.max(norms)):.6f}, Mean norm: {float(np.mean(norms)):.6f}."
        )

    return {
        "num_vectors": num_vectors,
        "dimension": dim,
        "dtype": str(embeddings.dtype),
        "min_norm": float(np.min(norms)),
        "max_norm": float(np.max(norms)),
        "mean_norm": float(np.mean(norms)),
        "nan_count": nan_count,
        "inf_count": inf_count,
        "zero_vector_count": zero_vector_count,
        "is_normalized": is_normalized,
    }


def cosine_similarity(
    vector_a: np.ndarray,
    vector_b: np.ndarray,
    assume_normalized: bool = True,
) -> float:
    """Compute mathematical cosine similarity between two single embedding vectors.

    Args:
        vector_a: First vector (1D array or 1xD 2D array).
        vector_b: Second vector (1D array or 1xD 2D array).
        assume_normalized: If True and vectors are unit-norm, uses fast dot product.

    Returns:
        Float cosine similarity value clamped to [-1.0, 1.0].

    Raises:
        TypeError: If inputs are not NumPy arrays with numeric dtype.
        ValueError: If shapes are invalid, dimensions mismatch, values are non-finite, or zero norm.
    """
    if not isinstance(vector_a, np.ndarray) or not isinstance(vector_b, np.ndarray):
        raise TypeError("Both vectors must be numpy.ndarray instances.")

    if not (np.issubdtype(vector_a.dtype, np.number) and np.issubdtype(vector_b.dtype, np.number)):
        raise TypeError("Both vectors must have numeric dtypes.")

    # Allow shape (1, D) or (D,)
    va = vector_a.squeeze()
    vb = vector_b.squeeze()

    if va.ndim != 1 or vb.ndim != 1:
        raise ValueError(
            f"Vectors must be 1-dimensional (or 1xD). Got shapes {vector_a.shape} and {vector_b.shape}."
        )

    if va.shape[0] != vb.shape[0]:
        raise ValueError(
            f"Dimension mismatch between vector A ({va.shape[0]}) and vector B ({vb.shape[0]})."
        )

    if not np.all(np.isfinite(va)) or not np.all(np.isfinite(vb)):
        raise ValueError("Vectors must contain only finite numbers (no NaN or Inf).")

    norm_a = float(np.linalg.norm(va))
    norm_b = float(np.linalg.norm(vb))

    if np.isclose(norm_a, 0.0, atol=1e-7) or np.isclose(norm_b, 0.0, atol=1e-7):
        raise ValueError("Cannot compute cosine similarity for a zero vector.")

    if (
        assume_normalized
        and np.isclose(norm_a, 1.0, atol=1e-4)
        and np.isclose(norm_b, 1.0, atol=1e-4)
    ):
        similarity = float(np.dot(va, vb))
    else:
        similarity = float(np.dot(va, vb) / (norm_a * norm_b))

    # Clamp within [-1.0, 1.0] to guard against floating-point imprecision
    return float(np.clip(similarity, -1.0, 1.0))


def run_semantic_sanity_check(encoder: Optional[EmbeddingEncoder] = None) -> Dict[str, Any]:
    """Execute a basic semantic sanity check comparing similar and distinct text pairs.

    This verifies that the embedding model produces a sensible relative similarity
    without acting as a benchmark or retrieval evaluation system.

    Args:
        encoder: Optional pre-loaded EmbeddingEncoder.

    Returns:
        Dictionary with semantic sanity check scores and pass/fail indicators.
    """
    if encoder is None:
        encoder = EmbeddingEncoder()

    emb_similar = encoder.encode_documents(list(SIMILAR_PAIR), normalize_embeddings=True)
    sim_similar = cosine_similarity(emb_similar[0], emb_similar[1])

    emb_different = encoder.encode_documents(list(DIFFERENT_PAIR), normalize_embeddings=True)
    sim_different = cosine_similarity(emb_different[0], emb_different[1])

    # Semantic sanity condition: similar pair should have strictly higher similarity than different pair
    # and satisfy reasonable baseline bounds for all-MiniLM-L6-v2
    similar_passed = bool(sim_similar > sim_different and sim_similar > 0.5)
    different_passed = bool(sim_different < sim_similar and sim_different < 0.8)
    overall_passed = bool(similar_passed and different_passed)

    return {
        "similar_pair": SIMILAR_PAIR,
        "similar_score": sim_similar,
        "similar_passed": similar_passed,
        "different_pair": DIFFERENT_PAIR,
        "different_score": sim_different,
        "different_passed": different_passed,
        "passed": overall_passed,
    }
