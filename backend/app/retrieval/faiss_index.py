"""FAISS Vector Index foundation for dense similarity search in TrustRAG."""

from pathlib import Path
from typing import Optional, Union
import faiss
import numpy as np

from backend.app.embeddings.validator import validate_embeddings


class FaissIndexError(Exception):
    """Raised when FAISS indexing, insertion, or operations fail."""
    pass


class FaissVectorIndex:
    """FAISS-based vector index utilizing inner-product (cosine similarity for normalized vectors)."""

    def __init__(self, dimension: int = 384) -> None:
        """Initialize an empty FAISS IndexFlatIP.

        Args:
            dimension: Dimensionality of indexed vectors (default: 384).

        Raises:
            ValueError: If dimension is non-positive.
        """
        if dimension <= 0:
            raise ValueError(f"Index dimension must be a positive integer, got {dimension}.")

        self._dimension = int(dimension)
        self._index: faiss.IndexFlatIP = faiss.IndexFlatIP(self._dimension)

    @property
    def dimension(self) -> int:
        """Return the vector dimension of the index."""
        return self._dimension

    @property
    def total_vectors(self) -> int:
        """Return the total number of indexed vectors."""
        return int(self._index.ntotal)

    @property
    def underlying_index(self) -> faiss.IndexFlatIP:
        """Return the underlying FAISS index instance."""
        return self._index

    def add_embeddings(
        self,
        embeddings: np.ndarray,
        check_normalized: bool = True,
    ) -> int:
        """Add dense vector embeddings to the FAISS index after rigorous validation.

        Args:
            embeddings: NumPy 2D array of shape (N, dimension).
            check_normalized: Whether to require unit L2 normalization (default: True).

        Returns:
            The total number of vectors in the index after addition.

        Raises:
            TypeError: If embeddings is not a numpy array.
            ValueError: If embeddings shape, dimension, finite values, or norms are invalid.
            FaissIndexError: If FAISS native addition fails.
        """
        # Validate embeddings using the centralized validator
        validate_embeddings(
            embeddings,
            expected_dim=self._dimension,
            check_normalized=check_normalized,
        )

        try:
            # FAISS requires contiguous float32 C-order arrays
            contiguous_embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
            self._index.add(contiguous_embeddings)
            return self.total_vectors
        except Exception as exc:
            raise FaissIndexError(f"Failed to add embeddings to FAISS index: {exc}") from exc
