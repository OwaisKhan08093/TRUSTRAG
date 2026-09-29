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

    def save(self, path: Union[str, Path]) -> None:
        """Persist the FAISS index to disk using native binary serialization.

        Args:
            path: Destination file path for the .faiss index file.

        Raises:
            FaissIndexError: If writing the index fails.
        """
        dest_path = Path(path)
        try:
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            faiss.write_index(self._index, str(dest_path))
        except Exception as exc:
            raise FaissIndexError(f"Failed to save FAISS index to '{dest_path}': {exc}") from exc

    @classmethod
    def load(cls, path: Union[str, Path]) -> "FaissVectorIndex":
        """Load a persisted FAISS index from disk.

        Args:
            path: Source file path of the .faiss index file.

        Returns:
            A new FaissVectorIndex instance wrapping the loaded index.

        Raises:
            FileNotFoundError: If the file does not exist.
            FaissIndexError: If reading or deserializing the index fails.
        """
        source_path = Path(path)
        if not source_path.exists():
            raise FileNotFoundError(f"FAISS index file not found at '{source_path}'.")

        try:
            raw_index = faiss.read_index(str(source_path))
            dimension = int(raw_index.d)
            instance = cls(dimension=dimension)
            instance._index = raw_index
            return instance
        except Exception as exc:
            raise FaissIndexError(f"Failed to load FAISS index from '{source_path}': {exc}") from exc
