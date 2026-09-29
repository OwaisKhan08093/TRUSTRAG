"""Embedding encoder foundation using SentenceTransformers."""

from typing import Optional, Sequence
import numpy as np
from sentence_transformers import SentenceTransformer

from backend.app.config import DEFAULT_EMBEDDING_BATCH_SIZE, DEFAULT_EMBEDDING_MODEL


class EmbeddingModelError(Exception):
    """Raised when an embedding model fails to initialize, load, or encode."""
    pass


class EmbeddingEncoder:
    """Wrapper around SentenceTransformer for generating vector embeddings."""

    def __init__(self, model_name: Optional[str] = None) -> None:
        """Initialize and load the embedding model.

        Args:
            model_name: Name or path of the sentence-transformers model.
                        Defaults to DEFAULT_EMBEDDING_MODEL ("sentence-transformers/all-MiniLM-L6-v2").

        Raises:
            EmbeddingModelError: If the model cannot be found or loaded.
        """
        self._model_name = model_name or DEFAULT_EMBEDDING_MODEL
        self._model: Optional[SentenceTransformer] = None
        self._load_model()

    def _load_model(self) -> None:
        """Load the SentenceTransformer model with error handling."""
        try:
            self._model = SentenceTransformer(self._model_name)
        except Exception as exc:
            raise EmbeddingModelError(
                f"Failed to load embedding model '{self._model_name}': {exc}"
            ) from exc

    @property
    def model_name(self) -> str:
        """Return the identifier of the loaded model."""
        return self._model_name

    @property
    def embedding_dimension(self) -> int:
        """Return the vector dimensionality produced by the model."""
        if self._model is None:
            raise EmbeddingModelError("Model is not loaded.")
        if hasattr(self._model, "get_embedding_dimension"):
            dim = self._model.get_embedding_dimension()
        else:
            dim = self._model.get_sentence_embedding_dimension()
        if dim is None:
            raise EmbeddingModelError(
                f"Could not determine embedding dimension for '{self._model_name}'."
            )
        return int(dim)

    @property
    def is_loaded(self) -> bool:
        """Verify whether the model is successfully loaded and ready."""
        return self._model is not None

    def encode_documents(
        self,
        texts: Sequence[str],
        batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE,
        normalize_embeddings: bool = True,
    ) -> np.ndarray:
        """Encode a sequence of document texts into dense vector embeddings.

        Args:
            texts: Sequence of strings representing chunk or document texts.
            batch_size: Number of texts to encode per batch (defaults to DEFAULT_EMBEDDING_BATCH_SIZE).
            normalize_embeddings: Whether to L2-normalize the resulting vectors (defaults to True).

        Returns:
            A NumPy 2D array of shape (len(texts), embedding_dimension) with float32 dtype.

        Raises:
            EmbeddingModelError: If encoding fails or the model is not ready.
        """
        if self._model is None:
            raise EmbeddingModelError("Model is not loaded.")

        if not texts:
            return np.empty((0, self.embedding_dimension), dtype=np.float32)

        text_list = [str(t) for t in texts]

        try:
            embeddings = self._model.encode(
                text_list,
                batch_size=batch_size,
                normalize_embeddings=normalize_embeddings,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            return np.asarray(embeddings, dtype=np.float32)
        except Exception as exc:
            raise EmbeddingModelError(
                f"Failed to generate embeddings: {exc}"
            ) from exc
