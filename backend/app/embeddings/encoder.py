"""Embedding encoder foundation using SentenceTransformers."""

from typing import Optional
from sentence_transformers import SentenceTransformer

from backend.app.config import DEFAULT_EMBEDDING_MODEL


class EmbeddingModelError(Exception):
    """Raised when an embedding model fails to initialize or load."""
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
