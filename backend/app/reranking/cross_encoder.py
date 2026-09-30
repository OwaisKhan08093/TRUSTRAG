"""Cross-Encoder model wrapper for pair scoring and neural reranking."""

from typing import Optional
from sentence_transformers import CrossEncoder

from backend.app.config import DEFAULT_RERANKER_MODEL


class CrossEncoderModelError(Exception):
    """Raised when the Cross-Encoder model fails to load or execute."""
    pass


class CrossEncoderReranker:
    """Wrapper around SentenceTransformers CrossEncoder for neural semantic relevance scoring."""

    def __init__(self, model_name: Optional[str] = None) -> None:
        """Initialize and load the Cross-Encoder model.

        Args:
            model_name: Model repository identifier or path.
                        Defaults to DEFAULT_RERANKER_MODEL ("cross-encoder/ms-marco-MiniLM-L-6-v2").

        Raises:
            ValueError: If model_name is explicitly passed as an empty/whitespace string.
            CrossEncoderModelError: If model fails to download or initialize.
        """
        if model_name is not None:
            if not isinstance(model_name, str) or not model_name.strip():
                raise ValueError("model_name must be a non-empty string.")
            self._model_name = model_name.strip()
        else:
            self._model_name = DEFAULT_RERANKER_MODEL

        self._model: Optional[CrossEncoder] = None
        self._load_model()

    def _load_model(self) -> None:
        """Instantiate the underlying CrossEncoder."""
        try:
            self._model = CrossEncoder(self._model_name)
        except Exception as exc:
            raise CrossEncoderModelError(
                f"Failed to load CrossEncoder model '{self._model_name}': {exc}"
            ) from exc

    @property
    def model_name(self) -> str:
        """Return the model name identifier."""
        return self._model_name

    @property
    def is_loaded(self) -> bool:
        """Check if the Cross-Encoder model is successfully loaded."""
        return self._model is not None

    @property
    def underlying_model(self) -> Optional[CrossEncoder]:
        """Return the underlying CrossEncoder instance."""
        return self._model
