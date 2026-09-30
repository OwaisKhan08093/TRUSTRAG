"""Cross-Encoder model wrapper for pair scoring and neural reranking."""

from typing import List, Optional, Sequence
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

    def score_pairs(
        self,
        query: str,
        documents: Sequence[str],
        batch_size: int = 16,
    ) -> List[float]:
        """Score semantic relevance of query-document pairs without re-ranking.

        Args:
            query: Non-empty user query string.
            documents: Sequence of document text strings.
            batch_size: Inference batch size (positive integer, default: 16).

        Returns:
            List of float relevance scores in exact 1:1 order with input documents.
            Returns empty list if documents is empty.

        Raises:
            TypeError: If query is not a string or documents is not a sequence of strings.
            ValueError: If query is empty or batch_size < 1.
            CrossEncoderModelError: If model is not loaded or scoring fails.
        """
        if not isinstance(query, str):
            raise TypeError(f"query must be a string, got {type(query).__name__}.")

        stripped_query = query.strip()
        if not stripped_query:
            raise ValueError("query cannot be empty or contain only whitespace.")

        if not isinstance(documents, (list, tuple)):
            raise TypeError(f"documents must be a sequence, got {type(documents).__name__}.")

        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError(f"batch_size must be a positive integer >= 1, got {batch_size}.")

        if len(documents) == 0:
            return []

        for idx, doc in enumerate(documents):
            if not isinstance(doc, str):
                raise TypeError(
                    f"Document at index {idx} must be a string, got {type(doc).__name__}."
                )

        if self._model is None:
            raise CrossEncoderModelError("Cross-Encoder model is not loaded.")

        # Construct (query, document) input pairs
        pairs = [[stripped_query, doc] for doc in documents]

        try:
            raw_scores = self._model.predict(
                pairs,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
        except Exception as exc:
            raise CrossEncoderModelError(f"Cross-Encoder prediction failed: {exc}") from exc

        # Handle scalar (single element) or 1D array
        if hasattr(raw_scores, "tolist"):
            scores_list = raw_scores.tolist()
            if isinstance(scores_list, float):
                scores_list = [scores_list]
        elif isinstance(raw_scores, (int, float)):
            scores_list = [float(raw_scores)]
        else:
            scores_list = list(raw_scores)

        if len(scores_list) != len(documents):
            raise CrossEncoderModelError(
                f"Score count mismatch: expected {len(documents)} scores, got {len(scores_list)}."
            )

        return [float(s) for s in scores_list]
