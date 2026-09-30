"""End-to-end reranking pipeline orchestrating hybrid retrieval and neural cross-encoder reranking."""

from typing import Any, Dict, List, Optional

from backend.app.config import (
    DEFAULT_RERANKER_BATCH_SIZE,
    DEFAULT_RERANKER_CANDIDATE_K,
    DEFAULT_RERANKER_MODEL,
    DEFAULT_RERANKER_TOP_K,
    MAX_RERANKER_TOP_K,
    validate_reranker_config,
)
from backend.app.reranking.cross_encoder import CrossEncoderReranker
from backend.app.reranking.reranker import ResultReranker
from backend.app.reranking.schema import RerankedChunk
from backend.app.retrieval.hybrid_retriever import HybridRetriever


class RerankingPipelineError(Exception):
    """Raised when pipeline orchestration or execution fails."""
    pass


class RerankingPipeline:
    """Orchestrates candidate retrieval via HybridRetriever followed by neural CrossEncoder reranking."""

    def __init__(
        self,
        retriever: Optional[HybridRetriever] = None,
        reranker: Optional[ResultReranker] = None,
        default_candidate_k: int = DEFAULT_RERANKER_CANDIDATE_K,
        default_final_k: int = DEFAULT_RERANKER_TOP_K,
        default_batch_size: int = DEFAULT_RERANKER_BATCH_SIZE,
        model_name: str = DEFAULT_RERANKER_MODEL,
    ) -> None:
        """Initialize the RerankingPipeline with component instances and default hyper-parameters.

        Args:
            retriever: Optional pre-configured HybridRetriever instance.
            reranker: Optional pre-configured ResultReranker instance.
            default_candidate_k: Number of candidate passages to fetch from hybrid retrieval.
            default_final_k: Final top-K reranked passages to return.
            default_batch_size: Inference batch size.
            model_name: Model identifier if instantiating default reranker.

        Raises:
            ValueError: If configuration values or relationships are invalid.
        """
        validate_reranker_config(
            candidate_k=default_candidate_k,
            final_k=default_final_k,
            batch_size=default_batch_size,
            model_name=model_name,
        )

        self._candidate_k = default_candidate_k
        self._final_k = default_final_k
        self._batch_size = default_batch_size
        self._model_name = model_name

        self._retriever = retriever or HybridRetriever()
        if reranker is not None:
            self._reranker = reranker
        else:
            encoder = CrossEncoderReranker(model_name=model_name)
            self._reranker = ResultReranker(
                cross_encoder=encoder,
                default_top_k=default_final_k,
                default_batch_size=default_batch_size,
            )

    @property
    def retriever(self) -> HybridRetriever:
        """Return the underlying HybridRetriever instance."""
        return self._retriever

    @property
    def reranker(self) -> ResultReranker:
        """Return the underlying ResultReranker instance."""
        return self._reranker

    @property
    def candidate_k(self) -> int:
        """Return the default candidate pool size."""
        return self._candidate_k

    @property
    def final_k(self) -> int:
        """Return the default final top-K limit."""
        return self._final_k

    @property
    def batch_size(self) -> int:
        """Return the default inference batch size."""
        return self._batch_size

    @property
    def model_name(self) -> str:
        """Return the model identifier."""
        return self._model_name

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        candidate_k: Optional[int] = None,
        batch_size: Optional[int] = None,
    ) -> List[RerankedChunk]:
        """Execute full pipeline: Hybrid Retrieval -> CrossEncoder Scoring -> Re-ranking -> RerankedChunk.

        Args:
            query: User query string.
            top_k: Final number of reranked passages to return (defaults to self.final_k).
            candidate_k: Number of hybrid candidates to retrieve (defaults to self.candidate_k).
            batch_size: Inference batch size (defaults to self.batch_size).

        Returns:
            List of RerankedChunk objects sorted descending by neural rerank_score.

        Raises:
            TypeError: If query is not a string.
            ValueError: If query is empty or configuration values are out of bounds.
            RerankingPipelineError: If retrieval or reranking fails.
        """
        if not isinstance(query, str):
            raise TypeError(f"query must be a string, got {type(query).__name__}.")

        stripped_query = query.strip()
        if not stripped_query:
            raise ValueError("query cannot be empty or contain only whitespace.")

        effective_candidate_k = self._candidate_k if candidate_k is None else candidate_k
        effective_final_k = self._final_k if top_k is None else top_k
        effective_batch_size = self._batch_size if batch_size is None else batch_size

        validate_reranker_config(
            candidate_k=effective_candidate_k,
            final_k=effective_final_k,
            batch_size=effective_batch_size,
            model_name=self._model_name,
        )

        # 1. Fetch candidates from HybridRetriever
        try:
            candidates = self._retriever.retrieve(
                query=stripped_query,
                top_k=effective_candidate_k,
            )
        except Exception as exc:
            raise RerankingPipelineError(f"Hybrid candidate retrieval failed: {exc}") from exc

        if not candidates:
            return []

        # 2. Re-rank candidates using neural Cross-Encoder
        try:
            reranked = self._reranker.rerank(
                query=stripped_query,
                results=candidates,
                top_k=effective_final_k,
                batch_size=effective_batch_size,
            )
        except Exception as exc:
            raise RerankingPipelineError(f"Cross-encoder reranking failed: {exc}") from exc

        return reranked
