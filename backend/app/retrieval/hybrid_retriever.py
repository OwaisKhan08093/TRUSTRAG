"""Hybrid retrieval engine orchestrating dense vector search, sparse BM25 search, and RRF."""

from typing import Any, Dict, List, Optional

from backend.app.config import (
    DEFAULT_DENSE_TOP_K,
    DEFAULT_FINAL_TOP_K,
    DEFAULT_RRF_K,
    DEFAULT_SPARSE_TOP_K,
    MAX_RRF_K,
    MAX_TOP_K,
    MIN_RRF_K,
)
from backend.app.retrieval.bm25_retriever import BM25Retriever
from backend.app.retrieval.retriever import VectorRetriever
from backend.app.retrieval.rrf import reciprocal_rank_fusion


class HybridRetrieverError(Exception):
    """Raised when hybrid retrieval execution or orchestration fails."""
    pass


class HybridRetriever:
    """Hybrid retrieval orchestrator combining dense semantic search, sparse BM25, and RRF."""

    def __init__(
        self,
        dense_retriever: Optional[VectorRetriever] = None,
        sparse_retriever: Optional[BM25Retriever] = None,
        default_top_k: int = DEFAULT_FINAL_TOP_K,
        default_dense_top_k: int = DEFAULT_DENSE_TOP_K,
        default_sparse_top_k: int = DEFAULT_SPARSE_TOP_K,
        rrf_k: int = DEFAULT_RRF_K,
        max_top_k: int = MAX_TOP_K,
    ) -> None:
        """Initialize the HybridRetriever with dense and sparse retrievers and RRF parameters.

        Args:
            dense_retriever: Optional pre-configured VectorRetriever.
            sparse_retriever: Optional pre-configured BM25Retriever.
            default_top_k: Number of final fused results to return by default.
            default_dense_top_k: Number of dense candidate chunks to retrieve.
            default_sparse_top_k: Number of sparse candidate chunks to retrieve.
            rrf_k: RRF smoothing constant (default: DEFAULT_RRF_K).
            max_top_k: Maximum allowed top_k value across all stages.

        Raises:
            ValueError: If any top_k is non-positive or exceeds max_top_k, or if rrf_k is out of bounds.
        """
        if default_top_k < 1 or default_top_k > max_top_k:
            raise ValueError(f"default_top_k must be between 1 and {max_top_k}, got {default_top_k}.")
        if default_dense_top_k < 1 or default_dense_top_k > max_top_k:
            raise ValueError(
                f"default_dense_top_k must be between 1 and {max_top_k}, got {default_dense_top_k}."
            )
        if default_sparse_top_k < 1 or default_sparse_top_k > max_top_k:
            raise ValueError(
                f"default_sparse_top_k must be between 1 and {max_top_k}, got {default_sparse_top_k}."
            )
        if rrf_k < MIN_RRF_K or rrf_k > MAX_RRF_K:
            raise ValueError(f"rrf_k must be an integer between {MIN_RRF_K} and {MAX_RRF_K}, got {rrf_k}.")

        self._default_top_k = default_top_k
        self._default_dense_top_k = default_dense_top_k
        self._default_sparse_top_k = default_sparse_top_k
        self._rrf_k = rrf_k
        self._max_top_k = max_top_k

        self._dense = dense_retriever or VectorRetriever()
        self._sparse = sparse_retriever or BM25Retriever()

    @property
    def dense_retriever(self) -> VectorRetriever:
        """Return the underlying dense VectorRetriever instance."""
        return self._dense

    @property
    def sparse_retriever(self) -> BM25Retriever:
        """Return the underlying sparse BM25Retriever instance."""
        return self._sparse

    @property
    def rrf_k(self) -> int:
        """Return the default RRF smoothing constant."""
        return self._rrf_k

    @property
    def default_top_k(self) -> int:
        """Return the default top_k value."""
        return self._default_top_k

    @property
    def default_dense_top_k(self) -> int:
        """Return the default dense top_k value."""
        return self._default_dense_top_k

    @property
    def default_sparse_top_k(self) -> int:
        """Return the default sparse top_k value."""
        return self._default_sparse_top_k

    @property
    def max_top_k(self) -> int:
        """Return the maximum allowed top_k value."""
        return self._max_top_k

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        dense_top_k: Optional[int] = None,
        sparse_top_k: Optional[int] = None,
        rrf_k: Optional[int] = None,
        dense_score_threshold: Optional[float] = None,
        sparse_score_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Execute hybrid retrieval combining dense FAISS search, sparse BM25 search, and RRF.

        Args:
            query: User query string.
            top_k: Number of final fused results to return.
            dense_top_k: Number of candidates to retrieve from dense retriever.
            sparse_top_k: Number of candidates to retrieve from sparse retriever.
            rrf_k: RRF constant for score computation.
            dense_score_threshold: Optional minimum cosine similarity threshold for dense results.
            sparse_score_threshold: Optional minimum score threshold for sparse results.

        Returns:
            List of fused document chunks sorted descending by RRF score.

        Raises:
            TypeError: If query is not a string.
            ValueError: If query is empty or if any parameter is out of bounds.
            HybridRetrieverError: If underlying retrieval pipelines fail.
        """
        if not isinstance(query, str):
            raise TypeError(f"Query must be a string, got {type(query).__name__}.")

        stripped_query = query.strip()
        if not stripped_query:
            raise ValueError("Query string cannot be empty or contain only whitespace.")

        effective_top_k = self._default_top_k if top_k is None else top_k
        effective_dense_k = self._default_dense_top_k if dense_top_k is None else dense_top_k
        effective_sparse_k = self._default_sparse_top_k if sparse_top_k is None else sparse_top_k
        effective_rrf_k = self._rrf_k if rrf_k is None else rrf_k

        if not isinstance(effective_top_k, int) or effective_top_k < 1:
            raise ValueError(f"top_k must be a positive integer >= 1, got {effective_top_k}.")
        if effective_top_k > self._max_top_k:
            raise ValueError(f"top_k ({effective_top_k}) exceeds maximum limit of {self._max_top_k}.")

        if not isinstance(effective_dense_k, int) or effective_dense_k < 1:
            raise ValueError(f"dense_top_k must be a positive integer >= 1, got {effective_dense_k}.")
        if effective_dense_k > self._max_top_k:
            raise ValueError(
                f"dense_top_k ({effective_dense_k}) exceeds maximum limit of {self._max_top_k}."
            )

        if not isinstance(effective_sparse_k, int) or effective_sparse_k < 1:
            raise ValueError(f"sparse_top_k must be a positive integer >= 1, got {effective_sparse_k}.")
        if effective_sparse_k > self._max_top_k:
            raise ValueError(
                f"sparse_top_k ({effective_sparse_k}) exceeds maximum limit of {self._max_top_k}."
            )

        if not isinstance(effective_rrf_k, int) or effective_rrf_k < MIN_RRF_K or effective_rrf_k > MAX_RRF_K:
            raise ValueError(
                f"rrf_k must be an integer between {MIN_RRF_K} and {MAX_RRF_K}, got {effective_rrf_k}."
            )

        # 1. Retrieve dense semantic candidates
        try:
            dense_results = self._dense.retrieve(
                query=stripped_query,
                top_k=effective_dense_k,
                score_threshold=dense_score_threshold,
            )
        except Exception as exc:
            raise HybridRetrieverError(f"Dense vector retrieval failed: {exc}") from exc

        # 2. Retrieve sparse BM25 candidates
        try:
            sparse_results = self._sparse.retrieve(
                query=stripped_query,
                top_k=effective_sparse_k,
                score_threshold=sparse_score_threshold,
            )
        except Exception as exc:
            raise HybridRetrieverError(f"Sparse BM25 retrieval failed: {exc}") from exc

        # 3. Fuse candidate rankings using Reciprocal Rank Fusion
        try:
            fused = reciprocal_rank_fusion(
                ranked_lists=[dense_results, sparse_results],
                k=effective_rrf_k,
                top_k=effective_top_k,
            )
        except Exception as exc:
            raise HybridRetrieverError(f"Reciprocal rank fusion failed: {exc}") from exc

        return fused
