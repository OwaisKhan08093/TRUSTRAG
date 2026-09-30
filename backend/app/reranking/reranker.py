"""Neural reranking logic for re-ordering hybrid retrieval candidates."""

from typing import Any, Dict, List, Optional, Sequence

from backend.app.config import (
    DEFAULT_RERANKER_BATCH_SIZE,
    DEFAULT_RERANKER_TOP_K,
    MAX_RERANKER_TOP_K,
)
from backend.app.reranking.cross_encoder import CrossEncoderReranker
from backend.app.reranking.schema import RerankedChunk


class RerankerError(Exception):
    """Raised when reranking execution or scoring fails."""
    pass


class ResultReranker:
    """Neural cross-encoder reranker for re-ordering candidate retrieval passages."""

    def __init__(
        self,
        cross_encoder: Optional[CrossEncoderReranker] = None,
        default_top_k: int = DEFAULT_RERANKER_TOP_K,
        default_batch_size: int = DEFAULT_RERANKER_BATCH_SIZE,
        max_top_k: int = MAX_RERANKER_TOP_K,
    ) -> None:
        """Initialize the ResultReranker with an underlying CrossEncoder.

        Args:
            cross_encoder: Optional pre-loaded CrossEncoderReranker instance.
            default_top_k: Default number of top results to retain after reranking.
            default_batch_size: Inference batch size for pair scoring.
            max_top_k: Hard upper bound on retrievable top_k.

        Raises:
            ValueError: If configuration values are invalid.
        """
        if default_top_k < 1 or default_top_k > max_top_k:
            raise ValueError(f"default_top_k must be between 1 and {max_top_k}, got {default_top_k}.")

        if default_batch_size < 1:
            raise ValueError(f"default_batch_size must be a positive integer, got {default_batch_size}.")

        self._encoder = cross_encoder or CrossEncoderReranker()
        self._default_top_k = default_top_k
        self._default_batch_size = default_batch_size
        self._max_top_k = max_top_k

    @property
    def cross_encoder(self) -> CrossEncoderReranker:
        """Return the underlying CrossEncoderReranker instance."""
        return self._encoder

    @property
    def default_top_k(self) -> int:
        """Return the default top_k limit."""
        return self._default_top_k

    @property
    def default_batch_size(self) -> int:
        """Return the default batch size."""
        return self._default_batch_size

    def rerank(
        self,
        query: str,
        results: Sequence[Dict[str, Any]],
        top_k: Optional[int] = None,
        batch_size: Optional[int] = None,
    ) -> List[RerankedChunk]:
        """Re-rank candidate retrieval results using cross-encoder relevance scores.

        Args:
            query: User query string.
            results: Sequence of candidate chunk dictionaries from retrieval.
            top_k: Maximum number of top reranked chunks to return.
            batch_size: Batch size for model inference.

        Returns:
            List of RerankedChunk objects sorted descending by rerank_score.

        Raises:
            TypeError: If query is not a string or results is not a sequence of dicts.
            ValueError: If query is empty or top_k/batch_size is invalid.
            RerankerError: If scoring or result construction fails.
        """
        if not isinstance(query, str):
            raise TypeError(f"query must be a string, got {type(query).__name__}.")

        stripped_query = query.strip()
        if not stripped_query:
            raise ValueError("query cannot be empty or contain only whitespace.")

        if not isinstance(results, (list, tuple)):
            raise TypeError(f"results must be a sequence of dictionaries, got {type(results).__name__}.")

        effective_top_k = self._default_top_k if top_k is None else top_k
        if not isinstance(effective_top_k, int) or effective_top_k < 1:
            raise ValueError(f"top_k must be a positive integer >= 1, got {effective_top_k}.")

        if effective_top_k > self._max_top_k:
            raise ValueError(
                f"Requested top_k ({effective_top_k}) exceeds configured maximum limit of {self._max_top_k}."
            )

        effective_batch_size = self._default_batch_size if batch_size is None else batch_size
        if not isinstance(effective_batch_size, int) or effective_batch_size < 1:
            raise ValueError(f"batch_size must be a positive integer >= 1, got {effective_batch_size}.")

        if len(results) == 0:
            return []

        # Validate candidate dictionaries and extract document texts
        doc_texts: List[str] = []
        for idx, res in enumerate(results):
            if not isinstance(res, dict):
                raise TypeError(f"Result at index {idx} must be a dictionary, got {type(res).__name__}.")
            if "text" not in res or not isinstance(res["text"], str):
                raise ValueError(f"Result at index {idx} is missing a valid string 'text' key.")
            doc_texts.append(res["text"])

        # Compute neural cross-encoder relevance scores
        try:
            scores = self._encoder.score_pairs(
                query=stripped_query,
                documents=doc_texts,
                batch_size=effective_batch_size,
            )
        except Exception as exc:
            raise RerankerError(f"Neural pair scoring failed: {exc}") from exc

        # Pair each candidate result with its rerank score and original index
        scored_candidates = []
        for orig_idx, (res, score) in enumerate(zip(results, scores)):
            chunk_id = str(res.get("chunk_id", f"chunk_{orig_idx}"))
            scored_candidates.append({
                "raw_result": res,
                "rerank_score": float(score),
                "chunk_id": chunk_id,
            })

        # Sort descending by rerank_score; tie-break ascending by chunk_id
        scored_candidates.sort(
            key=lambda item: (-item["rerank_score"], item["chunk_id"])
        )

        # Slice to requested top_k (or all available if candidate count < top_k)
        selected_candidates = scored_candidates[:effective_top_k]

        # Build final RerankedChunk list preserving full metadata and provenance
        reranked_results: List[RerankedChunk] = []
        for rank_idx, item in enumerate(selected_candidates, start=1):
            try:
                chunk_obj = RerankedChunk.from_retrieval_result(
                    result=item["raw_result"],
                    rerank_score=item["rerank_score"],
                    final_rank=rank_idx,
                )
                reranked_results.append(chunk_obj)
            except Exception as exc:
                raise RerankerError(f"Failed to construct RerankedChunk: {exc}") from exc

        return reranked_results
