"""Sparse lexical retriever orchestrating BM25 search and chunk metadata resolution."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from backend.app.config import (
    CHUNKS_OUTPUT_FILE,
    DEFAULT_TOP_K,
    EMBEDDING_METADATA_FILE,
    MAX_TOP_K,
)
from backend.app.retrieval.bm25_index import BM25Index
from backend.app.retrieval.metadata import ChunkMetadataResolver


class BM25RetrieverError(Exception):
    """Raised when BM25 retrieval fails."""
    pass


class BM25Retriever:
    """Sparse keyword retriever combining BM25Okapi scoring with chunk provenance metadata."""

    def __init__(
        self,
        index: Optional[BM25Index] = None,
        metadata_resolver: Optional[ChunkMetadataResolver] = None,
        chunks_file: Union[str, Path] = CHUNKS_OUTPUT_FILE,
        metadata_file: Union[str, Path] = EMBEDDING_METADATA_FILE,
        default_top_k: int = DEFAULT_TOP_K,
        max_top_k: int = MAX_TOP_K,
        default_score_threshold: Optional[float] = None,
    ) -> None:
        """Initialize the BM25 retrieval pipeline with injected or loaded components.

        Args:
            index: Optional pre-loaded BM25Index.
            metadata_resolver: Optional pre-loaded ChunkMetadataResolver.
            chunks_file: Path to chunks.json if resolver or index is not provided.
            metadata_file: Path to embedding_metadata.json if resolver is not provided.
            default_top_k: Default number of items to retrieve (default: DEFAULT_TOP_K).
            max_top_k: Hard ceiling on number of items retrievable per query (default: MAX_TOP_K).
            default_score_threshold: Optional minimum BM25 score threshold.

        Raises:
            ValueError: If default_top_k or score_threshold is invalid.
        """
        if default_top_k < 1 or default_top_k > max_top_k:
            raise ValueError(f"default_top_k must be between 1 and {max_top_k}, got {default_top_k}.")

        if default_score_threshold is not None:
            if not isinstance(default_score_threshold, (int, float)) or default_score_threshold < 0:
                raise ValueError(
                    f"default_score_threshold must be a non-negative float, got {default_score_threshold}."
                )

        self._default_top_k = default_top_k
        self._max_top_k = max_top_k
        self._default_score_threshold = default_score_threshold

        self._index = index or BM25Index.from_chunks_file(chunks_file)
        self._resolver = metadata_resolver or ChunkMetadataResolver(
            chunks_file=chunks_file,
            metadata_file=metadata_file,
        )

    @property
    def total_indexed_chunks(self) -> int:
        """Return total indexed document chunks available for retrieval."""
        return self._index.corpus_size

    @property
    def default_top_k(self) -> int:
        """Return the default top_k value."""
        return self._default_top_k

    @property
    def max_top_k(self) -> int:
        """Return the maximum allowed top_k value."""
        return self._max_top_k

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve top-K ranked relevant document chunks for a keyword query.

        Args:
            query: User search query string.
            top_k: Number of highest-scoring chunks to return (defaults to self.default_top_k).
            score_threshold: Optional minimum BM25 score to filter results.

        Returns:
            List of structured chunk results with keys:
            rank, chunk_id, score, document_id, document_name, page_start, page_end, text.

        Raises:
            TypeError: If query is not a string.
            ValueError: If query is empty, top_k is invalid or exceeds max_top_k, or score_threshold < 0.
            BM25RetrieverError: If BM25 search or metadata lookup fails.
        """
        if not isinstance(query, str):
            raise TypeError(f"Query must be a string, got {type(query).__name__}.")

        stripped_query = query.strip()
        if not stripped_query:
            raise ValueError("Query string cannot be empty or contain only whitespace.")

        effective_top_k = self._default_top_k if top_k is None else top_k
        if not isinstance(effective_top_k, int) or effective_top_k < 1:
            raise ValueError(f"top_k must be a positive integer >= 1, got {effective_top_k}.")

        if effective_top_k > self._max_top_k:
            raise ValueError(
                f"Requested top_k ({effective_top_k}) exceeds configured maximum limit of {self._max_top_k}."
            )

        threshold = score_threshold if score_threshold is not None else self._default_score_threshold
        if threshold is not None:
            if not isinstance(threshold, (int, float)):
                raise TypeError(f"score_threshold must be numeric, got {type(threshold).__name__}.")
            if threshold < 0:
                raise ValueError(f"score_threshold must be non-negative, got {threshold}.")

        # 1. Perform BM25 search
        try:
            search_results = self._index.search(stripped_query, top_k=effective_top_k)
        except Exception as exc:
            raise BM25RetrieverError(f"BM25 search failed: {exc}") from exc

        # 2. Resolve metadata, filter by score threshold, and assemble structured response
        retrieved: List[Dict[str, Any]] = []
        rank_idx = 1
        for match in search_results:
            bm25_idx = int(match["index"])
            score = float(match["score"])

            if threshold is not None and score < threshold:
                continue

            try:
                meta = self._resolver.resolve_index(bm25_idx)
            except Exception as exc:
                raise BM25RetrieverError(
                    f"Failed to resolve metadata for BM25 index {bm25_idx}: {exc}"
                ) from exc

            retrieved.append({
                "rank": rank_idx,
                "chunk_id": meta["chunk_id"],
                "score": score,
                "document_id": meta["document_id"],
                "document_name": meta["document_name"],
                "page_start": meta["page_start"],
                "page_end": meta["page_end"],
                "text": meta["text"],
            })
            rank_idx += 1

        return retrieved
