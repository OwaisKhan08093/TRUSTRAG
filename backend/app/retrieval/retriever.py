"""Vector retrieval service orchestrating query encoding, FAISS search, and metadata resolution."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from backend.app.config import (
    CHUNKS_OUTPUT_FILE,
    DEFAULT_SIMILARITY_THRESHOLD,
    DEFAULT_TOP_K,
    EMBEDDING_METADATA_FILE,
    FAISS_INDEX_FILE,
    MAX_TOP_K,
)
from backend.app.embeddings.encoder import EmbeddingEncoder
from backend.app.retrieval.faiss_index import FaissVectorIndex
from backend.app.retrieval.metadata import ChunkMetadataResolver


class VectorRetrieverError(Exception):
    """Raised when vector retrieval fails."""
    pass


class VectorRetriever:
    """Semantic vector retriever combining query embedding, FAISS index, and chunk provenance."""

    def __init__(
        self,
        index: Optional[FaissVectorIndex] = None,
        encoder: Optional[EmbeddingEncoder] = None,
        metadata_resolver: Optional[ChunkMetadataResolver] = None,
        index_file: Union[str, Path] = FAISS_INDEX_FILE,
        chunks_file: Union[str, Path] = CHUNKS_OUTPUT_FILE,
        metadata_file: Union[str, Path] = EMBEDDING_METADATA_FILE,
        default_top_k: int = DEFAULT_TOP_K,
        max_top_k: int = MAX_TOP_K,
        default_score_threshold: Optional[float] = None,
    ) -> None:
        """Initialize the retrieval pipeline with injected or loaded components and configurations.

        Args:
            index: Optional pre-loaded FaissVectorIndex.
            encoder: Optional pre-loaded EmbeddingEncoder.
            metadata_resolver: Optional pre-loaded ChunkMetadataResolver.
            index_file: Path to load .faiss file if index is not provided.
            chunks_file: Path to chunks.json if resolver is not provided.
            metadata_file: Path to embedding_metadata.json if resolver is not provided.
            default_top_k: Default number of items to retrieve (default: DEFAULT_TOP_K).
            max_top_k: Hard ceiling on number of items retrievable per query (default: MAX_TOP_K).
            default_score_threshold: Optional minimum cosine similarity threshold.
        """
        if default_top_k < 1 or default_top_k > max_top_k:
            raise ValueError(f"default_top_k must be between 1 and {max_top_k}, got {default_top_k}.")

        if default_score_threshold is not None:
            if not (-1.0 <= default_score_threshold <= 1.0):
                raise ValueError(
                    f"default_score_threshold must be between -1.0 and 1.0, got {default_score_threshold}."
                )

        self._default_top_k = default_top_k
        self._max_top_k = max_top_k
        self._default_score_threshold = default_score_threshold

        self._encoder = encoder or EmbeddingEncoder()
        self._index = index or FaissVectorIndex.load(index_file)
        self._resolver = metadata_resolver or ChunkMetadataResolver(
            chunks_file=chunks_file,
            metadata_file=metadata_file,
        )

        if self._index.dimension != self._encoder.embedding_dimension:
            raise VectorRetrieverError(
                f"Dimension mismatch between FAISS index ({self._index.dimension}) "
                f"and embedding encoder ({self._encoder.embedding_dimension})."
            )

    @property
    def total_indexed_vectors(self) -> int:
        """Return total indexed vectors available for retrieval."""
        return self._index.total_vectors

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
        """Retrieve top-K ranked relevant document chunks for a natural language user query.

        Note:
            The similarity threshold acts solely as a geometric/vector retrieval filter.
            Passing a similarity threshold does not prove or guarantee factual correctness.

        Args:
            query: User search query string.
            top_k: Number of highest-similarity chunks to return (defaults to self.default_top_k).
            score_threshold: Optional minimum cosine similarity score [-1.0, 1.0] to filter results.

        Returns:
            List of structured chunk results with keys:
            rank, chunk_id, score, document_id, document_name, page_start, page_end, text.

        Raises:
            TypeError: If query is not a string.
            ValueError: If query is empty, top_k is invalid or exceeds max_top_k, or score_threshold is out of bounds.
            VectorRetrieverError: If query encoding, FAISS search, or metadata lookup fails.
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
                raise TypeError(f"score_threshold must be a numeric float, got {type(threshold).__name__}.")
            if not (-1.0 <= threshold <= 1.0):
                raise ValueError(f"score_threshold must be between -1.0 and 1.0, got {threshold}.")

        # 1. Generate normalized query embedding
        try:
            query_vector = self._encoder.encode_query(stripped_query, normalize_embeddings=True)
        except Exception as exc:
            raise VectorRetrieverError(f"Failed to encode query: {exc}") from exc

        # 2. Perform FAISS similarity search
        try:
            search_results = self._index.search(query_vector, top_k=effective_top_k)
        except Exception as exc:
            raise VectorRetrieverError(f"FAISS search failed: {exc}") from exc

        # 3. Resolve metadata, filter by score threshold, and assemble structured response
        retrieved: List[Dict[str, Any]] = []
        rank_idx = 1
        for match in search_results:
            faiss_idx = int(match["index"])
            similarity_score = float(match["score"])

            # Filter by score threshold if specified
            if threshold is not None and similarity_score < threshold:
                continue

            try:
                meta = self._resolver.resolve_index(faiss_idx)
            except Exception as exc:
                raise VectorRetrieverError(
                    f"Failed to resolve metadata for FAISS index {faiss_idx}: {exc}"
                ) from exc

            retrieved.append({
                "rank": rank_idx,
                "chunk_id": meta["chunk_id"],
                "score": similarity_score,
                "document_id": meta["document_id"],
                "document_name": meta["document_name"],
                "page_start": meta["page_start"],
                "page_end": meta["page_end"],
                "text": meta["text"],
            })
            rank_idx += 1

        return retrieved

