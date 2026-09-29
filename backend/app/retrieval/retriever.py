"""Vector retrieval service orchestrating query encoding, FAISS search, and metadata resolution."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from backend.app.config import (
    CHUNKS_OUTPUT_FILE,
    EMBEDDING_METADATA_FILE,
    FAISS_INDEX_FILE,
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
    ) -> None:
        """Initialize the retrieval pipeline with injected or loaded components.

        Args:
            index: Optional pre-loaded FaissVectorIndex.
            encoder: Optional pre-loaded EmbeddingEncoder.
            metadata_resolver: Optional pre-loaded ChunkMetadataResolver.
            index_file: Path to load .faiss file if index is not provided.
            chunks_file: Path to chunks.json if resolver is not provided.
            metadata_file: Path to embedding_metadata.json if resolver is not provided.
        """
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

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """Retrieve top-K ranked relevant document chunks for a natural language user query.

        Args:
            query: User search query string.
            top_k: Number of highest-similarity chunks to return (default: 5, must be >= 1).

        Returns:
            List of structured chunk results with keys:
            rank, chunk_id, score, document_id, document_name, page_start, page_end, text.

        Raises:
            TypeError: If query is not a string.
            ValueError: If query is empty or top_k < 1.
            VectorRetrieverError: If query encoding, FAISS search, or metadata lookup fails.
        """
        if not isinstance(query, str):
            raise TypeError(f"Query must be a string, got {type(query).__name__}.")

        stripped_query = query.strip()
        if not stripped_query:
            raise ValueError("Query string cannot be empty or contain only whitespace.")

        if not isinstance(top_k, int) or top_k < 1:
            raise ValueError(f"top_k must be a positive integer >= 1, got {top_k}.")

        # 1. Generate normalized query embedding
        try:
            query_vector = self._encoder.encode_query(stripped_query, normalize_embeddings=True)
        except Exception as exc:
            raise VectorRetrieverError(f"Failed to encode query: {exc}") from exc

        # 2. Perform FAISS similarity search
        try:
            search_results = self._index.search(query_vector, top_k=top_k)
        except Exception as exc:
            raise VectorRetrieverError(f"FAISS search failed: {exc}") from exc

        # 3. Resolve metadata and assemble structured response
        retrieved: List[Dict[str, Any]] = []
        for rank_idx, match in enumerate(search_results, start=1):
            faiss_idx = int(match["index"])
            similarity_score = float(match["score"])

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

        return retrieved
