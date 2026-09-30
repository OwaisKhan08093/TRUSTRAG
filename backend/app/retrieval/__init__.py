"""TrustRAG Retrieval Subsystem.

Provides dense vector search (FAISS), sparse keyword search (BM25),
rank fusion (RRF), chunk metadata resolution, and unified hybrid retrieval.
"""

from backend.app.retrieval.bm25_index import (
    BM25Index,
    BM25IndexError,
    default_tokenizer,
)
from backend.app.retrieval.bm25_retriever import (
    BM25Retriever,
    BM25RetrieverError,
)
from backend.app.retrieval.faiss_index import (
    FaissIndexError,
    FaissVectorIndex,
)
from backend.app.retrieval.hybrid_retriever import (
    HybridRetriever,
    HybridRetrieverError,
)
from backend.app.retrieval.metadata import (
    ChunkMetadataResolver,
    MetadataResolutionError,
)
from backend.app.retrieval.retriever import (
    VectorRetriever,
    VectorRetrieverError,
)
from backend.app.retrieval.rrf import (
    RRFError,
    reciprocal_rank_fusion,
)

__all__ = [
    "FaissVectorIndex",
    "FaissIndexError",
    "ChunkMetadataResolver",
    "MetadataResolutionError",
    "VectorRetriever",
    "VectorRetrieverError",
    "BM25Index",
    "BM25IndexError",
    "default_tokenizer",
    "BM25Retriever",
    "BM25RetrieverError",
    "reciprocal_rank_fusion",
    "RRFError",
    "HybridRetriever",
    "HybridRetrieverError",
]
