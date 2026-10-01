"""TrustRAG Neural Cross-Encoder Reranking Subsystem (Phase 4).

Architecture:
    User Query
        │
        ▼
    HybridRetriever
        │
        ├── FAISS (Dense Retrieval)
        └── BM25 (Sparse Retrieval)
             │
             ▼
            RRF (Reciprocal Rank Fusion)
             │
             ▼
    Candidate Evidence (candidate_k)
             │
             ▼
    Cross-Encoder Reranker (ms-marco-MiniLM-L-6-v2)
             │
             ▼
    Re-ranked Evidence (final_k)

Responsibilities & Boundaries:
- Cross-encoder reranking performs joint query-document cross-attention scoring.
- It re-orders hybrid candidate passages by deep semantic relevance.
- Preserves complete document and chunk provenance metadata.
- What reranking DOES NOT do:
    * It does NOT generate answers or synthesize text.
    * It does NOT prove factual correctness.
    * It does NOT replace verification, citations, or confidence scoring.
"""

from backend.app.reranking.config import (
    RerankerConfig,
    RerankerConfigurationError,
)
from backend.app.reranking.cross_encoder import (
    CrossEncoderModelError,
    CrossEncoderReranker,
)
from backend.app.reranking.pipeline import (
    RerankingPipeline,
    RerankingPipelineError,
)
from backend.app.reranking.reranker import (
    RerankerError,
    ResultReranker,
)
from backend.app.reranking.schema import (
    RerankedChunk,
    SchemaValidationError,
)

__all__ = [
    "CrossEncoderReranker",
    "CrossEncoderModelError",
    "ResultReranker",
    "RerankerError",
    "RerankedChunk",
    "SchemaValidationError",
    "RerankerConfig",
    "RerankerConfigurationError",
    "RerankingPipeline",
    "RerankingPipelineError",
]
