"""Specialized agent coordinating hybrid dense and sparse document retrieval."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from backend.app.agents.base import AgentInputError, BaseAgent
from backend.app.retrieval.hybrid_retriever import HybridRetriever


@dataclass(frozen=True)
class RetrievalAgentResult:
    """Immutable typed container for outputs emitted by RetrievalAgent.

    Attributes:
        query: Evaluated search query.
        results: Ranked list of retrieved candidate dictionaries.
        candidate_count: Number of candidates retrieved.
        top_k: Requested top-k limit.
        retrieval_mode: Strategy utilized (e.g. 'hybrid_rrf').
        metadata: Execution details.
    """

    query: str
    results: List[Dict[str, Any]]
    candidate_count: int
    top_k: int
    retrieval_mode: str = "hybrid_rrf"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize retrieval agent result to dictionary."""
        return {
            "query": self.query,
            "results": list(self.results),
            "candidate_count": self.candidate_count,
            "top_k": self.top_k,
            "retrieval_mode": self.retrieval_mode,
            "metadata": dict(self.metadata),
        }


class RetrievalAgent(BaseAgent):
    """Specialized agent coordinating HybridRetriever (FAISS + BM25 + RRF).

    Design: Reuses existing HybridRetriever without reimplementing indexing or fusion.
    """

    def __init__(
        self,
        retriever: Optional[HybridRetriever] = None,
        default_top_k: int = 5,
    ) -> None:
        """Initialize RetrievalAgent.

        Args:
            retriever: Optional pre-configured HybridRetriever instance.
            default_top_k: Default candidate pool size to retrieve.
        """
        self._retriever = retriever
        self.default_top_k = default_top_k

    @property
    def name(self) -> str:
        return "RetrievalAgent"

    @property
    def description(self) -> str:
        return "Coordinates dense vector semantic search and BM25 sparse keyword retrieval with RRF fusion."

    @property
    def retriever(self) -> HybridRetriever:
        """Lazy-initialize HybridRetriever if not injected."""
        if self._retriever is None:
            self._retriever = HybridRetriever()
        return self._retriever

    def execute(
        self,
        query: str,
        top_k: Optional[int] = None,
        dense_top_k: Optional[int] = None,
        sparse_top_k: Optional[int] = None,
    ) -> RetrievalAgentResult:
        """Execute hybrid search for a query.

        Args:
            query: Search query string.
            top_k: Number of fused results to return.
            dense_top_k: Optional dense candidate limit.
            sparse_top_k: Optional sparse candidate limit.

        Returns:
            RetrievalAgentResult containing fused candidate items.

        Raises:
            AgentInputError: If query is empty or malformed.
        """
        if not isinstance(query, str):
            raise AgentInputError(f"query must be a string, got {type(query).__name__}.")

        stripped_query = query.strip()
        if not stripped_query:
            raise AgentInputError("query cannot be empty or whitespace only.")

        limit = top_k if top_k is not None else self.default_top_k
        if not isinstance(limit, int) or limit < 1:
            raise AgentInputError(f"top_k must be a positive integer >= 1, got {limit}.")

        candidates = self.retriever.retrieve(
            query=stripped_query,
            top_k=limit,
            dense_top_k=dense_top_k,
            sparse_top_k=sparse_top_k,
        )

        return RetrievalAgentResult(
            query=stripped_query,
            results=candidates,
            candidate_count=len(candidates),
            top_k=limit,
            retrieval_mode="hybrid_rrf",
            metadata={
                "dense_top_k": dense_top_k,
                "sparse_top_k": sparse_top_k,
            },
        )
