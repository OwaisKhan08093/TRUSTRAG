"""Specialized agent coordinating neural cross-encoder evidence reranking."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from backend.app.agents.base import AgentExecutionError, AgentInputError, BaseAgent
from backend.app.config import DEFAULT_RERANKER_TOP_K
from backend.app.reranking.reranker import ResultReranker
from backend.app.reranking.schema import RerankedChunk


@dataclass(frozen=True)
class EvidenceAgentResult:
    """Immutable typed container for outputs emitted by EvidenceAgent.

    Attributes:
        query: Evaluated search query.
        evidence: Ranked list of RerankedChunk evidence objects.
        evidence_count: Number of reranked evidence items returned.
        top_k: Requested top-k limit.
        metadata: Execution details.
    """

    query: str
    evidence: List[RerankedChunk]
    evidence_count: int
    top_k: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize evidence agent result to dictionary."""
        return {
            "query": self.query,
            "evidence": [item.to_dict() for item in self.evidence],
            "evidence_count": self.evidence_count,
            "top_k": self.top_k,
            "metadata": dict(self.metadata),
        }


class EvidenceAgent(BaseAgent):
    """Specialized agent coordinating ResultReranker (neural cross-encoder).

    Design: Reuses existing ResultReranker without reimplementing model inference or scoring.
    """

    def __init__(
        self,
        reranker: Optional[ResultReranker] = None,
        default_top_k: int = DEFAULT_RERANKER_TOP_K,
        batch_size: Optional[int] = None,
    ) -> None:
        """Initialize EvidenceAgent.

        Args:
            reranker: Optional pre-configured ResultReranker instance.
            default_top_k: Default evidence count to retain.
            batch_size: Optional batch size for cross-encoder inference.
        """
        self._reranker = reranker
        self.default_top_k = default_top_k
        self.batch_size = batch_size

    @property
    def name(self) -> str:
        return "EvidenceAgent"

    @property
    def description(self) -> str:
        return "Reranks retrieved candidate passages using neural cross-encoder cross-attention to isolate high-relevance evidence."

    @property
    def reranker(self) -> ResultReranker:
        """Lazy-initialize ResultReranker if not injected."""
        if self._reranker is None:
            self._reranker = ResultReranker()
        return self._reranker

    def execute(
        self,
        query: str,
        candidates: Sequence[Dict[str, Any]],
        top_k: Optional[int] = None,
        batch_size: Optional[int] = None,
    ) -> EvidenceAgentResult:
        """Execute neural reranking on candidate retrieval results.

        Args:
            query: Search query string.
            candidates: Sequence of candidate chunk dictionaries.
            top_k: Maximum number of reranked chunks to return.
            batch_size: Inference batch size override.

        Returns:
            EvidenceAgentResult containing ranked RerankedChunk evidence.

        Raises:
            AgentInputError: If query or candidates are invalid.
            AgentExecutionError: If neural reranker encounters runtime error.
        """
        if not isinstance(query, str):
            raise AgentInputError(f"query must be a string, got {type(query).__name__}.")

        stripped_query = query.strip()
        if not stripped_query:
            raise AgentInputError("query cannot be empty or whitespace only.")

        if not isinstance(candidates, (list, tuple)):
            raise AgentInputError(
                f"candidates must be a sequence of dictionaries, got {type(candidates).__name__}."
            )

        limit = top_k if top_k is not None else self.default_top_k
        if not isinstance(limit, int) or limit < 1:
            raise AgentInputError(f"top_k must be a positive integer >= 1, got {limit}.")

        effective_batch_size = batch_size if batch_size is not None else self.batch_size

        try:
            reranked_chunks = self.reranker.rerank(
                query=stripped_query,
                results=candidates,
                top_k=limit,
                batch_size=effective_batch_size,
            )
        except (TypeError, ValueError) as exc:
            raise AgentInputError(f"Invalid candidate data provided to reranker: {exc}") from exc
        except Exception as exc:
            raise AgentExecutionError(f"Cross-encoder reranking execution failed: {exc}") from exc

        return EvidenceAgentResult(
            query=stripped_query,
            evidence=reranked_chunks,
            evidence_count=len(reranked_chunks),
            top_k=limit,
            metadata={
                "batch_size": effective_batch_size,
                "input_candidate_count": len(candidates),
            },
        )
