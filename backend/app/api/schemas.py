"""Pydantic request and response schemas for TrustRAG FastAPI backend."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


class QueryRequest(BaseModel):
    """User query request submitted to TrustRAG API.

    Attributes:
        query: User question or search query string.
        retrieval_top_k: Optional number of candidate passages to retrieve.
        evidence_top_k: Optional number of reranked evidence chunks to retain.
        max_new_tokens: Optional LLM token generation budget.
        temperature: Optional sampling temperature (0.0 for deterministic).
        filter_cited_only: Whether citation block contains only cited references.
        session_id: Optional tracking identifier.
    """

    query: str = Field(
        ...,
        description="User question or query string.",
        min_length=1,
    )
    retrieval_top_k: Optional[int] = Field(
        default=None,
        ge=1,
        le=50,
        description="Candidate retrieval limit.",
    )
    evidence_top_k: Optional[int] = Field(
        default=None,
        ge=1,
        le=20,
        description="Neural reranking evidence limit.",
    )
    max_new_tokens: Optional[int] = Field(
        default=None,
        ge=1,
        le=2048,
        description="Token generation limit.",
    )
    temperature: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=2.0,
        description="Sampling temperature.",
    )
    filter_cited_only: bool = Field(
        default=True,
        description="Filter returned citations to only those cited in the response prose.",
    )
    session_id: Optional[str] = Field(
        default=None,
        description="Optional session tracking identifier.",
    )

    @field_validator("query")
    @classmethod
    def validate_query_not_empty_or_whitespace(cls, v: str) -> str:
        """Ensure query is not empty or composed solely of whitespace."""
        if not v or not v.strip():
            raise ValueError("query cannot be empty or whitespace only.")
        return v.strip()


class CitationItem(BaseModel):
    """Verifiable citation reference derived directly from document provenance."""

    index: int = Field(..., description="1-based citation index marker.")
    chunk_id: str = Field(..., description="Identifier of the supporting evidence chunk.")
    document_id: str = Field(..., description="Source document identifier.")
    document_name: str = Field(..., description="File name of the source document.")
    page_start: int = Field(..., description="Starting 1-based page number.")
    page_end: int = Field(..., description="Ending 1-based page number.")
    text_snippet: str = Field(..., description="Preview snippet of the supporting passage.")
    formatted_reference: str = Field(..., description="Formatted reference string (e.g. [1] doc.pdf, page 5).")


class TrustMetrics(BaseModel):
    """Multi-signal trust grounding and confidence measurements."""

    confidence_score: float = Field(..., description="Composite system trust confidence in [0.0, 1.0].")
    groundedness_score: float = Field(..., description="Composite evidence groundedness in [0.0, 1.0].")
    relevance_score: float = Field(..., description="Aggregate neural relevance score in [0.0, 1.0].")
    coverage_score: float = Field(..., description="Lexical query term coverage ratio in [0.0, 1.0].")
    provenance_valid: bool = Field(..., description="Whether all evidence chunks have valid provenance.")


class TraceEventItem(BaseModel):
    """Execution event record of a specialized agent within the pipeline."""

    agent_name: str = Field(..., description="Name of the executing specialized agent.")
    status: str = Field(..., description="Status (STARTED, COMPLETED, SKIPPED, FAILED).")
    latency_seconds: float = Field(default=0.0, description="Agent execution duration.")
    reason: Optional[str] = Field(default=None, description="Context note if skipped or failed.")


class QueryResponse(BaseModel):
    """Structured response container emitted by TrustRAG API.

    Attributes:
        query: Original user query.
        answer: Factual generated prose or structured refusal statement.
        decision: Trust gating decision (SUPPORTED or INSUFFICIENT_EVIDENCE).
        is_refusal: True if system abstained due to inadequate grounding.
        citations: List of verified CitationItem references backing the response.
        formatted_response: Formatted markdown representation with citations if supported.
        trust_metrics: Multi-signal trust and confidence metrics.
        latency_seconds: Total pipeline execution latency in seconds.
        trace_events: Chronological sequence of agent lifecycle events.
        metadata: Execution and diagnostic metadata.
    """

    query: str
    answer: str
    decision: str
    is_refusal: bool
    citations: List[CitationItem] = Field(default_factory=list)
    formatted_response: str
    trust_metrics: TrustMetrics
    latency_seconds: float = Field(default=0.0)
    trace_events: List[TraceEventItem] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
