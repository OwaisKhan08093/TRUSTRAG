"""Unit tests for Pydantic API schemas (Phase 8 Milestone 3)."""

import pytest
from pydantic import ValidationError

from backend.app.api.schemas import (
    CitationItem,
    QueryRequest,
    QueryResponse,
    TraceEventItem,
    TrustMetrics,
)


def test_query_request_valid():
    """Verify valid QueryRequest model instantiation."""
    req = QueryRequest(
        query="What notice is required under Section 5?",
        retrieval_top_k=10,
        evidence_top_k=5,
        max_new_tokens=256,
        temperature=0.0,
        filter_cited_only=True,
    )
    assert req.query == "What notice is required under Section 5?"
    assert req.retrieval_top_k == 10
    assert req.evidence_top_k == 5
    assert req.max_new_tokens == 256
    assert req.temperature == 0.0


def test_query_request_empty_or_whitespace():
    """Verify empty and whitespace queries raise ValidationError."""
    with pytest.raises(ValidationError):
        QueryRequest(query="")

    with pytest.raises(ValidationError):
        QueryRequest(query="    ")


def test_query_request_bounds_validation():
    """Verify out-of-bounds parameters raise ValidationError."""
    with pytest.raises(ValidationError):
        QueryRequest(query="valid", retrieval_top_k=0)

    with pytest.raises(ValidationError):
        QueryRequest(query="valid", retrieval_top_k=100)

    with pytest.raises(ValidationError):
        QueryRequest(query="valid", max_new_tokens=5000)

    with pytest.raises(ValidationError):
        QueryRequest(query="valid", temperature=-0.5)


def test_query_response_serialization():
    """Verify QueryResponse model instantiation and dictionary serialization."""
    citation = CitationItem(
        index=1,
        chunk_id="chunk_01",
        document_id="doc_dpdp",
        document_name="dpdp.pdf",
        page_start=5,
        page_end=5,
        text_snippet="Notice is required.",
        formatted_reference="[1] dpdp.pdf, page 5",
    )
    metrics = TrustMetrics(
        confidence_score=0.89,
        groundedness_score=0.91,
        relevance_score=0.95,
        coverage_score=1.0,
        provenance_valid=True,
    )
    event = TraceEventItem(
        agent_name="RetrievalAgent",
        status="COMPLETED",
        latency_seconds=0.02,
    )

    resp = QueryResponse(
        query="What notice is required?",
        answer="Notice is required [1].",
        decision="SUPPORTED",
        is_refusal=False,
        citations=[citation],
        formatted_response="Notice is required [1].\n\n### References\n- [1] dpdp.pdf, page 5",
        trust_metrics=metrics,
        latency_seconds=0.15,
        trace_events=[event],
        metadata={"gating_status": "SUPPORTED_AND_GENERATED"},
    )

    d = resp.model_dump()
    assert d["query"] == "What notice is required?"
    assert d["decision"] == "SUPPORTED"
    assert d["is_refusal"] is False
    assert len(d["citations"]) == 1
    assert d["trust_metrics"]["confidence_score"] == 0.89
    assert len(d["trace_events"]) == 1
