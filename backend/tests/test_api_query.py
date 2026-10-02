"""Unit tests for POST /query endpoint (Phase 8 Milestone 4)."""

from unittest.mock import MagicMock
import pytest
from fastapi.testclient import TestClient

from backend.app.agents.orchestrator import OrchestratorResult, TrustRAGOrchestrator
from backend.app.agents.trace import AgentEventStatus, ExecutionTrace
from backend.app.api.app import app
from backend.app.api.dependencies import get_orchestrator
from backend.app.generation.citations import Citation
from backend.app.generation.generator import STANDARD_ABSTENTION_MESSAGE
from backend.app.trust.engine import TrustDecision


@pytest.fixture
def mock_orchestrator():
    """Fixture providing a mocked TrustRAGOrchestrator."""
    return MagicMock(spec=TrustRAGOrchestrator)


@pytest.fixture
def client(mock_orchestrator):
    """Fixture providing a TestClient with overridden get_orchestrator dependency."""
    app.dependency_overrides[get_orchestrator] = lambda: mock_orchestrator
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_query_endpoint_supported(client, mock_orchestrator):
    """Verify POST /query returns 200 with grounded answer and citations on SUPPORTED query."""
    citation = Citation(
        index=1,
        chunk_id="chunk_01",
        document_id="doc_dpdp",
        document_name="dpdp.pdf",
        page_start=5,
        page_end=5,
        text_snippet="Notice is required.",
        formatted_reference="[1] dpdp.pdf, page 5",
    )
    trace = ExecutionTrace(query="What notice is required?")
    trace.record_event("RetrievalAgent", AgentEventStatus.COMPLETED, 0.01)
    trace.record_event("EvidenceAgent", AgentEventStatus.COMPLETED, 0.01)
    trace.record_event("TrustAgent", AgentEventStatus.COMPLETED, 0.01)
    trace.record_event("GenerationAgent", AgentEventStatus.COMPLETED, 0.05)
    trace.record_event("CitationAgent", AgentEventStatus.COMPLETED, 0.01)

    mock_result = OrchestratorResult(
        query="What notice is required?",
        answer="Notice must be given to Data Principal [1].",
        decision=TrustDecision.SUPPORTED,
        is_refusal=False,
        citations=[citation],
        formatted_response="Notice must be given to Data Principal [1].\n\n### References\n- [1] dpdp.pdf, page 5",
        confidence_score=0.92,
        groundedness_score=0.90,
        relevance_score=0.95,
        coverage_score=1.0,
        provenance_valid=True,
        trace=trace,
        latency_seconds=0.10,
    )
    mock_orchestrator.execute.return_value = mock_result

    response = client.post(
        "/query",
        json={"query": "What notice is required?", "retrieval_top_k": 5},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["query"] == "What notice is required?"
    assert data["decision"] == "SUPPORTED"
    assert data["is_refusal"] is False
    assert len(data["citations"]) == 1
    assert data["citations"][0]["document_name"] == "dpdp.pdf"
    assert data["trust_metrics"]["confidence_score"] == 0.92
    assert len(data["trace_events"]) == 5


def test_query_endpoint_insufficient_evidence(client, mock_orchestrator):
    """Verify POST /query returns 200 with refusal and no citations on INSUFFICIENT_EVIDENCE."""
    trace = ExecutionTrace(query="Explain quantum law")
    trace.record_event("RetrievalAgent", AgentEventStatus.COMPLETED, 0.01)
    trace.record_event("EvidenceAgent", AgentEventStatus.COMPLETED, 0.01)
    trace.record_event("TrustAgent", AgentEventStatus.COMPLETED, 0.01)
    trace.record_event("GenerationAgent", AgentEventStatus.SKIPPED, 0.0, reason="Gated")
    trace.record_event("CitationAgent", AgentEventStatus.SKIPPED, 0.0, reason="Gated")

    mock_result = OrchestratorResult(
        query="Explain quantum law",
        answer=STANDARD_ABSTENTION_MESSAGE,
        decision=TrustDecision.INSUFFICIENT_EVIDENCE,
        is_refusal=True,
        citations=[],
        formatted_response=f"**[Status: INSUFFICIENT_EVIDENCE]**\n\n{STANDARD_ABSTENTION_MESSAGE}",
        confidence_score=0.05,
        groundedness_score=0.0,
        relevance_score=0.1,
        coverage_score=0.0,
        provenance_valid=True,
        trace=trace,
        latency_seconds=0.03,
    )
    mock_orchestrator.execute.return_value = mock_result

    response = client.post(
        "/query",
        json={"query": "Explain quantum law"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["decision"] == "INSUFFICIENT_EVIDENCE"
    assert data["is_refusal"] is True
    assert data["citations"] == []
    assert data["answer"] == STANDARD_ABSTENTION_MESSAGE


def test_query_endpoint_invalid_body(client):
    """Verify POST /query returns 422 for empty query."""
    response = client.post("/query", json={"query": ""})
    assert response.status_code == 422
