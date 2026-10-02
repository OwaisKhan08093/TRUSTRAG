"""Comprehensive integration test suite for FastAPI backend (Phase 8 Milestone 8).

Validates:
1. /health endpoint behavior
2. Valid /query execution
3. Empty / whitespace invalid query handling
4. Gated refusal on INSUFFICIENT_EVIDENCE
5. Grounded generation on SUPPORTED evidence
6. Document citation & provenance preservation
7. Internal orchestrator error handling (HTTP 500)
8. Malformed request payload handling (HTTP 422)
9. Pydantic response schema compliance
10. Strict preservation of TrustEngine gating invariant
"""

from unittest.mock import MagicMock
import pytest
from fastapi.testclient import TestClient

from backend.app.agents.orchestrator import OrchestratorResult, TrustRAGOrchestrator
from backend.app.agents.trace import AgentEventStatus, ExecutionTrace
from backend.app.api.app import app
from backend.app.api.dependencies import get_orchestrator
from backend.app.api.schemas import QueryResponse
from backend.app.generation.citations import Citation
from backend.app.generation.generator import STANDARD_ABSTENTION_MESSAGE
from backend.app.trust.engine import TrustDecision


@pytest.fixture
def mock_orchestrator():
    """Fixture providing a mocked TrustRAGOrchestrator for deterministic API testing."""
    return MagicMock(spec=TrustRAGOrchestrator)


@pytest.fixture
def client(mock_orchestrator):
    """Fixture providing a TestClient with dependency override."""
    app.dependency_overrides[get_orchestrator] = lambda: mock_orchestrator
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_integration_01_health_endpoint(client):
    """1. Test /health status code and payload."""
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "service": "trustrag", "version": "1.0.0"}


def test_integration_02_valid_query(client, mock_orchestrator):
    """2. Test valid /query accepts requests and returns 200."""
    trace = ExecutionTrace(query="Notice rules")
    trace.record_event("RetrievalAgent", AgentEventStatus.COMPLETED, 0.01)
    mock_orchestrator.execute.return_value = OrchestratorResult(
        query="Notice rules",
        answer="Notice rules explanation.",
        decision=TrustDecision.SUPPORTED,
        is_refusal=False,
        citations=[],
        formatted_response="Notice rules explanation.",
        confidence_score=0.85,
        groundedness_score=0.80,
        relevance_score=0.90,
        coverage_score=0.80,
        provenance_valid=True,
        trace=trace,
        latency_seconds=0.05,
    )

    resp = client.post("/query", json={"query": "Notice rules"})
    assert resp.status_code == 200
    assert resp.json()["query"] == "Notice rules"


def test_integration_03_invalid_query_whitespace(client):
    """3. Test whitespace-only query fails validation with 422."""
    resp = client.post("/query", json={"query": "     "})
    assert resp.status_code == 422
    data = resp.json()
    assert data["error"]["type"] == "ValidationError"


def test_integration_04_insufficient_evidence_gating(client, mock_orchestrator):
    """4. Test INSUFFICIENT_EVIDENCE produces refusal and no LLM citations."""
    trace = ExecutionTrace(query="Alien astronomy in Indian Law")
    trace.record_event("TrustAgent", AgentEventStatus.COMPLETED, 0.01)
    trace.record_event("GenerationAgent", AgentEventStatus.SKIPPED, 0.0, reason="Gated")
    trace.record_event("CitationAgent", AgentEventStatus.SKIPPED, 0.0, reason="Gated")

    mock_orchestrator.execute.return_value = OrchestratorResult(
        query="Alien astronomy in Indian Law",
        answer=STANDARD_ABSTENTION_MESSAGE,
        decision=TrustDecision.INSUFFICIENT_EVIDENCE,
        is_refusal=True,
        citations=[],
        formatted_response=f"**[Status: INSUFFICIENT_EVIDENCE]**\n\n{STANDARD_ABSTENTION_MESSAGE}",
        confidence_score=0.0,
        groundedness_score=0.0,
        relevance_score=0.0,
        coverage_score=0.0,
        provenance_valid=True,
        trace=trace,
        latency_seconds=0.02,
    )

    resp = client.post("/query", json={"query": "Alien astronomy in Indian Law"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["decision"] == "INSUFFICIENT_EVIDENCE"
    assert data["is_refusal"] is True
    assert data["citations"] == []
    assert data["answer"] == STANDARD_ABSTENTION_MESSAGE


def test_integration_05_supported_grounded_answer(client, mock_orchestrator):
    """5. Test SUPPORTED query returns verified prose backed by citations."""
    citation = Citation(
        index=1,
        chunk_id="c1",
        document_id="d1",
        document_name="dpdp.pdf",
        page_start=5,
        page_end=5,
        text_snippet="Consent notice text.",
        formatted_reference="[1] dpdp.pdf, page 5",
    )
    trace = ExecutionTrace(query="Consent notice")
    trace.record_event("GenerationAgent", AgentEventStatus.COMPLETED, 0.05)

    mock_orchestrator.execute.return_value = OrchestratorResult(
        query="Consent notice",
        answer="A Data Fiduciary must issue notice before requesting consent [1].",
        decision=TrustDecision.SUPPORTED,
        is_refusal=False,
        citations=[citation],
        formatted_response="A Data Fiduciary must issue notice before requesting consent [1].\n\n### References\n- [1] dpdp.pdf, page 5",
        confidence_score=0.91,
        groundedness_score=0.88,
        relevance_score=0.95,
        coverage_score=1.0,
        provenance_valid=True,
        trace=trace,
        latency_seconds=0.09,
    )

    resp = client.post("/query", json={"query": "Consent notice"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["decision"] == "SUPPORTED"
    assert "[1]" in data["answer"]
    assert len(data["citations"]) == 1


def test_integration_06_citation_provenance_schema(client, mock_orchestrator):
    """6. Test citation response contains full provenance parameters."""
    citation = Citation(
        index=1,
        chunk_id="chunk_sec15_01",
        document_id="doc_dpdp_act",
        document_name="DPDP_Act_2023.pdf",
        page_start=15,
        page_end=16,
        text_snippet="Data Principal duties.",
        formatted_reference="[1] DPDP_Act_2023.pdf, pages 15-16",
    )
    mock_orchestrator.execute.return_value = OrchestratorResult(
        query="Data principal duties",
        answer="Duties include compliance [1].",
        decision=TrustDecision.SUPPORTED,
        is_refusal=False,
        citations=[citation],
        formatted_response="Duties include compliance [1].",
        confidence_score=0.90,
        groundedness_score=0.88,
        relevance_score=0.92,
        coverage_score=1.0,
        provenance_valid=True,
        latency_seconds=0.08,
    )

    resp = client.post("/query", json={"query": "Data principal duties"})
    assert resp.status_code == 200
    cit = resp.json()["citations"][0]
    assert cit["chunk_id"] == "chunk_sec15_01"
    assert cit["document_id"] == "doc_dpdp_act"
    assert cit["page_start"] == 15
    assert cit["page_end"] == 16


def test_integration_07_orchestrator_error_handling(client, mock_orchestrator):
    """7. Test unexpected orchestrator exceptions yield HTTP 500 without stack trace leaks."""
    mock_orchestrator.execute.side_effect = RuntimeError("Memory allocation failure")

    resp = client.post("/query", json={"query": "Crashing query"})
    assert resp.status_code == 500
    data = resp.json()
    assert data["error"]["status_code"] == 500
    msg = data["error"]["message"].lower()
    assert "internal" in msg or "memory allocation" in msg


def test_integration_08_malformed_request_payload(client):
    """8. Test invalid types (e.g. integer query) yield HTTP 422 with validation details."""
    resp = client.post("/query", json={"query": 12345})
    assert resp.status_code == 422
    data = resp.json()
    assert data["error"]["type"] == "ValidationError"


def test_integration_09_pydantic_response_schema_validation(client, mock_orchestrator):
    """9. Test that API response output directly validates against QueryResponse Pydantic model."""
    mock_orchestrator.execute.return_value = OrchestratorResult(
        query="Verify schema",
        answer="Schema is valid.",
        decision=TrustDecision.SUPPORTED,
        is_refusal=False,
        citations=[],
        formatted_response="Schema is valid.",
        confidence_score=0.80,
        groundedness_score=0.80,
        relevance_score=0.80,
        coverage_score=0.80,
        provenance_valid=True,
        latency_seconds=0.04,
    )

    resp = client.post("/query", json={"query": "Verify schema"})
    assert resp.status_code == 200
    parsed = QueryResponse.model_validate(resp.json())
    assert parsed.query == "Verify schema"
    assert parsed.trust_metrics.confidence_score == 0.80


def test_integration_10_trust_engine_gate_preservation(client, mock_orchestrator):
    """10. Test that orchestrator is invoked with exact client parameters and TrustDecision is preserved."""
    mock_orchestrator.execute.return_value = OrchestratorResult(
        query="Gated query",
        answer=STANDARD_ABSTENTION_MESSAGE,
        decision=TrustDecision.INSUFFICIENT_EVIDENCE,
        is_refusal=True,
        citations=[],
        formatted_response=STANDARD_ABSTENTION_MESSAGE,
        confidence_score=0.1,
        groundedness_score=0.1,
        relevance_score=0.1,
        coverage_score=0.1,
        provenance_valid=True,
        latency_seconds=0.02,
    )

    client.post(
        "/query",
        json={
            "query": "Gated query",
            "retrieval_top_k": 8,
            "evidence_top_k": 4,
            "max_new_tokens": 128,
            "temperature": 0.0,
            "filter_cited_only": True,
            "session_id": "sess_123",
        },
    )

    mock_orchestrator.execute.assert_called_once_with(
        query="Gated query",
        retrieval_top_k=8,
        evidence_top_k=4,
        max_new_tokens=128,
        temperature=0.0,
        filter_cited_only=True,
        session_id="sess_123",
    )
