"""Serialization and verification tests for trust metrics, provenance, and citations in API responses (Phase 8 Milestone 7)."""

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


def test_api_preserves_multiple_citations_provenance(client, mock_orchestrator):
    """Verify API exposes all citation fields with precise provenance mapping."""
    citations = [
        Citation(
            index=1,
            chunk_id="chunk_sec15_01",
            document_id="doc_dpdp_act",
            document_name="DPDP_Act_2023.pdf",
            page_start=15,
            page_end=15,
            text_snippet="Data Principal duties snippet.",
            formatted_reference="[1] DPDP_Act_2023.pdf, page 15",
        ),
        Citation(
            index=2,
            chunk_id="chunk_sec33_01",
            document_id="doc_dpdp_act",
            document_name="DPDP_Act_2023.pdf",
            page_start=28,
            page_end=28,
            text_snippet="Penalty schedule snippet.",
            formatted_reference="[2] DPDP_Act_2023.pdf, page 28",
        ),
    ]

    trace = ExecutionTrace(query="What are duties and penalties?")
    trace.record_event("RetrievalAgent", AgentEventStatus.COMPLETED, 0.02)
    trace.record_event("EvidenceAgent", AgentEventStatus.COMPLETED, 0.02)
    trace.record_event("TrustAgent", AgentEventStatus.COMPLETED, 0.01)
    trace.record_event("GenerationAgent", AgentEventStatus.COMPLETED, 0.08)
    trace.record_event("CitationAgent", AgentEventStatus.COMPLETED, 0.01)

    mock_result = OrchestratorResult(
        query="What are duties and penalties?",
        answer="Data Principal has duties [1] and faces penalties [2].",
        decision=TrustDecision.SUPPORTED,
        is_refusal=False,
        citations=citations,
        formatted_response="Data Principal has duties [1] and faces penalties [2].\n\n### References\n- [1] DPDP_Act_2023.pdf, page 15\n- [2] DPDP_Act_2023.pdf, page 28",
        confidence_score=0.945,
        groundedness_score=0.920,
        relevance_score=0.960,
        coverage_score=1.0,
        provenance_valid=True,
        trace=trace,
        latency_seconds=0.14,
        metadata={"gating_status": "SUPPORTED_AND_GENERATED"},
    )
    mock_orchestrator.execute.return_value = mock_result

    response = client.post("/query", json={"query": "What are duties and penalties?"})
    assert response.status_code == 200
    data = response.json()

    # Verify trust metrics
    metrics = data["trust_metrics"]
    assert metrics["confidence_score"] == 0.945
    assert metrics["groundedness_score"] == 0.920
    assert metrics["relevance_score"] == 0.960
    assert metrics["coverage_score"] == 1.0
    assert metrics["provenance_valid"] is True

    # Verify citations & provenance
    assert len(data["citations"]) == 2
    c1 = data["citations"][0]
    assert c1["index"] == 1
    assert c1["chunk_id"] == "chunk_sec15_01"
    assert c1["document_name"] == "DPDP_Act_2023.pdf"
    assert c1["page_start"] == 15
    assert c1["page_end"] == 15
    assert c1["formatted_reference"] == "[1] DPDP_Act_2023.pdf, page 15"

    c2 = data["citations"][1]
    assert c2["index"] == 2
    assert c2["chunk_id"] == "chunk_sec33_01"
    assert c2["document_name"] == "DPDP_Act_2023.pdf"
    assert c2["page_start"] == 28
    assert c2["page_end"] == 28

    # Verify formatted response and trace events
    assert "### References" in data["formatted_response"]
    assert len(data["trace_events"]) == 5
    assert data["trace_events"][3]["agent_name"] == "GenerationAgent"
    assert data["trace_events"][3]["status"] == "COMPLETED"


def test_api_preserves_refusal_and_skipped_trace(client, mock_orchestrator):
    """Verify refusal response exposes INSUFFICIENT_EVIDENCE status, empty citations, and SKIPPED trace events."""
    trace = ExecutionTrace(query="Explain ungrounded topic")
    trace.record_event("RetrievalAgent", AgentEventStatus.COMPLETED, 0.01)
    trace.record_event("EvidenceAgent", AgentEventStatus.COMPLETED, 0.01)
    trace.record_event("TrustAgent", AgentEventStatus.COMPLETED, 0.01)
    trace.record_event("GenerationAgent", AgentEventStatus.SKIPPED, 0.0, reason="Gated by TrustEngine: INSUFFICIENT_EVIDENCE")
    trace.record_event("CitationAgent", AgentEventStatus.SKIPPED, 0.0, reason="Gated by TrustEngine: INSUFFICIENT_EVIDENCE")

    mock_result = OrchestratorResult(
        query="Explain ungrounded topic",
        answer=STANDARD_ABSTENTION_MESSAGE,
        decision=TrustDecision.INSUFFICIENT_EVIDENCE,
        is_refusal=True,
        citations=[],
        formatted_response=f"**[Status: INSUFFICIENT_EVIDENCE]**\n\n{STANDARD_ABSTENTION_MESSAGE}",
        confidence_score=0.12,
        groundedness_score=0.08,
        relevance_score=0.15,
        coverage_score=0.0,
        provenance_valid=True,
        trace=trace,
        latency_seconds=0.03,
        metadata={"gating_status": "GATED_REFUSAL"},
    )
    mock_orchestrator.execute.return_value = mock_result

    response = client.post("/query", json={"query": "Explain ungrounded topic"})
    assert response.status_code == 200
    data = response.json()

    assert data["decision"] == "INSUFFICIENT_EVIDENCE"
    assert data["is_refusal"] is True
    assert data["answer"] == STANDARD_ABSTENTION_MESSAGE
    assert data["citations"] == []
    assert "**[Status: INSUFFICIENT_EVIDENCE]**" in data["formatted_response"]

    # Verify trace events reflect skipping downstream generation
    event_map = {e["agent_name"]: e for e in data["trace_events"]}
    assert event_map["GenerationAgent"]["status"] == "SKIPPED"
    assert "INSUFFICIENT_EVIDENCE" in event_map["GenerationAgent"]["reason"]
    assert event_map["CitationAgent"]["status"] == "SKIPPED"
    assert "INSUFFICIENT_EVIDENCE" in event_map["CitationAgent"]["reason"]
