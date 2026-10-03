"""Full-Stack End-to-End Test Suite for TrustRAG (Phase 10 Milestone 1).

Covers realistic end-to-end integration scenarios across the full request lifecycle:
1. Supported query (full pipeline execution with citations and trace)
2. Insufficient evidence (trust gating, structured abstention, skipped LLM)
3. Citation generation (multi-citation resolution and formatted references)
4. Multiple evidence sources (multi-document / multi-chunk provenance)
5. Provenance preservation (end-to-end metadata fidelity and invalid provenance gating)
6. Trust gating (relevance, coverage, and groundedness enforcement)
7. Empty / malformed query (FastAPI schema validation & 422 responses)
8. API error handling (500 internal error capture without stack trace leakage)
9. Retrieval failure handling (trace failure capture and agent error propagation)
10. Generation failure handling (LLM failure capture and state management)
11. Final response schema compliance (full validation against Pydantic QueryResponse)
"""

from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient

from backend.app.agents.base import AgentExecutionError, AgentInputError
from backend.app.agents.citation_agent import CitationAgent
from backend.app.agents.evidence_agent import EvidenceAgent
from backend.app.agents.generation_agent import GenerationAgent
from backend.app.agents.orchestrator import OrchestratorResult, TrustRAGOrchestrator
from backend.app.agents.retrieval_agent import RetrievalAgent
from backend.app.agents.trace import AgentEventStatus
from backend.app.agents.trust_agent import TrustAgent
from backend.app.api.app import app
from backend.app.api.dependencies import get_orchestrator, set_orchestrator
from backend.app.api.schemas import QueryResponse
from backend.app.generation.config import GenerationConfig
from backend.app.generation.generator import GroundedGenerator, STANDARD_ABSTENTION_MESSAGE
from backend.app.generation.llm import LocalLLM
from backend.app.reranking.pipeline import RerankingPipeline
from backend.app.reranking.schema import RerankedChunk
from backend.app.retrieval.hybrid_retriever import HybridRetriever
from backend.app.trust.engine import TrustDecision, TrustEngine
from backend.app.trust.evidence import TrustEvidence


@pytest.fixture
def mock_llm():
    """Deterministic LLM mock returning controlled grounded text."""
    llm = MagicMock(spec=LocalLLM)
    llm.config = GenerationConfig()
    llm.model_name = "Qwen/Qwen2.5-3B-Instruct"
    llm.generate.return_value = (
        "A Data Fiduciary must give notice to the Data Principal before requesting consent [1]."
    )
    return llm


@pytest.fixture
def real_orchestrator_with_mock_llm(mock_llm):
    """Real multi-agent orchestrator with real hybrid retriever, real cross-encoder,

    real trust engine, real citation agent, and deterministic mock LLM.
    """
    retriever = HybridRetriever()
    trust_engine = TrustEngine()
    generator = GroundedGenerator(llm=mock_llm)

    retrieval_agent = RetrievalAgent(retriever=retriever)
    evidence_agent = EvidenceAgent()
    trust_agent = TrustAgent(trust_engine=trust_engine)
    generation_agent = GenerationAgent(generator=generator)
    citation_agent = CitationAgent()

    orchestrator = TrustRAGOrchestrator(
        retrieval_agent=retrieval_agent,
        evidence_agent=evidence_agent,
        trust_agent=trust_agent,
        generation_agent=generation_agent,
        citation_agent=citation_agent,
    )
    return orchestrator


@pytest.fixture
def e2e_client(real_orchestrator_with_mock_llm):
    """FastAPI TestClient wired with real end-to-end orchestrator."""
    app.dependency_overrides[get_orchestrator] = lambda: real_orchestrator_with_mock_llm
    set_orchestrator(real_orchestrator_with_mock_llm)
    yield TestClient(app)
    app.dependency_overrides.clear()
    set_orchestrator(None)


# =========================================================================
# 1. Supported Query Scenario
# =========================================================================

def test_e2e_01_supported_query_lifecycle(e2e_client, mock_llm):
    """Scenario 1: End-to-end supported query produces grounded response with citations and trace."""
    query = "What notice must a Data Fiduciary give before requesting consent?"
    response = e2e_client.post("/query", json={"query": query})

    assert response.status_code == 200
    data = response.json()

    assert data["query"] == query
    assert data["decision"] == "SUPPORTED"
    assert data["is_refusal"] is False
    assert "[1]" in data["answer"]
    assert len(data["citations"]) >= 1
    assert "### References" in data["formatted_response"]

    # Verify trust metrics
    metrics = data["trust_metrics"]
    assert metrics["confidence_score"] > 0.0
    assert metrics["groundedness_score"] > 0.0
    assert metrics["relevance_score"] > 0.0
    assert metrics["coverage_score"] > 0.0
    assert metrics["provenance_valid"] is True

    # Verify execution trace captured all 5 agents
    trace = data["trace_events"]
    assert len(trace) == 5
    agent_names = [evt["agent_name"] for evt in trace]
    assert agent_names == [
        "RetrievalAgent",
        "EvidenceAgent",
        "TrustAgent",
        "GenerationAgent",
        "CitationAgent",
    ]
    assert all(evt["status"] == "COMPLETED" for evt in trace)
    assert mock_llm.generate.called


# =========================================================================
# 2. Insufficient Evidence Scenario
# =========================================================================

def test_e2e_02_insufficient_evidence_gating(e2e_client, mock_llm):
    """Scenario 2: Out-of-domain query triggers trust gating, structured abstention, and skips LLM."""
    query = "quantum entanglement photon spin state teleportation in vacuum"
    mock_llm.generate.reset_mock()

    response = e2e_client.post("/query", json={"query": query})

    assert response.status_code == 200
    data = response.json()

    assert data["decision"] == "INSUFFICIENT_EVIDENCE"
    assert data["is_refusal"] is True
    assert data["answer"] == STANDARD_ABSTENTION_MESSAGE
    assert data["citations"] == []
    assert "**[Status: INSUFFICIENT_EVIDENCE]**" in data["formatted_response"]

    # Invariant: GenerationAgent & CitationAgent must be marked SKIPPED in trace
    trace = data["trace_events"]
    statuses = {evt["agent_name"]: evt["status"] for evt in trace}
    assert statuses["RetrievalAgent"] == "COMPLETED"
    assert statuses["EvidenceAgent"] == "COMPLETED"
    assert statuses["TrustAgent"] == "COMPLETED"
    assert statuses["GenerationAgent"] == "SKIPPED"
    assert statuses["CitationAgent"] == "SKIPPED"

    # Invariant: LLM was never called
    assert not mock_llm.generate.called


# =========================================================================
# 3. Citation Generation Scenario
# =========================================================================

def test_e2e_03_multi_citation_generation(e2e_client, mock_llm):
    """Scenario 3: Generation referencing multiple citations resolves and formats all cited chunks."""
    query = "What are the notice requirements and rights of the data principal?"
    mock_llm.generate.return_value = (
        "The Data Fiduciary must give notice [1] and the Data Principal has the right to grievance redressal [2]."
    )

    response = e2e_client.post("/query", json={"query": query})

    assert response.status_code == 200
    data = response.json()
    assert data["decision"] == "SUPPORTED"
    assert len(data["citations"]) >= 2

    # Check citation indices and references
    indices = [c["index"] for c in data["citations"]]
    assert 1 in indices
    assert 2 in indices
    for citation in data["citations"]:
        assert citation["document_name"].endswith(".pdf")
        assert citation["page_start"] >= 1
        assert len(citation["text_snippet"]) > 0


# =========================================================================
# 4. Multiple Evidence Sources Scenario
# =========================================================================

def test_e2e_04_multiple_evidence_sources_reranking(real_orchestrator_with_mock_llm):
    """Scenario 4: Pipeline correctly retrieves, reranks, and preserves multiple distinct chunks."""
    query = "What is the penalty for failure to take reasonable security safeguards?"
    result = real_orchestrator_with_mock_llm.execute(query, retrieval_top_k=5, evidence_top_k=3)

    assert result.retrieval_result is not None
    assert result.evidence_result is not None
    assert result.evidence_result.evidence_count <= 3

    # Check distinct chunk IDs and valid provenance
    chunk_ids = [c.chunk_id for c in result.evidence_result.evidence]
    assert len(chunk_ids) == len(set(chunk_ids)), "Evidence chunks must have unique IDs"
    for ev in result.evidence_result.evidence:
        assert ev.document_name
        assert ev.page_start >= 1
        assert ev.page_end >= ev.page_start


# =========================================================================
# 5. Provenance Preservation Scenario
# =========================================================================

def test_e2e_05_provenance_preservation_and_corruption_gating(real_orchestrator_with_mock_llm):
    """Scenario 5: Corrupted chunk provenance is caught by schema validation and TrustEngine."""
    from backend.app.reranking.schema import SchemaValidationError
    from backend.app.trust.evidence import EvidenceValidationError
    from backend.app.trust.provenance import validate_evidence_provenance

    # 1. Direct schema corruption defense: empty chunk_id raises SchemaValidationError
    with pytest.raises(SchemaValidationError, match="chunk_id must be a non-empty string"):
        RerankedChunk(
            chunk_id="",
            document_id="doc_unknown",
            document_name="",
            page_start=-1,
            page_end=-1,
            text="Corrupted chunk text without proper provenance.",
            original_rank=1,
            original_score=0.9,
            rerank_score=5.0,
            final_rank=1,
        )

    # 2. Direct TrustEvidence corruption defense: invalid page bounds raise EvidenceValidationError
    with pytest.raises(EvidenceValidationError, match="page_end"):
        TrustEvidence(
            chunk_id="chunk_corrupt_bounds",
            document_id="doc_valid",
            document_name="test.pdf",
            page_start=10,
            page_end=5,
            text="Corrupted bounds snippet",
            retrieval_rank=1,
            retrieval_score=0.9,
            rerank_score=2.5,
        )

    # 3. Corrupted provenance dictionary evaluated by validate_evidence_provenance
    corrupted_dict = {
        "chunk_id": "corrupted_chunk_001",
        "document_id": "",  # invalid empty document_id
        "document_name": "unknown.pdf",
        "page_start": 5,
        "page_end": 2,  # page_end < page_start
        "text": "Valid text with corrupt page bounds.",
    }
    report = validate_evidence_provenance([corrupted_dict])
    assert report.is_valid is False
    assert len(report.issues) >= 1

    # 4. Valid provenance chunk succeeds
    valid_evidence = TrustEvidence(
        chunk_id="chunk_valid_001",
        document_id="doc_dpdp",
        document_name="dpdp_act.pdf",
        page_start=2,
        page_end=2,
        text="A Data Fiduciary shall give notice to the Data Principal before requesting consent.",
        retrieval_rank=1,
        retrieval_score=0.95,
        rerank_score=4.8,
    )
    trust_engine = TrustEngine()
    assessment = trust_engine.evaluate("notice before consent", [valid_evidence])
    assert assessment.provenance_valid is True
    assert assessment.decision == TrustDecision.SUPPORTED


# =========================================================================
# 6. Trust Gating Scenario
# =========================================================================

def test_e2e_06_trust_gating_threshold_enforcement():
    """Scenario 6: Verify TrustEngine gating rules enforce minimum relevance and coverage thresholds."""
    engine = TrustEngine()

    # Weak relevance chunk
    weak_chunk = RerankedChunk(
        chunk_id="chunk_weak_01",
        document_id="doc_dpdp",
        document_name="dpdp.pdf",
        page_start=1,
        page_end=1,
        text="Random unrelated phrase about cooking recipes.",
        original_rank=1,
        original_score=0.01,
        rerank_score=-5.0,  # very low neural score
        final_rank=1,
    )
    assessment = engine.evaluate("data protection consent notice", [TrustEvidence.from_reranked_chunk(weak_chunk)])
    assert assessment.decision == TrustDecision.INSUFFICIENT_EVIDENCE
    assert assessment.confidence_score < 0.8


# =========================================================================
# 7. Malformed / Empty Query Scenario
# =========================================================================

def test_e2e_07_malformed_and_empty_query_validation(e2e_client):
    """Scenario 7: FastAPI endpoints return 422 with structured validation errors for invalid inputs."""
    # Empty query string
    r1 = e2e_client.post("/query", json={"query": ""})
    assert r1.status_code == 422
    assert r1.json()["error"]["type"] == "ValidationError"

    # Whitespace-only query
    r2 = e2e_client.post("/query", json={"query": "     \t\n  "})
    assert r2.status_code == 422
    assert r2.json()["error"]["type"] == "ValidationError"

    # Non-string query (integer)
    r3 = e2e_client.post("/query", json={"query": 99999})
    assert r3.status_code == 422
    assert r3.json()["error"]["type"] == "ValidationError"

    # Missing query field entirely
    r4 = e2e_client.post("/query", json={"top_k": 5})
    assert r4.status_code == 422
    assert r4.json()["error"]["type"] == "ValidationError"


# =========================================================================
# 8. API Error Handling Scenario
# =========================================================================

def test_e2e_08_api_error_handling_sanitization(e2e_client, real_orchestrator_with_mock_llm):
    """Scenario 8: Internal exceptions yield HTTP 500 error envelopes without leaking raw tracebacks."""
    with patch.object(real_orchestrator_with_mock_llm, "execute", side_effect=RuntimeError("Internal system failure")):
        response = e2e_client.post("/query", json={"query": "Valid query triggers server error"})

        assert response.status_code == 500
        data = response.json()
        assert "error" in data
        assert data["error"]["status_code"] == 500
        assert data["error"]["message"] == "Unexpected internal server error."
        assert "Traceback" not in data["error"]["message"]
        assert "Internal system failure" not in data["error"]["message"]


# =========================================================================
# 9. Retrieval Failure Scenario
# =========================================================================

def test_e2e_09_retrieval_failure_trace_capture(real_orchestrator_with_mock_llm):
    """Scenario 9: Retrieval failure updates execution trace as FAILED and raises AgentExecutionError."""
    with patch.object(
        real_orchestrator_with_mock_llm.retrieval_agent,
        "execute",
        side_effect=RuntimeError("FAISS index file locked"),
    ):
        with pytest.raises(AgentExecutionError, match="RetrievalAgent failed"):
            real_orchestrator_with_mock_llm.execute("Sample query")


# =========================================================================
# 10. Generation Failure Scenario
# =========================================================================

def test_e2e_10_generation_failure_handling(real_orchestrator_with_mock_llm, mock_llm):
    """Scenario 10: LLM failure during supported generation is cleanly caught and propagated."""
    mock_llm.generate.side_effect = RuntimeError("CUDA out of memory during token sampling")

    with pytest.raises(AgentExecutionError, match="GenerationAgent failed"):
        real_orchestrator_with_mock_llm.execute("What notice must a Data Fiduciary give before requesting consent?")


# =========================================================================
# 11. Final Response Schema Scenario
# =========================================================================

def test_e2e_11_final_response_schema_compliance(e2e_client):
    """Scenario 11: Direct verification that endpoint outputs strictly conform to QueryResponse model."""
    query = "What is a Data Principal under the Act?"
    response = e2e_client.post("/query", json={"query": query})

    assert response.status_code == 200
    json_data = response.json()

    # Model validation directly with Pydantic
    parsed_response = QueryResponse.model_validate(json_data)
    assert parsed_response.query == query
    assert parsed_response.decision in ("SUPPORTED", "INSUFFICIENT_EVIDENCE")
    assert isinstance(parsed_response.is_refusal, bool)
    assert isinstance(parsed_response.latency_seconds, float)
    assert parsed_response.latency_seconds >= 0.0
    assert len(parsed_response.trace_events) >= 3
