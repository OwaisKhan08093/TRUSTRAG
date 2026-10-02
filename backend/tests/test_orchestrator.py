"""Unit tests for TrustRAGOrchestrator (Phase 7 Milestone 7)."""

from unittest.mock import MagicMock
import pytest

from backend.app.agents.base import AgentExecutionError, AgentInputError, AgentResult
from backend.app.agents.citation_agent import CitationAgent, CitationAgentResult
from backend.app.agents.evidence_agent import EvidenceAgent, EvidenceAgentResult
from backend.app.agents.generation_agent import GenerationAgent, GenerationAgentResult
from backend.app.agents.orchestrator import OrchestratorResult, TrustRAGOrchestrator
from backend.app.agents.retrieval_agent import RetrievalAgent, RetrievalAgentResult
from backend.app.agents.trust_agent import TrustAgent, TrustAgentResult
from backend.app.generation.citations import Citation
from backend.app.generation.generator import STANDARD_ABSTENTION_MESSAGE
from backend.app.generation.models import GenerationResult
from backend.app.generation.response import GroundedAnswer
from backend.app.reranking.schema import RerankedChunk
from backend.app.trust.engine import TrustDecision, TrustEngine
from backend.app.trust.evidence import TrustEvidence


@pytest.fixture
def mock_subagents():
    """Fixture providing mock specialized agents."""
    retrieval_agent = MagicMock(spec=RetrievalAgent)
    evidence_agent = MagicMock(spec=EvidenceAgent)
    trust_agent = MagicMock(spec=TrustAgent)
    generation_agent = MagicMock(spec=GenerationAgent)
    citation_agent = MagicMock(spec=CitationAgent)

    return {
        "retrieval": retrieval_agent,
        "evidence": evidence_agent,
        "trust": trust_agent,
        "generation": generation_agent,
        "citation": citation_agent,
    }


def test_orchestrator_supported_flow(mock_subagents):
    """Verify complete pipeline execution when evidence is SUPPORTED."""
    chunk = RerankedChunk(
        chunk_id="chunk_01",
        document_id="doc_dpdp",
        document_name="dpdp_act.pdf",
        page_start=5,
        page_end=5,
        text="Notice is required prior to processing personal data.",
        original_rank=1,
        original_score=0.15,
        rerank_score=3.8,
        final_rank=1,
    )

    mock_subagents["retrieval"].execute.return_value = RetrievalAgentResult(
        query="notice rules",
        results=[{"chunk_id": "chunk_01", "text": "Notice is required"}],
        candidate_count=1,
        top_k=5,
    )

    mock_subagents["evidence"].execute.return_value = EvidenceAgentResult(
        query="notice rules",
        evidence=[chunk],
        evidence_count=1,
        top_k=3,
    )

    engine = TrustEngine()
    assessment = engine.evaluate("notice rules", [chunk])

    mock_subagents["trust"].execute.return_value = TrustAgentResult(
        query="notice rules",
        decision=TrustDecision.SUPPORTED,
        is_supported=True,
        assessment=assessment,
        confidence_score=0.88,
        groundedness_score=0.90,
        relevance_score=0.92,
        coverage_score=1.0,
        provenance_valid=True,
        decision_reasons=["All criteria met"],
    )

    gen_res = GenerationResult(
        query="notice rules",
        answer="Notice is required before data processing [1].",
        model_name="qwen-test",
        evidence_ids=["chunk_01"],
        is_refusal=False,
    )

    mock_subagents["generation"].execute.return_value = GenerationAgentResult(
        query="notice rules",
        answer_text="Notice is required before data processing [1].",
        is_refusal=False,
        is_supported=True,
        generation_result=gen_res,
    )

    cit = Citation(
        index=1,
        chunk_id="chunk_01",
        document_id="doc_dpdp",
        document_name="dpdp_act.pdf",
        page_start=5,
        page_end=5,
        text_snippet="Notice is required",
        formatted_reference="[1] dpdp_act.pdf, page 5",
    )

    mock_subagents["citation"].execute.return_value = CitationAgentResult(
        query="notice rules",
        citations=[cit],
        citation_count=1,
        formatted_citations="### References\n- **[1] dpdp_act.pdf, page 5**",
    )

    orchestrator = TrustRAGOrchestrator(
        retrieval_agent=mock_subagents["retrieval"],
        evidence_agent=mock_subagents["evidence"],
        trust_agent=mock_subagents["trust"],
        generation_agent=mock_subagents["generation"],
        citation_agent=mock_subagents["citation"],
    )

    assert orchestrator.name == "TrustRAGOrchestrator"

    result = orchestrator.execute("notice rules")

    assert isinstance(result, OrchestratorResult)
    assert result.decision == TrustDecision.SUPPORTED
    assert result.is_refusal is False
    assert len(result.citations) == 1
    assert "Notice is required before data processing" in result.answer
    assert mock_subagents["retrieval"].execute.called
    assert mock_subagents["evidence"].execute.called
    assert mock_subagents["trust"].execute.called
    assert mock_subagents["generation"].execute.called
    assert mock_subagents["citation"].execute.called


def test_orchestrator_insufficient_evidence_refusal(mock_subagents):
    """Verify gating strictly prevents GenerationAgent and CitationAgent calls on INSUFFICIENT_EVIDENCE."""
    mock_subagents["retrieval"].execute.return_value = RetrievalAgentResult(
        query="unsupported query",
        results=[],
        candidate_count=0,
        top_k=5,
    )

    mock_subagents["evidence"].execute.return_value = EvidenceAgentResult(
        query="unsupported query",
        evidence=[],
        evidence_count=0,
        top_k=3,
    )

    engine = TrustEngine()
    assessment = engine.evaluate("unsupported query", [])

    mock_subagents["trust"].execute.return_value = TrustAgentResult(
        query="unsupported query",
        decision=TrustDecision.INSUFFICIENT_EVIDENCE,
        is_supported=False,
        assessment=assessment,
        confidence_score=0.0,
        groundedness_score=0.0,
        relevance_score=0.0,
        coverage_score=0.0,
        provenance_valid=True,
        decision_reasons=["Empty evidence"],
    )

    orchestrator = TrustRAGOrchestrator(
        retrieval_agent=mock_subagents["retrieval"],
        evidence_agent=mock_subagents["evidence"],
        trust_agent=mock_subagents["trust"],
        generation_agent=mock_subagents["generation"],
        citation_agent=mock_subagents["citation"],
    )

    result = orchestrator.execute("unsupported query")

    assert isinstance(result, OrchestratorResult)
    assert result.decision == TrustDecision.INSUFFICIENT_EVIDENCE
    assert result.is_refusal is True
    assert result.answer == STANDARD_ABSTENTION_MESSAGE
    assert result.citations == []
    assert result.generation_result is None
    assert result.citation_result is None

    # CRITICAL INVARIANT: GenerationAgent and CitationAgent MUST NOT BE CALLED
    assert not mock_subagents["generation"].execute.called
    assert not mock_subagents["citation"].execute.called


def test_orchestrator_run_wrapper(mock_subagents):
    """Verify BaseAgent.run wrapper captures orchestrator pipeline."""
    mock_subagents["retrieval"].execute.return_value = RetrievalAgentResult(
        query="test query", results=[], candidate_count=0, top_k=5
    )
    mock_subagents["evidence"].execute.return_value = EvidenceAgentResult(
        query="test query", evidence=[], evidence_count=0, top_k=3
    )
    engine = TrustEngine()
    assessment = engine.evaluate("test query", [])
    mock_subagents["trust"].execute.return_value = TrustAgentResult(
        query="test query",
        decision=TrustDecision.INSUFFICIENT_EVIDENCE,
        is_supported=False,
        assessment=assessment,
        confidence_score=0.0,
        groundedness_score=0.0,
        relevance_score=0.0,
        coverage_score=0.0,
        provenance_valid=True,
        decision_reasons=["Empty"],
    )

    orchestrator = TrustRAGOrchestrator(
        retrieval_agent=mock_subagents["retrieval"],
        evidence_agent=mock_subagents["evidence"],
        trust_agent=mock_subagents["trust"],
        generation_agent=mock_subagents["generation"],
        citation_agent=mock_subagents["citation"],
    )

    res = orchestrator.run("test query")
    assert isinstance(res, AgentResult)
    assert res.success is True
    assert res.agent_name == "TrustRAGOrchestrator"
    assert isinstance(res.data, OrchestratorResult)


def test_orchestrator_input_validation(mock_subagents):
    """Verify input validation handles empty or malformed queries."""
    orchestrator = TrustRAGOrchestrator(
        retrieval_agent=mock_subagents["retrieval"],
        evidence_agent=mock_subagents["evidence"],
        trust_agent=mock_subagents["trust"],
        generation_agent=mock_subagents["generation"],
        citation_agent=mock_subagents["citation"],
    )

    with pytest.raises(AgentInputError, match="query must be a string"):
        orchestrator.execute(123)  # type: ignore

    with pytest.raises(AgentInputError, match="query cannot be empty"):
        orchestrator.execute("   ")


def test_orchestrator_subagent_error_handling(mock_subagents):
    """Verify subagent runtime failures are wrapped in AgentExecutionError."""
    mock_subagents["retrieval"].execute.side_effect = RuntimeError("FAISS error")

    orchestrator = TrustRAGOrchestrator(
        retrieval_agent=mock_subagents["retrieval"],
        evidence_agent=mock_subagents["evidence"],
        trust_agent=mock_subagents["trust"],
        generation_agent=mock_subagents["generation"],
        citation_agent=mock_subagents["citation"],
    )

    with pytest.raises(AgentExecutionError, match="RetrievalAgent failed"):
        orchestrator.execute("valid query")
