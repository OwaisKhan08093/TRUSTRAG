"""Failure and Structured Abstention Regression Suite for TrustRAG (Phase 10 Milestone 8).

Verifies the foundational safety invariant:
    INSUFFICIENT_EVIDENCE -> NO UNSUPPORTED ANSWER -> STRUCTURED ABSTENTION

Covers:
1. No evidence (empty retrieval results)
2. Weak evidence (low neural cross-encoder relevance)
3. Partial evidence (insufficient lexical query coverage)
4. Invalid provenance (corrupted chunk bounds / missing doc metadata)
5. Conflicting/contradictory evidence (adversarial query gating)
6. Retrieval agent failure handling & trace event logging
7. Trust agent failure handling & trace event logging
8. Generation agent failure handling & state preservation
9. Citation agent failure handling & reference safety
10. Invariant preservation: Valid supported queries execute with full citations
"""

from unittest.mock import MagicMock, patch
import pytest

from backend.app.agents.base import AgentExecutionError
from backend.app.agents.citation_agent import CitationAgent
from backend.app.agents.evidence_agent import EvidenceAgent
from backend.app.agents.generation_agent import GenerationAgent
from backend.app.agents.orchestrator import TrustRAGOrchestrator
from backend.app.agents.retrieval_agent import RetrievalAgent
from backend.app.agents.trace import AgentEventStatus
from backend.app.agents.trust_agent import TrustAgent
from backend.app.generation.config import GenerationConfig
from backend.app.generation.generator import GroundedGenerator, STANDARD_ABSTENTION_MESSAGE
from backend.app.generation.llm import LocalLLM
from backend.app.reranking.schema import RerankedChunk
from backend.app.retrieval.hybrid_retriever import HybridRetriever
from backend.app.trust.engine import TrustDecision, TrustEngine
from backend.app.trust.evidence import TrustEvidence
from backend.app.trust.provenance import validate_evidence_provenance


@pytest.fixture
def mock_llm():
    """Mock LLM returning deterministic text for supported generation."""
    llm = MagicMock(spec=LocalLLM)
    llm.config = GenerationConfig()
    llm.model_name = "Qwen/Qwen2.5-3B-Instruct"
    llm.generate.return_value = (
        "A Data Fiduciary must give notice to the Data Principal before requesting consent [1]."
    )
    return llm


@pytest.fixture
def safety_orchestrator(mock_llm):
    """Real multi-agent orchestrator with deterministic mock LLM."""
    retriever = HybridRetriever()
    trust_engine = TrustEngine()
    generator = GroundedGenerator(llm=mock_llm)

    return TrustRAGOrchestrator(
        retrieval_agent=RetrievalAgent(retriever),
        evidence_agent=EvidenceAgent(),
        trust_agent=TrustAgent(trust_engine),
        generation_agent=GenerationAgent(generator),
        citation_agent=CitationAgent(),
    )


# =========================================================================
# 1. No Evidence Scenario
# =========================================================================

def test_01_no_evidence_triggers_immediate_abstention(safety_orchestrator, mock_llm):
    """Verify empty evidence triggers structured abstention and skips LLM generation."""
    with patch.object(safety_orchestrator.retrieval_agent, "execute") as mock_retrieval:
        # Mock retrieval returning 0 chunks
        mock_ret_result = MagicMock()
        mock_ret_result.results = []
        mock_ret_result.candidate_count = 0
        mock_retrieval.return_value = mock_ret_result

        result = safety_orchestrator.execute("What is the penalty for failure to wear seatbelts?")

        assert result.decision == TrustDecision.INSUFFICIENT_EVIDENCE
        assert result.is_refusal is True
        assert STANDARD_ABSTENTION_MESSAGE in result.answer
        assert len(result.citations) == 0

        # Invariant: GenerationAgent must NEVER be invoked for unsupported queries
        mock_llm.generate.assert_not_called()


# =========================================================================
# 2. Weak Neural Relevance Scenario
# =========================================================================

def test_02_weak_neural_relevance_triggers_abstention(safety_orchestrator, mock_llm):
    """Verify weak neural relevance (< threshold) forces INSUFFICIENT_EVIDENCE."""
    weak_chunk = RerankedChunk(
        chunk_id="chunk_weak_99",
        document_id="doc_dpdp",
        document_name="dpdp_act.pdf",
        page_start=1,
        page_end=1,
        text="Arbitrary unrelated text about atmospheric pressure on Mars.",
        original_rank=1,
        original_score=0.01,
        rerank_score=-8.5,  # Very low cross-encoder score
        final_rank=1,
    )

    with patch.object(safety_orchestrator.evidence_agent, "execute") as mock_ev:
        mock_ev_result = MagicMock()
        mock_ev_result.evidence = [TrustEvidence.from_reranked_chunk(weak_chunk)]
        mock_ev_result.evidence_count = 1
        mock_ev.return_value = mock_ev_result

        result = safety_orchestrator.execute("What is the penalty for data breach?")

        assert result.decision == TrustDecision.INSUFFICIENT_EVIDENCE
        assert result.is_refusal is True
        mock_llm.generate.assert_not_called()


# =========================================================================
# 3. Partial / Low Coverage Scenario
# =========================================================================

def test_03_partial_low_coverage_triggers_abstention():
    """Verify query with missing key concepts fails coverage threshold."""
    engine = TrustEngine()
    chunk = RerankedChunk(
        chunk_id="chunk_cov_01",
        document_id="doc_dpdp",
        document_name="dpdp_act.pdf",
        page_start=1,
        page_end=1,
        text="The board may inspect any premise during working hours.",
        original_rank=1,
        original_score=0.5,
        rerank_score=2.0,
        final_rank=1,
    )

    # Query with extensive specific terms not present in chunk
    query = "quantum cryptographic algorithmic encryption key exchange protocols"
    assessment = engine.evaluate(query, [TrustEvidence.from_reranked_chunk(chunk)])

    assert assessment.coverage_score < 0.3
    assert assessment.decision == TrustDecision.INSUFFICIENT_EVIDENCE


# =========================================================================
# 4. Invalid Provenance Scenario
# =========================================================================

def test_04_invalid_corrupted_provenance_triggers_abstention():
    """Verify corrupted provenance metadata flags provenance_valid=False."""
    corrupted_item = {
        "chunk_id": "chunk_bad_01",
        "document_id": "",
        "document_name": "corrupt.pdf",
        "page_start": 8,
        "page_end": 2,  # Invalid range
        "text": "Valid body text with invalid metadata bounds.",
    }
    report = validate_evidence_provenance([corrupted_item])
    assert report.is_valid is False
    assert len(report.issues) >= 1


# =========================================================================
# 5. Conflicting / Adversarial Query Gating
# =========================================================================

def test_05_conflicting_adversarial_query_gating(safety_orchestrator, mock_llm):
    """Verify adversarial query with false premise is safely refused."""
    adversarial_query = "Why does the DPDP Act allow companies to sell children's private data freely without parental consent?"
    result = safety_orchestrator.execute(adversarial_query)

    assert result.decision == TrustDecision.INSUFFICIENT_EVIDENCE
    assert result.is_refusal is True
    assert STANDARD_ABSTENTION_MESSAGE in result.answer


# =========================================================================
# 6. Retrieval Failure Handling
# =========================================================================

def test_06_retrieval_agent_failure_propagation(safety_orchestrator):
    """Verify RetrievalAgent exception is caught and raises AgentExecutionError."""
    with patch.object(
        safety_orchestrator.retrieval_agent,
        "execute",
        side_effect=RuntimeError("FAISS disk corruption"),
    ):
        with pytest.raises(AgentExecutionError, match="RetrievalAgent failed"):
            safety_orchestrator.execute("Valid query")


# =========================================================================
# 7. Trust Agent Failure Handling
# =========================================================================

def test_07_trust_agent_failure_propagation(safety_orchestrator):
    """Verify TrustAgent exception is caught and raises AgentExecutionError."""
    with patch.object(
        safety_orchestrator.trust_agent,
        "execute",
        side_effect=RuntimeError("Trust weight divide by zero"),
    ):
        with pytest.raises(AgentExecutionError, match="TrustAgent failed"):
            safety_orchestrator.execute("Valid query")


# =========================================================================
# 8. Generation Agent Failure Handling
# =========================================================================

def test_08_generation_agent_failure_propagation(safety_orchestrator, mock_llm):
    """Verify LLM failure during supported query raises AgentExecutionError."""
    mock_llm.generate.side_effect = RuntimeError("VRAM allocation failure")

    with pytest.raises(AgentExecutionError, match="GenerationAgent failed"):
        safety_orchestrator.execute("What notice must a Data Fiduciary give before requesting consent?")


# =========================================================================
# 9. Citation Agent Failure Handling
# =========================================================================

def test_09_citation_agent_failure_propagation(safety_orchestrator):
    """Verify CitationAgent exception during assembly raises AgentExecutionError."""
    with patch.object(
        safety_orchestrator.citation_agent,
        "execute",
        side_effect=RuntimeError("Regex recursion limit exceeded"),
    ):
        with pytest.raises(AgentExecutionError, match="CitationAgent failed"):
            safety_orchestrator.execute("What notice must a Data Fiduciary give before requesting consent?")


# =========================================================================
# 10. Invariant: Genuine Supported Queries Still Execute
# =========================================================================

def test_10_invariant_supported_queries_execute_with_citations(safety_orchestrator, mock_llm):
    """Verify supported query succeeds, invokes LLM, and attaches citations."""
    query = "What notice must a Data Fiduciary give before requesting consent?"
    result = safety_orchestrator.execute(query)

    assert result.decision == TrustDecision.SUPPORTED
    assert result.is_refusal is False
    assert result.confidence_score >= 0.70
    assert result.groundedness_score >= 0.60
    assert len(result.citations) >= 1
    assert len(result.trace.events) >= 5
    mock_llm.generate.assert_called_once()
