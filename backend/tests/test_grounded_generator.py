"""Unit tests for GroundedGenerator (Phase 6 Milestone 5)."""

from unittest.mock import MagicMock
import pytest

from backend.app.generation.config import GenerationConfig
from backend.app.generation.generator import (
    STANDARD_ABSTENTION_MESSAGE,
    GeneratorError,
    GroundedGenerator,
)
from backend.app.generation.llm import LocalLLM
from backend.app.generation.models import GenerationResult
from backend.app.trust.coverage import CoverageAssessment
from backend.app.trust.engine import TrustAssessment, TrustDecision
from backend.app.trust.evidence import TrustEvidence
from backend.app.trust.groundedness import GroundednessAssessment
from backend.app.trust.provenance import ProvenanceReport
from backend.app.trust.relevance import RelevanceAssessment


@pytest.fixture
def mock_llm():
    """Fixture providing a mocked LocalLLM."""
    mock = MagicMock(spec=LocalLLM)
    mock.config = GenerationConfig()
    mock.model_name = "Qwen/Qwen2.5-3B-Instruct"
    mock.generate.return_value = "A Data Fiduciary must give notice before processing [1]."
    return mock


@pytest.fixture
def supported_assessment():
    """Fixture providing a SUPPORTED TrustAssessment."""
    rel = RelevanceAssessment(
        top_score=3.5,
        top_probability=0.97,
        mean_probability=0.97,
        weighted_relevance=0.97,
        individual_scores=[3.5],
        individual_probabilities=[0.97],
    )
    cov = CoverageAssessment(
        query_terms=["data", "fiduciary", "notice"],
        covered_terms=["data", "fiduciary", "notice"],
        uncovered_terms=[],
        coverage_ratio=1.0,
        chunk_coverage_ratios=[1.0],
        evidence_count=1,
    )
    prov = ProvenanceReport(
        is_valid=True,
        total_chunks=1,
        valid_chunks_count=1,
        invalid_chunks_count=0,
        issues=[],
    )
    ground = GroundednessAssessment(
        groundedness_score=0.90,
        relevance_score=0.97,
        coverage_score=1.0,
        provenance_score=1.0,
        weights={"relevance": 0.55, "coverage": 0.35, "provenance": 0.10},
        evidence_count=1,
        relevance_details=rel,
        coverage_details=cov,
        provenance_details=prov,
    )
    from backend.app.trust.confidence import ConfidenceAssessment
    conf = ConfidenceAssessment(
        confidence_score=0.92,
        groundedness_score=0.90,
        top_relevance_prob=0.97,
        coverage_score=1.0,
        volume_factor=1.0,
        provenance_score=1.0,
        evidence_count=1,
        groundedness_details=ground,
    )

    return TrustAssessment(
        query="What are notice obligations for Data Fiduciary?",
        decision=TrustDecision.SUPPORTED,
        relevance_score=0.97,
        coverage_score=1.0,
        groundedness_score=0.90,
        confidence_score=0.92,
        provenance_valid=True,
        evidence_count=1,
        decision_reasons=["Evidence meets all grounding thresholds."],
        relevance_details=rel,
        coverage_details=cov,
        groundedness_details=ground,
        confidence_details=conf,
        provenance_details=prov,
    )


@pytest.fixture
def insufficient_assessment():
    """Fixture providing an INSUFFICIENT_EVIDENCE TrustAssessment."""
    rel = RelevanceAssessment(0.0, 0.0, 0.0, 0.0, [], [])
    cov = CoverageAssessment([], [], [], 0.0, [], 0)
    prov = ProvenanceReport(True, 0, 0, 0, [])
    ground = GroundednessAssessment(0.0, 0.0, 0.0, 1.0, {}, 0, rel, cov, prov)
    from backend.app.trust.confidence import ConfidenceAssessment
    conf = ConfidenceAssessment(0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0, ground)

    return TrustAssessment(
        query="Unrelated topic query",
        decision=TrustDecision.INSUFFICIENT_EVIDENCE,
        relevance_score=0.0,
        coverage_score=0.0,
        groundedness_score=0.0,
        confidence_score=0.0,
        provenance_valid=True,
        evidence_count=0,
        decision_reasons=["No evidence chunks were provided."],
        relevance_details=rel,
        coverage_details=cov,
        groundedness_details=ground,
        confidence_details=conf,
        provenance_details=prov,
    )


def test_grounded_generator_supported_path(mock_llm, supported_assessment):
    """Verify generator invokes LLM and returns answer when assessment is SUPPORTED."""
    generator = GroundedGenerator(llm=mock_llm)
    evidence = [
        TrustEvidence(
            chunk_id="chunk_01",
            document_id="doc_dpdp",
            document_name="dpdp.pdf",
            page_start=5,
            page_end=5,
            text="The Data Fiduciary must give notice before requesting consent.",
            retrieval_rank=1,
            retrieval_score=0.15,
            rerank_score=3.5,
        )
    ]

    result = generator.generate(
        query="What are notice obligations for Data Fiduciary?",
        evidence=evidence,
        assessment=supported_assessment,
    )

    assert isinstance(result, GenerationResult)
    assert result.is_refusal is False
    assert result.refusal_reason is None
    assert result.answer == "A Data Fiduciary must give notice before processing [1]."
    assert result.evidence_ids == ["chunk_01"]
    assert mock_llm.generate.called
    assert result.generation_metadata["gated_by_trust_engine"] is False


def test_grounded_generator_insufficient_evidence_gating(mock_llm, insufficient_assessment):
    """Verify generator DOES NOT invoke LLM when assessment is INSUFFICIENT_EVIDENCE."""
    generator = GroundedGenerator(llm=mock_llm)

    result = generator.generate(
        query="Unrelated topic query",
        evidence=[],
        assessment=insufficient_assessment,
    )

    assert isinstance(result, GenerationResult)
    assert result.is_refusal is True
    assert result.answer == STANDARD_ABSTENTION_MESSAGE
    assert "INSUFFICIENT_EVIDENCE" in result.refusal_reason
    assert result.evidence_ids == []
    # CRITICAL: Verify LLM was NOT called
    assert not mock_llm.generate.called
    assert result.generation_metadata["gated_by_trust_engine"] is True


def test_grounded_generator_input_validation(mock_llm, supported_assessment):
    """Verify input argument type checking."""
    generator = GroundedGenerator(llm=mock_llm)

    with pytest.raises(TypeError, match="query must be a string"):
        generator.generate(1234, [], supported_assessment)  # type: ignore

    with pytest.raises(GeneratorError, match="query cannot be empty"):
        generator.generate("   ", [], supported_assessment)

    with pytest.raises(TypeError, match="evidence must be a sequence"):
        generator.generate("query", "not a list", supported_assessment)  # type: ignore

    with pytest.raises(TypeError, match="assessment must be an instance of TrustAssessment"):
        generator.generate("query", [], "not an assessment")  # type: ignore
