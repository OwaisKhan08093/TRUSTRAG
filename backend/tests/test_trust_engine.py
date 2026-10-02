"""Unit and integration tests for TrustEngine (Phase 5 Milestone 7)."""

import pytest

from backend.app.reranking.schema import RerankedChunk
from backend.app.trust.engine import (
    TrustAssessment,
    TrustDecision,
    TrustEngine,
    TrustEngineError,
)
from backend.app.trust.evidence import TrustEvidence


@pytest.fixture
def sample_valid_evidence() -> list:
    """Fixture providing high-quality grounded evidence."""
    return [
        TrustEvidence(
            chunk_id="chunk_01",
            document_id="doc_dpdp",
            document_name="dpdp_act.pdf",
            page_start=5,
            page_end=5,
            text="The Data Fiduciary shall give notice to the Data Principal before requesting consent.",
            retrieval_rank=1,
            retrieval_score=0.15,
            rerank_score=3.8,  # high relevance
        ),
        TrustEvidence(
            chunk_id="chunk_02",
            document_id="doc_dpdp",
            document_name="dpdp_act.pdf",
            page_start=6,
            page_end=6,
            text="Notice must contain details of personal data to be processed and purpose.",
            retrieval_rank=2,
            retrieval_score=0.10,
            rerank_score=2.5,
        ),
    ]


def test_trust_engine_supported_decision(sample_valid_evidence):
    """Verify high-quality relevant evidence produces a SUPPORTED decision."""
    engine = TrustEngine()
    query = "Data Fiduciary notice requirements for consent"
    assessment = engine.evaluate(query, sample_valid_evidence)

    assert isinstance(assessment, TrustAssessment)
    assert assessment.decision == TrustDecision.SUPPORTED
    assert assessment.provenance_valid is True
    assert assessment.evidence_count == 2
    assert assessment.coverage_score > 0.5
    assert assessment.groundedness_score > 0.6
    assert assessment.confidence_score > 0.6
    assert "Evidence meets all grounding" in assessment.decision_reasons[0]


def test_trust_engine_empty_evidence():
    """Verify empty evidence results in INSUFFICIENT_EVIDENCE."""
    engine = TrustEngine()
    assessment = engine.evaluate("Any query", [])

    assert assessment.decision == TrustDecision.INSUFFICIENT_EVIDENCE
    assert assessment.evidence_count == 0
    assert assessment.confidence_score == 0.0
    assert "No evidence chunks were provided." in assessment.decision_reasons


def test_trust_engine_empty_query(sample_valid_evidence):
    """Verify empty query string results in INSUFFICIENT_EVIDENCE."""
    engine = TrustEngine()
    assessment = engine.evaluate("   ", sample_valid_evidence)

    assert assessment.decision == TrustDecision.INSUFFICIENT_EVIDENCE
    assert "Query is empty" in assessment.decision_reasons[0]


def test_trust_engine_unrelated_evidence():
    """Verify evidence with low relevance/coverage results in INSUFFICIENT_EVIDENCE."""
    engine = TrustEngine()
    unrelated_evidence = [
        TrustEvidence(
            chunk_id="chunk_astro",
            document_id="doc_astro",
            document_name="astronomy.pdf",
            page_start=1,
            page_end=1,
            text="Galaxies are massive systems of stars and dark matter.",
            retrieval_rank=1,
            retrieval_score=0.01,
            rerank_score=-4.0,  # Negative logit -> very low probability
        )
    ]
    query = "penalties imposed by Data Protection Board"
    assessment = engine.evaluate(query, unrelated_evidence)

    assert assessment.decision == TrustDecision.INSUFFICIENT_EVIDENCE
    assert assessment.coverage_score == 0.0
    assert assessment.groundedness_score < 0.3


def test_trust_engine_provenance_failure_gate():
    """Verify provenance corruption triggers INSUFFICIENT_EVIDENCE when required."""
    engine = TrustEngine(require_valid_provenance=True)
    corrupt_evidence = [
        {
            "chunk_id": "c1",
            "document_id": "d1",
            "document_name": "doc.pdf",
            "page_start": 5,
            "page_end": 2,  # Invalid range
            "text": "Data Fiduciary notice requirements for consent.",
            "rerank_score": 4.0,
        }
    ]
    assessment = engine.evaluate("Data Fiduciary notice", corrupt_evidence)
    assert assessment.decision == TrustDecision.INSUFFICIENT_EVIDENCE
    assert assessment.provenance_valid is False
    assert any("Provenance validation failed" in r for r in assessment.decision_reasons)


def test_trust_engine_threshold_configuration():
    """Verify custom strict thresholds reject borderline evidence."""
    strict_engine = TrustEngine(
        min_confidence_threshold=0.99,
        min_groundedness_threshold=0.99,
    )
    evidence = [
        TrustEvidence(
            chunk_id="c1",
            document_id="d1",
            document_name="d.pdf",
            page_start=1,
            page_end=1,
            text="General notice obligations.",
            retrieval_rank=1,
            retrieval_score=0.1,
            rerank_score=1.0,  # Moderate score
        )
    ]
    assessment = strict_engine.evaluate("notice obligations", evidence)
    assert assessment.decision == TrustDecision.INSUFFICIENT_EVIDENCE


def test_trust_engine_validation_errors():
    """Verify parameter type/value errors."""
    with pytest.raises(TrustEngineError, match="must be a float bounded in"):
        TrustEngine(min_confidence_threshold=1.5)

    with pytest.raises(TrustEngineError, match="min_evidence_count must be an integer >= 1"):
        TrustEngine(min_evidence_count=0)

    engine = TrustEngine()
    with pytest.raises(TypeError, match="query must be a string"):
        engine.evaluate(1234, [])  # type: ignore

    with pytest.raises(TypeError, match="evidence must be a sequence"):
        engine.evaluate("query", "not a list")  # type: ignore


def test_trust_assessment_to_dict(sample_valid_evidence):
    """Verify complete dictionary serialization."""
    engine = TrustEngine()
    assessment = engine.evaluate("Data Fiduciary notice", sample_valid_evidence)
    d = assessment.to_dict()

    assert d["decision"] == "SUPPORTED"
    assert "relevance_score" in d
    assert "coverage_score" in d
    assert "groundedness_score" in d
    assert "confidence_score" in d
    assert "decision_reasons" in d
    assert "relevance_details" in d
    assert "coverage_details" in d
    assert "groundedness_details" in d
    assert "confidence_details" in d
    assert "provenance_details" in d
