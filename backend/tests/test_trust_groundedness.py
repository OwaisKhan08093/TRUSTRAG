"""Unit tests for Evidence Groundedness scoring (Phase 5 Milestone 5)."""

import pytest

from backend.app.reranking.schema import RerankedChunk
from backend.app.trust.evidence import TrustEvidence
from backend.app.trust.groundedness import (
    GroundednessAssessment,
    GroundednessScoringError,
    calculate_groundedness,
)


def test_calculate_groundedness_high_quality_evidence():
    """Verify strong relevance, full coverage, and valid provenance yields high groundedness."""
    query = "Data Fiduciary notice obligations"
    evidence = [
        TrustEvidence(
            chunk_id="chunk_1",
            document_id="doc_1",
            document_name="dpdp.pdf",
            page_start=1,
            page_end=1,
            text="The Data Fiduciary must issue notice of obligations before processing personal data.",
            retrieval_rank=1,
            retrieval_score=0.1,
            rerank_score=4.0,  # High neural logit -> sigmoid ~ 0.982
        )
    ]

    result = calculate_groundedness(query, evidence)
    assert isinstance(result, GroundednessAssessment)
    assert result.relevance_score > 0.9
    assert result.coverage_score == 1.0
    assert result.provenance_score == 1.0
    assert result.groundedness_score > 0.9
    assert 0.0 <= result.groundedness_score <= 1.0


def test_calculate_groundedness_empty_evidence():
    """Verify empty evidence sequence yields groundedness score 0.0."""
    result = calculate_groundedness("some query", [])
    assert result.groundedness_score == 0.0
    assert result.relevance_score == 0.0
    assert result.coverage_score == 0.0
    assert result.evidence_count == 0


def test_calculate_groundedness_empty_query():
    """Verify empty query raises error or yields 0.0."""
    result = calculate_groundedness("", [
        TrustEvidence(
            chunk_id="c1",
            document_id="d1",
            document_name="d.pdf",
            page_start=1,
            page_end=1,
            text="Some text",
            retrieval_rank=1,
            retrieval_score=0.1,
            rerank_score=1.0,
        )
    ])
    assert result.groundedness_score == 0.0


def test_calculate_groundedness_unrelated_evidence():
    """Verify low relevance and zero coverage yields low groundedness."""
    query = "biometric processing consent requirements"
    evidence = [
        {
            "chunk_id": "c1",
            "document_id": "d1",
            "document_name": "astronomy.pdf",
            "page_start=1": 1,
            "page_start": 1,
            "page_end": 1,
            "text": "The interstellar medium consists of cosmic dust and gas clouds.",
            "rerank_score": -4.0,  # Very low logit -> sigmoid ~ 0.018
        }
    ]
    result = calculate_groundedness(query, evidence)
    assert result.coverage_score == 0.0
    assert result.relevance_score < 0.05
    assert result.groundedness_score < 0.20


def test_calculate_groundedness_weight_customization():
    """Verify custom weights are correctly normalized and applied."""
    query = "notice requirement"
    evidence = [
        TrustEvidence(
            chunk_id="c1",
            document_id="d1",
            document_name="doc.pdf",
            page_start=1,
            page_end=1,
            text="notice requirement details",
            retrieval_rank=1,
            retrieval_score=0.1,
            rerank_score=0.0,  # sigmoid(0.0) = 0.5
        )
    ]
    # Equal weighting: 1/3 each
    result = calculate_groundedness(
        query,
        evidence,
        weight_relevance=1.0,
        weight_coverage=1.0,
        weight_provenance=1.0,
    )
    # relevance ~ 0.5, coverage = 1.0, provenance = 1.0
    expected = (0.5 + 1.0 + 1.0) / 3.0
    assert pytest.approx(result.groundedness_score, rel=1e-2) == expected
    assert pytest.approx(result.weights["relevance"], rel=1e-3) == 1.0 / 3.0


def test_calculate_groundedness_invalid_weights():
    """Verify negative or non-finite weights raise GroundednessScoringError."""
    with pytest.raises(GroundednessScoringError, match="must be a non-negative finite float"):
        calculate_groundedness("query", [], weight_relevance=-0.5)

    with pytest.raises(GroundednessScoringError, match="Sum of weights must be strictly positive"):
        calculate_groundedness("query", [], weight_relevance=0.0, weight_coverage=0.0, weight_provenance=0.0)


def test_groundedness_assessment_to_dict():
    """Verify serialization to dictionary."""
    result = calculate_groundedness("query", [])
    d = result.to_dict()
    assert "groundedness_score" in d
    assert "relevance_details" in d
    assert "coverage_details" in d
    assert "provenance_details" in d
