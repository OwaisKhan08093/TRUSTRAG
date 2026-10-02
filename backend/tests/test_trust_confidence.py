"""Unit tests for Evidence Confidence scoring (Phase 5 Milestone 6)."""

import pytest

from backend.app.trust.confidence import (
    ConfidenceAssessment,
    ConfidenceScoringError,
    calculate_confidence,
)
from backend.app.trust.evidence import TrustEvidence
from backend.app.trust.groundedness import calculate_groundedness


def test_calculate_confidence_strong_evidence():
    """Verify strong evidence with high relevance and coverage yields high confidence."""
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

    conf = calculate_confidence(query, evidence, min_supporting_chunks=1)
    assert isinstance(conf, ConfidenceAssessment)
    assert conf.confidence_score > 0.85
    assert conf.volume_factor == 1.0
    assert conf.provenance_score == 1.0
    assert 0.0 <= conf.confidence_score <= 1.0


def test_calculate_confidence_empty_evidence():
    """Verify empty evidence yields confidence 0.0."""
    conf = calculate_confidence("any query", [])
    assert conf.confidence_score == 0.0
    assert conf.groundedness_score == 0.0
    assert conf.evidence_count == 0


def test_calculate_confidence_empty_query():
    """Verify empty query yields confidence 0.0."""
    evidence = [
        TrustEvidence(
            chunk_id="chunk_1",
            document_id="doc_1",
            document_name="dpdp.pdf",
            page_start=1,
            page_end=1,
            text="Some text",
            retrieval_rank=1,
            retrieval_score=0.1,
            rerank_score=2.0,
        )
    ]
    conf = calculate_confidence("", evidence)
    assert conf.confidence_score == 0.0


def test_calculate_confidence_volume_scaling():
    """Verify min_supporting_chunks scales volume factor when chunks < threshold."""
    query = "Data Fiduciary notice"
    evidence = [
        TrustEvidence(
            chunk_id="chunk_1",
            document_id="doc_1",
            document_name="dpdp.pdf",
            page_start=1,
            page_end=1,
            text="The Data Fiduciary must issue notice.",
            retrieval_rank=1,
            retrieval_score=0.1,
            rerank_score=3.0,
        )
    ]
    conf_1 = calculate_confidence(query, evidence, min_supporting_chunks=1)
    conf_3 = calculate_confidence(query, evidence, min_supporting_chunks=3)

    assert conf_1.volume_factor == 1.0
    assert pytest.approx(conf_3.volume_factor, rel=1e-3) == 1.0 / 3.0
    assert conf_3.confidence_score < conf_1.confidence_score


def test_calculate_confidence_with_precomputed_groundedness():
    """Verify precomputed GroundednessAssessment can be passed in directly."""
    query = "Data Fiduciary notice obligations"
    evidence = [
        TrustEvidence(
            chunk_id="c1",
            document_id="d1",
            document_name="d.pdf",
            page_start=1,
            page_end=1,
            text="Data Fiduciary notice obligations text.",
            retrieval_rank=1,
            retrieval_score=0.1,
            rerank_score=2.5,
        )
    ]
    g_assess = calculate_groundedness(query, evidence)
    conf = calculate_confidence(query, evidence, groundedness_assessment=g_assess)
    assert conf.groundedness_score == g_assess.groundedness_score
    assert conf.confidence_score > 0.0


def test_calculate_confidence_validation_errors():
    """Verify invalid weights and parameters raise ConfidenceScoringError."""
    with pytest.raises(ConfidenceScoringError, match="min_supporting_chunks must be an integer >= 1"):
        calculate_confidence("query", [], min_supporting_chunks=0)

    with pytest.raises(ConfidenceScoringError, match="must be a non-negative finite float"):
        calculate_confidence("query", [], weight_groundedness=-1.0)

    with pytest.raises(ConfidenceScoringError, match="Sum of confidence weights must be strictly positive"):
        calculate_confidence(
            "query",
            [],
            weight_groundedness=0.0,
            weight_top_relevance=0.0,
            weight_coverage=0.0,
            weight_volume=0.0,
        )


def test_confidence_assessment_to_dict():
    """Verify dictionary serialization of assessment."""
    conf = calculate_confidence("query", [])
    d = conf.to_dict()
    assert "confidence_score" in d
    assert "groundedness_details" in d
    assert "volume_factor" in d
