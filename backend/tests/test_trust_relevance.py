"""Unit tests for Evidence Relevance Aggregation (Phase 5 Milestone 2)."""

import math
import pytest

from backend.app.reranking.schema import RerankedChunk
from backend.app.trust.evidence import TrustEvidence
from backend.app.trust.relevance import (
    RelevanceAssessment,
    RelevanceScoringError,
    aggregate_evidence_relevance,
    sigmoid,
)


def test_sigmoid_values():
    """Verify sigmoid behaves correctly across known boundary and center values."""
    assert sigmoid(0.0) == 0.5
    assert sigmoid(100.0) == pytest.approx(1.0, abs=1e-6)
    assert sigmoid(-100.0) == pytest.approx(0.0, abs=1e-6)
    assert 0.0 < sigmoid(2.0) < 1.0
    assert 0.0 < sigmoid(-2.0) < 1.0


def test_sigmoid_rejects_non_finite():
    """Verify sigmoid rejects NaN and infinite values."""
    with pytest.raises(RelevanceScoringError):
        sigmoid(float("nan"))
    with pytest.raises(RelevanceScoringError):
        sigmoid(float("inf"))


def test_aggregate_evidence_relevance_empty():
    """Verify empty evidence sequence yields safe zero assessment."""
    assessment = aggregate_evidence_relevance([])
    assert assessment.top_score == 0.0
    assert assessment.top_probability == 0.0
    assert assessment.mean_probability == 0.0
    assert assessment.weighted_relevance == 0.0
    assert assessment.individual_scores == []
    assert assessment.individual_probabilities == []


def test_aggregate_evidence_relevance_with_trust_evidence():
    """Verify relevance aggregation using TrustEvidence items."""
    ev1 = TrustEvidence(
        chunk_id="c1",
        document_id="d1",
        document_name="d.pdf",
        page_start=1,
        page_end=1,
        text="Text 1",
        retrieval_rank=1,
        retrieval_score=0.05,
        rerank_score=2.0,
    )
    ev2 = TrustEvidence(
        chunk_id="c2",
        document_id="d1",
        document_name="d.pdf",
        page_start=2,
        page_end=2,
        text="Text 2",
        retrieval_rank=2,
        retrieval_score=0.04,
        rerank_score=0.0,
    )

    assessment = aggregate_evidence_relevance([ev1, ev2], decay_factor=0.5)

    assert assessment.top_score == 2.0
    assert assessment.top_probability == pytest.approx(sigmoid(2.0), abs=1e-5)
    assert assessment.individual_scores == [2.0, 0.0]
    assert assessment.individual_probabilities == [
        pytest.approx(sigmoid(2.0), abs=1e-5),
        pytest.approx(sigmoid(0.0), abs=1e-5),
    ]
    # Weighted relevance with weights [1.0, 0.5]: (p1 * 1.0 + 0.5 * 0.5) / 1.5
    p1 = sigmoid(2.0)
    p2 = 0.5
    expected_weighted = (p1 * 1.0 + p2 * 0.5) / 1.5
    assert assessment.weighted_relevance == pytest.approx(expected_weighted, abs=1e-5)
    assert 0.0 <= assessment.weighted_relevance <= 1.0


def test_aggregate_evidence_relevance_with_reranked_chunks():
    """Verify relevance aggregation handles RerankedChunk instances."""
    c1 = RerankedChunk(
        chunk_id="c1",
        document_id="d1",
        document_name="d.pdf",
        page_start=1,
        page_end=1,
        text="text",
        original_rank=1,
        original_score=0.1,
        rerank_score=3.5,
        final_rank=1,
    )
    assessment = aggregate_evidence_relevance([c1])
    assert assessment.top_score == 3.5
    assert assessment.weighted_relevance == pytest.approx(sigmoid(3.5), abs=1e-5)


def test_aggregate_evidence_relevance_with_dicts():
    """Verify relevance aggregation handles raw dict objects."""
    d = [{"rerank_score": 1.0}]
    assessment = aggregate_evidence_relevance(d)
    assert assessment.top_score == 1.0
    assert assessment.top_probability == pytest.approx(sigmoid(1.0), abs=1e-5)


def test_aggregate_evidence_relevance_validation_errors():
    """Verify invalid inputs raise appropriate errors."""
    with pytest.raises(TypeError):
        aggregate_evidence_relevance("not a list")  # type: ignore

    with pytest.raises(RelevanceScoringError, match="decay_factor"):
        aggregate_evidence_relevance([], decay_factor=0.0)

    with pytest.raises(RelevanceScoringError, match="decay_factor"):
        aggregate_evidence_relevance([], decay_factor=1.5)

    with pytest.raises(RelevanceScoringError, match="missing 'rerank_score'"):
        aggregate_evidence_relevance([{"text": "missing score"}])

    with pytest.raises(RelevanceScoringError, match="must be a finite float"):
        aggregate_evidence_relevance([{"rerank_score": float("nan")}])

    with pytest.raises(TypeError, match="must be TrustEvidence, RerankedChunk, or dict"):
        aggregate_evidence_relevance([12345])  # type: ignore


def test_relevance_assessment_to_dict():
    """Verify RelevanceAssessment.to_dict serialization."""
    assessment = RelevanceAssessment(
        top_score=1.0,
        top_probability=0.73,
        mean_probability=0.73,
        weighted_relevance=0.73,
        individual_scores=[1.0],
        individual_probabilities=[0.73],
    )
    d = assessment.to_dict()
    assert d["top_score"] == 1.0
    assert d["top_probability"] == 0.73
    assert d["individual_scores"] == [1.0]
