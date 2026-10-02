"""Unit tests for Evidence Coverage evaluation (Phase 5 Milestone 3)."""

import pytest

from backend.app.reranking.schema import RerankedChunk
from backend.app.trust.coverage import (
    CoverageAssessment,
    CoverageScoringError,
    calculate_evidence_coverage,
    extract_query_terms,
)
from backend.app.trust.evidence import TrustEvidence


def test_extract_query_terms_normal():
    """Verify term extraction strips stopwords, punctuation, and deduplicates."""
    terms = extract_query_terms("What is the definition of Data Fiduciary and Data Principal?")
    assert "definition" in terms
    assert "data" in terms
    assert "fiduciary" in terms
    assert "principal" in terms
    # Stop words like "what", "is", "the", "of", "and" should be removed
    assert "what" not in terms
    assert "is" not in terms
    # "data" should only appear once
    assert terms.count("data") == 1


def test_extract_query_terms_stopwords_only_fallback():
    """Verify term extraction keeps terms if query consists entirely of stop words."""
    terms = extract_query_terms("who what where when why")
    assert len(terms) > 0
    assert "who" in terms or "what" in terms


def test_calculate_evidence_coverage_high_coverage():
    """Verify 100% coverage when all query terms appear across evidence passages."""
    query = "Data Fiduciary notice obligations"
    evidence = [
        TrustEvidence(
            chunk_id="c1",
            document_id="d1",
            document_name="doc.pdf",
            page_start=1,
            page_end=1,
            text="The Data Fiduciary must fulfill all notice obligations prior to processing.",
            retrieval_rank=1,
            retrieval_score=0.05,
            rerank_score=3.0,
        )
    ]
    assessment = calculate_evidence_coverage(query, evidence)
    assert assessment.coverage_ratio == 1.0
    assert len(assessment.uncovered_terms) == 0
    assert set(assessment.covered_terms) == set(assessment.query_terms)
    assert assessment.chunk_coverage_ratios == [1.0]
    assert assessment.evidence_count == 1


def test_calculate_evidence_coverage_partial_coverage():
    """Verify partial coverage calculation across multiple chunks."""
    query = "Data Principal right correction erasure grievance redressal"
    # Chunk 1 contains: data, principal, right, correction, erasure
    # Chunk 2 contains: grievance, redressal
    c1 = RerankedChunk(
        chunk_id="c1",
        document_id="d1",
        document_name="d.pdf",
        page_start=1,
        page_end=1,
        text="A Data Principal has the right to correction and erasure.",
        original_rank=1,
        original_score=0.1,
        rerank_score=2.5,
        final_rank=1,
    )
    c2 = RerankedChunk(
        chunk_id="c2",
        document_id="d1",
        document_name="d.pdf",
        page_start=2,
        page_end=2,
        text="Procedures for grievance redressal with the officer.",
        original_rank=2,
        original_score=0.08,
        rerank_score=1.5,
        final_rank=2,
    )

    assessment = calculate_evidence_coverage(query, [c1, c2])
    assert assessment.coverage_ratio == 1.0  # Union covers all terms
    assert assessment.chunk_coverage_ratios[0] > 0.0
    assert assessment.chunk_coverage_ratios[1] > 0.0
    assert assessment.chunk_coverage_ratios[0] < 1.0
    assert assessment.evidence_count == 2


def test_calculate_evidence_coverage_unrelated_evidence():
    """Verify 0% coverage when evidence has no query terms."""
    query = "digital personal data protection board penalties"
    evidence = [
        {"text": "Quantum mechanics and atomic structure in modern physics.", "rerank_score": -1.0}
    ]
    assessment = calculate_evidence_coverage(query, evidence)
    assert assessment.coverage_ratio == 0.0
    assert len(assessment.covered_terms) == 0
    assert len(assessment.uncovered_terms) == len(assessment.query_terms)
    assert assessment.chunk_coverage_ratios == [0.0]


def test_calculate_evidence_coverage_empty_evidence():
    """Verify empty evidence sequence returns 0 coverage safely."""
    query = "consent manager registration"
    assessment = calculate_evidence_coverage(query, [])
    assert assessment.coverage_ratio == 0.0
    assert assessment.evidence_count == 0
    assert len(assessment.covered_terms) == 0
    assert len(assessment.uncovered_terms) == len(assessment.query_terms)


def test_calculate_evidence_coverage_duplicate_evidence():
    """Verify duplicate evidence items do not distort overall union coverage."""
    query = "notice consent"
    ev = TrustEvidence(
        chunk_id="c1",
        document_id="d1",
        document_name="d.pdf",
        page_start=1,
        page_end=1,
        text="Notice and consent requirements.",
        retrieval_rank=1,
        retrieval_score=0.5,
        rerank_score=2.0,
    )
    assessment = calculate_evidence_coverage(query, [ev, ev])
    assert assessment.coverage_ratio == 1.0
    assert assessment.evidence_count == 2
    assert assessment.chunk_coverage_ratios == [1.0, 1.0]


def test_calculate_evidence_coverage_validation_errors():
    """Verify invalid inputs raise appropriate errors."""
    with pytest.raises(TypeError, match="query must be a string"):
        calculate_evidence_coverage(1234, [])  # type: ignore

    with pytest.raises(CoverageScoringError, match="query cannot be empty"):
        calculate_evidence_coverage("   ", [])

    with pytest.raises(TypeError, match="evidence must be a sequence"):
        calculate_evidence_coverage("valid query", "not a list")  # type: ignore

    with pytest.raises(CoverageScoringError, match="missing valid 'text' key"):
        calculate_evidence_coverage("query", [{"missing": "text"}])


def test_coverage_assessment_to_dict():
    """Verify CoverageAssessment.to_dict serialization."""
    assessment = CoverageAssessment(
        query_terms=["data", "fiduciary"],
        covered_terms=["data"],
        uncovered_terms=["fiduciary"],
        coverage_ratio=0.5,
        chunk_coverage_ratios=[0.5],
        evidence_count=1,
    )
    d = assessment.to_dict()
    assert d["query_terms"] == ["data", "fiduciary"]
    assert d["coverage_ratio"] == 0.5
    assert d["evidence_count"] == 1
