"""Unit tests for Evidence Provenance Integrity validation (Phase 5 Milestone 4)."""

import pytest

from backend.app.reranking.schema import RerankedChunk
from backend.app.trust.evidence import TrustEvidence
from backend.app.trust.provenance import (
    ProvenanceIssue,
    ProvenanceReport,
    validate_evidence_provenance,
    validate_single_evidence_provenance,
)


def test_validate_provenance_valid_trust_evidence():
    """Verify clean TrustEvidence items pass provenance checks."""
    evidence = [
        TrustEvidence(
            chunk_id="chunk_001",
            document_id="doc_dpdp",
            document_name="dpdp_act_2023.pdf",
            page_start=1,
            page_end=2,
            text="Obligations of the Data Fiduciary.",
            retrieval_rank=1,
            retrieval_score=0.15,
            rerank_score=2.8,
        ),
        TrustEvidence(
            chunk_id="chunk_002",
            document_id="doc_dpdp",
            document_name="dpdp_act_2023.pdf",
            page_start=3,
            page_end=3,
            text="Rights of the Data Principal.",
            retrieval_rank=2,
            retrieval_score=0.12,
            rerank_score=1.9,
        ),
    ]

    report = validate_evidence_provenance(evidence)
    assert report.is_valid is True
    assert report.total_chunks == 2
    assert report.valid_chunks_count == 2
    assert report.invalid_chunks_count == 0
    assert len(report.issues) == 0


def test_validate_provenance_empty_list():
    """Verify empty evidence sequence is valid with zero counts."""
    report = validate_evidence_provenance([])
    assert report.is_valid is True
    assert report.total_chunks == 0
    assert report.valid_chunks_count == 0
    assert report.invalid_chunks_count == 0
    assert len(report.issues) == 0


def test_validate_provenance_invalid_page_range():
    """Verify page_end < page_start or page_start < 1 triggers issue."""
    bad_item = {
        "chunk_id": "c1",
        "document_id": "d1",
        "document_name": "doc.pdf",
        "page_start": 5,
        "page_end": 2,  # page_end < page_start
        "text": "Some text",
    }
    report = validate_evidence_provenance([bad_item])
    assert report.is_valid is False
    assert report.invalid_chunks_count == 1
    assert any(i.issue_type == "invalid_page_range" for i in report.issues)


def test_validate_provenance_duplicate_chunk_ids():
    """Verify duplicate chunk IDs across evidence items are caught."""
    item1 = {
        "chunk_id": "dup_chunk",
        "document_id": "d1",
        "document_name": "doc1.pdf",
        "page_start": 1,
        "page_end": 1,
        "text": "Text 1",
    }
    item2 = {
        "chunk_id": "dup_chunk",
        "document_id": "d1",
        "document_name": "doc1.pdf",
        "page_start": 2,
        "page_end": 2,
        "text": "Text 2",
    }
    report = validate_evidence_provenance([item1, item2])
    assert report.is_valid is False
    assert report.total_chunks == 2
    assert report.valid_chunks_count == 1
    assert report.invalid_chunks_count == 1
    assert any(i.issue_type == "duplicate_chunk_id" for i in report.issues)


def test_validate_provenance_missing_or_empty_metadata():
    """Verify missing document_id, document_name, chunk_id, or text triggers issues."""
    bad_items = [
        {
            "chunk_id": "",
            "document_id": "d1",
            "document_name": "doc.pdf",
            "page_start": 1,
            "page_end": 1,
            "text": "Text",
        },
        {
            "chunk_id": "c2",
            "document_id": "   ",
            "document_name": "doc.pdf",
            "page_start": 1,
            "page_end": 1,
            "text": "Text",
        },
        {
            "chunk_id": "c3",
            "document_id": "d3",
            "document_name": "",
            "page_start": 1,
            "page_end": 1,
            "text": "Text",
        },
        {
            "chunk_id": "c4",
            "document_id": "d4",
            "document_name": "doc.pdf",
            "page_start": 1,
            "page_end": 1,
            "text": "   ",
        },
    ]
    report = validate_evidence_provenance(bad_items)
    assert report.is_valid is False
    assert report.total_chunks == 4
    assert report.invalid_chunks_count == 4
    issue_types = [i.issue_type for i in report.issues]
    assert "invalid_chunk_id" in issue_types
    assert "invalid_document_id" in issue_types
    assert "invalid_document_name" in issue_types
    assert "empty_text" in issue_types


def test_validate_provenance_unsupported_type():
    """Verify non-sequence and invalid item types."""
    with pytest.raises(TypeError, match="evidence must be a sequence"):
        validate_evidence_provenance(12345)  # type: ignore

    report = validate_evidence_provenance([12345])  # type: ignore
    assert report.is_valid is False
    assert any(i.issue_type == "invalid_type" for i in report.issues)


def test_provenance_report_and_issue_to_dict():
    """Verify serialization to dictionary."""
    issue = ProvenanceIssue(
        chunk_index=0,
        chunk_id="c1",
        issue_type="invalid_page_start",
        message="Invalid page",
    )
    d_issue = issue.to_dict()
    assert d_issue["chunk_index"] == 0
    assert d_issue["issue_type"] == "invalid_page_start"

    report = ProvenanceReport(
        is_valid=False,
        total_chunks=1,
        valid_chunks_count=0,
        invalid_chunks_count=1,
        issues=[issue],
    )
    d_report = report.to_dict()
    assert d_report["is_valid"] is False
    assert len(d_report["issues"]) == 1
