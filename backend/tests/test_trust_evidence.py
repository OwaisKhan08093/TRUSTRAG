"""Unit tests for TrustEvidence quality model (Phase 5 Milestone 1)."""

import math
import pytest

from backend.app.reranking.schema import RerankedChunk
from backend.app.trust.evidence import EvidenceValidationError, TrustEvidence


def test_trust_evidence_valid_creation():
    """Verify TrustEvidence instantiates properly with valid inputs."""
    ev = TrustEvidence(
        chunk_id="chunk_1",
        document_id="doc_1",
        document_name="doc.pdf",
        page_start=1,
        page_end=2,
        text="Sample evidence text content.",
        retrieval_rank=1,
        retrieval_score=0.033,
        rerank_score=2.85,
    )
    assert ev.chunk_id == "chunk_1"
    assert ev.document_id == "doc_1"
    assert ev.document_name == "doc.pdf"
    assert ev.page_start == 1
    assert ev.page_end == 2
    assert ev.text == "Sample evidence text content."
    assert ev.retrieval_rank == 1
    assert ev.retrieval_score == 0.033
    assert ev.rerank_score == 2.85


def test_trust_evidence_to_and_from_dict():
    """Verify serialization and deserialization via to_dict and from_dict."""
    ev = TrustEvidence(
        chunk_id="chunk_1",
        document_id="doc_1",
        document_name="doc.pdf",
        page_start=3,
        page_end=3,
        text="Sample text.",
        retrieval_rank=2,
        retrieval_score=0.016,
        rerank_score=1.5,
    )
    d = ev.to_dict()
    assert d["chunk_id"] == "chunk_1"
    assert d["page_start"] == 3
    assert d["page_end"] == 3
    assert d["rerank_score"] == 1.5

    reconstructed = TrustEvidence.from_dict(d)
    assert reconstructed == ev


def test_trust_evidence_from_reranked_chunk():
    """Verify construction from RerankedChunk instance."""
    chunk = RerankedChunk(
        chunk_id="c1",
        document_id="d1",
        document_name="doc.pdf",
        page_start=1,
        page_end=1,
        text="Reranked passage text.",
        original_rank=3,
        original_score=0.02,
        rerank_score=3.14,
        final_rank=1,
    )
    ev = TrustEvidence.from_reranked_chunk(chunk)
    assert ev.chunk_id == "c1"
    assert ev.document_id == "d1"
    assert ev.retrieval_rank == 3
    assert ev.retrieval_score == 0.02
    assert ev.rerank_score == 3.14
    assert ev.text == "Reranked passage text."


def test_trust_evidence_validation_empty_strings():
    """Verify validation rejects empty strings in IDs and text."""
    with pytest.raises(EvidenceValidationError, match="chunk_id must be a non-empty string"):
        TrustEvidence(
            chunk_id="  ",
            document_id="doc_1",
            document_name="doc.pdf",
            page_start=1,
            page_end=1,
            text="text",
            retrieval_rank=1,
            retrieval_score=0.5,
            rerank_score=1.0,
        )

    with pytest.raises(EvidenceValidationError, match="text must be a non-empty string"):
        TrustEvidence(
            chunk_id="c1",
            document_id="doc_1",
            document_name="doc.pdf",
            page_start=1,
            page_end=1,
            text="",
            retrieval_rank=1,
            retrieval_score=0.5,
            rerank_score=1.0,
        )


def test_trust_evidence_validation_page_numbers():
    """Verify validation rejects invalid page numbers."""
    with pytest.raises(EvidenceValidationError, match="page_start must be a positive integer >= 1"):
        TrustEvidence(
            chunk_id="c1",
            document_id="doc_1",
            document_name="doc.pdf",
            page_start=0,
            page_end=1,
            text="text",
            retrieval_rank=1,
            retrieval_score=0.5,
            rerank_score=1.0,
        )

    with pytest.raises(EvidenceValidationError, match="page_end .* must be an integer >= page_start"):
        TrustEvidence(
            chunk_id="c1",
            document_id="doc_1",
            document_name="doc.pdf",
            page_start=5,
            page_end=4,
            text="text",
            retrieval_rank=1,
            retrieval_score=0.5,
            rerank_score=1.0,
        )


def test_trust_evidence_validation_scores():
    """Verify validation rejects non-finite scores."""
    with pytest.raises(EvidenceValidationError, match="rerank_score must be a finite float"):
        TrustEvidence(
            chunk_id="c1",
            document_id="doc_1",
            document_name="doc.pdf",
            page_start=1,
            page_end=1,
            text="text",
            retrieval_rank=1,
            retrieval_score=0.5,
            rerank_score=float("nan"),
        )

    with pytest.raises(EvidenceValidationError, match="retrieval_score must be a finite float"):
        TrustEvidence(
            chunk_id="c1",
            document_id="doc_1",
            document_name="doc.pdf",
            page_start=1,
            page_end=1,
            text="text",
            retrieval_rank=1,
            retrieval_score=float("inf"),
            rerank_score=1.0,
        )


def test_trust_evidence_from_dict_missing_key():
    """Verify from_dict raises EvidenceValidationError when a key is absent."""
    with pytest.raises(EvidenceValidationError, match="Missing required key"):
        TrustEvidence.from_dict({"chunk_id": "c1"})
