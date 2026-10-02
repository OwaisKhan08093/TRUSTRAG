"""Unit tests for Citation Reference System (Phase 6 Milestone 6)."""

import pytest

from backend.app.generation.citations import (
    Citation,
    CitationError,
    build_citation,
    build_citation_references,
    extract_cited_indices,
    format_citation_reference_string,
    format_citations_markdown,
)
from backend.app.trust.evidence import TrustEvidence


def test_build_citation_single_evidence():
    """Verify clean citation creation from a single TrustEvidence chunk."""
    ev = TrustEvidence(
        chunk_id="chunk_dpdp_01",
        document_id="doc_dpdp",
        document_name="DPDP_Act_2023.pdf",
        page_start=5,
        page_end=5,
        text="Section 5 notice obligations for Data Fiduciary.",
        retrieval_rank=1,
        retrieval_score=0.15,
        rerank_score=3.5,
    )

    cit = build_citation(index=1, item=ev)
    assert isinstance(cit, Citation)
    assert cit.index == 1
    assert cit.chunk_id == "chunk_dpdp_01"
    assert cit.document_name == "DPDP_Act_2023.pdf"
    assert cit.page_start == 5
    assert cit.page_end == 5
    assert cit.formatted_reference == "[1] DPDP_Act_2023.pdf, page 5"
    assert "Section 5 notice obligations" in cit.text_snippet


def test_build_citation_multi_page_range():
    """Verify page range formatting (e.g. pages 5-6)."""
    item = {
        "chunk_id": "c2",
        "document_id": "d1",
        "document_name": "rules.pdf",
        "page_start": 5,
        "page_end": 6,
        "text": "Multi page rule content.",
    }
    cit = build_citation(index=2, item=item)
    assert cit.formatted_reference == "[2] rules.pdf, pages 5-6"


def test_build_citation_references_deduplication():
    """Verify duplicate chunks are deduplicated by chunk_id."""
    ev1 = TrustEvidence(
        chunk_id="c1",
        document_id="d1",
        document_name="doc.pdf",
        page_start=1,
        page_end=1,
        text="Text 1",
        retrieval_rank=1,
        retrieval_score=0.1,
        rerank_score=2.0,
    )
    ev2 = TrustEvidence(
        chunk_id="c2",
        document_id="d1",
        document_name="doc.pdf",
        page_start=2,
        page_end=2,
        text="Text 2",
        retrieval_rank=2,
        retrieval_score=0.1,
        rerank_score=1.5,
    )

    # Pass duplicates of ev1
    citations = build_citation_references([ev1, ev1, ev2])
    assert len(citations) == 2
    assert citations[0].index == 1
    assert citations[0].chunk_id == "c1"
    assert citations[1].index == 2
    assert citations[1].chunk_id == "c2"


def test_build_citation_invalid_provenance():
    """Verify missing fields or inverted page numbers raise CitationError."""
    with pytest.raises(CitationError, match="Missing or invalid chunk_id"):
        build_citation(1, {"document_id": "d", "document_name": "n", "page_start": 1, "page_end": 1})

    with pytest.raises(CitationError, match="Missing or invalid document_name"):
        build_citation(1, {"chunk_id": "c", "document_id": "d", "document_name": "", "page_start": 1, "page_end": 1})

    with pytest.raises(CitationError, match="Invalid page_end"):
        build_citation(1, {"chunk_id": "c", "document_id": "d", "document_name": "n", "page_start": 5, "page_end": 2})


def test_extract_cited_indices():
    """Verify extracting bracketed citation numbers from generated text."""
    text = "The Data Fiduciary has duties [1] and rights of the principal are outlined in [2]. Also see [1]."
    indices = extract_cited_indices(text)
    assert indices == [1, 2]

    # No citations in text
    assert extract_cited_indices("Plain text with no numbers") == []


def test_format_citations_markdown():
    """Verify markdown output formatting."""
    cit = Citation(
        index=1,
        chunk_id="c1",
        document_id="d1",
        document_name="dpdp.pdf",
        page_start=3,
        page_end=3,
        text_snippet="Sample text content.",
        formatted_reference="[1] dpdp.pdf, page 3",
    )
    md = format_citations_markdown([cit])
    assert "### References" in md
    assert "- **[1] dpdp.pdf, page 3**" in md
    assert "> *\"Sample text content.\"*" in md


def test_citation_to_dict():
    """Verify dictionary serialization."""
    cit = Citation(
        index=1,
        chunk_id="c1",
        document_id="d1",
        document_name="doc.pdf",
        page_start=1,
        page_end=1,
        text_snippet="snippet",
        formatted_reference="[1] doc.pdf, page 1",
    )
    d = cit.to_dict()
    assert d["index"] == 1
    assert d["chunk_id"] == "c1"
    assert d["formatted_reference"] == "[1] doc.pdf, page 1"
