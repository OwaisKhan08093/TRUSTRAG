"""Unit tests for the deterministic, metadata-preserving chunker."""

import pytest
from backend.app.ingestion.pdf_loader import PageRecord
from backend.app.ingestion.chunker import chunk_document, chunk_multiple_documents


@pytest.fixture
def multi_page_records() -> list[PageRecord]:
    """Create simulated multi-page document records."""
    # Page 1: 50 words
    page1_text = " ".join([f"word_p1_{i}" for i in range(1, 51)])
    # Page 2: 50 words
    page2_text = " ".join([f"word_p2_{i}" for i in range(1, 51)])
    # Page 3: 50 words
    page3_text = " ".join([f"word_p3_{i}" for i in range(1, 51)])

    return [
        {"document_id": "test_act", "document_name": "test_act.pdf", "page_number": 1, "text": page1_text},
        {"document_id": "test_act", "document_name": "test_act.pdf", "page_number": 2, "text": page2_text},
        {"document_id": "test_act", "document_name": "test_act.pdf", "page_number": 3, "text": page3_text},
    ]


def test_chunker_creates_chunks(multi_page_records: list[PageRecord]):
    """Test 7: Chunker creates chunks from page records."""
    # Total words = 150. Chunk size = 60, overlap = 20 -> step = 40
    chunks = chunk_document(multi_page_records, chunk_size=60, chunk_overlap=20)
    assert len(chunks) > 0


def test_chunk_overlap_works(multi_page_records: list[PageRecord]):
    """Test 8: Chunk overlap works deterministically."""
    chunks = chunk_document(multi_page_records, chunk_size=60, chunk_overlap=20)
    assert len(chunks) >= 2

    # Words of chunk 0 and chunk 1
    words_c0 = chunks[0]["text"].split()
    words_c1 = chunks[1]["text"].split()

    # The last 20 words of chunk 0 should match the first 20 words of chunk 1
    overlap_from_c0 = words_c0[-20:]
    overlap_from_c1 = words_c1[:20]
    assert overlap_from_c0 == overlap_from_c1


def test_chunk_metadata_schema(multi_page_records: list[PageRecord]):
    """Test 9: Every chunk contains all required metadata fields."""
    chunks = chunk_document(multi_page_records, chunk_size=60, chunk_overlap=20)
    required_keys = {"chunk_id", "document_id", "document_name", "page_start", "page_end", "text"}

    for chunk in chunks:
        assert required_keys.issubset(chunk.keys())
        assert chunk["document_id"] == "test_act"
        assert chunk["document_name"] == "test_act.pdf"
        assert isinstance(chunk["page_start"], int)
        assert isinstance(chunk["page_end"], int)
        assert chunk["page_start"] <= chunk["page_end"]
        assert chunk["chunk_id"].startswith("test_act_page")


def test_no_chunk_has_empty_text(multi_page_records: list[PageRecord]):
    """Test 10: No emitted chunk has empty or whitespace-only text."""
    # Add an empty page record in the middle
    records_with_empty = multi_page_records + [
        {"document_id": "test_act", "document_name": "test_act.pdf", "page_number": 4, "text": "   "}
    ]
    chunks = chunk_document(records_with_empty, chunk_size=60, chunk_overlap=20)

    for chunk in chunks:
        assert len(chunk["text"].strip()) > 0


def test_page_spanning_chunk_preserves_page_start_and_end(multi_page_records: list[PageRecord]):
    """Verify that chunks spanning across page boundaries report correct page_start and page_end."""
    # Chunk size of 80 words will span from page 1 (50 words) into page 2 (30 words)
    chunks = chunk_document(multi_page_records, chunk_size=80, chunk_overlap=0)

    assert chunks[0]["page_start"] == 1
    assert chunks[0]["page_end"] == 2


def test_multiple_documents_never_mixed():
    """Verify pages from distinct documents are never merged into a single chunk."""
    doc1_pages: list[PageRecord] = [
        {"document_id": "doc1", "document_name": "doc1.pdf", "page_number": 1, "text": "doc1 word1 word2 word3"}
    ]
    doc2_pages: list[PageRecord] = [
        {"document_id": "doc2", "document_name": "doc2.pdf", "page_number": 1, "text": "doc2 wordA wordB wordC"}
    ]

    all_chunks = chunk_multiple_documents(doc1_pages + doc2_pages, chunk_size=50, chunk_overlap=10)
    assert len(all_chunks) == 2
    assert all_chunks[0]["document_id"] == "doc1"
    assert all_chunks[1]["document_id"] == "doc2"
    assert "wordA" not in all_chunks[0]["text"]
    assert "word1" not in all_chunks[1]["text"]
