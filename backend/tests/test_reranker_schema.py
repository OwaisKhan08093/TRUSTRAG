"""Unit tests for RerankedChunk result schema and provenance preservation."""

import pytest

from backend.app.reranking.schema import RerankedChunk, SchemaValidationError


def test_reranked_chunk_instantiation():
    """Verify RerankedChunk initializes and preserves all required provenance fields."""
    chunk = RerankedChunk(
        chunk_id="c1",
        document_id="doc1",
        document_name="doc1.pdf",
        page_start=1,
        page_end=2,
        text="Sample text content.",
        original_rank=3,
        original_score=0.032,
        rerank_score=4.85,
        final_rank=1,
    )
    assert chunk.chunk_id == "c1"
    assert chunk.document_id == "doc1"
    assert chunk.document_name == "doc1.pdf"
    assert chunk.page_start == 1
    assert chunk.page_end == 2
    assert chunk.text == "Sample text content."
    assert chunk.original_rank == 3
    assert chunk.original_score == 0.032
    assert chunk.rerank_score == 4.85
    assert chunk.final_rank == 1


def test_reranked_chunk_to_and_from_dict():
    """Verify serialization to dict and round-trip deserialization."""
    original = RerankedChunk(
        chunk_id="c2",
        document_id="doc2",
        document_name="doc2.pdf",
        page_start=5,
        page_end=5,
        text="Important text.",
        original_rank=1,
        original_score=0.95,
        rerank_score=7.12,
        final_rank=1,
    )
    d = original.to_dict()
    assert isinstance(d, dict)
    assert d["chunk_id"] == "c2"
    assert d["original_score"] == 0.95
    assert d["rerank_score"] == 7.12

    restored = RerankedChunk.from_dict(d)
    assert restored == original


def test_reranked_chunk_from_retrieval_result():
    """Verify construction from standard retrieval dictionary."""
    retrieval_output = {
        "rank": 2,
        "chunk_id": "c10",
        "score": 0.81,
        "document_id": "act_2023",
        "document_name": "act.pdf",
        "page_start": 2,
        "page_end": 3,
        "text": "Legal section text.",
    }
    chunk = RerankedChunk.from_retrieval_result(
        result=retrieval_output,
        rerank_score=5.5,
        final_rank=1,
    )
    assert chunk.chunk_id == "c10"
    assert chunk.original_rank == 2
    assert chunk.original_score == 0.81
    assert chunk.rerank_score == 5.5
    assert chunk.final_rank == 1
    assert chunk.document_name == "act.pdf"


def test_reranked_chunk_validation_errors():
    """Verify SchemaValidationError is raised for invalid values."""
    valid_args = {
        "chunk_id": "c1",
        "document_id": "doc1",
        "document_name": "doc1.pdf",
        "page_start": 1,
        "page_end": 1,
        "text": "Some text",
        "original_rank": 1,
        "original_score": 0.5,
        "rerank_score": 1.0,
        "final_rank": 1,
    }

    # Empty chunk_id
    with pytest.raises(SchemaValidationError, match="chunk_id must be a non-empty string"):
        RerankedChunk(**{**valid_args, "chunk_id": ""})

    # Empty text
    with pytest.raises(SchemaValidationError, match="text must be a non-empty string"):
        RerankedChunk(**{**valid_args, "text": "   "})

    # Invalid page bounds (page_end < page_start)
    with pytest.raises(SchemaValidationError, match="page_end .* must be an integer >= page_start"):
        RerankedChunk(**{**valid_args, "page_start": 5, "page_end": 2})

    # Non-positive original_rank
    with pytest.raises(SchemaValidationError, match="original_rank must be a positive integer"):
        RerankedChunk(**{**valid_args, "original_rank": 0})

    # Non-positive final_rank
    with pytest.raises(SchemaValidationError, match="final_rank must be a positive integer"):
        RerankedChunk(**{**valid_args, "final_rank": -1})

    # Missing keys in from_dict
    with pytest.raises(SchemaValidationError, match="Missing required key"):
        RerankedChunk.from_dict({"chunk_id": "c1"})
