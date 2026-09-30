"""Unit tests for BM25Retriever."""

import json
from pathlib import Path
import pytest

from backend.app.config import CHUNKS_OUTPUT_FILE, EMBEDDING_METADATA_FILE
from backend.app.retrieval.bm25_index import BM25Index
from backend.app.retrieval.bm25_retriever import BM25Retriever, BM25RetrieverError
from backend.app.retrieval.metadata import ChunkMetadataResolver


@pytest.fixture
def sample_dataset(tmp_path: Path):
    """Create a temporary chunks.json and embedding_metadata.json fixture."""
    chunks = [
        {
            "chunk_id": "c1",
            "document_id": "doc1",
            "document_name": "doc1.pdf",
            "page_start": 1,
            "page_end": 1,
            "text": "Digital Personal Data Protection Act establishes the Data Protection Board.",
        },
        {
            "chunk_id": "c2",
            "document_id": "doc1",
            "document_name": "doc1.pdf",
            "page_start": 2,
            "page_end": 2,
            "text": "Obligations of Data Fiduciary and processing grounds for personal consent.",
        },
        {
            "chunk_id": "c3",
            "document_id": "doc1",
            "document_name": "doc1.pdf",
            "page_start": 3,
            "page_end": 3,
            "text": "Right of grievance redressal for Data Principals.",
        },
    ]
    metadata_map = [
        {"embedding_index": 0, "chunk_id": "c1"},
        {"embedding_index": 1, "chunk_id": "c2"},
        {"embedding_index": 2, "chunk_id": "c3"},
    ]

    chunks_path = tmp_path / "chunks.json"
    meta_path = tmp_path / "metadata.json"

    chunks_path.write_text(json.dumps(chunks), encoding="utf-8")
    meta_path.write_text(json.dumps(metadata_map), encoding="utf-8")

    return chunks_path, meta_path


def test_bm25_retriever_init(sample_dataset):
    """Verify BM25Retriever initializes properly with custom and default parameters."""
    chunks_path, meta_path = sample_dataset
    retriever = BM25Retriever(
        chunks_file=chunks_path,
        metadata_file=meta_path,
        default_top_k=2,
        max_top_k=10,
    )
    assert retriever.default_top_k == 2
    assert retriever.max_top_k == 10
    assert retriever.total_indexed_chunks == 3


def test_bm25_retriever_invalid_config(sample_dataset):
    """Verify BM25Retriever rejects invalid constructor arguments."""
    chunks_path, meta_path = sample_dataset

    with pytest.raises(ValueError, match="default_top_k must be between"):
        BM25Retriever(chunks_file=chunks_path, metadata_file=meta_path, default_top_k=0)

    with pytest.raises(ValueError, match="default_top_k must be between"):
        BM25Retriever(chunks_file=chunks_path, metadata_file=meta_path, default_top_k=50, max_top_k=20)

    with pytest.raises(ValueError, match="default_score_threshold must be a non-negative"):
        BM25Retriever(chunks_file=chunks_path, metadata_file=meta_path, default_score_threshold=-1.0)


def test_bm25_retriever_retrieve_schema(sample_dataset):
    """Verify retrieve output matches the standard retrieval schema."""
    chunks_path, meta_path = sample_dataset
    retriever = BM25Retriever(chunks_file=chunks_path, metadata_file=meta_path)

    results = retriever.retrieve("Data Protection Board", top_k=2)
    assert len(results) == 2

    top = results[0]
    expected_keys = {
        "rank", "chunk_id", "score", "document_id",
        "document_name", "page_start", "page_end", "text"
    }
    assert set(top.keys()) == expected_keys
    assert top["rank"] == 1
    assert top["chunk_id"] == "c1"
    assert results[1]["rank"] == 2
    assert top["score"] >= results[1]["score"]


def test_bm25_retriever_query_validation(sample_dataset):
    """Verify input validation on retrieve queries."""
    chunks_path, meta_path = sample_dataset
    retriever = BM25Retriever(chunks_file=chunks_path, metadata_file=meta_path)

    with pytest.raises(TypeError, match="Query must be a string"):
        retriever.retrieve(None)  # type: ignore

    with pytest.raises(ValueError, match="cannot be empty"):
        retriever.retrieve("   ")

    with pytest.raises(ValueError, match="top_k must be a positive integer"):
        retriever.retrieve("valid", top_k=0)

    with pytest.raises(ValueError, match="exceeds configured maximum limit"):
        retriever.retrieve("valid", top_k=100)


def test_bm25_retriever_score_threshold(sample_dataset):
    """Verify score_threshold filters out low-scoring or zero-scoring results."""
    chunks_path, meta_path = sample_dataset
    retriever = BM25Retriever(chunks_file=chunks_path, metadata_file=meta_path)

    # All results
    all_res = retriever.retrieve("grievance", top_k=3)
    assert len(all_res) == 3

    # With threshold requiring positive match score
    filtered = retriever.retrieve("grievance", top_k=3, score_threshold=0.01)
    assert len(filtered) == 1
    assert filtered[0]["chunk_id"] == "c3"


def test_bm25_retriever_real_data():
    """Verify BM25Retriever operates against the project processed data."""
    if CHUNKS_OUTPUT_FILE.exists() and EMBEDDING_METADATA_FILE.exists():
        retriever = BM25Retriever()
        results = retriever.retrieve("consent notice data fiduciary", top_k=3)
        assert len(results) == 3
        assert results[0]["rank"] == 1
        assert "consent" in results[0]["text"].lower() or "fiduciary" in results[0]["text"].lower()
