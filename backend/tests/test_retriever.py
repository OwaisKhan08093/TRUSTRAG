"""Unit and integration tests for VectorRetriever (Milestone 7)."""

import json
from pathlib import Path
import numpy as np
import pytest

from backend.app.config import (
    CHUNKS_OUTPUT_FILE,
    EMBEDDING_METADATA_FILE,
    FAISS_INDEX_FILE,
)
from backend.app.embeddings.encoder import EmbeddingEncoder
from backend.app.retrieval.faiss_index import FaissVectorIndex
from backend.app.retrieval.metadata import ChunkMetadataResolver
from backend.app.retrieval.retriever import VectorRetriever, VectorRetrieverError


@pytest.fixture(scope="module")
def encoder() -> EmbeddingEncoder:
    """Shared EmbeddingEncoder fixture."""
    return EmbeddingEncoder()


@pytest.fixture
def mock_retriever_setup(tmp_path: Path, encoder: EmbeddingEncoder):
    """Fixture providing a self-contained VectorRetriever with synthetic data."""
    chunks = [
        {
            "chunk_id": "doc1_page1_chunk1",
            "document_id": "doc1",
            "document_name": "doc1.pdf",
            "page_start": 1,
            "page_end": 1,
            "text": "Personal data breach means any unauthorized processing of personal data.",
        },
        {
            "chunk_id": "doc1_page2_chunk2",
            "document_id": "doc1",
            "document_name": "doc1.pdf",
            "page_start": 2,
            "page_end": 2,
            "text": "Data fiduciary shall implement appropriate technical and organizational measures.",
        },
        {
            "chunk_id": "doc1_page3_chunk3",
            "document_id": "doc1",
            "document_name": "doc1.pdf",
            "page_start": 3,
            "page_end": 3,
            "text": "Penalties for non-compliance with provisions of this Act may extend to crores.",
        },
    ]

    chunks_file = tmp_path / "chunks.json"
    with open(chunks_file, "w", encoding="utf-8") as f:
        json.dump(chunks, f)

    metadata = [
        {"embedding_index": 0, "chunk_id": "doc1_page1_chunk1"},
        {"embedding_index": 1, "chunk_id": "doc1_page2_chunk2"},
        {"embedding_index": 2, "chunk_id": "doc1_page3_chunk3"},
    ]
    metadata_file = tmp_path / "embedding_metadata.json"
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f)

    # Encode chunk texts
    texts = [c["text"] for c in chunks]
    embeddings = encoder.encode_documents(texts, normalize_embeddings=True)

    index = FaissVectorIndex(dimension=384)
    index.add_embeddings(embeddings)

    resolver = ChunkMetadataResolver(chunks_file, metadata_file)
    retriever = VectorRetriever(index=index, encoder=encoder, metadata_resolver=resolver)

    return retriever


def test_retriever_query_validation(mock_retriever_setup: VectorRetriever):
    """Verify retriever validates query string types and empty content."""
    retriever = mock_retriever_setup

    with pytest.raises(TypeError):
        retriever.retrieve(None)  # type: ignore

    with pytest.raises(ValueError):
        retriever.retrieve("")

    with pytest.raises(ValueError):
        retriever.retrieve("   \t\n  ")


def test_retriever_top_k_validation(mock_retriever_setup: VectorRetriever):
    """Verify retriever validates top_k parameter."""
    retriever = mock_retriever_setup

    with pytest.raises(ValueError):
        retriever.retrieve("data breach", top_k=0)

    with pytest.raises(ValueError):
        retriever.retrieve("data breach", top_k=-1)


def test_retriever_structured_result_fields(mock_retriever_setup: VectorRetriever):
    """Verify retrieved results contain all required schema fields in descending score order."""
    retriever = mock_retriever_setup
    results = retriever.retrieve("What is a personal data breach?", top_k=2)

    assert len(results) == 2
    assert results[0]["rank"] == 1
    assert results[1]["rank"] == 2
    assert results[0]["score"] >= results[1]["score"]

    required_keys = {"rank", "chunk_id", "score", "document_id", "document_name", "page_start", "page_end", "text"}
    for res in results:
        assert required_keys.issubset(res.keys())
        assert isinstance(res["rank"], int)
        assert isinstance(res["score"], float)
        assert isinstance(res["text"], str)

    # The first result should be the breach chunk
    assert results[0]["chunk_id"] == "doc1_page1_chunk1"


def test_retriever_dimension_mismatch_raises(encoder: EmbeddingEncoder):
    """Verify VectorRetriever raises error if FAISS dimension doesn't match encoder."""
    wrong_index = FaissVectorIndex(dimension=128)
    with pytest.raises(VectorRetrieverError) as exc_info:
        VectorRetriever(index=wrong_index, encoder=encoder)
    assert "Dimension mismatch" in str(exc_info.value)


def test_retriever_integration_real_data(encoder: EmbeddingEncoder):
    """Verify end-to-end retrieval with the actual artifacts in data/processed."""
    if FAISS_INDEX_FILE.exists() and CHUNKS_OUTPUT_FILE.exists() and EMBEDDING_METADATA_FILE.exists():
        retriever = VectorRetriever(encoder=encoder)
        assert retriever.total_indexed_vectors > 0

        query = "What is the penalty for failure to take reasonable security safeguards?"
        results = retriever.retrieve(query, top_k=3)

        assert len(results) > 0
        assert results[0]["rank"] == 1
        assert "chunk_id" in results[0]
        assert "text" in results[0]
        assert -1.0 <= results[0]["score"] <= 1.0
