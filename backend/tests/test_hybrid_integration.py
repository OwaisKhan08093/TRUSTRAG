"""Comprehensive integration tests for end-to-end hybrid retrieval engine."""

import json
from pathlib import Path
import numpy as np
import pytest

from backend.app.embeddings.encoder import EmbeddingEncoder
from backend.app.retrieval.bm25_index import BM25Index
from backend.app.retrieval.bm25_retriever import BM25Retriever
from backend.app.retrieval.faiss_index import FaissVectorIndex
from backend.app.retrieval.hybrid_retriever import HybridRetriever, HybridRetrieverError
from backend.app.retrieval.metadata import ChunkMetadataResolver
from backend.app.retrieval.retriever import VectorRetriever
from backend.app.retrieval.rrf import reciprocal_rank_fusion


@pytest.fixture(scope="module")
def shared_encoder() -> EmbeddingEncoder:
    """Shared EmbeddingEncoder across integration tests."""
    return EmbeddingEncoder()


@pytest.fixture
def test_environment(tmp_path: Path, shared_encoder: EmbeddingEncoder):
    """Set up an isolated end-to-end environment with chunks, FAISS index, BM25, and metadata."""
    chunks = [
        {
            "chunk_id": "act_p1_c1",
            "document_id": "DPDP_Act",
            "document_name": "DPDP_Act.pdf",
            "page_start": 1,
            "page_end": 1,
            "text": "The Digital Personal Data Protection Act 2023 preliminary rules and definitions.",
        },
        {
            "chunk_id": "act_p2_c2",
            "document_id": "DPDP_Act",
            "document_name": "DPDP_Act.pdf",
            "page_start": 2,
            "page_end": 2,
            "text": "Grounds for processing personal data and obligations of Data Fiduciary regarding notice.",
        },
        {
            "chunk_id": "act_p3_c3",
            "document_id": "DPDP_Act",
            "document_name": "DPDP_Act.pdf",
            "page_start": 3,
            "page_end": 3,
            "text": "Rights of Data Principal including correction, erasure, and grievance redressal.",
        },
    ]

    chunks_file = tmp_path / "chunks.json"
    meta_file = tmp_path / "embedding_metadata.json"
    faiss_file = tmp_path / "index.faiss"

    chunks_file.write_text(json.dumps(chunks), encoding="utf-8")

    # Build embeddings & FAISS index
    texts = [c["text"] for c in chunks]
    embeddings = shared_encoder.encode_documents(texts, normalize_embeddings=True)
    faiss_idx = FaissVectorIndex(dimension=embeddings.shape[1])
    faiss_idx.add_embeddings(embeddings)
    faiss_idx.save(faiss_file)

    meta_entries = [{"embedding_index": i, "chunk_id": chunks[i]["chunk_id"]} for i in range(len(chunks))]
    meta_file.write_text(json.dumps(meta_entries), encoding="utf-8")

    resolver = ChunkMetadataResolver(chunks_file=chunks_file, metadata_file=meta_file)
    dense_retriever = VectorRetriever(
        index=faiss_idx,
        encoder=shared_encoder,
        metadata_resolver=resolver,
    )
    bm25_idx = BM25Index.from_chunks_file(chunks_file)
    sparse_retriever = BM25Retriever(
        index=bm25_idx,
        metadata_resolver=resolver,
    )
    hybrid_retriever = HybridRetriever(
        dense_retriever=dense_retriever,
        sparse_retriever=sparse_retriever,
        default_top_k=3,
        rrf_k=60,
    )

    return {
        "hybrid": hybrid_retriever,
        "dense": dense_retriever,
        "sparse": sparse_retriever,
        "resolver": resolver,
        "chunks": chunks,
    }


def test_end_to_end_hybrid_flow(test_environment):
    """Verify full end-to-end pipeline: query -> dense + sparse -> RRF -> metadata -> final result."""
    hybrid = test_environment["hybrid"]
    results = hybrid.retrieve("grievance redressal rights of data principal", top_k=2)

    assert len(results) == 2
    top = results[0]
    assert top["rank"] == 1
    assert top["chunk_id"] == "act_p3_c3"
    assert top["document_name"] == "DPDP_Act.pdf"
    assert top["page_start"] == 3
    assert top["page_end"] == 3
    assert "grievance redressal" in top["text"]
    assert isinstance(top["score"], float)
    assert top["score"] > 0.0


def test_hybrid_edge_case_unknown_query(test_environment):
    """Verify hybrid retriever behavior on totally unknown / gibberish query."""
    hybrid = test_environment["hybrid"]
    # Semantic search still finds geometric nearest neighbors; BM25 has 0 scores
    results = hybrid.retrieve("XylophoneZebraQuantum9999", top_k=3)
    assert len(results) == 3
    # Top ranks are still properly resolved
    assert all("chunk_id" in r for r in results)
    assert [r["rank"] for r in results] == [1, 2, 3]


def test_hybrid_edge_case_dense_filter_all_out(test_environment):
    """Verify hybrid retriever works when dense retriever filters all results with high threshold."""
    hybrid = test_environment["hybrid"]
    # Cosine threshold 0.99 filters out all dense matches; sparse still returns results
    results = hybrid.retrieve(
        "definitions",
        top_k=2,
        dense_score_threshold=0.99,
    )
    assert len(results) >= 1
    assert results[0]["chunk_id"] == "act_p1_c1"


def test_hybrid_edge_case_sparse_filter_all_out(test_environment):
    """Verify hybrid retriever works when sparse retriever filters all results with high threshold."""
    hybrid = test_environment["hybrid"]
    # BM25 threshold 100.0 filters out sparse matches; dense still returns results
    results = hybrid.retrieve(
        "definitions",
        top_k=2,
        sparse_score_threshold=100.0,
    )
    assert len(results) == 2
    assert results[0]["chunk_id"] == "act_p1_c1"


def test_hybrid_edge_case_top_k_exceeding_corpus(test_environment):
    """Verify top_k requested larger than corpus returns at most total available unique chunks."""
    hybrid = test_environment["hybrid"]
    results = hybrid.retrieve("data", top_k=10, dense_top_k=10, sparse_top_k=10)
    assert len(results) == 3  # Corpus size is 3


def test_hybrid_invalid_query_rejection(test_environment):
    """Verify empty and invalid query strings are rejected."""
    hybrid = test_environment["hybrid"]

    with pytest.raises(ValueError, match="cannot be empty"):
        hybrid.retrieve("")

    with pytest.raises(ValueError, match="cannot be empty"):
        hybrid.retrieve("     ")

    with pytest.raises(TypeError, match="Query must be a string"):
        hybrid.retrieve(12345)  # type: ignore


def test_hybrid_invalid_parameters(test_environment):
    """Verify invalid top_k, dense_top_k, sparse_top_k, rrf_k raise ValueError."""
    hybrid = test_environment["hybrid"]

    with pytest.raises(ValueError, match="top_k must be a positive integer"):
        hybrid.retrieve("test", top_k=0)

    with pytest.raises(ValueError, match="dense_top_k must be a positive integer"):
        hybrid.retrieve("test", dense_top_k=0)

    with pytest.raises(ValueError, match="sparse_top_k must be a positive integer"):
        hybrid.retrieve("test", sparse_top_k=-2)

    with pytest.raises(ValueError, match="rrf_k must be an integer between"):
        hybrid.retrieve("test", rrf_k=0)


def test_rrf_malformed_result_lists():
    """Verify RRF handles malformed list inputs gracefully."""
    with pytest.raises(TypeError, match="must be a sequence"):
        reciprocal_rank_fusion(None)  # type: ignore

    with pytest.raises(TypeError, match="must be a dictionary"):
        reciprocal_rank_fusion([[1, 2, 3]])  # type: ignore


def test_rrf_deduplication_exactness():
    """Verify that chunks appearing in both dense and sparse results are merged exactly once."""
    list_a = [
        {"chunk_id": "c1", "document_id": "d1", "page_start": 1, "page_end": 1, "text": "text 1"},
        {"chunk_id": "c2", "document_id": "d1", "page_start": 2, "page_end": 2, "text": "text 2"},
    ]
    list_b = [
        {"chunk_id": "c1", "document_id": "d1", "page_start": 1, "page_end": 1, "text": "text 1"},
        {"chunk_id": "c2", "document_id": "d1", "page_start": 2, "page_end": 2, "text": "text 2"},
    ]
    fused = reciprocal_rank_fusion([list_a, list_b], k=60)
    assert len(fused) == 2
    assert [f["chunk_id"] for f in fused] == ["c1", "c2"]
    assert fused[0]["rank"] == 1
    assert fused[1]["rank"] == 2
    assert fused[0]["score"] == pytest.approx(2 * (1 / 61))
    assert fused[1]["score"] == pytest.approx(2 * (1 / 62))
