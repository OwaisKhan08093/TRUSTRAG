"""Unit tests connecting BM25 search indices to chunk metadata resolution."""

import json
from pathlib import Path
import pytest

from backend.app.config import CHUNKS_OUTPUT_FILE, EMBEDDING_METADATA_FILE
from backend.app.retrieval.bm25_index import BM25Index
from backend.app.retrieval.metadata import ChunkMetadataResolver


def test_bm25_search_to_metadata_resolution(tmp_path: Path):
    """Verify BM25 search result indices resolve to complete chunk metadata records."""
    chunks = [
        {
            "chunk_id": "doc1_p1_c0",
            "document_id": "doc1",
            "document_name": "doc1.pdf",
            "page_start": 1,
            "page_end": 1,
            "text": "The Digital Personal Data Protection Act covers data fiduciary compliance.",
        },
        {
            "chunk_id": "doc1_p2_c1",
            "document_id": "doc1",
            "document_name": "doc1.pdf",
            "page_start": 2,
            "page_end": 2,
            "text": "Appellate Tribunal procedures and penalty schedules.",
        },
    ]
    metadata_map = [
        {"embedding_index": 0, "chunk_id": "doc1_p1_c0"},
        {"embedding_index": 1, "chunk_id": "doc1_p2_c1"},
    ]

    chunks_file = tmp_path / "chunks.json"
    meta_file = tmp_path / "metadata.json"

    chunks_file.write_text(json.dumps(chunks), encoding="utf-8")
    meta_file.write_text(json.dumps(metadata_map), encoding="utf-8")

    bm25_idx = BM25Index.from_chunks_file(chunks_file)
    resolver = ChunkMetadataResolver(chunks_file=chunks_file, metadata_file=meta_file)

    # Search for keyword present only in chunk 0
    results = bm25_idx.search("fiduciary compliance", top_k=1)
    assert len(results) == 1
    top_result = results[0]

    # Resolve metadata
    resolved = resolver.resolve_index(top_result["index"])

    assert resolved["chunk_id"] == "doc1_p1_c0"
    assert resolved["document_id"] == "doc1"
    assert resolved["document_name"] == "doc1.pdf"
    assert resolved["page_start"] == 1
    assert resolved["page_end"] == 1
    assert "fiduciary compliance" in resolved["text"]


def test_bm25_metadata_resolution_with_real_dataset():
    """Verify BM25 results map accurately to metadata on project chunks.json."""
    if CHUNKS_OUTPUT_FILE.exists() and EMBEDDING_METADATA_FILE.exists():
        bm25_idx = BM25Index.from_chunks_file(CHUNKS_OUTPUT_FILE)
        resolver = ChunkMetadataResolver(
            chunks_file=CHUNKS_OUTPUT_FILE,
            metadata_file=EMBEDDING_METADATA_FILE,
        )

        results = bm25_idx.search("grievance redressal", top_k=2)
        assert len(results) == 2

        resolved_list = [resolver.resolve_index(r["index"]) for r in results]
        for res in resolved_list:
            assert "chunk_id" in res
            assert "document_id" in res
            assert "document_name" in res
            assert "page_start" in res
            assert "page_end" in res
            assert "text" in res
            assert isinstance(res["page_start"], int)
            assert isinstance(res["page_end"], int)


def test_resolve_chunk_id():
    """Verify ChunkMetadataResolver.resolve_chunk_id returns correct chunk metadata directly."""
    if CHUNKS_OUTPUT_FILE.exists() and EMBEDDING_METADATA_FILE.exists():
        resolver = ChunkMetadataResolver(
            chunks_file=CHUNKS_OUTPUT_FILE,
            metadata_file=EMBEDDING_METADATA_FILE,
        )
        meta = resolver.resolve_chunk_id("DPDP_Act_2023_page1_chunk1")
        assert meta["chunk_id"] == "DPDP_Act_2023_page1_chunk1"
        assert meta["document_id"] == "DPDP_Act_2023"
        assert meta["page_start"] == 1

        with pytest.raises(KeyError, match="not found in metadata registry"):
            resolver.resolve_chunk_id("non_existent_chunk_id")

        with pytest.raises(TypeError, match="chunk_id must be a string"):
            resolver.resolve_chunk_id(123)  # type: ignore
