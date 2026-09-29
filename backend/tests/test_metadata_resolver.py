"""Unit tests for chunk metadata resolver (Milestone 6)."""

import json
from pathlib import Path
import pytest

from backend.app.config import CHUNKS_OUTPUT_FILE, EMBEDDING_METADATA_FILE
from backend.app.retrieval.metadata import (
    ChunkMetadataResolver,
    MetadataResolutionError,
)


@pytest.fixture
def sample_data_paths(tmp_path: Path):
    """Fixture providing valid sample chunks and metadata files."""
    chunks_file = tmp_path / "chunks.json"
    metadata_file = tmp_path / "embedding_metadata.json"

    chunks = [
        {
            "chunk_id": "DPDP_Act_2023_page1_chunk1",
            "document_id": "DPDP_Act_2023",
            "document_name": "DPDP_Act_2023.pdf",
            "page_start": 1,
            "page_end": 1,
            "text": "First chunk text on preliminary provisions.",
        },
        {
            "chunk_id": "DPDP_Act_2023_page1_chunk2",
            "document_id": "DPDP_Act_2023",
            "document_name": "DPDP_Act_2023.pdf",
            "page_start": 1,
            "page_end": 2,
            "text": "Second chunk text on definitions of data principal.",
        },
        {
            "chunk_id": "DPDP_Act_2023_page2_chunk3",
            "document_id": "DPDP_Act_2023",
            "document_name": "DPDP_Act_2023.pdf",
            "page_start": 2,
            "page_end": 2,
            "text": "Third chunk text on data fiduciary obligations.",
        },
    ]

    metadata = [
        {"embedding_index": 0, "chunk_id": "DPDP_Act_2023_page1_chunk1"},
        {"embedding_index": 1, "chunk_id": "DPDP_Act_2023_page1_chunk2"},
        {"embedding_index": 2, "chunk_id": "DPDP_Act_2023_page2_chunk3"},
    ]

    with open(chunks_file, "w", encoding="utf-8") as f:
        json.dump(chunks, f)

    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f)

    return chunks_file, metadata_file


def test_resolver_loads_and_resolves(sample_data_paths):
    """Verify resolver successfully initializes and resolves single index."""
    chunks_file, metadata_file = sample_data_paths
    resolver = ChunkMetadataResolver(chunks_file, metadata_file)

    assert resolver.total_chunks == 3
    assert resolver.total_mappings == 3

    chunk_0 = resolver.resolve_index(0)
    assert chunk_0["chunk_id"] == "DPDP_Act_2023_page1_chunk1"
    assert chunk_0["document_name"] == "DPDP_Act_2023.pdf"
    assert chunk_0["page_start"] == 1
    assert chunk_0["page_end"] == 1
    assert "preliminary provisions" in chunk_0["text"]


def test_resolver_multiple_indices(sample_data_paths):
    """Verify resolve_indices resolves multiple indices in given sequence."""
    chunks_file, metadata_file = sample_data_paths
    resolver = ChunkMetadataResolver(chunks_file, metadata_file)

    resolved = resolver.resolve_indices([2, 0])
    assert len(resolved) == 2
    assert resolved[0]["chunk_id"] == "DPDP_Act_2023_page2_chunk3"
    assert resolved[1]["chunk_id"] == "DPDP_Act_2023_page1_chunk1"


def test_resolver_out_of_bounds_index_raises(sample_data_paths):
    """Verify index out of bounds raises IndexError."""
    chunks_file, metadata_file = sample_data_paths
    resolver = ChunkMetadataResolver(chunks_file, metadata_file)

    with pytest.raises(IndexError) as exc_info:
        resolver.resolve_index(10)
    assert "out of bounds" in str(exc_info.value)

    with pytest.raises(IndexError):
        resolver.resolve_index(-1)


def test_resolver_invalid_index_type_raises(sample_data_paths):
    """Verify non-integer index raises TypeError."""
    chunks_file, metadata_file = sample_data_paths
    resolver = ChunkMetadataResolver(chunks_file, metadata_file)

    with pytest.raises(TypeError) as exc_info:
        resolver.resolve_index("0")  # type: ignore
    assert "must be an integer" in str(exc_info.value)


def test_resolver_missing_files_raises(tmp_path: Path):
    """Verify missing chunks or metadata file raises FileNotFoundError."""
    missing_chunks = tmp_path / "missing_chunks.json"
    missing_metadata = tmp_path / "missing_metadata.json"

    with pytest.raises(FileNotFoundError):
        ChunkMetadataResolver(missing_chunks, missing_metadata)


def test_resolver_duplicate_chunk_ids_raises(tmp_path: Path):
    """Verify duplicate chunk_id in chunks.json is detected and rejected."""
    chunks_file = tmp_path / "chunks.json"
    metadata_file = tmp_path / "embedding_metadata.json"

    duplicate_chunks = [
        {"chunk_id": "dup_chunk", "document_id": "d1", "document_name": "f1.pdf", "page_start": 1, "page_end": 1, "text": "A"},
        {"chunk_id": "dup_chunk", "document_id": "d1", "document_name": "f1.pdf", "page_start": 1, "page_end": 1, "text": "B"},
    ]
    metadata = [
        {"embedding_index": 0, "chunk_id": "dup_chunk"}
    ]

    with open(chunks_file, "w", encoding="utf-8") as f:
        json.dump(duplicate_chunks, f)
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f)

    with pytest.raises(MetadataResolutionError) as exc_info:
        ChunkMetadataResolver(chunks_file, metadata_file)
    assert "Duplicate chunk_id" in str(exc_info.value)


def test_resolver_missing_chunk_reference_raises(tmp_path: Path):
    """Verify metadata referencing a non-existent chunk_id raises error."""
    chunks_file = tmp_path / "chunks.json"
    metadata_file = tmp_path / "embedding_metadata.json"

    chunks = [
        {"chunk_id": "chunk_A", "document_id": "d1", "document_name": "f1.pdf", "page_start": 1, "page_end": 1, "text": "A"}
    ]
    metadata = [
        {"embedding_index": 0, "chunk_id": "chunk_UNKNOWN"}
    ]

    with open(chunks_file, "w", encoding="utf-8") as f:
        json.dump(chunks, f)
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f)

    with pytest.raises(MetadataResolutionError) as exc_info:
        ChunkMetadataResolver(chunks_file, metadata_file)
    assert "does not exist in chunks file" in str(exc_info.value)


def test_resolver_real_data():
    """Verify resolver works with the actual processed artifacts in data/processed."""
    if CHUNKS_OUTPUT_FILE.exists() and EMBEDDING_METADATA_FILE.exists():
        resolver = ChunkMetadataResolver()
        assert resolver.total_chunks > 0
        assert resolver.total_mappings == resolver.total_chunks
        first = resolver.resolve_index(0)
        assert "chunk_id" in first
        assert "text" in first
