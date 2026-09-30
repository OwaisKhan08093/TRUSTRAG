"""Unit tests for BM25 retrieval foundation and index construction."""

import json
from pathlib import Path
import pytest

from backend.app.config import CHUNKS_OUTPUT_FILE
from backend.app.retrieval.bm25_index import (
    BM25Index,
    BM25IndexError,
    default_tokenizer,
)


def test_default_tokenizer():
    """Verify default_tokenizer lowers text and extracts alphanumeric words."""
    assert default_tokenizer("Hello, World! 123") == ["hello", "world", "123"]
    assert default_tokenizer("") == []
    assert default_tokenizer(123) == []  # type: ignore


def test_bm25_index_init_with_dicts():
    """Verify BM25Index initializes properly with chunk dictionaries."""
    chunks = [
        {"chunk_id": "c1", "text": "First chunk text about privacy."},
        {"chunk_id": "c2", "text": "Second chunk text about data protection."},
    ]
    index = BM25Index(chunks)
    assert index.corpus_size == 2
    assert len(index.tokenized_corpus) == 2
    assert "privacy" in index.tokenized_corpus[0]
    assert "protection" in index.tokenized_corpus[1]


def test_bm25_index_init_with_strings():
    """Verify BM25Index initializes properly with a sequence of raw strings."""
    chunks = ["Document one contents.", "Document two contents."]
    index = BM25Index(chunks)
    assert index.corpus_size == 2
    assert index.tokenized_corpus[0] == ["document", "one", "contents"]


def test_bm25_index_custom_tokenizer():
    """Verify BM25Index works with a custom tokenizer function."""
    custom_tok = lambda s: s.split()
    chunks = ["Alpha Beta Gamma", "Delta Epsilon"]
    index = BM25Index(chunks, tokenizer=custom_tok)
    assert index.corpus_size == 2
    assert index.tokenized_corpus[0] == ["Alpha", "Beta", "Gamma"]


def test_bm25_index_empty_corpus():
    """Verify BM25Index raises ValueError on empty corpus."""
    with pytest.raises(ValueError, match="Corpus cannot be empty"):
        BM25Index([])


def test_bm25_index_invalid_types():
    """Verify BM25Index rejects invalid corpus or chunk types."""
    with pytest.raises(TypeError, match="must be a list or tuple"):
        BM25Index("not a list")  # type: ignore

    with pytest.raises(TypeError, match="must be a dict or string"):
        BM25Index([123, 456])  # type: ignore

    with pytest.raises(ValueError, match="missing required 'text' key"):
        BM25Index([{"chunk_id": "c1"}])

    with pytest.raises(TypeError, match="must be a string"):
        BM25Index([{"text": 12345}])


def test_bm25_from_chunks_file(tmp_path: Path):
    """Verify BM25Index.from_chunks_file loads correctly from a valid file."""
    data = [
        {"chunk_id": "chk1", "text": "Sample text for chunk 1"},
        {"chunk_id": "chk2", "text": "Sample text for chunk 2"},
    ]
    file_path = tmp_path / "test_chunks.json"
    file_path.write_text(json.dumps(data), encoding="utf-8")

    index = BM25Index.from_chunks_file(file_path)
    assert index.corpus_size == 2


def test_bm25_from_chunks_file_not_found(tmp_path: Path):
    """Verify from_chunks_file raises FileNotFoundError for missing file."""
    missing = tmp_path / "does_not_exist.json"
    with pytest.raises(FileNotFoundError):
        BM25Index.from_chunks_file(missing)


def test_bm25_from_chunks_file_invalid_json(tmp_path: Path):
    """Verify from_chunks_file raises BM25IndexError on invalid JSON syntax."""
    bad_file = tmp_path / "bad.json"
    bad_file.write_text("{ broken json", encoding="utf-8")
    with pytest.raises(BM25IndexError, match="Failed to parse JSON"):
        BM25Index.from_chunks_file(bad_file)


def test_bm25_from_chunks_file_non_list(tmp_path: Path):
    """Verify from_chunks_file raises ValueError if JSON is not a list."""
    obj_file = tmp_path / "obj.json"
    obj_file.write_text('{"key": "value"}', encoding="utf-8")
    with pytest.raises(ValueError, match="must contain a JSON list"):
        BM25Index.from_chunks_file(obj_file)


def test_bm25_with_processed_data():
    """Verify BM25Index loads existing project processed chunks.json if available."""
    if CHUNKS_OUTPUT_FILE.exists():
        index = BM25Index.from_chunks_file(CHUNKS_OUTPUT_FILE)
        assert index.corpus_size > 0


def test_bm25_search_basic():
    """Verify BM25 search scores and ranks matching documents highest."""
    chunks = [
        {"chunk_id": "c0", "text": "Apples and oranges in the orchard."},
        {"chunk_id": "c1", "text": "Data fiduciary and personal data consent obligations."},
        {"chunk_id": "c2", "text": "Grievance redressal mechanism under the data act."},
    ]
    index = BM25Index(chunks)
    results = index.search("fiduciary consent", top_k=2)

    assert len(results) == 2
    assert results[0]["index"] == 1
    assert results[0]["score"] > results[1]["score"]
    assert isinstance(results[0]["index"], int)
    assert isinstance(results[0]["score"], float)


def test_bm25_search_top_k_bounds():
    """Verify BM25 search never returns more results than corpus size."""
    chunks = ["Document A", "Document B"]
    index = BM25Index(chunks)
    results = index.search("Document", top_k=10)
    assert len(results) == 2


def test_bm25_search_empty_or_invalid_query():
    """Verify BM25 search rejects invalid or empty query inputs."""
    chunks = ["Some text here"]
    index = BM25Index(chunks)

    with pytest.raises(ValueError, match="Query string cannot be empty"):
        index.search("")

    with pytest.raises(ValueError, match="Query string cannot be empty"):
        index.search("   ")

    with pytest.raises(TypeError, match="Query must be a string"):
        index.search(None)  # type: ignore

    with pytest.raises(TypeError, match="Query must be a string"):
        index.search(["keyword"])  # type: ignore


def test_bm25_search_invalid_top_k():
    """Verify BM25 search rejects invalid top_k."""
    chunks = ["Some text here"]
    index = BM25Index(chunks)

    with pytest.raises(ValueError, match="top_k must be a positive integer"):
        index.search("text", top_k=0)

    with pytest.raises(ValueError, match="top_k must be a positive integer"):
        index.search("text", top_k=-5)

    with pytest.raises(ValueError, match="top_k must be a positive integer"):
        index.search("text", top_k=3.5)  # type: ignore


def test_bm25_search_zero_match_query():
    """Verify BM25 search handles queries with zero matching terms gracefully."""
    chunks = ["Alpha Beta Gamma", "Delta Epsilon Zeta"]
    index = BM25Index(chunks)
    results = index.search("UnrelatedXyzToken", top_k=2)
    assert len(results) == 2
    assert all(r["score"] == 0.0 for r in results)
