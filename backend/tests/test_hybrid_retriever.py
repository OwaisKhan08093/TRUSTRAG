"""Unit tests for HybridRetriever foundation."""

from unittest.mock import MagicMock
import pytest

from backend.app.config import CHUNKS_OUTPUT_FILE, FAISS_INDEX_FILE
from backend.app.retrieval.bm25_retriever import BM25Retriever
from backend.app.retrieval.hybrid_retriever import HybridRetriever, HybridRetrieverError
from backend.app.retrieval.retriever import VectorRetriever


@pytest.fixture
def mock_retrievers():
    """Create mock dense and sparse retrievers."""
    dense_mock = MagicMock(spec=VectorRetriever)
    sparse_mock = MagicMock(spec=BM25Retriever)
    return dense_mock, sparse_mock


def test_hybrid_retriever_init(mock_retrievers):
    """Verify HybridRetriever initializes with injected retrievers and parameter properties."""
    dense, sparse = mock_retrievers
    hybrid = HybridRetriever(
        dense_retriever=dense,
        sparse_retriever=sparse,
        default_top_k=3,
        default_dense_top_k=4,
        default_sparse_top_k=4,
        rrf_k=30,
        max_top_k=15,
    )
    assert hybrid.dense_retriever is dense
    assert hybrid.sparse_retriever is sparse
    assert hybrid.default_top_k == 3
    assert hybrid.default_dense_top_k == 4
    assert hybrid.default_sparse_top_k == 4
    assert hybrid.rrf_k == 30
    assert hybrid.max_top_k == 15


def test_hybrid_retriever_init_validation(mock_retrievers):
    """Verify constructor parameter validations and bounds."""
    dense, sparse = mock_retrievers

    with pytest.raises(ValueError, match="default_top_k must be between"):
        HybridRetriever(dense_retriever=dense, sparse_retriever=sparse, default_top_k=0)

    with pytest.raises(ValueError, match="default_dense_top_k must be between"):
        HybridRetriever(dense_retriever=dense, sparse_retriever=sparse, default_dense_top_k=30, max_top_k=10)

    with pytest.raises(ValueError, match="default_sparse_top_k must be between"):
        HybridRetriever(dense_retriever=dense, sparse_retriever=sparse, default_sparse_top_k=0)

    with pytest.raises(ValueError, match="rrf_k must be an integer between"):
        HybridRetriever(dense_retriever=dense, sparse_retriever=sparse, rrf_k=0)

    with pytest.raises(ValueError, match="rrf_k must be an integer between"):
        HybridRetriever(dense_retriever=dense, sparse_retriever=sparse, rrf_k=5000)


def test_hybrid_retriever_retrieve_logic(mock_retrievers):
    """Verify retrieve invokes dense and sparse retrievers and merges via RRF."""
    dense, sparse = mock_retrievers

    dense.retrieve.return_value = [
        {"rank": 1, "chunk_id": "c1", "score": 0.85, "text": "Semantic text 1"},
        {"rank": 2, "chunk_id": "c2", "score": 0.75, "text": "Semantic text 2"},
    ]
    sparse.retrieve.return_value = [
        {"rank": 1, "chunk_id": "c2", "score": 8.5, "text": "Semantic text 2"},
        {"rank": 2, "chunk_id": "c3", "score": 6.2, "text": "Lexical text 3"},
    ]

    hybrid = HybridRetriever(
        dense_retriever=dense,
        sparse_retriever=sparse,
        rrf_k=60,
    )

    results = hybrid.retrieve("data protection query", top_k=3)

    dense.retrieve.assert_called_once_with(
        query="data protection query",
        top_k=5,
        score_threshold=None,
    )
    sparse.retrieve.assert_called_once_with(
        query="data protection query",
        top_k=5,
        score_threshold=None,
    )

    # c2 was rank 2 dense and rank 1 sparse -> highest RRF score
    assert len(results) == 3
    assert results[0]["chunk_id"] == "c2"
    assert results[0]["rank"] == 1
    assert results[1]["chunk_id"] == "c1"
    assert results[1]["rank"] == 2
    assert results[2]["chunk_id"] == "c3"
    assert results[2]["rank"] == 3


def test_hybrid_retriever_input_validation(mock_retrievers):
    """Verify input validation for retrieve arguments."""
    dense, sparse = mock_retrievers
    hybrid = HybridRetriever(dense_retriever=dense, sparse_retriever=sparse)

    with pytest.raises(TypeError, match="Query must be a string"):
        hybrid.retrieve(123)  # type: ignore

    with pytest.raises(ValueError, match="cannot be empty"):
        hybrid.retrieve("   ")

    with pytest.raises(ValueError, match="top_k must be a positive integer"):
        hybrid.retrieve("query", top_k=0)

    with pytest.raises(ValueError, match="exceeds maximum limit"):
        hybrid.retrieve("query", top_k=100)


def test_hybrid_retriever_error_wrapping(mock_retrievers):
    """Verify underlying retrieval exceptions are wrapped into HybridRetrieverError."""
    dense, sparse = mock_retrievers
    dense.retrieve.side_effect = RuntimeError("FAISS crashed")

    hybrid = HybridRetriever(dense_retriever=dense, sparse_retriever=sparse)

    with pytest.raises(HybridRetrieverError, match="Dense vector retrieval failed"):
        hybrid.retrieve("test query")


def test_hybrid_retriever_real_execution():
    """Verify HybridRetriever executes on local real indices if present."""
    if CHUNKS_OUTPUT_FILE.exists() and FAISS_INDEX_FILE.exists():
        hybrid = HybridRetriever()
        results = hybrid.retrieve("Digital Personal Data Protection Act", top_k=2)
        assert len(results) == 2
        assert results[0]["rank"] == 1
        assert "chunk_id" in results[0]
        assert "score" in results[0]
