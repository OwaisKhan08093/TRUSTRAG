"""Unit tests for ResultReranker component."""

from unittest.mock import MagicMock
import pytest

from backend.app.reranking.cross_encoder import CrossEncoderReranker
from backend.app.reranking.reranker import ResultReranker, RerankerError
from backend.app.reranking.schema import RerankedChunk


@pytest.fixture
def mock_encoder():
    """Create a mock CrossEncoderReranker."""
    encoder = MagicMock(spec=CrossEncoderReranker)
    return encoder


@pytest.fixture
def sample_candidates():
    """Sample candidate chunks as returned by HybridRetriever."""
    return [
        {
            "rank": 1,
            "chunk_id": "c1",
            "score": 0.032,
            "document_id": "doc1",
            "document_name": "doc1.pdf",
            "page_start": 1,
            "page_end": 1,
            "text": "First passage about general terms.",
        },
        {
            "rank": 2,
            "chunk_id": "c2",
            "score": 0.025,
            "document_id": "doc1",
            "document_name": "doc1.pdf",
            "page_start": 2,
            "page_end": 2,
            "text": "Second passage specifically about penalty notices and compliance.",
        },
        {
            "rank": 3,
            "chunk_id": "c3",
            "score": 0.018,
            "document_id": "doc1",
            "document_name": "doc1.pdf",
            "page_start": 3,
            "page_end": 3,
            "text": "Third passage about unrelated topics.",
        },
    ]


def test_reranker_init(mock_encoder):
    """Verify ResultReranker initializes with custom and default parameters."""
    reranker = ResultReranker(
        cross_encoder=mock_encoder,
        default_top_k=2,
        default_batch_size=8,
        max_top_k=15,
    )
    assert reranker.cross_encoder is mock_encoder
    assert reranker.default_top_k == 2
    assert reranker.default_batch_size == 8


def test_reranker_init_validation(mock_encoder):
    """Verify ResultReranker rejects invalid parameters in __init__."""
    with pytest.raises(ValueError, match="default_top_k must be between"):
        ResultReranker(cross_encoder=mock_encoder, default_top_k=0)

    with pytest.raises(ValueError, match="default_batch_size must be a positive integer"):
        ResultReranker(cross_encoder=mock_encoder, default_batch_size=0)


def test_rerank_reordering_and_provenance(mock_encoder, sample_candidates):
    """Verify rerank correctly re-orders candidates by neural score and preserves original provenance."""
    # c2 gets highest score (8.5), c1 gets middle score (4.2), c3 gets lowest (1.1)
    mock_encoder.score_pairs.return_value = [4.2, 8.5, 1.1]

    reranker = ResultReranker(cross_encoder=mock_encoder, default_top_k=2)
    results = reranker.rerank("penalty compliance", sample_candidates, top_k=2)

    assert len(results) == 2
    assert all(isinstance(r, RerankedChunk) for r in results)

    # c2 should now be final_rank 1
    top = results[0]
    assert top.chunk_id == "c2"
    assert top.final_rank == 1
    assert top.rerank_score == 8.5
    assert top.original_rank == 2
    assert top.original_score == 0.025
    assert top.document_name == "doc1.pdf"

    # c1 should be final_rank 2
    second = results[1]
    assert second.chunk_id == "c1"
    assert second.final_rank == 2
    assert second.rerank_score == 4.2
    assert second.original_rank == 1
    assert second.original_score == 0.032


def test_rerank_candidate_count_smaller_than_top_k(mock_encoder, sample_candidates):
    """Verify rerank handles candidate lists smaller than requested top_k."""
    mock_encoder.score_pairs.return_value = [5.0, 7.0, 2.0]

    reranker = ResultReranker(cross_encoder=mock_encoder)
    results = reranker.rerank("query", sample_candidates, top_k=10)

    assert len(results) == 3
    assert [r.final_rank for r in results] == [1, 2, 3]


def test_rerank_empty_results(mock_encoder):
    """Verify rerank returns empty list when given empty candidates."""
    reranker = ResultReranker(cross_encoder=mock_encoder)
    assert reranker.rerank("query", []) == []


def test_rerank_validation_errors(mock_encoder):
    """Verify input validation on rerank arguments."""
    reranker = ResultReranker(cross_encoder=mock_encoder)

    with pytest.raises(TypeError, match="query must be a string"):
        reranker.rerank(None, [{"text": "sample"}])  # type: ignore

    with pytest.raises(ValueError, match="query cannot be empty"):
        reranker.rerank("   ", [{"text": "sample"}])

    with pytest.raises(TypeError, match="must be a sequence of dictionaries"):
        reranker.rerank("query", "not a list")  # type: ignore

    with pytest.raises(TypeError, match="must be a dictionary"):
        reranker.rerank("query", ["not a dict"])  # type: ignore

    with pytest.raises(ValueError, match="missing a valid string 'text' key"):
        reranker.rerank("query", [{"chunk_id": "c1"}])

    with pytest.raises(ValueError, match="top_k must be a positive integer"):
        reranker.rerank("query", [{"text": "sample"}], top_k=0)


def test_rerank_tie_breaking(mock_encoder):
    """Verify deterministic tie breaking by chunk_id when scores are equal."""
    candidates = [
        {"chunk_id": "z_chunk", "text": "text 1", "rank": 1, "score": 0.5, "document_id": "d1", "document_name": "d.pdf", "page_start": 1, "page_end": 1},
        {"chunk_id": "a_chunk", "text": "text 2", "rank": 2, "score": 0.4, "document_id": "d1", "document_name": "d.pdf", "page_start": 1, "page_end": 1},
    ]
    # Identical score 5.0 for both
    mock_encoder.score_pairs.return_value = [5.0, 5.0]

    reranker = ResultReranker(cross_encoder=mock_encoder)
    results = reranker.rerank("query", candidates, top_k=2)

    assert len(results) == 2
    assert results[0].chunk_id == "a_chunk"
    assert results[1].chunk_id == "z_chunk"
