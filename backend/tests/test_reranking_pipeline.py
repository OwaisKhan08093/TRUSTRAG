"""Unit tests for RerankingPipeline orchestration."""

from unittest.mock import MagicMock
import pytest

from backend.app.reranking.pipeline import RerankingPipeline, RerankingPipelineError
from backend.app.reranking.reranker import ResultReranker
from backend.app.reranking.schema import RerankedChunk
from backend.app.retrieval.hybrid_retriever import HybridRetriever


@pytest.fixture
def mock_components():
    """Create mock HybridRetriever and ResultReranker."""
    retriever = MagicMock(spec=HybridRetriever)
    reranker = MagicMock(spec=ResultReranker)
    return retriever, reranker


def test_pipeline_init(mock_components):
    """Verify RerankingPipeline initializes with custom and default parameters."""
    retriever, reranker = mock_components
    pipeline = RerankingPipeline(
        retriever=retriever,
        reranker=reranker,
        default_candidate_k=10,
        default_final_k=5,
        default_batch_size=8,
        model_name="test-model",
    )
    assert pipeline.retriever is retriever
    assert pipeline.reranker is reranker
    assert pipeline.candidate_k == 10
    assert pipeline.final_k == 5
    assert pipeline.batch_size == 8
    assert pipeline.model_name == "test-model"


def test_pipeline_init_validation(mock_components):
    """Verify constructor validations."""
    retriever, reranker = mock_components

    # final_k > candidate_k
    with pytest.raises(ValueError, match="cannot exceed candidate_k"):
        RerankingPipeline(
            retriever=retriever,
            reranker=reranker,
            default_candidate_k=3,
            default_final_k=5,
        )


def test_pipeline_retrieve_flow(mock_components):
    """Verify pipeline coordinates hybrid retrieval followed by neural reranking."""
    retriever, reranker = mock_components

    raw_candidates = [
        {"chunk_id": "c1", "text": "text 1", "rank": 1, "score": 0.03, "document_id": "d1", "document_name": "d.pdf", "page_start": 1, "page_end": 1},
        {"chunk_id": "c2", "text": "text 2", "rank": 2, "score": 0.02, "document_id": "d1", "document_name": "d.pdf", "page_start": 2, "page_end": 2},
    ]
    retriever.retrieve.return_value = raw_candidates

    reranked_chunks = [
        RerankedChunk(
            chunk_id="c2",
            document_id="d1",
            document_name="d.pdf",
            page_start=2,
            page_end=2,
            text="text 2",
            original_rank=2,
            original_score=0.02,
            rerank_score=9.1,
            final_rank=1,
        )
    ]
    reranker.rerank.return_value = reranked_chunks

    pipeline = RerankingPipeline(
        retriever=retriever,
        reranker=reranker,
        default_candidate_k=5,
        default_final_k=1,
    )

    results = pipeline.retrieve("test query", top_k=1, candidate_k=4)

    retriever.retrieve.assert_called_once_with(query="test query", top_k=4)
    reranker.rerank.assert_called_once_with(
        query="test query",
        results=raw_candidates,
        top_k=1,
        batch_size=16,
    )

    assert len(results) == 1
    assert results[0].chunk_id == "c2"
    assert results[0].final_rank == 1


def test_pipeline_retrieve_empty_candidates(mock_components):
    """Verify pipeline returns empty list when retrieval finds 0 candidates."""
    retriever, reranker = mock_components
    retriever.retrieve.return_value = []

    pipeline = RerankingPipeline(retriever=retriever, reranker=reranker)
    results = pipeline.retrieve("no match query")

    assert results == []
    reranker.rerank.assert_not_called()


def test_pipeline_error_wrapping(mock_components):
    """Verify underlying retrieval and reranking exceptions are wrapped."""
    retriever, reranker = mock_components

    # Retrieval failure
    retriever.retrieve.side_effect = RuntimeError("FAISS error")
    pipeline = RerankingPipeline(retriever=retriever, reranker=reranker)
    with pytest.raises(RerankingPipelineError, match="Hybrid candidate retrieval failed"):
        pipeline.retrieve("query")

    # Reranking failure
    retriever.retrieve.side_effect = None
    retriever.retrieve.return_value = [{"chunk_id": "c1", "text": "sample"}]
    reranker.rerank.side_effect = RuntimeError("Cross-encoder OOM")
    with pytest.raises(RerankingPipelineError, match="Cross-encoder reranking failed"):
        pipeline.retrieve("query")
