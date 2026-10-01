"""Unit tests for Cross-Encoder Reranking evaluation (Milestone 8)."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from backend.app.config import DEFAULT_RERANKER_CANDIDATE_K, DEFAULT_RERANKER_TOP_K
from backend.app.reranking.pipeline import RerankingPipeline
from backend.app.reranking.schema import RerankedChunk
from backend.app.retrieval.hybrid_retriever import HybridRetriever
from scripts.evaluate_reranking import (
    evaluate_reranked_pipeline,
    run_reranking_evaluation_cli,
)
from scripts.evaluate_retrieval import DEFAULT_EVALUATION_FILE


def test_evaluate_reranked_pipeline_missing_file(tmp_path: Path):
    """Verify evaluate_reranked_pipeline raises FileNotFoundError for missing path."""
    non_existent = tmp_path / "non_existent_queries.json"
    with pytest.raises(FileNotFoundError):
        evaluate_reranked_pipeline(queries_file=non_existent)


def test_evaluate_reranked_pipeline_empty_file(tmp_path: Path):
    """Verify evaluate_reranked_pipeline raises ValueError for empty query list."""
    empty_file = tmp_path / "empty_queries.json"
    with open(empty_file, "w", encoding="utf-8") as f:
        json.dump([], f)

    with pytest.raises(ValueError, match="empty or not a valid JSON list"):
        evaluate_reranked_pipeline(queries_file=empty_file)


def test_evaluate_reranked_pipeline_with_mocks(tmp_path: Path):
    """Verify evaluation calculations against mock hybrid and pipeline responses."""
    queries = [
        {"query": "query 1", "expected_chunk_id": "chunk_1", "expected_page": 1},
        {"query": "query 2", "expected_chunk_id": "chunk_2", "expected_page": 2},
    ]
    query_file = tmp_path / "test_queries.json"
    with open(query_file, "w", encoding="utf-8") as f:
        json.dump(queries, f)

    mock_hybrid = MagicMock(spec=HybridRetriever)
    mock_pipeline = MagicMock(spec=RerankingPipeline)

    # Query 1: hybrid hits rank 2, reranker hits rank 1
    # Query 2: hybrid hits rank 1, reranker hits rank 1
    mock_hybrid.retrieve.side_effect = [
        [
            {"chunk_id": "chunk_other", "rank": 1, "score": 0.05, "page_start": 5, "page_end": 5},
            {"chunk_id": "chunk_1", "rank": 2, "score": 0.04, "page_start": 1, "page_end": 1},
        ],
        [
            {"chunk_id": "chunk_2", "rank": 1, "score": 0.06, "page_start": 2, "page_end": 2},
        ],
    ]

    mock_pipeline.retrieve.side_effect = [
        [
            RerankedChunk(
                chunk_id="chunk_1",
                document_id="doc1",
                document_name="doc.pdf",
                page_start=1,
                page_end=1,
                text="text 1",
                original_rank=2,
                original_score=0.04,
                rerank_score=0.95,
                final_rank=1,
            ),
            RerankedChunk(
                chunk_id="chunk_other",
                document_id="doc1",
                document_name="doc.pdf",
                page_start=5,
                page_end=5,
                text="text other",
                original_rank=1,
                original_score=0.05,
                rerank_score=0.30,
                final_rank=2,
            ),
        ],
        [
            RerankedChunk(
                chunk_id="chunk_2",
                document_id="doc1",
                document_name="doc.pdf",
                page_start=2,
                page_end=2,
                text="text 2",
                original_rank=1,
                original_score=0.06,
                rerank_score=0.99,
                final_rank=1,
            ),
        ],
    ]

    result = evaluate_reranked_pipeline(
        queries_file=query_file,
        pipeline=mock_pipeline,
        hybrid_retriever=mock_hybrid,
        candidate_k=5,
        final_k=2,
    )

    assert result["num_queries"] == 2
    assert result["candidate_k"] == 5
    assert result["final_k"] == 2

    # Hybrid metrics: Q1 rank 2, Q2 rank 1 -> Hit@1 = 0.5, Hit@3 = 1.0, MRR = (0.5 + 1.0)/2 = 0.75
    h_m = result["hybrid"]["metrics"]
    assert h_m["hit_at_1"] == 0.5
    assert h_m["hit_at_3"] == 1.0
    assert h_m["mrr"] == 0.75

    # Reranked metrics: Q1 rank 1, Q2 rank 1 -> Hit@1 = 1.0, Hit@3 = 1.0, MRR = 1.0
    r_m = result["reranked"]["metrics"]
    assert r_m["hit_at_1"] == 1.0
    assert r_m["hit_at_3"] == 1.0
    assert r_m["mrr"] == 1.0


def test_run_reranking_evaluation_cli(tmp_path: Path, capsys):
    """Verify CLI runner handles successful execution output."""
    queries = [{"query": "test query", "expected_chunk_id": "c1", "expected_page": 1}]
    query_file = tmp_path / "cli_queries.json"
    with open(query_file, "w", encoding="utf-8") as f:
        json.dump(queries, f)

    mock_hybrid = MagicMock(spec=HybridRetriever)
    mock_pipeline = MagicMock(spec=RerankingPipeline)
    mock_hybrid.retrieve.return_value = [{"chunk_id": "c1", "rank": 1, "score": 0.05, "page_start": 1, "page_end": 1}]
    mock_pipeline.retrieve.return_value = [
        RerankedChunk(
            chunk_id="c1",
            document_id="d1",
            document_name="d.pdf",
            page_start=1,
            page_end=1,
            text="text",
            original_rank=1,
            original_score=0.05,
            rerank_score=0.95,
            final_rank=1,
        )
    ]

    with patch("scripts.evaluate_reranking.HybridRetriever", return_value=mock_hybrid), \
         patch("scripts.evaluate_reranking.RerankingPipeline", return_value=mock_pipeline):
        exit_code = run_reranking_evaluation_cli(queries_file=query_file)

    assert exit_code == 0
    captured = capsys.readouterr()
    assert "TrustRAG Cross-Encoder Reranking Evaluation" in captured.out
    assert "Hit@1" in captured.out
    assert "MRR" in captured.out
    assert "Overall Status: SUCCESS" in captured.out
