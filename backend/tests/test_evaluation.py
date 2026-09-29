"""Unit tests for semantic retrieval evaluation (Milestone 9)."""

import json
from pathlib import Path
import pytest

from backend.app.embeddings.encoder import EmbeddingEncoder
from backend.app.retrieval.retriever import VectorRetriever
from scripts.evaluate_retrieval import (
    DEFAULT_EVALUATION_FILE,
    compute_metrics,
    evaluate_retrieval,
    run_evaluation_cli,
)


@pytest.fixture(scope="module")
def encoder() -> EmbeddingEncoder:
    """Shared EmbeddingEncoder fixture."""
    return EmbeddingEncoder()


def test_compute_metrics_perfect_ranking():
    """Verify compute_metrics calculates 100% hits and MRR=1.0 for rank 1 matches."""
    records = [
        {"rank": 1},
        {"rank": 1},
        {"rank": 1},
        {"rank": 1},
    ]
    metrics = compute_metrics(records)
    assert metrics["hit_at_1"] == 1.0
    assert metrics["hit_at_3"] == 1.0
    assert metrics["hit_at_5"] == 1.0
    assert metrics["mrr"] == 1.0


def test_compute_metrics_mixed_and_misses():
    """Verify compute_metrics correctly computes Hit@K and MRR with mixed ranks and misses."""
    records = [
        {"rank": 1},      # Reciprocal = 1.0
        {"rank": 2},      # Reciprocal = 0.5
        {"rank": 4},      # Reciprocal = 0.25 (Hit@5 but not Hit@3)
        {"rank": None},   # Miss -> Reciprocal = 0.0
    ]
    metrics = compute_metrics(records)

    # Total = 4 queries
    assert metrics["hit_at_1"] == 1 / 4   # 0.25
    assert metrics["hit_at_3"] == 2 / 4   # 0.50
    assert metrics["hit_at_5"] == 3 / 4   # 0.75
    expected_mrr = (1.0 + 0.5 + 0.25 + 0.0) / 4  # 0.4375
    assert metrics["mrr"] == pytest.approx(expected_mrr, abs=1e-4)


def test_compute_metrics_empty():
    """Verify compute_metrics handles empty records safely."""
    metrics = compute_metrics([])
    assert metrics["hit_at_1"] == 0.0
    assert metrics["mrr"] == 0.0


def test_evaluate_retrieval_missing_file(tmp_path: Path):
    """Verify evaluate_retrieval raises FileNotFoundError for missing path."""
    non_existent = tmp_path / "missing_queries.json"
    with pytest.raises(FileNotFoundError):
        evaluate_retrieval(queries_file=non_existent)


def test_evaluate_retrieval_integration(encoder: EmbeddingEncoder):
    """Verify evaluate_retrieval executes against the real dataset successfully."""
    if DEFAULT_EVALUATION_FILE.exists():
        retriever = VectorRetriever(encoder=encoder)
        eval_result = evaluate_retrieval(
            queries_file=DEFAULT_EVALUATION_FILE,
            retriever=retriever,
            top_k=4,
        )

        assert eval_result["num_queries"] >= 4
        assert "metrics" in eval_result
        metrics = eval_result["metrics"]
        assert 0.0 <= metrics["hit_at_1"] <= 1.0
        assert 0.0 <= metrics["mrr"] <= 1.0
        assert len(eval_result["records"]) == eval_result["num_queries"]


def test_run_evaluation_cli():
    """Verify run_evaluation_cli exits with code 0 on real evaluation dataset."""
    if DEFAULT_EVALUATION_FILE.exists():
        exit_code = run_evaluation_cli(queries_file=DEFAULT_EVALUATION_FILE, top_k=4)
        assert exit_code == 0
