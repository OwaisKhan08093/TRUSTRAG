"""Regression tests for performance benchmarking suite."""

import pytest
from backend.app.agents.citation_agent import CitationAgent
from backend.app.reranking.pipeline import RerankingPipeline
from backend.app.retrieval.hybrid_retriever import HybridRetriever
from backend.app.trust.engine import TrustEngine
from scripts.benchmark_performance import (
    measure_citation_assembly_latency,
    measure_orchestrator_cold_and_warm,
    measure_reranking_latency,
    measure_retrieval_latency,
    measure_trust_engine_latency,
)


def test_retrieval_latency_benchmark():
    """Verify hybrid retrieval benchmarking computes valid stats."""
    retriever = HybridRetriever()
    res = measure_retrieval_latency(retriever, ["test query"], iterations=1)
    assert res["mean_ms"] > 0
    assert res["min_ms"] > 0
    assert res["max_ms"] >= res["min_ms"]
    assert res["total_samples"] == 1


def test_trust_engine_latency_benchmark():
    """Verify trust engine latency benchmarking executes cleanly."""
    retriever = HybridRetriever()
    pipeline = RerankingPipeline(retriever=retriever)
    engine = TrustEngine()
    res = measure_trust_engine_latency(engine, pipeline, ["test query"], iterations=1)
    assert res["mean_ms"] > 0
    assert res["total_samples"] == 1


def test_citation_assembly_latency_benchmark():
    """Verify citation assembly benchmarking executes cleanly."""
    citation_agent = CitationAgent()
    res = measure_citation_assembly_latency(citation_agent, iterations=5)
    assert res["mean_ms"] > 0
    assert res["total_samples"] == 5


def test_orchestrator_cold_warm_benchmark():
    """Verify cold and warm orchestrator benchmarking executes cleanly."""
    res = measure_orchestrator_cold_and_warm(["notice before consent", "rights of data principal"])
    assert res["cold_request_ms"] > 0
    assert res["warm_mean_ms"] > 0
    assert res["warm_samples"] == 2
