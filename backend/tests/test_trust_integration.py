"""Integration tests for the complete Trust Engine workflow (Retrieval -> Reranking -> Trust)."""

import pytest

from backend.app.reranking.pipeline import RerankingPipeline
from backend.app.retrieval.hybrid_retriever import HybridRetriever
from backend.app.trust.config import TrustConfig
from backend.app.trust.engine import TrustAssessment, TrustDecision, TrustEngine
from backend.app.trust.evidence import TrustEvidence
from scripts.evaluate_trust import DEV_EVALUATION_SCENARIOS, evaluate_trust_scenarios


def test_trust_engine_evaluation_scenarios_all_pass():
    """Verify all internal development evaluation scenarios produce expected decisions."""
    results = evaluate_trust_scenarios()
    assert results["total_scenarios"] == 5
    assert results["correct_decisions"] == 5
    assert results["accuracy"] == 1.0


def test_full_pipeline_retrieval_rerank_to_trust():
    """Verify end-to-end flow from HybridRetriever through RerankingPipeline to TrustEngine."""
    retriever = HybridRetriever()
    pipeline = RerankingPipeline(retriever=retriever)
    engine = TrustEngine()

    query = "What are the obligations of a Data Fiduciary regarding notice and consent?"
    # 1. Retrieve candidates & rerank with cross-encoder
    reranked_chunks = pipeline.retrieve(query, top_k=3, candidate_k=5)

    assert len(reranked_chunks) > 0

    # 2. Convert to strongly-typed TrustEvidence
    evidence_items = [TrustEvidence.from_reranked_chunk(c) for c in reranked_chunks]

    # 3. Evaluate evidence grounding with TrustEngine
    assessment = engine.evaluate(query, evidence_items)

    assert isinstance(assessment, TrustAssessment)
    assert assessment.evidence_count == len(evidence_items)
    assert assessment.provenance_valid is True
    assert 0.0 <= assessment.groundedness_score <= 1.0
    assert 0.0 <= assessment.confidence_score <= 1.0
    assert assessment.decision in {TrustDecision.SUPPORTED, TrustDecision.INSUFFICIENT_EVIDENCE}


def test_full_pipeline_unrelated_query_insufficient_evidence():
    """Verify that an unrelated query retrieved on the DPDP index is flagged as INSUFFICIENT_EVIDENCE."""
    retriever = HybridRetriever()
    pipeline = RerankingPipeline(retriever=retriever)
    engine = TrustEngine()

    # Query completely outside the scope of DPDP documents
    query = "photosynthesis chlorophyll light absorption mechanism in botany plants"
    reranked_chunks = pipeline.retrieve(query, top_k=3, candidate_k=5)

    evidence_items = [TrustEvidence.from_reranked_chunk(c) for c in reranked_chunks]
    assessment = engine.evaluate(query, evidence_items)

    assert assessment.decision == TrustDecision.INSUFFICIENT_EVIDENCE
    assert assessment.groundedness_score < 0.45 or assessment.coverage_score < 0.30
