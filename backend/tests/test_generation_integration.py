"""Integration tests for the complete generation pipeline (Retrieval -> Rerank -> Trust -> Gen -> Citation)."""

from unittest.mock import MagicMock
import pytest

from backend.app.generation.config import GenerationConfig
from backend.app.generation.generator import GroundedGenerator
from backend.app.generation.llm import LocalLLM
from backend.app.generation.models import GenerationResult
from backend.app.generation.response import GroundedAnswer, assemble_grounded_answer
from backend.app.reranking.pipeline import RerankingPipeline
from backend.app.retrieval.hybrid_retriever import HybridRetriever
from backend.app.trust.engine import TrustDecision, TrustEngine
from backend.app.trust.evidence import TrustEvidence
from scripts.evaluate_generation import evaluate_generation_scenarios


def test_generation_evaluation_scenarios_pass():
    """Verify all internal development generation evaluation scenarios pass."""
    results = evaluate_generation_scenarios()
    assert results["total_scenarios"] == 4
    assert results["correct_gating_count"] == 4
    assert results["gating_accuracy"] == 1.0


def test_full_pipeline_from_retrieval_to_grounded_answer():
    """Verify end-to-end execution from HybridRetriever through TrustEngine to GroundedAnswer."""
    retriever = HybridRetriever()
    pipeline = RerankingPipeline(retriever=retriever)
    engine = TrustEngine()

    mock_llm = MagicMock(spec=LocalLLM)
    mock_llm.config = GenerationConfig()
    mock_llm.model_name = "Qwen/Qwen2.5-3B-Instruct"
    mock_llm.generate.return_value = (
        "A Data Fiduciary must give notice to the Data Principal before requesting consent [1]."
    )
    generator = GroundedGenerator(llm=mock_llm)

    query = "What notice must a Data Fiduciary give before requesting consent?"

    # 1. Retrieval & Reranking
    reranked = pipeline.retrieve(query, top_k=3, candidate_k=5)
    evidence = [TrustEvidence.from_reranked_chunk(c) for c in reranked]

    # 2. Trust Assessment
    assessment = engine.evaluate(query, evidence)
    assert assessment.decision == TrustDecision.SUPPORTED

    # 3. Grounded Generation
    gen_result = generator.generate(query, evidence, assessment)
    assert gen_result.is_refusal is False
    assert mock_llm.generate.called

    # 4. Answer + Citation Assembly
    grounded_ans = assemble_grounded_answer(gen_result, assessment, evidence)

    assert isinstance(grounded_ans, GroundedAnswer)
    assert grounded_ans.trust_decision == TrustDecision.SUPPORTED
    assert len(grounded_ans.citations) > 0
    assert "### References" in grounded_ans.formatted_response
    assert "[1]" in grounded_ans.formatted_response


def test_full_pipeline_unrelated_query_refusal():
    """Verify unrelated queries retrieved from index are safely refused by GroundedGenerator."""
    retriever = HybridRetriever()
    pipeline = RerankingPipeline(retriever=retriever)
    engine = TrustEngine()

    mock_llm = MagicMock(spec=LocalLLM)
    mock_llm.config = GenerationConfig()
    mock_llm.model_name = "Qwen/Qwen2.5-3B-Instruct"
    generator = GroundedGenerator(llm=mock_llm)

    unrelated_query = "photosynthesis chlorophyll electron transport in green plants"

    reranked = pipeline.retrieve(unrelated_query, top_k=3, candidate_k=5)
    evidence = [TrustEvidence.from_reranked_chunk(c) for c in reranked]

    assessment = engine.evaluate(unrelated_query, evidence)
    assert assessment.decision == TrustDecision.INSUFFICIENT_EVIDENCE

    gen_result = generator.generate(unrelated_query, evidence, assessment)
    assert gen_result.is_refusal is True
    # Verify LLM was NOT invoked
    assert not mock_llm.generate.called

    grounded_ans = assemble_grounded_answer(gen_result, assessment, evidence)
    assert grounded_ans.is_refusal is True
    assert grounded_ans.citations == []
