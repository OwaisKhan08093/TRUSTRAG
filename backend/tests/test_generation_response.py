"""Unit tests for GroundedAnswer assembly (Phase 6 Milestone 7)."""

import pytest

from backend.app.generation.models import GenerationResult
from backend.app.generation.response import (
    AnswerAssemblyError,
    GroundedAnswer,
    assemble_grounded_answer,
)
from backend.app.trust.coverage import CoverageAssessment
from backend.app.trust.engine import TrustAssessment, TrustDecision
from backend.app.trust.evidence import TrustEvidence
from backend.app.trust.groundedness import GroundednessAssessment
from backend.app.trust.provenance import ProvenanceReport
from backend.app.trust.relevance import RelevanceAssessment


@pytest.fixture
def sample_assessment():
    """Fixture providing a valid SUPPORTED assessment."""
    rel = RelevanceAssessment(3.0, 0.95, 0.95, 0.95, [3.0], [0.95])
    cov = CoverageAssessment(["notice"], ["notice"], [], 1.0, [1.0], 1)
    prov = ProvenanceReport(True, 1, 1, 0, [])
    ground = GroundednessAssessment(0.95, 0.95, 1.0, 1.0, {}, 1, rel, cov, prov)
    from backend.app.trust.confidence import ConfidenceAssessment
    conf = ConfidenceAssessment(0.95, 0.95, 0.95, 1.0, 1.0, 1.0, 1, ground)

    return TrustAssessment(
        query="What is the notice rule?",
        decision=TrustDecision.SUPPORTED,
        relevance_score=0.95,
        coverage_score=1.0,
        groundedness_score=0.95,
        confidence_score=0.95,
        provenance_valid=True,
        evidence_count=1,
        decision_reasons=["Fully grounded."],
        relevance_details=rel,
        coverage_details=cov,
        groundedness_details=ground,
        confidence_details=conf,
        provenance_details=prov,
    )


def test_assemble_grounded_answer_supported(sample_assessment):
    """Verify complete assembly of answer with citations and confidence metrics."""
    ev = TrustEvidence(
        chunk_id="c1",
        document_id="d1",
        document_name="dpdp_act.pdf",
        page_start=5,
        page_end=5,
        text="Data Fiduciary notice obligations.",
        retrieval_rank=1,
        retrieval_score=0.1,
        rerank_score=3.0,
    )

    gen_res = GenerationResult(
        query="What is the notice rule?",
        answer="A Data Fiduciary must give notice [1].",
        model_name="Qwen/Qwen2.5-3B-Instruct",
        evidence_ids=["c1"],
        is_refusal=False,
    )

    grounded_ans = assemble_grounded_answer(
        generation_result=gen_res,
        assessment=sample_assessment,
        evidence=[ev],
    )

    assert isinstance(grounded_ans, GroundedAnswer)
    assert grounded_ans.is_refusal is False
    assert len(grounded_ans.citations) == 1
    assert grounded_ans.citations[0].document_name == "dpdp_act.pdf"
    assert grounded_ans.confidence_score == 0.95
    assert grounded_ans.trust_decision == TrustDecision.SUPPORTED
    assert "### References" in grounded_ans.formatted_response
    assert "[1] dpdp_act.pdf, page 5" in grounded_ans.formatted_response


def test_assemble_grounded_answer_refusal():
    """Verify assembly of refusal when TrustEngine rejected generation."""
    rel = RelevanceAssessment(0.0, 0.0, 0.0, 0.0, [], [])
    cov = CoverageAssessment([], [], [], 0.0, [], 0)
    prov = ProvenanceReport(True, 0, 0, 0, [])
    ground = GroundednessAssessment(0.0, 0.0, 0.0, 1.0, {}, 0, rel, cov, prov)
    from backend.app.trust.confidence import ConfidenceAssessment
    conf = ConfidenceAssessment(0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0, ground)

    assessment = TrustAssessment(
        query="unsupported query",
        decision=TrustDecision.INSUFFICIENT_EVIDENCE,
        relevance_score=0.0,
        coverage_score=0.0,
        groundedness_score=0.0,
        confidence_score=0.0,
        provenance_valid=True,
        evidence_count=0,
        decision_reasons=["No evidence"],
        relevance_details=rel,
        coverage_details=cov,
        groundedness_details=ground,
        confidence_details=conf,
        provenance_details=prov,
    )

    gen_res = GenerationResult(
        query="unsupported query",
        answer="I am unable to answer because evidence is insufficient.",
        model_name="Qwen/Qwen2.5-3B-Instruct",
        evidence_ids=[],
        is_refusal=True,
        refusal_reason="Gated by TrustEngine.",
    )

    grounded_ans = assemble_grounded_answer(
        generation_result=gen_res,
        assessment=assessment,
        evidence=[],
    )

    assert grounded_ans.is_refusal is True
    assert grounded_ans.citations == []
    assert "[Status: INSUFFICIENT_EVIDENCE]" in grounded_ans.formatted_response


def test_assemble_grounded_answer_to_dict(sample_assessment):
    """Verify dictionary serialization."""
    gen_res = GenerationResult(
        query="query",
        answer="answer",
        model_name="m",
        evidence_ids=[],
    )
    ans = assemble_grounded_answer(gen_res, sample_assessment, [])
    d = ans.to_dict()
    assert d["query"] == "query"
    assert d["answer"] == "answer"
    assert "assessment" in d
    assert "generation_result" in d
