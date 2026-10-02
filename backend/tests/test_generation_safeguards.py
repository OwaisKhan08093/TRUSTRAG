"""Edge-case tests and validation for generation safeguards (Phase 6 Milestone 8)."""

from unittest.mock import MagicMock
import pytest

from backend.app.generation.citations import CitationError, build_citation, build_citation_references
from backend.app.generation.config import GenerationConfig, GenerationConfigError
from backend.app.generation.generator import GroundedGenerator
from backend.app.generation.models import GenerationRequest, GenerationResult, ModelValidationError
from backend.app.generation.prompt import PromptBuildingError, build_grounded_prompt
from backend.app.trust.coverage import CoverageAssessment
from backend.app.trust.engine import TrustAssessment, TrustDecision
from backend.app.trust.evidence import TrustEvidence
from backend.app.trust.groundedness import GroundednessAssessment
from backend.app.trust.provenance import ProvenanceReport
from backend.app.trust.relevance import RelevanceAssessment


def test_generation_config_all_parameter_safeguards():
    """Verify all parameter invariants are strictly guarded."""
    # max_new_tokens > 0
    with pytest.raises(GenerationConfigError):
        GenerationConfig(max_new_tokens=0)
    with pytest.raises(GenerationConfigError):
        GenerationConfig(max_new_tokens=-10)

    # temperature >= 0.0
    with pytest.raises(GenerationConfigError):
        GenerationConfig(temperature=-0.01)

    # top_p in (0.0, 1.0]
    with pytest.raises(GenerationConfigError):
        GenerationConfig(top_p=0.0)
    with pytest.raises(GenerationConfigError):
        GenerationConfig(top_p=1.01)

    # repetition_penalty > 0.0
    with pytest.raises(GenerationConfigError):
        GenerationConfig(repetition_penalty=0.0)
    with pytest.raises(GenerationConfigError):
        GenerationConfig(repetition_penalty=-1.0)

    # model_name non-empty
    with pytest.raises(GenerationConfigError):
        GenerationConfig(model_name="   ")


def test_prompt_building_excessive_context_safeguard():
    """Verify prompt truncation safeguard when context exceeds max_prompt_chars."""
    huge_evidence = [
        TrustEvidence(
            chunk_id="chunk_huge",
            document_id="doc_huge",
            document_name="huge_doc.pdf",
            page_start=1,
            page_end=1,
            text="A" * 5000,
            retrieval_rank=1,
            retrieval_score=0.1,
            rerank_score=2.0,
        )
    ]
    prompt = build_grounded_prompt(
        query="What is A?",
        evidence=huge_evidence,
        max_prompt_chars=500,
    )
    assert len(prompt.evidence_context) < 600
    assert "[... truncated for context limits ...]" in prompt.evidence_context


def test_generator_gating_safeguards():
    """Verify generator refuses execution for any non-SUPPORTED status."""
    mock_llm = MagicMock()
    mock_llm.config = GenerationConfig()
    mock_llm.model_name = "Qwen/Qwen2.5-3B-Instruct"
    generator = GroundedGenerator(llm=mock_llm)

    rel = RelevanceAssessment(0.0, 0.0, 0.0, 0.0, [], [])
    cov = CoverageAssessment([], [], [], 0.0, [], 0)
    prov = ProvenanceReport(False, 1, 0, 1, [])
    ground = GroundednessAssessment(0.0, 0.0, 0.0, 0.0, {}, 1, rel, cov, prov)
    from backend.app.trust.confidence import ConfidenceAssessment
    conf = ConfidenceAssessment(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1, ground)

    bad_assessment = TrustAssessment(
        query="Query with corrupt provenance",
        decision=TrustDecision.INSUFFICIENT_EVIDENCE,
        relevance_score=0.0,
        coverage_score=0.0,
        groundedness_score=0.0,
        confidence_score=0.0,
        provenance_valid=False,
        evidence_count=1,
        decision_reasons=["Provenance validation failed."],
        relevance_details=rel,
        coverage_details=cov,
        groundedness_details=ground,
        confidence_details=conf,
        provenance_details=prov,
    )

    result = generator.generate(
        query="Query with corrupt provenance",
        evidence=[],
        assessment=bad_assessment,
    )

    assert result.is_refusal is True
    assert "INSUFFICIENT_EVIDENCE" in result.refusal_reason
    assert not mock_llm.generate.called


def test_citation_missing_provenance_safeguard():
    """Verify malformed evidence items cannot generate fake citations."""
    corrupt_item = {
        "chunk_id": "c1",
        # missing document_id and document_name
        "page_start": 1,
        "page_end": 1,
        "text": "Some text",
    }
    with pytest.raises(CitationError):
        build_citation(1, corrupt_item)
