"""Unit tests for GenerationAgent (Phase 7 Milestone 5)."""

from unittest.mock import MagicMock
import pytest

from backend.app.agents.base import AgentExecutionError, AgentInputError, AgentResult
from backend.app.agents.generation_agent import GenerationAgent, GenerationAgentResult
from backend.app.generation.generator import GroundedGenerator, STANDARD_ABSTENTION_MESSAGE
from backend.app.generation.llm import LocalLLM
from backend.app.generation.models import GenerationResult
from backend.app.trust.engine import TrustAssessment, TrustDecision, TrustEngine
from backend.app.trust.evidence import TrustEvidence


@pytest.fixture
def mock_evidence_and_assessment():
    """Fixture providing evidence and real TrustAssessment."""
    evidence = [
        TrustEvidence(
            chunk_id="chunk_01",
            document_id="doc_dpdp",
            document_name="dpdp_act.pdf",
            page_start=5,
            page_end=5,
            text="The Data Fiduciary shall give notice to the Data Principal before requesting consent.",
            retrieval_rank=1,
            retrieval_score=0.15,
            rerank_score=3.8,
        )
    ]
    engine = TrustEngine()
    assessment = engine.evaluate("Data Fiduciary notice requirements for consent", evidence)
    return evidence, assessment


@pytest.fixture
def mock_insufficient_assessment():
    """Fixture providing empty evidence and INSUFFICIENT_EVIDENCE assessment."""
    engine = TrustEngine()
    assessment = engine.evaluate("Explain unknown topic", [])
    return assessment


def test_generation_agent_supported_generation(mock_evidence_and_assessment):
    """Verify GenerationAgent successfully coordinates answer generation for SUPPORTED assessments."""
    evidence, assessment = mock_evidence_and_assessment

    mock_generator = MagicMock(spec=GroundedGenerator)
    mock_gen_result = GenerationResult(
        query="Data Fiduciary notice requirements for consent",
        answer="According to Section 5 [1], notice is required before requesting consent.",
        model_name="test-model",
        evidence_ids=["chunk_01"],
        is_refusal=False,
    )
    mock_generator.generate.return_value = mock_gen_result

    agent = GenerationAgent(generator=mock_generator)
    assert agent.name == "GenerationAgent"
    assert "gated by TrustEngine" in agent.description

    res = agent.execute(
        query="Data Fiduciary notice requirements for consent",
        evidence=evidence,
        assessment=assessment,
    )

    assert isinstance(res, GenerationAgentResult)
    assert res.is_supported is True
    assert res.is_refusal is False
    assert "[1]" in res.answer_text
    assert mock_generator.generate.called


def test_generation_agent_refusal_when_insufficient(mock_insufficient_assessment):
    """Verify GroundedGenerator structured refusal is returned and LLM is not called."""
    mock_llm = MagicMock(spec=LocalLLM)
    # GroundedGenerator initialized with mock LLM:
    generator = GroundedGenerator(llm=mock_llm)
    agent = GenerationAgent(generator=generator)

    res = agent.execute(
        query="Explain unknown topic",
        evidence=[],
        assessment=mock_insufficient_assessment,
    )

    assert isinstance(res, GenerationAgentResult)
    assert res.is_supported is False
    assert res.is_refusal is True
    assert res.answer_text == STANDARD_ABSTENTION_MESSAGE
    # LLM generate MUST NOT be called:
    assert not mock_llm.generate.called


def test_generation_agent_run_wrapper(mock_evidence_and_assessment):
    """Verify BaseAgent.run wrapper captures GenerationAgent execution."""
    evidence, assessment = mock_evidence_and_assessment
    mock_generator = MagicMock(spec=GroundedGenerator)
    mock_generator.generate.return_value = GenerationResult(
        query="Data Fiduciary notice requirements",
        answer="Notice answer [1].",
        model_name="test-model",
        evidence_ids=["chunk_01"],
        is_refusal=False,
    )
    agent = GenerationAgent(generator=mock_generator)
    result = agent.run(
        query="Data Fiduciary notice requirements",
        evidence=evidence,
        assessment=assessment,
    )

    assert isinstance(result, AgentResult)
    assert result.success is True
    assert result.agent_name == "GenerationAgent"
    assert isinstance(result.data, GenerationAgentResult)


def test_generation_agent_input_validation(mock_evidence_and_assessment):
    """Verify input validation for invalid query, evidence, and assessment."""
    evidence, assessment = mock_evidence_and_assessment
    agent = GenerationAgent()

    with pytest.raises(AgentInputError, match="query must be a string"):
        agent.execute(query=123, evidence=evidence, assessment=assessment)  # type: ignore

    with pytest.raises(AgentInputError, match="query cannot be empty"):
        agent.execute(query="   ", evidence=evidence, assessment=assessment)

    with pytest.raises(AgentInputError, match="evidence must be a sequence"):
        agent.execute(query="valid", evidence="bad", assessment=assessment)  # type: ignore

    with pytest.raises(AgentInputError, match="assessment must be an instance of TrustAssessment"):
        agent.execute(query="valid", evidence=evidence, assessment="not_an_assessment")  # type: ignore


def test_generation_agent_execution_error(mock_evidence_and_assessment):
    """Verify unexpected runtime errors during generation are wrapped."""
    evidence, assessment = mock_evidence_and_assessment
    mock_generator = MagicMock(spec=GroundedGenerator)
    mock_generator.generate.side_effect = RuntimeError("Generation failed")

    agent = GenerationAgent(generator=mock_generator)
    with pytest.raises(AgentExecutionError, match="Grounded generation failed"):
        agent.execute(query="valid query", evidence=evidence, assessment=assessment)


def test_generation_agent_result_to_dict():
    """Verify serialization of GenerationAgentResult to dictionary."""
    gen_res = GenerationResult(
        query="sample query",
        answer="sample answer",
        model_name="test-llm",
        evidence_ids=["chunk_01"],
        is_refusal=False,
    )
    res = GenerationAgentResult(
        query="sample query",
        answer_text="sample answer",
        is_refusal=False,
        is_supported=True,
        generation_result=gen_res,
        metadata={"model": "test-llm"},
    )
    d = res.to_dict()
    assert d["query"] == "sample query"
    assert d["answer_text"] == "sample answer"
    assert d["is_refusal"] is False
    assert d["is_supported"] is True
    assert isinstance(d["generation_result"], dict)
