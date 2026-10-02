"""Unit tests for CitationAgent (Phase 7 Milestone 6)."""

from unittest.mock import MagicMock
import pytest

from backend.app.agents.base import AgentExecutionError, AgentInputError, AgentResult
from backend.app.agents.citation_agent import CitationAgent, CitationAgentResult
from backend.app.generation.citations import Citation
from backend.app.generation.models import GenerationResult
from backend.app.generation.response import GroundedAnswer
from backend.app.trust.engine import TrustEngine
from backend.app.trust.evidence import TrustEvidence


@pytest.fixture
def sample_evidence():
    """Fixture providing sample evidence chunks."""
    return [
        TrustEvidence(
            chunk_id="chunk_01",
            document_id="doc_dpdp",
            document_name="dpdp_act.pdf",
            page_start=5,
            page_end=5,
            text="The Data Fiduciary shall give notice before processing.",
            retrieval_rank=1,
            retrieval_score=0.15,
            rerank_score=3.8,
        ),
        TrustEvidence(
            chunk_id="chunk_02",
            document_id="doc_dpdp",
            document_name="dpdp_act.pdf",
            page_start=6,
            page_end=6,
            text="Notice must specify categories of data.",
            retrieval_rank=2,
            retrieval_score=0.10,
            rerank_score=2.5,
        ),
    ]


def test_citation_agent_extract_citations(sample_evidence):
    """Verify CitationAgent builds citations for cited markers [1]."""
    agent = CitationAgent()
    assert agent.name == "CitationAgent"
    assert "provenance" in agent.description

    answer_text = "Notice is required prior to processing personal data [1]."
    res = agent.execute(
        query="What notice is required?",
        answer_text=answer_text,
        evidence=sample_evidence,
        filter_cited_only=True,
    )

    assert isinstance(res, CitationAgentResult)
    assert res.citation_count == 1
    assert res.citations[0].index == 1
    assert res.citations[0].document_name == "dpdp_act.pdf"
    assert res.citations[0].page_start == 5
    assert "### References" in res.formatted_citations


def test_citation_agent_with_grounded_answer(sample_evidence):
    """Verify CitationAgent assembles full GroundedAnswer when assessment and generation_result are provided."""
    engine = TrustEngine()
    query = "What notice is required?"
    assessment = engine.evaluate(query, sample_evidence)

    gen_res = GenerationResult(
        query=query,
        answer="Notice is required prior to processing personal data [1].",
        model_name="qwen-test",
        evidence_ids=["chunk_01"],
        is_refusal=False,
    )

    agent = CitationAgent()
    res = agent.execute(
        query=query,
        answer_text=gen_res.answer,
        evidence=sample_evidence,
        assessment=assessment,
        generation_result=gen_res,
        filter_cited_only=True,
    )

    assert res.citation_count == 1
    assert len(res.citations) == 1
    assert isinstance(res.grounded_answer, GroundedAnswer)
    assert res.grounded_answer.confidence_score > 0.5
    assert res.grounded_answer.is_refusal is False
    assert len(res.grounded_answer.citations) == 2


def test_citation_agent_run_wrapper(sample_evidence):
    """Verify BaseAgent.run wrapper captures CitationAgent execution."""
    agent = CitationAgent()
    result = agent.run(
        query="What notice is required?",
        answer_text="Notice is required [1].",
        evidence=sample_evidence,
    )

    assert isinstance(result, AgentResult)
    assert result.success is True
    assert result.agent_name == "CitationAgent"
    assert isinstance(result.data, CitationAgentResult)
    assert result.data.citation_count == 1


def test_citation_agent_input_validation(sample_evidence):
    """Verify input validation for invalid arguments."""
    agent = CitationAgent()

    with pytest.raises(AgentInputError, match="query must be a string"):
        agent.execute(query=123, answer_text="text", evidence=sample_evidence)  # type: ignore

    with pytest.raises(AgentInputError, match="query cannot be empty"):
        agent.execute(query="   ", answer_text="text", evidence=sample_evidence)

    with pytest.raises(AgentInputError, match="answer_text must be a string"):
        agent.execute(query="valid", answer_text=123, evidence=sample_evidence)  # type: ignore

    with pytest.raises(AgentInputError, match="evidence must be a sequence"):
        agent.execute(query="valid", answer_text="text", evidence="bad")  # type: ignore


def test_citation_agent_result_to_dict():
    """Verify serialization of CitationAgentResult to dictionary."""
    cit = Citation(
        index=1,
        chunk_id="c1",
        document_id="d1",
        document_name="dpdp.pdf",
        page_start=1,
        page_end=1,
        text_snippet="Sample text",
        formatted_reference="[1] dpdp.pdf, page 1",
    )
    res = CitationAgentResult(
        query="test query",
        citations=[cit],
        citation_count=1,
        formatted_citations="[1] dpdp.pdf, page 1",
    )
    d = res.to_dict()
    assert d["query"] == "test query"
    assert d["citation_count"] == 1
    assert len(d["citations"]) == 1
    assert d["citations"][0]["document_name"] == "dpdp.pdf"
