"""Unit tests for TrustAgent (Phase 7 Milestone 4)."""

from unittest.mock import MagicMock
import pytest

from backend.app.agents.base import AgentExecutionError, AgentInputError, AgentResult
from backend.app.agents.trust_agent import TrustAgent, TrustAgentResult
from backend.app.trust.engine import TrustDecision, TrustEngine
from backend.app.trust.evidence import TrustEvidence


@pytest.fixture
def sample_valid_evidence() -> list:
    """Fixture providing high-quality grounded evidence."""
    return [
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
        ),
        TrustEvidence(
            chunk_id="chunk_02",
            document_id="doc_dpdp",
            document_name="dpdp_act.pdf",
            page_start=6,
            page_end=6,
            text="Notice must contain details of personal data to be processed and purpose.",
            retrieval_rank=2,
            retrieval_score=0.10,
            rerank_score=2.5,
        ),
    ]


def test_trust_agent_supported_decision(sample_valid_evidence):
    """Verify TrustAgent correctly issues SUPPORTED decision with real TrustEngine."""
    agent = TrustAgent()
    assert agent.name == "TrustAgent"
    assert "deterministic SUPPORTED vs INSUFFICIENT_EVIDENCE" in agent.description

    res = agent.execute(
        query="Data Fiduciary notice requirements for consent",
        evidence=sample_valid_evidence,
    )

    assert isinstance(res, TrustAgentResult)
    assert res.decision == TrustDecision.SUPPORTED
    assert res.is_supported is True
    assert res.confidence_score > 0.5
    assert res.groundedness_score > 0.5
    assert res.provenance_valid is True
    assert len(res.decision_reasons) > 0


def test_trust_agent_insufficient_evidence_decision():
    """Verify TrustAgent correctly issues INSUFFICIENT_EVIDENCE decision on empty/unrelated evidence."""
    agent = TrustAgent()
    res = agent.execute(
        query="Explain quantum gravity string theory in Indian law",
        evidence=[],
    )

    assert isinstance(res, TrustAgentResult)
    assert res.decision == TrustDecision.INSUFFICIENT_EVIDENCE
    assert res.is_supported is False
    assert res.confidence_score == 0.0


def test_trust_agent_run_wrapper(sample_valid_evidence):
    """Verify BaseAgent.run wrapper captures TrustAgent outputs."""
    agent = TrustAgent()
    result = agent.run(
        query="Data Fiduciary notice requirements",
        evidence=sample_valid_evidence,
    )

    assert isinstance(result, AgentResult)
    assert result.success is True
    assert result.agent_name == "TrustAgent"
    assert isinstance(result.data, TrustAgentResult)
    assert result.data.is_supported is True


def test_trust_agent_input_validation():
    """Verify input validation for invalid query and evidence inputs."""
    agent = TrustAgent()

    with pytest.raises(AgentInputError, match="query must be a string"):
        agent.execute(query=1234, evidence=[])  # type: ignore

    with pytest.raises(AgentInputError, match="query cannot be empty"):
        agent.execute(query="   ", evidence=[])

    with pytest.raises(AgentInputError, match="evidence must be a sequence"):
        agent.execute(query="valid", evidence="not_a_list")  # type: ignore


def test_trust_agent_execution_error():
    """Verify unexpected runtime errors during evaluation are caught."""
    engine = MagicMock(spec=TrustEngine)
    engine.evaluate.side_effect = RuntimeError("Evaluation crash")
    agent = TrustAgent(trust_engine=engine)

    with pytest.raises(AgentExecutionError, match="Trust evaluation failed"):
        agent.execute(query="valid", evidence=[])


def test_trust_agent_result_to_dict(sample_valid_evidence):
    """Verify serialization of TrustAgentResult to dictionary."""
    agent = TrustAgent()
    res = agent.execute(
        query="Data Fiduciary notice requirements",
        evidence=sample_valid_evidence,
    )
    d = res.to_dict()
    assert d["query"] == "Data Fiduciary notice requirements"
    assert d["decision"] == "SUPPORTED"
    assert d["is_supported"] is True
    assert isinstance(d["confidence_score"], float)
    assert isinstance(d["assessment"], dict)
