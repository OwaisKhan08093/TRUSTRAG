"""Unit tests for EvidenceAgent (Phase 7 Milestone 3)."""

from unittest.mock import MagicMock
import pytest

from backend.app.agents.base import AgentExecutionError, AgentInputError, AgentResult
from backend.app.agents.evidence_agent import EvidenceAgent, EvidenceAgentResult
from backend.app.reranking.reranker import ResultReranker
from backend.app.reranking.schema import RerankedChunk


@pytest.fixture
def mock_reranker():
    """Fixture providing a mocked ResultReranker."""
    mock = MagicMock(spec=ResultReranker)
    chunk = RerankedChunk(
        chunk_id="chunk_01",
        document_id="doc_dpdp",
        document_name="dpdp.pdf",
        page_start=5,
        page_end=5,
        text="Section 5 notice requirements for data processing.",
        original_rank=1,
        original_score=0.032,
        rerank_score=0.95,
        final_rank=1,
    )
    mock.rerank.return_value = [chunk]
    return mock


def test_evidence_agent_execute_success(mock_reranker):
    """Verify EvidenceAgent executes neural reranking and returns EvidenceAgentResult."""
    agent = EvidenceAgent(reranker=mock_reranker, default_top_k=3)
    assert agent.name == "EvidenceAgent"
    assert "cross-encoder" in agent.description

    candidates = [
        {
            "rank": 1,
            "chunk_id": "chunk_01",
            "document_id": "doc_dpdp",
            "document_name": "dpdp.pdf",
            "page_start": 5,
            "page_end": 5,
            "text": "Section 5 notice requirements for data processing.",
            "score": 0.032,
        }
    ]

    res = agent.execute(query="What are notice rules?", candidates=candidates, top_k=2)
    assert isinstance(res, EvidenceAgentResult)
    assert res.query == "What are notice rules?"
    assert res.evidence_count == 1
    assert res.top_k == 2
    assert res.evidence[0].chunk_id == "chunk_01"
    assert res.evidence[0].rerank_score == 0.95
    assert mock_reranker.rerank.called


def test_evidence_agent_run_wrapper(mock_reranker):
    """Verify BaseAgent.run wrapper captures EvidenceAgent output."""
    agent = EvidenceAgent(reranker=mock_reranker)
    result = agent.run(
        query="data principal rights",
        candidates=[{
            "rank": 1,
            "chunk_id": "chunk_01",
            "document_id": "doc_dpdp",
            "document_name": "dpdp.pdf",
            "page_start": 5,
            "page_end": 5,
            "text": "Sample text",
            "score": 0.032,
        }],
    )

    assert isinstance(result, AgentResult)
    assert result.success is True
    assert result.agent_name == "EvidenceAgent"
    assert isinstance(result.data, EvidenceAgentResult)
    assert result.data.evidence_count == 1


def test_evidence_agent_input_validation(mock_reranker):
    """Verify input validation handles invalid queries and candidate lists."""
    agent = EvidenceAgent(reranker=mock_reranker)

    with pytest.raises(AgentInputError, match="query must be a string"):
        agent.execute(query=123, candidates=[])  # type: ignore

    with pytest.raises(AgentInputError, match="query cannot be empty"):
        agent.execute(query="   ", candidates=[])

    with pytest.raises(AgentInputError, match="candidates must be a sequence"):
        agent.execute(query="valid", candidates="not_a_list")  # type: ignore

    with pytest.raises(AgentInputError, match="top_k must be a positive integer"):
        agent.execute(query="valid", candidates=[], top_k=0)


def test_evidence_agent_execution_error_handling(mock_reranker):
    """Verify that reranker runtime errors are wrapped in AgentExecutionError."""
    mock_reranker.rerank.side_effect = RuntimeError("GPU memory error")
    agent = EvidenceAgent(reranker=mock_reranker)

    with pytest.raises(AgentExecutionError, match="Cross-encoder reranking execution failed"):
        agent.execute(query="test", candidates=[{"chunk_id": "c1", "text": "foo"}])


def test_evidence_agent_result_to_dict():
    """Verify dictionary serialization for EvidenceAgentResult."""
    chunk = RerankedChunk(
        chunk_id="chunk_01",
        document_id="doc_dpdp",
        document_name="dpdp.pdf",
        page_start=1,
        page_end=1,
        text="Sample text",
        original_rank=1,
        original_score=0.03,
        rerank_score=0.88,
        final_rank=1,
    )
    res = EvidenceAgentResult(
        query="test query",
        evidence=[chunk],
        evidence_count=1,
        top_k=3,
        metadata={"batch_size": 16},
    )
    d = res.to_dict()
    assert d["query"] == "test query"
    assert d["evidence_count"] == 1
    assert d["top_k"] == 3
    assert len(d["evidence"]) == 1
    assert d["evidence"][0]["chunk_id"] == "chunk_01"
