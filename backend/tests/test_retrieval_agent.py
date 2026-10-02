"""Unit tests for RetrievalAgent (Phase 7 Milestone 2)."""

from unittest.mock import MagicMock
import pytest

from backend.app.agents.base import AgentInputError, AgentResult
from backend.app.agents.retrieval_agent import RetrievalAgent, RetrievalAgentResult
from backend.app.retrieval.hybrid_retriever import HybridRetriever


@pytest.fixture
def mock_hybrid_retriever():
    """Fixture providing a mocked HybridRetriever."""
    mock = MagicMock(spec=HybridRetriever)
    mock.retrieve.return_value = [
        {
            "rank": 1,
            "chunk_id": "chunk_01",
            "document_id": "doc_dpdp",
            "document_name": "dpdp.pdf",
            "page_start": 5,
            "page_end": 5,
            "text": "Notice requirements under Section 5.",
            "score": 0.032,
            "dense_rank": 1,
            "sparse_rank": 2,
        }
    ]
    return mock


def test_retrieval_agent_execute_success(mock_hybrid_retriever):
    """Verify RetrievalAgent executes hybrid retrieval and returns typed result."""
    agent = RetrievalAgent(retriever=mock_hybrid_retriever, default_top_k=5)
    assert agent.name == "RetrievalAgent"
    assert "dense vector semantic search" in agent.description

    res = agent.execute("What are notice requirements?", top_k=3)
    assert isinstance(res, RetrievalAgentResult)
    assert res.query == "What are notice requirements?"
    assert res.candidate_count == 1
    assert res.top_k == 3
    assert res.results[0]["chunk_id"] == "chunk_01"
    assert mock_hybrid_retriever.retrieve.called


def test_retrieval_agent_run_wrapper(mock_hybrid_retriever):
    """Verify BaseAgent.run wrapper captures RetrievalAgent output."""
    agent = RetrievalAgent(retriever=mock_hybrid_retriever)
    result = agent.run("data principal rights")

    assert isinstance(result, AgentResult)
    assert result.success is True
    assert result.agent_name == "RetrievalAgent"
    assert isinstance(result.data, RetrievalAgentResult)
    assert result.data.candidate_count == 1


def test_retrieval_agent_input_validation(mock_hybrid_retriever):
    """Verify input argument errors."""
    agent = RetrievalAgent(retriever=mock_hybrid_retriever)

    with pytest.raises(AgentInputError, match="query must be a string"):
        agent.execute(1234)  # type: ignore

    with pytest.raises(AgentInputError, match="query cannot be empty"):
        agent.execute("   ")

    with pytest.raises(AgentInputError, match="top_k must be a positive integer"):
        agent.execute("valid", top_k=0)


def test_retrieval_agent_result_to_dict():
    """Verify dictionary serialization."""
    res = RetrievalAgentResult(
        query="query",
        results=[{"chunk_id": "c1"}],
        candidate_count=1,
        top_k=5,
    )
    d = res.to_dict()
    assert d["query"] == "query"
    assert d["candidate_count"] == 1
    assert d["retrieval_mode"] == "hybrid_rrf"
