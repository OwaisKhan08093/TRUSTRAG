"""Unit tests for Agent State and Execution Trace (Phase 7 Milestone 8)."""

from unittest.mock import MagicMock
import pytest

from backend.app.agents.orchestrator import TrustRAGOrchestrator
from backend.app.agents.retrieval_agent import RetrievalAgent, RetrievalAgentResult
from backend.app.agents.evidence_agent import EvidenceAgent, EvidenceAgentResult
from backend.app.agents.trust_agent import TrustAgent, TrustAgentResult
from backend.app.agents.generation_agent import GenerationAgent, GenerationAgentResult
from backend.app.agents.citation_agent import CitationAgent, CitationAgentResult
from backend.app.agents.state import AgentState, PipelineStatus
from backend.app.agents.trace import AgentEventStatus, AgentTraceEvent, ExecutionTrace
from backend.app.generation.models import GenerationResult
from backend.app.reranking.schema import RerankedChunk
from backend.app.trust.engine import TrustDecision, TrustEngine


def test_agent_state_lifecycle():
    """Verify AgentState transitions and serialization."""
    state = AgentState(query="What is consent under DPDP?")
    assert state.status == PipelineStatus.INITIALIZED
    assert state.final_answer is None
    assert state.is_refusal is False

    state.mark_in_progress()
    assert state.status == PipelineStatus.IN_PROGRESS

    state.mark_completed(final_answer="Consent must be free and informed.", is_refusal=False)
    assert state.status == PipelineStatus.COMPLETED
    assert state.final_answer == "Consent must be free and informed."

    d = state.to_dict()
    assert d["query"] == "What is consent under DPDP?"
    assert d["status"] == "COMPLETED"
    assert d["final_answer"] == "Consent must be free and informed."


def test_agent_state_failure():
    """Verify AgentState failure marking."""
    state = AgentState(query="invalid")
    state.mark_failed("Database connection timed out")
    assert state.status == PipelineStatus.FAILED
    assert state.metadata["error"] == "Database connection timed out"


def test_execution_trace_recording():
    """Verify ExecutionTrace records events and formats trace summary."""
    trace = ExecutionTrace(query="test query")
    trace.record_event(
        agent_name="RetrievalAgent",
        status=AgentEventStatus.COMPLETED,
        latency_seconds=0.05,
        output_summary={"candidate_count": 5},
    )
    trace.record_event(
        agent_name="TrustAgent",
        status=AgentEventStatus.COMPLETED,
        latency_seconds=0.01,
        output_summary={"decision": "INSUFFICIENT_EVIDENCE"},
    )
    trace.record_event(
        agent_name="GenerationAgent",
        status=AgentEventStatus.SKIPPED,
        reason="Gated by TrustEngine: INSUFFICIENT_EVIDENCE",
    )
    trace.finish()

    assert len(trace.events) == 3
    assert trace.end_time is not None
    assert trace.total_latency_seconds >= 0.0

    summary = trace.format_trace_summary()
    assert "RetrievalAgent: COMPLETED" in summary
    assert "GenerationAgent: SKIPPED (Gated by TrustEngine: INSUFFICIENT_EVIDENCE)" in summary

    d = trace.to_dict()
    assert d["query"] == "test query"
    assert len(d["events"]) == 3
    assert d["events"][2]["status"] == "SKIPPED"


def test_orchestrator_trace_refusal_path():
    """Verify Orchestrator execution trace on INSUFFICIENT_EVIDENCE shows explicit SKIPPED agents."""
    retrieval_agent = MagicMock(spec=RetrievalAgent)
    retrieval_agent.execute.return_value = RetrievalAgentResult(
        query="unsupported query", results=[], candidate_count=0, top_k=5
    )
    evidence_agent = MagicMock(spec=EvidenceAgent)
    evidence_agent.execute.return_value = EvidenceAgentResult(
        query="unsupported query", evidence=[], evidence_count=0, top_k=3
    )
    engine = TrustEngine()
    assessment = engine.evaluate("unsupported query", [])
    trust_agent = MagicMock(spec=TrustAgent)
    trust_agent.execute.return_value = TrustAgentResult(
        query="unsupported query",
        decision=TrustDecision.INSUFFICIENT_EVIDENCE,
        is_supported=False,
        assessment=assessment,
        confidence_score=0.0,
        groundedness_score=0.0,
        relevance_score=0.0,
        coverage_score=0.0,
        provenance_valid=True,
        decision_reasons=["Empty"],
    )

    generation_agent = MagicMock(spec=GenerationAgent)
    citation_agent = MagicMock(spec=CitationAgent)

    orchestrator = TrustRAGOrchestrator(
        retrieval_agent=retrieval_agent,
        evidence_agent=evidence_agent,
        trust_agent=trust_agent,
        generation_agent=generation_agent,
        citation_agent=citation_agent,
    )

    res = orchestrator.execute("unsupported query")

    assert res.state is not None
    assert res.state.status == PipelineStatus.REFUSED
    assert res.trace is not None

    event_map = {e.agent_name: e for e in res.trace.events}
    assert event_map["RetrievalAgent"].status == AgentEventStatus.COMPLETED
    assert event_map["EvidenceAgent"].status == AgentEventStatus.COMPLETED
    assert event_map["TrustAgent"].status == AgentEventStatus.COMPLETED

    # Critical requirement: GenerationAgent and CitationAgent must be SKIPPED in trace
    assert event_map["GenerationAgent"].status == AgentEventStatus.SKIPPED
    assert "INSUFFICIENT_EVIDENCE" in event_map["GenerationAgent"].reason
    assert event_map["CitationAgent"].status == AgentEventStatus.SKIPPED
    assert "INSUFFICIENT_EVIDENCE" in event_map["CitationAgent"].reason
