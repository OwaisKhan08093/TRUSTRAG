"""Execution trace and observability for TrustRAG Multi-Agent Orchestration."""

from dataclasses import dataclass, field
from enum import Enum
import time
from typing import Any, Dict, List, Optional
import uuid


class AgentEventStatus(str, Enum):
    """Lifecycle status of an individual agent execution event."""

    STARTED = "STARTED"
    COMPLETED = "COMPLETED"
    SKIPPED = "SKIPPED"
    FAILED = "FAILED"


@dataclass(frozen=True)
class AgentTraceEvent:
    """Immutable record of a discrete agent execution event.

    Attributes:
        agent_name: Name of the specialized agent.
        status: Event status (STARTED, COMPLETED, SKIPPED, FAILED).
        timestamp: Epoch timestamp when event occurred.
        latency_seconds: Duration taken if completed.
        input_summary: High-level summary of inputs provided to agent.
        output_summary: High-level summary of agent output payload.
        reason: Explanatory context (e.g. why an agent was SKIPPED).
        metadata: Extra diagnostic attributes.
    """

    agent_name: str
    status: AgentEventStatus
    timestamp: float = field(default_factory=time.time)
    latency_seconds: float = 0.0
    input_summary: Optional[Dict[str, Any]] = None
    output_summary: Optional[Dict[str, Any]] = None
    reason: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize trace event to dictionary."""
        return {
            "agent_name": self.agent_name,
            "status": self.status.value,
            "timestamp": self.timestamp,
            "latency_seconds": self.latency_seconds,
            "input_summary": dict(self.input_summary) if self.input_summary else None,
            "output_summary": dict(self.output_summary) if self.output_summary else None,
            "reason": self.reason,
            "metadata": dict(self.metadata),
        }


@dataclass
class ExecutionTrace:
    """Comprehensive execution trace capturing full lineage and agent event chronology.

    Attributes:
        trace_id: Unique trace identifier.
        query: User input query.
        events: Chronological sequence of recorded AgentTraceEvent instances.
        start_time: Start timestamp of pipeline execution.
        end_time: End timestamp of pipeline execution.
        total_latency_seconds: Total pipeline duration.
        metadata: Pipeline metadata.
    """

    query: str
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    events: List[AgentTraceEvent] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    total_latency_seconds: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def record_event(
        self,
        agent_name: str,
        status: AgentEventStatus,
        latency_seconds: float = 0.0,
        input_summary: Optional[Dict[str, Any]] = None,
        output_summary: Optional[Dict[str, Any]] = None,
        reason: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AgentTraceEvent:
        """Record and append an agent event to the execution trace."""
        event = AgentTraceEvent(
            agent_name=agent_name,
            status=status,
            timestamp=time.time(),
            latency_seconds=latency_seconds,
            input_summary=input_summary,
            output_summary=output_summary,
            reason=reason,
            metadata=metadata or {},
        )
        self.events.append(event)
        return event

    def finish(self) -> None:
        """Finalize execution trace timing."""
        self.end_time = time.time()
        self.total_latency_seconds = max(0.0, self.end_time - self.start_time)

    def format_trace_summary(self) -> str:
        """Generate human-readable execution trace summary."""
        lines = [
            f"=== Execution Trace [{self.trace_id[:8]}] ===",
            f"Query: {self.query}",
            f"Total Latency: {self.total_latency_seconds:.3f}s",
            "Agent Events:",
        ]
        for evt in self.events:
            reason_str = f" ({evt.reason})" if evt.reason else ""
            latency_str = f" [{evt.latency_seconds:.3f}s]" if evt.latency_seconds > 0 else ""
            lines.append(f"  - {evt.agent_name}: {evt.status.value}{reason_str}{latency_str}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize full execution trace to dictionary."""
        return {
            "trace_id": self.trace_id,
            "query": self.query,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "total_latency_seconds": self.total_latency_seconds,
            "events": [e.to_dict() for e in self.events],
            "metadata": dict(self.metadata),
        }
