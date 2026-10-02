"""Multi-Agent Orchestration subsystem for TrustRAG."""

from backend.app.agents.base import (
    AgentError,
    AgentExecutionError,
    AgentInputError,
    AgentResult,
    BaseAgent,
)
from backend.app.agents.citation_agent import (
    CitationAgent,
    CitationAgentResult,
)
from backend.app.agents.evidence_agent import (
    EvidenceAgent,
    EvidenceAgentResult,
)
from backend.app.agents.generation_agent import (
    GenerationAgent,
    GenerationAgentResult,
)
from backend.app.agents.orchestrator import (
    OrchestratorResult,
    TrustRAGOrchestrator,
)
from backend.app.agents.retrieval_agent import (
    RetrievalAgent,
    RetrievalAgentResult,
)
from backend.app.agents.state import (
    AgentState,
    PipelineStatus,
)
from backend.app.agents.trace import (
    AgentEventStatus,
    AgentTraceEvent,
    ExecutionTrace,
)
from backend.app.agents.trust_agent import (
    TrustAgent,
    TrustAgentResult,
)

__all__ = [
    "AgentError",
    "AgentEventStatus",
    "AgentExecutionError",
    "AgentInputError",
    "AgentResult",
    "AgentState",
    "AgentTraceEvent",
    "BaseAgent",
    "CitationAgent",
    "CitationAgentResult",
    "EvidenceAgent",
    "EvidenceAgentResult",
    "ExecutionTrace",
    "GenerationAgent",
    "GenerationAgentResult",
    "OrchestratorResult",
    "PipelineStatus",
    "RetrievalAgent",
    "RetrievalAgentResult",
    "TrustAgent",
    "TrustAgentResult",
    "TrustRAGOrchestrator",
]
