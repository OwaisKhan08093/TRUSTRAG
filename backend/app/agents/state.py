"""Structured agent pipeline state management for TrustRAG."""

from dataclasses import dataclass, field
from enum import Enum
import time
from typing import Any, Dict, List, Optional

from backend.app.agents.citation_agent import CitationAgentResult
from backend.app.agents.evidence_agent import EvidenceAgentResult
from backend.app.agents.generation_agent import GenerationAgentResult
from backend.app.agents.retrieval_agent import RetrievalAgentResult
from backend.app.agents.trust_agent import TrustAgentResult
from backend.app.generation.response import GroundedAnswer
from backend.app.trust.engine import TrustDecision


class PipelineStatus(str, Enum):
    """Execution status of the TrustRAG pipeline state."""

    INITIALIZED = "INITIALIZED"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    REFUSED = "REFUSED"
    FAILED = "FAILED"


@dataclass
class AgentState:
    """Mutable structured context tracking state across pipeline stages.

    Attributes:
        query: User input query.
        session_id: Optional tracking identifier.
        status: Current pipeline lifecycle status.
        retrieval_result: Output from RetrievalAgent.
        evidence_result: Output from EvidenceAgent.
        trust_result: Output from TrustAgent.
        generation_result: Output from GenerationAgent.
        citation_result: Output from CitationAgent.
        grounded_answer: Unified assembled GroundedAnswer.
        final_answer: Final formatted or text response string.
        is_refusal: Whether execution culminated in structured refusal.
        created_at: Epoch timestamp of state creation.
        updated_at: Epoch timestamp of latest update.
        metadata: Arbitrary diagnostic dictionary.
    """

    query: str
    session_id: Optional[str] = None
    status: PipelineStatus = PipelineStatus.INITIALIZED
    retrieval_result: Optional[RetrievalAgentResult] = None
    evidence_result: Optional[EvidenceAgentResult] = None
    trust_result: Optional[TrustAgentResult] = None
    generation_result: Optional[GenerationAgentResult] = None
    citation_result: Optional[CitationAgentResult] = None
    grounded_answer: Optional[GroundedAnswer] = None
    final_answer: Optional[str] = None
    is_refusal: bool = False
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def mark_in_progress(self) -> None:
        """Update status to IN_PROGRESS."""
        self.status = PipelineStatus.IN_PROGRESS
        self.updated_at = time.time()

    def mark_completed(self, final_answer: str, is_refusal: bool = False) -> None:
        """Mark pipeline as completed."""
        self.final_answer = final_answer
        self.is_refusal = is_refusal
        self.status = PipelineStatus.REFUSED if is_refusal else PipelineStatus.COMPLETED
        self.updated_at = time.time()

    def mark_failed(self, error_message: str) -> None:
        """Mark pipeline as failed with error context."""
        self.status = PipelineStatus.FAILED
        self.metadata["error"] = error_message
        self.updated_at = time.time()

    def to_dict(self) -> Dict[str, Any]:
        """Serialize state to dictionary."""
        return {
            "query": self.query,
            "session_id": self.session_id,
            "status": self.status.value,
            "final_answer": self.final_answer,
            "is_refusal": self.is_refusal,
            "retrieval_result": self.retrieval_result.to_dict() if self.retrieval_result else None,
            "evidence_result": self.evidence_result.to_dict() if self.evidence_result else None,
            "trust_result": self.trust_result.to_dict() if self.trust_result else None,
            "generation_result": self.generation_result.to_dict() if self.generation_result else None,
            "citation_result": self.citation_result.to_dict() if self.citation_result else None,
            "grounded_answer": self.grounded_answer.to_dict() if self.grounded_answer else None,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": dict(self.metadata),
        }
