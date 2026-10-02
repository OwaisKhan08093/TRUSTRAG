"""Specialized agent coordinating deterministic evidence trust gating and verification."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Union

from backend.app.agents.base import AgentExecutionError, AgentInputError, BaseAgent
from backend.app.reranking.schema import RerankedChunk
from backend.app.trust.config import TrustConfig
from backend.app.trust.engine import TrustAssessment, TrustDecision, TrustEngine
from backend.app.trust.evidence import TrustEvidence


@dataclass(frozen=True)
class TrustAgentResult:
    """Immutable typed container for outputs emitted by TrustAgent.

    Attributes:
        query: Evaluated user query.
        decision: Discrete trust decision (SUPPORTED or INSUFFICIENT_EVIDENCE).
        is_supported: Convenience boolean indicating if decision is SUPPORTED.
        assessment: Full TrustAssessment object.
        confidence_score: Composite system trust confidence in [0.0, 1.0].
        groundedness_score: Composite evidence groundedness in [0.0, 1.0].
        relevance_score: Weighted aggregate relevance score in [0.0, 1.0].
        coverage_score: Lexical query term coverage ratio in [0.0, 1.0].
        provenance_valid: Boolean indicating if all evidence chunks have valid provenance.
        decision_reasons: List of explanatory factors supporting the trust decision.
        metadata: Execution details.
    """

    query: str
    decision: TrustDecision
    is_supported: bool
    assessment: TrustAssessment
    confidence_score: float
    groundedness_score: float
    relevance_score: float
    coverage_score: float
    provenance_valid: bool
    decision_reasons: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize trust agent result to dictionary."""
        return {
            "query": self.query,
            "decision": self.decision.value,
            "is_supported": self.is_supported,
            "confidence_score": self.confidence_score,
            "groundedness_score": self.groundedness_score,
            "relevance_score": self.relevance_score,
            "coverage_score": self.coverage_score,
            "provenance_valid": self.provenance_valid,
            "decision_reasons": list(self.decision_reasons),
            "assessment": self.assessment.to_dict(),
            "metadata": dict(self.metadata),
        }


class TrustAgent(BaseAgent):
    """Specialized agent coordinating TrustEngine gating.

    Design: Reuses existing TrustEngine without bypassing any trust validation rules.
    """

    def __init__(
        self,
        trust_engine: Optional[TrustEngine] = None,
        config: Optional[TrustConfig] = None,
    ) -> None:
        """Initialize TrustAgent.

        Args:
            trust_engine: Optional pre-configured TrustEngine instance.
            config: Optional TrustConfig if engine needs to be instantiated.
        """
        self._trust_engine = trust_engine
        self._config = config

    @property
    def name(self) -> str:
        return "TrustAgent"

    @property
    def description(self) -> str:
        return "Evaluates evidence grounding, relevance, coverage, and provenance to issue deterministic SUPPORTED vs INSUFFICIENT_EVIDENCE trust gating decisions."

    @property
    def trust_engine(self) -> TrustEngine:
        """Lazy-initialize TrustEngine if not injected."""
        if self._trust_engine is None:
            self._trust_engine = TrustEngine(config=self._config)
        return self._trust_engine

    def execute(
        self,
        query: str,
        evidence: Sequence[Union[TrustEvidence, RerankedChunk, Dict[str, Any]]],
    ) -> TrustAgentResult:
        """Evaluate evidence sufficiency and issue discrete gating decision.

        Args:
            query: User query string.
            evidence: Sequence of evidence objects (TrustEvidence, RerankedChunk, or dicts).

        Returns:
            TrustAgentResult containing TrustAssessment and decision flags.

        Raises:
            AgentInputError: If query or evidence format is invalid.
            AgentExecutionError: If trust evaluation encounters an unexpected runtime error.
        """
        if not isinstance(query, str):
            raise AgentInputError(f"query must be a string, got {type(query).__name__}.")

        stripped_query = query.strip()
        if not stripped_query:
            raise AgentInputError("query cannot be empty or whitespace only.")

        if not isinstance(evidence, (list, tuple)):
            raise AgentInputError(
                f"evidence must be a sequence, got {type(evidence).__name__}."
            )

        try:
            assessment = self.trust_engine.evaluate(
                query=stripped_query,
                evidence=evidence,
            )
        except (TypeError, ValueError) as exc:
            raise AgentInputError(f"Invalid input provided to TrustEngine: {exc}") from exc
        except Exception as exc:
            raise AgentExecutionError(f"Trust evaluation failed: {exc}") from exc

        return TrustAgentResult(
            query=stripped_query,
            decision=assessment.decision,
            is_supported=(assessment.decision == TrustDecision.SUPPORTED),
            assessment=assessment,
            confidence_score=assessment.confidence_score,
            groundedness_score=assessment.groundedness_score,
            relevance_score=assessment.relevance_score,
            coverage_score=assessment.coverage_score,
            provenance_valid=assessment.provenance_valid,
            decision_reasons=assessment.decision_reasons,
            metadata={"evidence_count": assessment.evidence_count},
        )
