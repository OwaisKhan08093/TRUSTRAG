"""Master Multi-Agent TrustRAG Orchestrator coordinating retrieval, evidence reranking, trust gating, generation, and citation agents."""

from dataclasses import dataclass, field
import logging
import time
from typing import Any, Dict, List, Optional, Sequence

from backend.app.agents.base import AgentError, AgentExecutionError, AgentInputError, AgentResult, BaseAgent
from backend.app.agents.citation_agent import CitationAgent, CitationAgentResult
from backend.app.agents.evidence_agent import EvidenceAgent, EvidenceAgentResult
from backend.app.agents.generation_agent import GenerationAgent, GenerationAgentResult
from backend.app.agents.retrieval_agent import RetrievalAgent, RetrievalAgentResult
from backend.app.agents.trust_agent import TrustAgent, TrustAgentResult
from backend.app.generation.citations import Citation
from backend.app.generation.generator import STANDARD_ABSTENTION_MESSAGE
from backend.app.generation.response import GroundedAnswer
from backend.app.trust.engine import TrustDecision

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OrchestratorResult:
    """Immutable typed container for outputs emitted by TrustRAGOrchestrator.

    Attributes:
        query: Evaluated user query.
        answer: Final answer text or structured refusal message.
        decision: Trust gating decision (SUPPORTED or INSUFFICIENT_EVIDENCE).
        is_refusal: True if response is a refusal/abstention.
        citations: List of verified Citation objects.
        formatted_response: Complete presentation string including citations if supported.
        confidence_score: Multi-signal system confidence index.
        groundedness_score: Composite evidence grounding metric.
        relevance_score: Aggregate neural relevance score.
        coverage_score: Lexical query term coverage ratio.
        provenance_valid: Boolean indicating if all cited chunks have valid metadata.
        retrieval_result: Optional output from RetrievalAgent.
        evidence_result: Optional output from EvidenceAgent.
        trust_result: Optional output from TrustAgent.
        generation_result: Optional output from GenerationAgent.
        citation_result: Optional output from CitationAgent.
        grounded_answer: Optional complete assembled GroundedAnswer.
        latency_seconds: Total pipeline execution duration.
        metadata: Pipeline diagnostic metadata.
    """

    query: str
    answer: str
    decision: TrustDecision
    is_refusal: bool
    citations: List[Citation]
    formatted_response: str
    confidence_score: float
    groundedness_score: float
    relevance_score: float
    coverage_score: float
    provenance_valid: bool
    retrieval_result: Optional[RetrievalAgentResult] = None
    evidence_result: Optional[EvidenceAgentResult] = None
    trust_result: Optional[TrustAgentResult] = None
    generation_result: Optional[GenerationAgentResult] = None
    citation_result: Optional[CitationAgentResult] = None
    grounded_answer: Optional[GroundedAnswer] = None
    latency_seconds: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize orchestrator result to dictionary."""
        return {
            "query": self.query,
            "answer": self.answer,
            "decision": self.decision.value,
            "is_refusal": self.is_refusal,
            "citations": [c.to_dict() for c in self.citations],
            "formatted_response": self.formatted_response,
            "confidence_score": self.confidence_score,
            "groundedness_score": self.groundedness_score,
            "relevance_score": self.relevance_score,
            "coverage_score": self.coverage_score,
            "provenance_valid": self.provenance_valid,
            "retrieval_result": self.retrieval_result.to_dict() if self.retrieval_result else None,
            "evidence_result": self.evidence_result.to_dict() if self.evidence_result else None,
            "trust_result": self.trust_result.to_dict() if self.trust_result else None,
            "generation_result": self.generation_result.to_dict() if self.generation_result else None,
            "citation_result": self.citation_result.to_dict() if self.citation_result else None,
            "grounded_answer": self.grounded_answer.to_dict() if self.grounded_answer else None,
            "latency_seconds": self.latency_seconds,
            "metadata": dict(self.metadata),
        }


class TrustRAGOrchestrator(BaseAgent):
    """Master orchestrator executing the deterministic Multi-Agent TrustRAG pipeline.

    Pipeline Flow:
        User Query
            ↓
        RetrievalAgent (FAISS + BM25 + RRF)
            ↓
        EvidenceAgent (Neural Cross-Encoder)
            ↓
        TrustAgent (TrustEngine Multi-Signal Gating)
            ↓
        ┌───────────────────────────────────┐
        │                                   │
     SUPPORTED                    INSUFFICIENT_EVIDENCE
        │                                   │
        ↓                                   ↓
     GenerationAgent (Local LLM)           STOP (Refusal)
        │
        ↓
     CitationAgent (Provenance Validation)
        │
        ↓
     Final Grounded Response
    """

    def __init__(
        self,
        retrieval_agent: Optional[RetrievalAgent] = None,
        evidence_agent: Optional[EvidenceAgent] = None,
        trust_agent: Optional[TrustAgent] = None,
        generation_agent: Optional[GenerationAgent] = None,
        citation_agent: Optional[CitationAgent] = None,
        abstention_message: Optional[str] = None,
    ) -> None:
        """Initialize TrustRAGOrchestrator with sub-agents."""
        self.retrieval_agent = retrieval_agent or RetrievalAgent()
        self.evidence_agent = evidence_agent or EvidenceAgent()
        self.trust_agent = trust_agent or TrustAgent()
        self.generation_agent = generation_agent or GenerationAgent(abstention_message=abstention_message)
        self.citation_agent = citation_agent or CitationAgent()
        self.abstention_message = abstention_message or STANDARD_ABSTENTION_MESSAGE

    @property
    def name(self) -> str:
        return "TrustRAGOrchestrator"

    @property
    def description(self) -> str:
        return "Master orchestrator coordinating hybrid retrieval, evidence reranking, deterministic trust gating, grounded generation, and citation assembly."

    def execute(
        self,
        query: str,
        retrieval_top_k: Optional[int] = None,
        evidence_top_k: Optional[int] = None,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        filter_cited_only: bool = True,
    ) -> OrchestratorResult:
        """Execute the full multi-agent pipeline for a user query.

        Args:
            query: User query string.
            retrieval_top_k: Optional candidate retrieval limit.
            evidence_top_k: Optional neural evidence rerank limit.
            max_new_tokens: Token generation budget.
            temperature: Generation temperature.
            filter_cited_only: Whether citation block contains only cited references.

        Returns:
            OrchestratorResult containing full execution details and final response.

        Raises:
            AgentInputError: If query is invalid.
            AgentExecutionError: If unrecoverable pipeline failure occurs.
        """
        start_time = time.perf_counter()

        if not isinstance(query, str):
            raise AgentInputError(f"query must be a string, got {type(query).__name__}.")

        stripped_query = query.strip()
        if not stripped_query:
            raise AgentInputError("query cannot be empty or whitespace only.")

        # Step 1: RetrievalAgent
        try:
            retrieval_res = self.retrieval_agent.execute(
                query=stripped_query,
                top_k=retrieval_top_k,
            )
        except Exception as exc:
            raise AgentExecutionError(f"RetrievalAgent failed: {exc}") from exc

        # Step 2: EvidenceAgent
        try:
            evidence_res = self.evidence_agent.execute(
                query=stripped_query,
                candidates=retrieval_res.results,
                top_k=evidence_top_k,
            )
        except Exception as exc:
            raise AgentExecutionError(f"EvidenceAgent failed: {exc}") from exc

        # Step 3: TrustAgent Gating
        try:
            trust_res = self.trust_agent.execute(
                query=stripped_query,
                evidence=evidence_res.evidence,
            )
        except Exception as exc:
            raise AgentExecutionError(f"TrustAgent failed: {exc}") from exc

        # Step 4: Gating Decision Check
        if not trust_res.is_supported:
            # INSUFFICIENT_EVIDENCE -> STOP: Do NOT call GenerationAgent LLM or CitationAgent
            elapsed = time.perf_counter() - start_time
            refusal_formatted = f"**[Status: {trust_res.decision.value}]**\n\n{self.abstention_message}"
            return OrchestratorResult(
                query=stripped_query,
                answer=self.abstention_message,
                decision=trust_res.decision,
                is_refusal=True,
                citations=[],
                formatted_response=refusal_formatted,
                confidence_score=trust_res.confidence_score,
                groundedness_score=trust_res.groundedness_score,
                relevance_score=trust_res.relevance_score,
                coverage_score=trust_res.coverage_score,
                provenance_valid=trust_res.provenance_valid,
                retrieval_result=retrieval_res,
                evidence_result=evidence_res,
                trust_result=trust_res,
                generation_result=None,
                citation_result=None,
                grounded_answer=None,
                latency_seconds=elapsed,
                metadata={
                    "gating_status": "GATED_REFUSAL",
                    "decision_reasons": trust_res.decision_reasons,
                },
            )

        # Step 5: GenerationAgent (Only for SUPPORTED)
        try:
            gen_res = self.generation_agent.execute(
                query=stripped_query,
                evidence=evidence_res.evidence,
                assessment=trust_res.assessment,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
        except Exception as exc:
            raise AgentExecutionError(f"GenerationAgent failed: {exc}") from exc

        # Step 6: CitationAgent
        try:
            citation_res = self.citation_agent.execute(
                query=stripped_query,
                answer_text=gen_res.answer_text,
                evidence=evidence_res.evidence,
                assessment=trust_res.assessment,
                generation_result=gen_res.generation_result,
                filter_cited_only=filter_cited_only,
            )
        except Exception as exc:
            raise AgentExecutionError(f"CitationAgent failed: {exc}") from exc

        elapsed = time.perf_counter() - start_time
        final_formatted = (
            citation_res.grounded_answer.formatted_response
            if citation_res.grounded_answer
            else (
                f"{gen_res.answer_text}\n\n{citation_res.formatted_citations}"
                if citation_res.formatted_citations
                else gen_res.answer_text
            )
        )

        return OrchestratorResult(
            query=stripped_query,
            answer=gen_res.answer_text,
            decision=trust_res.decision,
            is_refusal=gen_res.is_refusal,
            citations=citation_res.citations,
            formatted_response=final_formatted,
            confidence_score=trust_res.confidence_score,
            groundedness_score=trust_res.groundedness_score,
            relevance_score=trust_res.relevance_score,
            coverage_score=trust_res.coverage_score,
            provenance_valid=trust_res.provenance_valid,
            retrieval_result=retrieval_res,
            evidence_result=evidence_res,
            trust_result=trust_res,
            generation_result=gen_res,
            citation_result=citation_res,
            grounded_answer=citation_res.grounded_answer,
            latency_seconds=elapsed,
            metadata={
                "gating_status": "SUPPORTED_AND_GENERATED",
                "decision_reasons": trust_res.decision_reasons,
            },
        )
