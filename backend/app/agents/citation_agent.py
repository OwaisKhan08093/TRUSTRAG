"""Specialized agent extracting, verifying, and formatting evidence citations."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Union

from backend.app.agents.base import AgentExecutionError, AgentInputError, BaseAgent
from backend.app.generation.citations import (
    Citation,
    CitationError,
    build_citation_references,
    extract_cited_indices,
    format_citations_markdown,
)
from backend.app.generation.models import GenerationResult
from backend.app.generation.response import GroundedAnswer, assemble_grounded_answer
from backend.app.reranking.schema import RerankedChunk
from backend.app.trust.engine import TrustAssessment
from backend.app.trust.evidence import TrustEvidence


@dataclass(frozen=True)
class CitationAgentResult:
    """Immutable typed container for outputs emitted by CitationAgent.

    Attributes:
        query: Evaluated user query.
        citations: List of Citation references backing the claims.
        citation_count: Number of citations referenced.
        formatted_citations: Markdown-formatted citation reference block.
        grounded_answer: Optional complete assembled GroundedAnswer object.
        metadata: Execution details.
    """

    query: str
    citations: List[Citation]
    citation_count: int
    formatted_citations: str
    grounded_answer: Optional[GroundedAnswer] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize citation agent result to dictionary."""
        return {
            "query": self.query,
            "citations": [c.to_dict() for c in self.citations],
            "citation_count": self.citation_count,
            "formatted_citations": self.formatted_citations,
            "grounded_answer": self.grounded_answer.to_dict() if self.grounded_answer else None,
            "metadata": dict(self.metadata),
        }


class CitationAgent(BaseAgent):
    """Specialized agent coordinating Citation extraction and GroundedAnswer assembly.

    Design: Reuses existing Citation subsystem and never hallucinates or invents provenance.
    """

    @property
    def name(self) -> str:
        return "CitationAgent"

    @property
    def description(self) -> str:
        return "Resolves and validates document provenance to assemble verified citations and structured grounded answers."

    def execute(
        self,
        query: str,
        answer_text: str,
        evidence: Sequence[Union[TrustEvidence, RerankedChunk, Dict[str, Any]]],
        assessment: Optional[TrustAssessment] = None,
        generation_result: Optional[GenerationResult] = None,
        filter_cited_only: bool = True,
    ) -> CitationAgentResult:
        """Extract and format citation references from generated text and evidence.

        Args:
            query: User query string.
            answer_text: Generated response text (may contain markers like [1]).
            evidence: Sequence of evidence chunks.
            assessment: Optional TrustAssessment from TrustEngine.
            generation_result: Optional GenerationResult from local LLM.
            filter_cited_only: If True, include only citations cited in answer_text.

        Returns:
            CitationAgentResult containing validated citations and optional GroundedAnswer.

        Raises:
            AgentInputError: If query, answer_text, or evidence is invalid.
            AgentExecutionError: If citation construction encounters an unexpected failure.
        """
        if not isinstance(query, str):
            raise AgentInputError(f"query must be a string, got {type(query).__name__}.")

        stripped_query = query.strip()
        if not stripped_query:
            raise AgentInputError("query cannot be empty or whitespace only.")

        if not isinstance(answer_text, str):
            raise AgentInputError(
                f"answer_text must be a string, got {type(answer_text).__name__}."
            )

        if not isinstance(evidence, (list, tuple)):
            raise AgentInputError(
                f"evidence must be a sequence, got {type(evidence).__name__}."
            )

        try:
            all_citations = build_citation_references(evidence=evidence)
            if filter_cited_only and answer_text.strip():
                cited_indices = set(extract_cited_indices(answer_text))
                citations = [c for c in all_citations if c.index in cited_indices]
            else:
                citations = all_citations

            formatted_md = format_citations_markdown(citations)

            grounded_answer: Optional[GroundedAnswer] = None
            if assessment is not None and generation_result is not None:
                grounded_answer = assemble_grounded_answer(
                    generation_result=generation_result,
                    assessment=assessment,
                    evidence=evidence,
                )
        except (CitationError, TypeError, ValueError) as exc:
            raise AgentInputError(f"Citation resolution failed with invalid data: {exc}") from exc
        except Exception as exc:
            raise AgentExecutionError(f"Citation agent execution failed: {exc}") from exc

        return CitationAgentResult(
            query=stripped_query,
            citations=citations,
            citation_count=len(citations),
            formatted_citations=formatted_md,
            grounded_answer=grounded_answer,
            metadata={
                "filter_cited_only": filter_cited_only,
                "evidence_count": len(evidence),
            },
        )
