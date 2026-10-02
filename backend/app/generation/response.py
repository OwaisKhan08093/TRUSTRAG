"""Structured assembly of grounded answers, citations, and trust assessments."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Union

from backend.app.generation.citations import (
    Citation,
    build_citation_references,
    format_citations_markdown,
)
from backend.app.generation.models import GenerationResult
from backend.app.reranking.schema import RerankedChunk
from backend.app.trust.engine import TrustAssessment, TrustDecision
from backend.app.trust.evidence import TrustEvidence


class AnswerAssemblyError(ValueError):
    """Raised when answer assembly encounters invalid component structures."""
    pass


@dataclass(frozen=True)
class GroundedAnswer:
    """Complete, end-to-end grounded response uniting generated prose, citations, and trust metrics.

    Attributes:
        query: Evaluated user query.
        answer: Generated factual text or transparent refusal statement.
        citations: List of Citation references backing the claims.
        trust_decision: Discrete verdict from TrustEngine (SUPPORTED vs INSUFFICIENT_EVIDENCE).
        confidence_score: Multi-signal system confidence index in [0.0, 1.0].
        groundedness_score: Composite evidence grounding metric in [0.0, 1.0].
        relevance_score: Aggregate neural relevance score in [0.0, 1.0].
        coverage_score: Lexical term coverage ratio in [0.0, 1.0].
        provenance_valid: Boolean indicating if all cited chunks have valid metadata.
        is_refusal: True if generation was abstained to prevent ungrounded output.
        refusal_reason: Explanatory reasons if is_refusal is True.
        model_name: Identifier of the local generation model.
        assessment: Full TrustAssessment object.
        generation_result: Underlying GenerationResult object.
    """

    query: str
    answer: str
    citations: List[Citation]
    trust_decision: TrustDecision
    confidence_score: float
    groundedness_score: float
    relevance_score: float
    coverage_score: float
    provenance_valid: bool
    is_refusal: bool
    refusal_reason: Optional[str]
    model_name: str
    assessment: TrustAssessment
    generation_result: GenerationResult

    @property
    def formatted_response(self) -> str:
        """Format final user-facing response with body prose and citation references."""
        if self.is_refusal:
            return f"**[Status: {self.trust_decision.value}]**\n\n{self.answer}"

        citations_md = format_citations_markdown(self.citations)
        if citations_md:
            return f"{self.answer}\n\n{citations_md}"
        return self.answer

    def to_dict(self) -> Dict[str, Any]:
        """Convert GroundedAnswer to serializable dictionary."""
        return {
            "query": self.query,
            "answer": self.answer,
            "citations": [c.to_dict() for c in self.citations],
            "trust_decision": self.trust_decision.value,
            "confidence_score": self.confidence_score,
            "groundedness_score": self.groundedness_score,
            "relevance_score": self.relevance_score,
            "coverage_score": self.coverage_score,
            "provenance_valid": self.provenance_valid,
            "is_refusal": self.is_refusal,
            "refusal_reason": self.refusal_reason,
            "model_name": self.model_name,
            "formatted_response": self.formatted_response,
            "assessment": self.assessment.to_dict(),
            "generation_result": self.generation_result.to_dict(),
        }


def assemble_grounded_answer(
    generation_result: GenerationResult,
    assessment: TrustAssessment,
    evidence: Sequence[Union[TrustEvidence, RerankedChunk, Dict[str, Any]]],
) -> GroundedAnswer:
    """Assemble a unified GroundedAnswer from generation and trust assessment components.

    Args:
        generation_result: Output from GroundedGenerator.
        assessment: Assessment from TrustEngine.
        evidence: Sequence of evidence items.

    Returns:
        GroundedAnswer instance with complete citation and trust preservation.

    Raises:
        TypeError: If input argument types are invalid.
    """
    if not isinstance(generation_result, GenerationResult):
        raise TypeError(
            f"generation_result must be a GenerationResult, got {type(generation_result).__name__}."
        )

    if not isinstance(assessment, TrustAssessment):
        raise TypeError(
            f"assessment must be a TrustAssessment, got {type(assessment).__name__}."
        )

    if not isinstance(evidence, (list, tuple)):
        raise TypeError(f"evidence must be a sequence, got {type(evidence).__name__}.")

    # Generate citations only if generation succeeded without refusal
    if not generation_result.is_refusal and assessment.decision == TrustDecision.SUPPORTED:
        citations = build_citation_references(evidence)
    else:
        citations = []

    return GroundedAnswer(
        query=generation_result.query,
        answer=generation_result.answer,
        citations=citations,
        trust_decision=assessment.decision,
        confidence_score=assessment.confidence_score,
        groundedness_score=assessment.groundedness_score,
        relevance_score=assessment.relevance_score,
        coverage_score=assessment.coverage_score,
        provenance_valid=assessment.provenance_valid,
        is_refusal=generation_result.is_refusal,
        refusal_reason=generation_result.refusal_reason,
        model_name=generation_result.model_name,
        assessment=assessment,
        generation_result=generation_result,
    )
