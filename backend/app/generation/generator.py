"""Evidence-constrained grounded answer generator with TrustEngine gating."""

import logging
import time
from typing import Any, Dict, List, Optional, Sequence, Union

from backend.app.generation.config import GenerationConfig
from backend.app.generation.llm import LocalLLM
from backend.app.generation.models import GenerationResult
from backend.app.generation.prompt import GroundedPrompt, build_grounded_prompt
from backend.app.reranking.schema import RerankedChunk
from backend.app.trust.engine import TrustAssessment, TrustDecision
from backend.app.trust.evidence import TrustEvidence

logger = logging.getLogger(__name__)

STANDARD_ABSTENTION_MESSAGE: str = (
    "I am unable to answer this question because the retrieved evidence is insufficient "
    "or lacks adequate factual grounding to support a verifiable response."
)


class GeneratorError(RuntimeError):
    """Raised when GroundedGenerator encounters configuration or execution errors."""
    pass


class GroundedGenerator:
    """Orchestrates evidence-grounded answer generation strictly gated by TrustEngine assessments.

    Architectural Invariant:
    If TrustAssessment is not SUPPORTED (e.g. INSUFFICIENT_EVIDENCE), the LLM is NEVER invoked.
    Instead, a structured refusal result is returned immediately to prevent hallucination.
    """

    def __init__(
        self,
        llm: Optional[LocalLLM] = None,
        config: Optional[GenerationConfig] = None,
        abstention_message: Optional[str] = None,
    ) -> None:
        """Initialize GroundedGenerator.

        Args:
            llm: Optional pre-configured LocalLLM instance.
            config: Optional GenerationConfig instance.
            abstention_message: Custom message returned when generation is gated.
        """
        if config is not None:
            self.config = config
        elif llm is not None and hasattr(llm, "config") and isinstance(llm.config, GenerationConfig):
            self.config = llm.config
        else:
            self.config = GenerationConfig()

        self.llm = llm or LocalLLM(config=self.config, lazy_load=True)
        self.abstention_message = abstention_message or STANDARD_ABSTENTION_MESSAGE

    def generate(
        self,
        query: str,
        evidence: Sequence[Union[TrustEvidence, RerankedChunk, Dict[str, Any]]],
        assessment: TrustAssessment,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
    ) -> GenerationResult:
        """Execute evidence-constrained generation governed strictly by TrustAssessment.

        Args:
            query: User query string.
            evidence: Sequence of evidence chunks retrieved and reranked.
            assessment: TrustAssessment emitted by the TrustEngine.
            max_new_tokens: Maximum tokens to generate (overrides config).
            temperature: Sampling temperature (overrides config).
            top_p: Nucleus sampling probability mass (overrides config).
            repetition_penalty: Repetition penalty (overrides config).

        Returns:
            GenerationResult containing either the grounded answer or explicit refusal.

        Raises:
            TypeError: If input arguments have invalid types.
            GeneratorError: If query is empty or prompt building fails.
        """
        if not isinstance(query, str):
            raise TypeError(f"query must be a string, got {type(query).__name__}.")

        stripped_query = query.strip()
        if not stripped_query:
            raise GeneratorError("query cannot be empty or whitespace only.")

        if not isinstance(evidence, (list, tuple)):
            raise TypeError(f"evidence must be a sequence, got {type(evidence).__name__}.")

        if not isinstance(assessment, TrustAssessment):
            raise TypeError(
                f"assessment must be an instance of TrustAssessment, got {type(assessment).__name__}."
            )

        # 1. HARD TRUST GATE: If evidence is not SUPPORTED, ABSTAIN WITHOUT CALLING LLM
        if assessment.decision != TrustDecision.SUPPORTED:
            logger.info(
                "TrustEngine decision '%s' blocked generation for query: %s",
                assessment.decision.value,
                stripped_query,
            )
            reasons_str = "; ".join(assessment.decision_reasons)
            return GenerationResult(
                query=stripped_query,
                answer=self.abstention_message,
                model_name=self.llm.model_name,
                evidence_ids=[],
                is_refusal=True,
                refusal_reason=(
                    f"TrustEngine decision was '{assessment.decision.value}'. "
                    f"Gating factors: {reasons_str}"
                ),
                generation_metadata={
                    "gated_by_trust_engine": True,
                    "trust_decision": assessment.decision.value,
                    "groundedness_score": assessment.groundedness_score,
                    "confidence_score": assessment.confidence_score,
                    "coverage_score": assessment.coverage_score,
                    "relevance_score": assessment.relevance_score,
                    "provenance_valid": assessment.provenance_valid,
                    "decision_reasons": list(assessment.decision_reasons),
                },
            )

        # 2. Build structured grounded prompt
        prompt: GroundedPrompt = build_grounded_prompt(
            query=stripped_query,
            evidence=evidence,
            system_instruction=self.config.system_prompt_template,
        )

        # 3. Perform local LLM inference
        start_time = time.perf_counter()
        raw_answer = self.llm.generate(
            prompt=prompt.user_prompt,
            system_prompt=prompt.system_instruction,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )
        elapsed_time = time.perf_counter() - start_time

        return GenerationResult(
            query=stripped_query,
            answer=raw_answer,
            model_name=self.llm.model_name,
            evidence_ids=prompt.evidence_ids,
            is_refusal=False,
            refusal_reason=None,
            generation_metadata={
                "gated_by_trust_engine": False,
                "trust_decision": assessment.decision.value,
                "groundedness_score": assessment.groundedness_score,
                "confidence_score": assessment.confidence_score,
                "coverage_score": assessment.coverage_score,
                "relevance_score": assessment.relevance_score,
                "evidence_count": prompt.evidence_count,
                "latency_seconds": elapsed_time,
                "model_name": self.llm.model_name,
            },
        )
