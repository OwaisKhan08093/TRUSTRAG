"""Specialized agent coordinating evidence-grounded answer generation under TrustEngine gating."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Union

from backend.app.agents.base import AgentExecutionError, AgentInputError, BaseAgent
from backend.app.generation.config import GenerationConfig
from backend.app.generation.generator import GroundedGenerator, STANDARD_ABSTENTION_MESSAGE
from backend.app.generation.models import GenerationResult
from backend.app.reranking.schema import RerankedChunk
from backend.app.trust.engine import TrustAssessment, TrustDecision
from backend.app.trust.evidence import TrustEvidence


@dataclass(frozen=True)
class GenerationAgentResult:
    """Immutable typed container for outputs emitted by GenerationAgent.

    Attributes:
        query: Evaluated user query.
        answer_text: Raw generated answer text or structured refusal message.
        is_refusal: Boolean indicating if the response is an explicit refusal/abstention.
        is_supported: Boolean indicating if generation was supported by TrustEngine.
        generation_result: Full GenerationResult object.
        metadata: Execution and token generation details.
    """

    query: str
    answer_text: str
    is_refusal: bool
    is_supported: bool
    generation_result: GenerationResult
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize generation agent result to dictionary."""
        return {
            "query": self.query,
            "answer_text": self.answer_text,
            "is_refusal": self.is_refusal,
            "is_supported": self.is_supported,
            "generation_result": self.generation_result.to_dict(),
            "metadata": dict(self.metadata),
        }


class GenerationAgent(BaseAgent):
    """Specialized agent coordinating GroundedGenerator.

    Architectural Invariant:
    If TrustAssessment is INSUFFICIENT_EVIDENCE, the LLM is NEVER called.
    Structured refusal is emitted immediately.
    """

    def __init__(
        self,
        generator: Optional[GroundedGenerator] = None,
        config: Optional[GenerationConfig] = None,
        abstention_message: Optional[str] = None,
    ) -> None:
        """Initialize GenerationAgent.

        Args:
            generator: Optional pre-configured GroundedGenerator instance.
            config: Optional GenerationConfig instance.
            abstention_message: Custom message returned when generation is gated.
        """
        self._generator = generator
        self._config = config
        self._abstention_message = abstention_message

    @property
    def name(self) -> str:
        return "GenerationAgent"

    @property
    def description(self) -> str:
        return "Generates evidence-grounded responses strictly gated by TrustEngine assessments with refusal safeguards."

    @property
    def generator(self) -> GroundedGenerator:
        """Lazy-initialize GroundedGenerator if not injected."""
        if self._generator is None:
            self._generator = GroundedGenerator(
                config=self._config,
                abstention_message=self._abstention_message,
            )
        return self._generator

    def execute(
        self,
        query: str,
        evidence: Sequence[Union[TrustEvidence, RerankedChunk, Dict[str, Any]]],
        assessment: TrustAssessment,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
    ) -> GenerationAgentResult:
        """Execute evidence-constrained generation under TrustEngine gating.

        Args:
            query: User query string.
            evidence: Sequence of evidence objects.
            assessment: TrustAssessment emitted by TrustEngine / TrustAgent.
            max_new_tokens: Optional token limit override.
            temperature: Optional sampling temperature override.
            top_p: Optional nucleus sampling override.
            repetition_penalty: Optional repetition penalty override.

        Returns:
            GenerationAgentResult containing answer and refusal status.

        Raises:
            AgentInputError: If inputs or assessment types are invalid.
            AgentExecutionError: If generation encounters unexpected runtime errors.
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

        if not isinstance(assessment, TrustAssessment):
            raise AgentInputError(
                f"assessment must be an instance of TrustAssessment, got {type(assessment).__name__}."
            )

        try:
            gen_res = self.generator.generate(
                query=stripped_query,
                evidence=evidence,
                assessment=assessment,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
            )
        except (TypeError, ValueError) as exc:
            raise AgentInputError(f"Invalid generation input: {exc}") from exc
        except Exception as exc:
            raise AgentExecutionError(f"Grounded generation failed: {exc}") from exc

        return GenerationAgentResult(
            query=stripped_query,
            answer_text=gen_res.answer,
            is_refusal=gen_res.is_refusal,
            is_supported=(assessment.decision == TrustDecision.SUPPORTED),
            generation_result=gen_res,
            metadata={
                "model_name": gen_res.model_name,
                "evidence_ids": gen_res.evidence_ids,
                "generation_metadata": gen_res.generation_metadata,
            },
        )
