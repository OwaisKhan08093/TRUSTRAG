"""Structured request and response data models for local LLM generation."""

from dataclasses import asdict, dataclass, field
import math
from typing import Any, Dict, List, Optional

from backend.app.generation.config import (
    DEFAULT_GENERATION_MODEL,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_REPETITION_PENALTY,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
)


class ModelValidationError(ValueError):
    """Raised when GenerationRequest or GenerationResult fields fail validation."""
    pass


@dataclass(frozen=True)
class GenerationRequest:
    """Immutable parameters submitted for an evidence-grounded generation query.

    Attributes:
        query: User search query string.
        evidence_ids: List of chunk identifiers providing trusted support.
        model_name: Identifier of the local LLM.
        max_new_tokens: Maximum token generation budget.
        temperature: Sampling temperature (0.0 for greedy).
        top_p: Nucleus sampling probability mass.
        repetition_penalty: Token repetition penalty.
    """

    query: str
    evidence_ids: List[str] = field(default_factory=list)
    model_name: str = DEFAULT_GENERATION_MODEL
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS
    temperature: float = DEFAULT_TEMPERATURE
    top_p: float = DEFAULT_TOP_P
    repetition_penalty: float = DEFAULT_REPETITION_PENALTY

    def __post_init__(self) -> None:
        """Validate request parameters."""
        if not isinstance(self.query, str) or not self.query.strip():
            raise ModelValidationError("query must be a non-empty string.")

        if not isinstance(self.evidence_ids, (list, tuple)):
            raise ModelValidationError(
                f"evidence_ids must be a list or tuple of strings, got {type(self.evidence_ids).__name__}."
            )

        for idx, eid in enumerate(self.evidence_ids):
            if not isinstance(eid, str) or not eid.strip():
                raise ModelValidationError(f"evidence_ids[{idx}] must be a non-empty string.")

        if not isinstance(self.model_name, str) or not self.model_name.strip():
            raise ModelValidationError("model_name must be a non-empty string.")

        if not isinstance(self.max_new_tokens, int) or isinstance(self.max_new_tokens, bool) or self.max_new_tokens <= 0:
            raise ModelValidationError(
                f"max_new_tokens must be an integer > 0, got {self.max_new_tokens}."
            )

        if not isinstance(self.temperature, (int, float)) or not math.isfinite(self.temperature) or self.temperature < 0.0:
            raise ModelValidationError(
                f"temperature must be a finite float >= 0.0, got {self.temperature}."
            )

        if not isinstance(self.top_p, (int, float)) or not math.isfinite(self.top_p) or self.top_p <= 0.0 or self.top_p > 1.0:
            raise ModelValidationError(f"top_p must be a float in (0.0, 1.0], got {self.top_p}.")

        if not isinstance(self.repetition_penalty, (int, float)) or not math.isfinite(self.repetition_penalty) or self.repetition_penalty <= 0.0:
            raise ModelValidationError(
                f"repetition_penalty must be a float > 0.0, got {self.repetition_penalty}."
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert request to dictionary."""
        return {
            "query": self.query,
            "evidence_ids": list(self.evidence_ids),
            "model_name": self.model_name,
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "repetition_penalty": self.repetition_penalty,
        }


@dataclass(frozen=True)
class GenerationResult:
    """Immutable structured result produced by the local LLM generation layer.

    Attributes:
        query: Original user query.
        answer: Generated text response (or explicit refusal notification).
        model_name: Identifier of the local LLM used.
        evidence_ids: List of chunk IDs referenced in the generation prompt.
        is_refusal: True if generation was safely abstained (e.g. insufficient evidence).
        refusal_reason: Explanation if is_refusal is True.
        generation_metadata: Execution metadata (e.g. temperature, latency, token count).
    """

    query: str
    answer: str
    model_name: str
    evidence_ids: List[str]
    is_refusal: bool = False
    refusal_reason: Optional[str] = None
    generation_metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate result attributes."""
        if not isinstance(self.query, str) or not self.query.strip():
            raise ModelValidationError("query must be a non-empty string.")

        if not isinstance(self.answer, str) or not self.answer.strip():
            raise ModelValidationError("answer must be a non-empty string.")

        if not isinstance(self.model_name, str) or not self.model_name.strip():
            raise ModelValidationError("model_name must be a non-empty string.")

        if not isinstance(self.evidence_ids, (list, tuple)):
            raise ModelValidationError(
                f"evidence_ids must be a sequence of strings, got {type(self.evidence_ids).__name__}."
            )

        if not isinstance(self.is_refusal, bool):
            raise ModelValidationError(
                f"is_refusal must be a boolean, got {type(self.is_refusal).__name__}."
            )

        if not isinstance(self.generation_metadata, dict):
            raise ModelValidationError(
                f"generation_metadata must be a dictionary, got {type(self.generation_metadata).__name__}."
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "query": self.query,
            "answer": self.answer,
            "model_name": self.model_name,
            "evidence_ids": list(self.evidence_ids),
            "is_refusal": self.is_refusal,
            "refusal_reason": self.refusal_reason,
            "generation_metadata": dict(self.generation_metadata),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GenerationResult":
        """Instantiate GenerationResult from a dictionary."""
        if not isinstance(data, dict):
            raise ModelValidationError(f"data must be a dictionary, got {type(data).__name__}.")

        required_keys = ["query", "answer", "model_name", "evidence_ids"]
        for k in required_keys:
            if k not in data:
                raise ModelValidationError(f"Missing required key '{k}' in result data.")

        return cls(
            query=str(data["query"]),
            answer=str(data["answer"]),
            model_name=str(data["model_name"]),
            evidence_ids=list(data["evidence_ids"]),
            is_refusal=bool(data.get("is_refusal", False)),
            refusal_reason=data.get("refusal_reason"),
            generation_metadata=dict(data.get("generation_metadata", {})),
        )
