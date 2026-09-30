"""Configuration model and validation for cross-encoder reranking."""

from dataclasses import asdict, dataclass
from typing import Any, Dict

from backend.app.config import (
    DEFAULT_RERANKER_BATCH_SIZE,
    DEFAULT_RERANKER_CANDIDATE_K,
    DEFAULT_RERANKER_MODEL,
    DEFAULT_RERANKER_TOP_K,
    MAX_RERANKER_TOP_K,
    MIN_RERANKER_TOP_K,
)


class RerankerConfigurationError(Exception):
    """Raised when cross-encoder reranker configuration parameters fail validation."""
    pass


@dataclass(frozen=True)
class RerankerConfig:
    """Immutable, validated configuration object for Cross-Encoder Reranking."""

    model_name: str = DEFAULT_RERANKER_MODEL
    candidate_k: int = DEFAULT_RERANKER_CANDIDATE_K
    final_k: int = DEFAULT_RERANKER_TOP_K
    batch_size: int = DEFAULT_RERANKER_BATCH_SIZE
    max_k: int = MAX_RERANKER_TOP_K

    def __post_init__(self) -> None:
        """Validate configuration parameters upon creation."""
        if not isinstance(self.model_name, str) or not self.model_name.strip():
            raise RerankerConfigurationError("model_name must be a non-empty string.")

        if not isinstance(self.candidate_k, int) or self.candidate_k < MIN_RERANKER_TOP_K:
            raise RerankerConfigurationError(
                f"candidate_k must be a positive integer >= {MIN_RERANKER_TOP_K}, got {self.candidate_k}."
            )

        if not isinstance(self.final_k, int) or self.final_k < MIN_RERANKER_TOP_K:
            raise RerankerConfigurationError(
                f"final_k must be a positive integer >= {MIN_RERANKER_TOP_K}, got {self.final_k}."
            )

        if self.final_k > self.candidate_k:
            raise RerankerConfigurationError(
                f"final_k ({self.final_k}) cannot exceed candidate_k ({self.candidate_k})."
            )

        if self.candidate_k > self.max_k:
            raise RerankerConfigurationError(
                f"candidate_k ({self.candidate_k}) cannot exceed maximum allowed limit of {self.max_k}."
            )

        if self.final_k > self.max_k:
            raise RerankerConfigurationError(
                f"final_k ({self.final_k}) cannot exceed maximum allowed limit of {self.max_k}."
            )

        if not isinstance(self.batch_size, int) or self.batch_size < 1:
            raise RerankerConfigurationError(
                f"batch_size must be a positive integer >= 1, got {self.batch_size}."
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RerankerConfig":
        """Instantiate RerankerConfig from dictionary."""
        if not isinstance(data, dict):
            raise RerankerConfigurationError(f"data must be a dictionary, got {type(data).__name__}.")
        return cls(**data)
