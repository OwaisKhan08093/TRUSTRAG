"""Configuration and hyperparameter models for local LLM generation."""

from dataclasses import asdict, dataclass
import math
from typing import Any, Dict, Optional

DEFAULT_GENERATION_MODEL: str = "Qwen/Qwen2.5-3B-Instruct"
DEFAULT_MAX_NEW_TOKENS: int = 512
DEFAULT_TEMPERATURE: float = 0.0  # Greedy/deterministic decoding by default for factual grounding
DEFAULT_TOP_P: float = 0.9
DEFAULT_REPETITION_PENALTY: float = 1.1


class GenerationConfigError(ValueError):
    """Raised when generation configuration parameters fail validation."""
    pass


@dataclass(frozen=True)
class GenerationConfig:
    """Immutable configuration for local LLM answer generation.

    Attributes:
        model_name: Hugging Face model identifier (default: Qwen/Qwen2.5-3B-Instruct).
        max_new_tokens: Maximum number of tokens to generate (> 0).
        temperature: Sampling temperature (>= 0.0). 0.0 specifies deterministic greedy decoding.
        top_p: Nucleus sampling probability mass (in (0.0, 1.0]).
        repetition_penalty: Penalty factor for token repetition (> 0.0).
        system_prompt_template: Optional custom system prompt string.
    """

    model_name: str = DEFAULT_GENERATION_MODEL
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS
    temperature: float = DEFAULT_TEMPERATURE
    top_p: float = DEFAULT_TOP_P
    repetition_penalty: float = DEFAULT_REPETITION_PENALTY
    system_prompt_template: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate all configuration attributes upon instantiation."""
        if not isinstance(self.model_name, str) or not self.model_name.strip():
            raise GenerationConfigError("model_name must be a non-empty string.")

        if not isinstance(self.max_new_tokens, int) or isinstance(self.max_new_tokens, bool) or self.max_new_tokens <= 0:
            raise GenerationConfigError(
                f"max_new_tokens must be a positive integer > 0, got {self.max_new_tokens}."
            )

        if not isinstance(self.temperature, (int, float)) or not math.isfinite(self.temperature) or self.temperature < 0.0:
            raise GenerationConfigError(
                f"temperature must be a non-negative finite float >= 0.0, got {self.temperature}."
            )

        if not isinstance(self.top_p, (int, float)) or not math.isfinite(self.top_p) or self.top_p <= 0.0 or self.top_p > 1.0:
            raise GenerationConfigError(
                f"top_p must be a float in (0.0, 1.0], got {self.top_p}."
            )

        if not isinstance(self.repetition_penalty, (int, float)) or not math.isfinite(self.repetition_penalty) or self.repetition_penalty <= 0.0:
            raise GenerationConfigError(
                f"repetition_penalty must be a positive float > 0.0, got {self.repetition_penalty}."
            )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize configuration to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GenerationConfig":
        """Instantiate GenerationConfig from a dictionary.

        Args:
            data: Key-value mapping of generation settings.

        Returns:
            Validated GenerationConfig instance.
        """
        if not isinstance(data, dict):
            raise GenerationConfigError(f"data must be a dictionary, got {type(data).__name__}.")

        valid_fields = cls.__dataclass_fields__.keys()
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)
