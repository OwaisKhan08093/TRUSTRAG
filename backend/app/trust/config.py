"""Centralized configuration for the Trust Engine subsystem."""

from dataclasses import asdict, dataclass
import math
from typing import Any, Dict, Optional


class TrustConfigError(ValueError):
    """Raised when trust configuration fails validation."""
    pass


@dataclass(frozen=True)
class TrustConfig:
    """Immutable configuration holding all trust thresholds, weights, and rules.

    Attributes:
        min_confidence_threshold: Minimum overall confidence required for SUPPORTED decision.
        min_groundedness_threshold: Minimum groundedness score required for SUPPORTED.
        min_coverage_threshold: Minimum query term coverage ratio required for SUPPORTED.
        min_evidence_count: Minimum number of evidence chunks required.
        require_valid_provenance: Whether provenance defects immediately disqualify evidence.
        relevance_decay_factor: Discount multiplier per rank index in relevance aggregation.
        weight_relevance: Relevance weight in groundedness score.
        weight_coverage: Coverage weight in groundedness score.
        weight_provenance: Provenance weight in groundedness score.
        weight_confidence_groundedness: Groundedness weight in confidence calculation.
        weight_confidence_top_relevance: Top-1 relevance weight in confidence calculation.
        weight_confidence_coverage: Coverage weight in confidence calculation.
        weight_confidence_volume: Evidence volume saturation weight in confidence calculation.
    """

    min_confidence_threshold: float = 0.50
    min_groundedness_threshold: float = 0.45
    min_coverage_threshold: float = 0.30
    min_evidence_count: int = 1
    require_valid_provenance: bool = True
    relevance_decay_factor: float = 0.50
    weight_relevance: float = 0.55
    weight_coverage: float = 0.35
    weight_provenance: float = 0.10
    weight_confidence_groundedness: float = 0.50
    weight_confidence_top_relevance: float = 0.30
    weight_confidence_coverage: float = 0.10
    weight_confidence_volume: float = 0.10

    def __post_init__(self) -> None:
        """Validate all configuration parameters upon instantiation."""
        # Validate probability / ratio thresholds in [0.0, 1.0]
        for name, val in [
            ("min_confidence_threshold", self.min_confidence_threshold),
            ("min_groundedness_threshold", self.min_groundedness_threshold),
            ("min_coverage_threshold", self.min_coverage_threshold),
            ("relevance_decay_factor", self.relevance_decay_factor),
        ]:
            if not isinstance(val, (int, float)) or not math.isfinite(val) or val < 0.0 or val > 1.0:
                raise TrustConfigError(f"{name} must be a float bounded in [0.0, 1.0], got {val}.")

        if not isinstance(self.min_evidence_count, int) or self.min_evidence_count < 1:
            raise TrustConfigError(
                f"min_evidence_count must be an integer >= 1, got {self.min_evidence_count}."
            )

        if not isinstance(self.require_valid_provenance, bool):
            raise TrustConfigError(
                f"require_valid_provenance must be a boolean, got {type(self.require_valid_provenance).__name__}."
            )

        # Validate groundedness weights
        for name, val in [
            ("weight_relevance", self.weight_relevance),
            ("weight_coverage", self.weight_coverage),
            ("weight_provenance", self.weight_provenance),
        ]:
            if not isinstance(val, (int, float)) or not math.isfinite(val) or val < 0.0:
                raise TrustConfigError(f"{name} must be a non-negative finite float, got {val}.")

        if (self.weight_relevance + self.weight_coverage + self.weight_provenance) <= 0.0:
            raise TrustConfigError("Sum of groundedness weights must be strictly positive.")

        # Validate confidence weights
        for name, val in [
            ("weight_confidence_groundedness", self.weight_confidence_groundedness),
            ("weight_confidence_top_relevance", self.weight_confidence_top_relevance),
            ("weight_confidence_coverage", self.weight_confidence_coverage),
            ("weight_confidence_volume", self.weight_confidence_volume),
        ]:
            if not isinstance(val, (int, float)) or not math.isfinite(val) or val < 0.0:
                raise TrustConfigError(f"{name} must be a non-negative finite float, got {val}.")

        sum_conf = (
            self.weight_confidence_groundedness
            + self.weight_confidence_top_relevance
            + self.weight_confidence_coverage
            + self.weight_confidence_volume
        )
        if sum_conf <= 0.0:
            raise TrustConfigError("Sum of confidence weights must be strictly positive.")

    def to_dict(self) -> Dict[str, Any]:
        """Serialize configuration to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrustConfig":
        """Instantiate TrustConfig from a dictionary of settings.

        Args:
            data: Dictionary containing configuration key-values.

        Returns:
            Validated TrustConfig instance.
        """
        if not isinstance(data, dict):
            raise TrustConfigError(f"data must be a dictionary, got {type(data).__name__}.")

        valid_fields = cls.__dataclass_fields__.keys()
        filtered_kwargs = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered_kwargs)
