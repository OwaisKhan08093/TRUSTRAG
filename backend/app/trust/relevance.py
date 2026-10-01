"""Evidence relevance scoring and aggregation for the Trust Engine."""

from dataclasses import asdict, dataclass
import math
from typing import Any, Dict, List, Optional, Sequence, Union

from backend.app.reranking.schema import RerankedChunk
from backend.app.trust.evidence import TrustEvidence


class RelevanceScoringError(ValueError):
    """Raised when relevance scoring encounters invalid inputs or non-finite scores."""
    pass


def sigmoid(x: float) -> float:
    """Compute numerically stable logistic sigmoid mapping logits (-inf, +inf) to (0, 1).

    Args:
        x: Real-valued logit / cross-encoder score.

    Returns:
        Float bounded in (0.0, 1.0).
    """
    if not isinstance(x, (int, float)) or not math.isfinite(x):
        raise RelevanceScoringError(f"Score must be a finite float, got {x}.")

    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        z = math.exp(x)
        return z / (1.0 + z)


@dataclass(frozen=True)
class RelevanceAssessment:
    """Immutable summary of evidence relevance signals.

    Attributes:
        top_score: Raw cross-encoder score of the top-ranked evidence chunk (0.0 if empty).
        top_probability: Sigmoid-transformed probability of the top chunk (0.0 if empty).
        mean_probability: Unweighted arithmetic mean of sigmoid probabilities across evidence.
        weighted_relevance: Rank-discounted aggregate relevance score bounded in [0.0, 1.0].
        individual_scores: List of raw cross-encoder scores in rank order.
        individual_probabilities: List of sigmoid-transformed probabilities in rank order.
    """

    top_score: float
    top_probability: float
    mean_probability: float
    weighted_relevance: float
    individual_scores: List[float]
    individual_probabilities: List[float]

    def to_dict(self) -> Dict[str, Any]:
        """Convert assessment to dictionary."""
        return {
            "top_score": self.top_score,
            "top_probability": self.top_probability,
            "mean_probability": self.mean_probability,
            "weighted_relevance": self.weighted_relevance,
            "individual_scores": list(self.individual_scores),
            "individual_probabilities": list(self.individual_probabilities),
        }


def aggregate_evidence_relevance(
    evidence: Sequence[Union[TrustEvidence, RerankedChunk, Dict[str, Any]]],
    decay_factor: float = 0.5,
) -> RelevanceAssessment:
    """Calculate deterministic aggregate relevance signals from a sequence of ranked evidence chunks.

    Methodology:
    1. Extract rerank_score from each evidence item.
    2. Map raw logits to calibrated probabilities via logistic sigmoid: p_i = sigmoid(s_i).
    3. Compute rank-discounted weighted aggregate relevance:
       weighted_relevance = sum(p_i * (decay_factor ** i)) / sum(decay_factor ** i) for i in 0..N-1.
       This gives highest priority to top-1 evidence while incorporating support from secondary chunks.

    Args:
        evidence: Sequence of TrustEvidence, RerankedChunk, or dicts with 'rerank_score'.
        decay_factor: Rank discount factor in (0.0, 1.0]. Defaults to 0.5 (half-life per rank step).

    Returns:
        RelevanceAssessment object with bounded metrics.

    Raises:
        RelevanceScoringError: If inputs are invalid or contain non-finite scores.
        TypeError: If evidence container is not a sequence.
    """
    if not isinstance(evidence, (list, tuple)):
        raise TypeError(f"evidence must be a sequence, got {type(evidence).__name__}.")

    if not isinstance(decay_factor, (int, float)) or decay_factor <= 0.0 or decay_factor > 1.0:
        raise RelevanceScoringError(f"decay_factor must be in (0.0, 1.0], got {decay_factor}.")

    if len(evidence) == 0:
        return RelevanceAssessment(
            top_score=0.0,
            top_probability=0.0,
            mean_probability=0.0,
            weighted_relevance=0.0,
            individual_scores=[],
            individual_probabilities=[],
        )

    raw_scores: List[float] = []
    for idx, item in enumerate(evidence):
        if isinstance(item, (TrustEvidence, RerankedChunk)):
            score = item.rerank_score
        elif isinstance(item, dict):
            if "rerank_score" not in item:
                raise RelevanceScoringError(f"Evidence dict at index {idx} missing 'rerank_score'.")
            score = item["rerank_score"]
        else:
            raise TypeError(
                f"Evidence item at index {idx} must be TrustEvidence, RerankedChunk, or dict, got {type(item).__name__}."
            )

        if not isinstance(score, (int, float)) or not math.isfinite(score):
            raise RelevanceScoringError(f"Score at index {idx} must be a finite float, got {score}.")

        raw_scores.append(float(score))

    probabilities = [sigmoid(s) for s in raw_scores]

    # Compute rank-discounted weighted average
    weights = [decay_factor ** i for i in range(len(probabilities))]
    total_weight = sum(weights)
    weighted_sum = sum(p * w for p, w in zip(probabilities, weights))
    weighted_rel = weighted_sum / total_weight if total_weight > 0 else 0.0

    # Ensure strictly bounded in [0.0, 1.0]
    weighted_rel = max(0.0, min(1.0, weighted_rel))
    mean_prob = max(0.0, min(1.0, sum(probabilities) / len(probabilities)))

    return RelevanceAssessment(
        top_score=raw_scores[0],
        top_probability=probabilities[0],
        mean_probability=mean_prob,
        weighted_relevance=weighted_rel,
        individual_scores=raw_scores,
        individual_probabilities=probabilities,
    )
