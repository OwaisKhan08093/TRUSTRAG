"""Deterministic confidence scoring derived from evidence signals for the Trust Engine."""

from dataclasses import dataclass
import math
from typing import Any, Dict, Optional, Sequence, Union

from backend.app.reranking.schema import RerankedChunk
from backend.app.trust.evidence import TrustEvidence
from backend.app.trust.groundedness import GroundednessAssessment, calculate_groundedness


class ConfidenceScoringError(ValueError):
    """Raised when confidence scoring receives invalid arguments or configuration."""
    pass


@dataclass(frozen=True)
class ConfidenceAssessment:
    """Immutable assessment of system evidence confidence.

    IMPORTANT NOTE:
    This confidence score is NOT a statistical probability of answer correctness.
    It is an auditable, deterministic trust index reflecting whether the retrieved
    and reranked evidence corpus provides sufficient, consistent, and well-grounded
    support for downstream reasoning.

    Attributes:
        confidence_score: Final bounded confidence index in [0.0, 1.0].
        groundedness_score: Groundedness component in [0.0, 1.0].
        top_relevance_prob: Sigmoid-transformed probability of the top-1 evidence item.
        coverage_score: Lexical term coverage ratio in [0.0, 1.0].
        volume_factor: Evidence chunk sufficiency saturation factor in [0.0, 1.0].
        provenance_score: Provenance integrity factor in [0.0, 1.0].
        evidence_count: Number of evidence chunks evaluated.
        groundedness_details: Full GroundednessAssessment.
    """

    confidence_score: float
    groundedness_score: float
    top_relevance_prob: float
    coverage_score: float
    volume_factor: float
    provenance_score: float
    evidence_count: int
    groundedness_details: GroundednessAssessment

    def to_dict(self) -> Dict[str, Any]:
        """Convert assessment to dictionary."""
        return {
            "confidence_score": self.confidence_score,
            "groundedness_score": self.groundedness_score,
            "top_relevance_prob": self.top_relevance_prob,
            "coverage_score": self.coverage_score,
            "volume_factor": self.volume_factor,
            "provenance_score": self.provenance_score,
            "evidence_count": self.evidence_count,
            "groundedness_details": self.groundedness_details.to_dict(),
        }


def calculate_confidence(
    query: str,
    evidence: Sequence[Union[TrustEvidence, RerankedChunk, Dict[str, Any]]],
    min_supporting_chunks: int = 1,
    weight_groundedness: float = 0.50,
    weight_top_relevance: float = 0.30,
    weight_coverage: float = 0.10,
    weight_volume: float = 0.10,
    groundedness_assessment: Optional[GroundednessAssessment] = None,
) -> ConfidenceAssessment:
    """Calculate deterministic confidence score from multi-signal evidence analysis.

    Methodology:
    1. Obtain GroundednessAssessment (computing it if not already provided).
    2. Extract top-1 reranker sigmoid probability $T$.
    3. Compute evidence volume saturation factor: $S = \\min(1.0, N / \\text{min_supporting_chunks})$.
    4. Compute weighted linear combination bounded in [0.0, 1.0].

    Args:
        query: Search query string.
        evidence: Sequence of evidence items.
        min_supporting_chunks: Number of supporting chunks desired for full volume saturation.
        weight_groundedness: Weight for composite groundedness (default 0.50).
        weight_top_relevance: Weight for top-1 reranker signal (default 0.30).
        weight_coverage: Weight for query coverage ratio (default 0.10).
        weight_volume: Weight for evidence volume sufficiency (default 0.10).
        groundedness_assessment: Optional precomputed GroundednessAssessment.

    Returns:
        ConfidenceAssessment instance.

    Raises:
        TypeError: If query or evidence types are invalid.
        ConfidenceScoringError: If weights or min_supporting_chunks are invalid.
    """
    if not isinstance(query, str):
        raise TypeError(f"query must be a string, got {type(query).__name__}.")

    if not isinstance(evidence, (list, tuple)):
        raise TypeError(f"evidence must be a sequence, got {type(evidence).__name__}.")

    if not isinstance(min_supporting_chunks, int) or min_supporting_chunks < 1:
        raise ConfidenceScoringError(
            f"min_supporting_chunks must be an integer >= 1, got {min_supporting_chunks}."
        )

    for w_name, w_val in [
        ("weight_groundedness", weight_groundedness),
        ("weight_top_relevance", weight_top_relevance),
        ("weight_coverage", weight_coverage),
        ("weight_volume", weight_volume),
    ]:
        if not isinstance(w_val, (int, float)) or not math.isfinite(w_val) or w_val < 0.0:
            raise ConfidenceScoringError(f"{w_name} must be a non-negative finite float, got {w_val}.")

    total_weight = weight_groundedness + weight_top_relevance + weight_coverage + weight_volume
    if total_weight <= 0.0:
        raise ConfidenceScoringError("Sum of confidence weights must be strictly positive.")

    # Obtain or compute groundedness
    if groundedness_assessment is None:
        groundedness_assessment = calculate_groundedness(query, evidence)

    n_chunks = len(evidence)
    if n_chunks == 0 or not query.strip():
        return ConfidenceAssessment(
            confidence_score=0.0,
            groundedness_score=0.0,
            top_relevance_prob=0.0,
            coverage_score=0.0,
            volume_factor=0.0,
            provenance_score=1.0 if n_chunks == 0 else 0.0,
            evidence_count=n_chunks,
            groundedness_details=groundedness_assessment,
        )

    g_score = groundedness_assessment.groundedness_score
    top_prob = groundedness_assessment.relevance_details.top_probability
    cov_score = groundedness_assessment.coverage_score
    prov_score = groundedness_assessment.provenance_score
    volume_factor = min(1.0, n_chunks / float(min_supporting_chunks))

    # Calculate normalized weighted composite
    w_g = weight_groundedness / total_weight
    w_t = weight_top_relevance / total_weight
    w_c = weight_coverage / total_weight
    w_v = weight_volume / total_weight

    raw_conf = (
        w_g * g_score +
        w_t * top_prob +
        w_c * cov_score +
        w_v * volume_factor
    )

    # Scale by provenance integrity if any provenance defects exist
    if prov_score < 1.0:
        raw_conf = raw_conf * prov_score

    confidence_score = max(0.0, min(1.0, float(raw_conf)))

    return ConfidenceAssessment(
        confidence_score=confidence_score,
        groundedness_score=g_score,
        top_relevance_prob=top_prob,
        coverage_score=cov_score,
        volume_factor=volume_factor,
        provenance_score=prov_score,
        evidence_count=n_chunks,
        groundedness_details=groundedness_assessment,
    )
