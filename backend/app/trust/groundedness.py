"""Deterministic groundedness estimation based on relevance, coverage, and provenance."""

from dataclasses import dataclass, field
import math
from typing import Any, Dict, Optional, Sequence, Union

from backend.app.reranking.schema import RerankedChunk
from backend.app.trust.coverage import CoverageAssessment, calculate_evidence_coverage
from backend.app.trust.evidence import TrustEvidence
from backend.app.trust.provenance import ProvenanceReport, validate_evidence_provenance
from backend.app.trust.relevance import RelevanceAssessment, aggregate_evidence_relevance


class GroundednessScoringError(ValueError):
    """Raised when groundedness calculation inputs are invalid or weights are improper."""
    pass


@dataclass(frozen=True)
class GroundednessAssessment:
    """Immutable result of deterministic groundedness scoring for query and evidence.

    NOTE ON METHODOLOGY & LIMITATIONS:
    This is a transparent, heuristic trust metric synthesized from:
    1. Cross-encoder neural relevance (calibrated sigmoid probabilities).
    2. Lexical query term coverage across evidence chunks.
    3. Provenance integrity (metadata completeness, valid page ranges, deduplication).

    It is NOT a learned deep-reasoning or NLI model, but an auditable heuristic signal.

    Attributes:
        groundedness_score: Final bounded composite score in [0.0, 1.0].
        relevance_score: Aggregate relevance component in [0.0, 1.0].
        coverage_score: Lexical term coverage ratio in [0.0, 1.0].
        provenance_score: Provenance integrity factor in [0.0, 1.0].
        weights: Dictionary of component weights used in calculation.
        evidence_count: Total number of evidence chunks analyzed.
        relevance_details: Full RelevanceAssessment.
        coverage_details: Full CoverageAssessment.
        provenance_details: Full ProvenanceReport.
    """

    groundedness_score: float
    relevance_score: float
    coverage_score: float
    provenance_score: float
    weights: Dict[str, float]
    evidence_count: int
    relevance_details: RelevanceAssessment
    coverage_details: CoverageAssessment
    provenance_details: ProvenanceReport

    def to_dict(self) -> Dict[str, Any]:
        """Convert assessment to dictionary."""
        return {
            "groundedness_score": self.groundedness_score,
            "relevance_score": self.relevance_score,
            "coverage_score": self.coverage_score,
            "provenance_score": self.provenance_score,
            "weights": dict(self.weights),
            "evidence_count": self.evidence_count,
            "relevance_details": self.relevance_details.to_dict(),
            "coverage_details": self.coverage_details.to_dict(),
            "provenance_details": self.provenance_details.to_dict(),
        }


def calculate_groundedness(
    query: str,
    evidence: Sequence[Union[TrustEvidence, RerankedChunk, Dict[str, Any]]],
    weight_relevance: float = 0.55,
    weight_coverage: float = 0.35,
    weight_provenance: float = 0.10,
    decay_factor: float = 0.5,
) -> GroundednessAssessment:
    """Calculate deterministic groundedness score from multi-signal evidence analysis.

    Formula:
        groundedness = (w_r * relevance_score + w_c * coverage_score + w_p * provenance_score)
        where weights sum to 1.0, and if evidence is empty, groundedness is 0.0.

    Args:
        query: User search query string.
        evidence: Sequence of retrieved & reranked evidence items.
        weight_relevance: Weight for aggregate neural relevance (default 0.55).
        weight_coverage: Weight for lexical term coverage (default 0.35).
        weight_provenance: Weight for provenance integrity (default 0.10).
        decay_factor: Rank discount factor for relevance aggregation.

    Returns:
        GroundednessAssessment instance containing all component scores and breakdown.

    Raises:
        TypeError: If input types are invalid.
        GroundednessScoringError: If weights are negative, don't sum to > 0, or inputs are invalid.
    """
    if not isinstance(query, str):
        raise TypeError(f"query must be a string, got {type(query).__name__}.")

    if not isinstance(evidence, (list, tuple)):
        raise TypeError(f"evidence must be a sequence, got {type(evidence).__name__}.")

    for w_name, w_val in [
        ("weight_relevance", weight_relevance),
        ("weight_coverage", weight_coverage),
        ("weight_provenance", weight_provenance),
    ]:
        if not isinstance(w_val, (int, float)) or not math.isfinite(w_val) or w_val < 0.0:
            raise GroundednessScoringError(f"{w_name} must be a non-negative finite float, got {w_val}.")

    total_weight = weight_relevance + weight_coverage + weight_provenance
    if total_weight <= 0.0:
        raise GroundednessScoringError("Sum of weights must be strictly positive.")

    # Normalize weights to sum to 1.0
    norm_w_r = weight_relevance / total_weight
    norm_w_c = weight_coverage / total_weight
    norm_w_p = weight_provenance / total_weight

    weights_dict = {
        "relevance": norm_w_r,
        "coverage": norm_w_c,
        "provenance": norm_w_p,
    }

    # If empty evidence or empty query
    if len(evidence) == 0 or not query.strip():
        rel_assess = aggregate_evidence_relevance([], decay_factor=decay_factor)
        cov_assess = calculate_evidence_coverage(
            query if query.strip() else "empty", []
        ) if query.strip() else CoverageAssessment(
            query_terms=[], covered_terms=[], uncovered_terms=[], coverage_ratio=0.0, chunk_coverage_ratios=[], evidence_count=0
        )
        prov_assess = validate_evidence_provenance([])

        return GroundednessAssessment(
            groundedness_score=0.0,
            relevance_score=0.0,
            coverage_score=0.0,
            provenance_score=1.0 if len(evidence) == 0 else 0.0,
            weights=weights_dict,
            evidence_count=len(evidence),
            relevance_details=rel_assess,
            coverage_details=cov_assess,
            provenance_details=prov_assess,
        )

    # 1. Relevance Assessment
    rel_assess = aggregate_evidence_relevance(evidence, decay_factor=decay_factor)
    rel_score = rel_assess.weighted_relevance

    # 2. Coverage Assessment
    cov_assess = calculate_evidence_coverage(query, evidence)
    cov_score = cov_assess.coverage_ratio

    # 3. Provenance Assessment
    prov_assess = validate_evidence_provenance(evidence)
    prov_score = (
        prov_assess.valid_chunks_count / prov_assess.total_chunks
        if prov_assess.total_chunks > 0
        else 0.0
    )

    # Composite Groundedness Formula
    raw_groundedness = (
        norm_w_r * rel_score +
        norm_w_c * cov_score +
        norm_w_p * prov_score
    )

    groundedness_score = max(0.0, min(1.0, float(raw_groundedness)))

    return GroundednessAssessment(
        groundedness_score=groundedness_score,
        relevance_score=rel_score,
        coverage_score=cov_score,
        provenance_score=prov_score,
        weights=weights_dict,
        evidence_count=len(evidence),
        relevance_details=rel_assess,
        coverage_details=cov_assess,
        provenance_details=prov_assess,
    )
