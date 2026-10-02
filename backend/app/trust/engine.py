"""Trust Decision Engine evaluating evidence sufficiency, groundedness, and confidence."""

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Any, Dict, List, Optional, Sequence, Union

from backend.app.reranking.schema import RerankedChunk
from backend.app.trust.confidence import ConfidenceAssessment, calculate_confidence
from backend.app.trust.coverage import CoverageAssessment, calculate_evidence_coverage
from backend.app.trust.evidence import TrustEvidence
from backend.app.trust.groundedness import GroundednessAssessment, calculate_groundedness
from backend.app.trust.provenance import ProvenanceReport, validate_evidence_provenance
from backend.app.trust.relevance import RelevanceAssessment, aggregate_evidence_relevance


class TrustDecision(str, Enum):
    """Discrete decision outcome regarding retrieved evidence grounding."""

    SUPPORTED = "SUPPORTED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


class TrustEngineError(ValueError):
    """Raised when TrustEngine encounters invalid inputs or configuration."""
    pass


@dataclass(frozen=True)
class TrustAssessment:
    """Comprehensive trust assessment for a query and its retrieved evidence.

    Attributes:
        query: Evaluated user query.
        decision: Discrete trust decision (SUPPORTED or INSUFFICIENT_EVIDENCE).
        relevance_score: Weighted aggregate relevance score in [0.0, 1.0].
        coverage_score: Lexical query term coverage ratio in [0.0, 1.0].
        groundedness_score: Composite evidence groundedness in [0.0, 1.0].
        confidence_score: Composite system trust confidence in [0.0, 1.0].
        provenance_valid: Boolean indicating if all evidence chunks have valid provenance.
        evidence_count: Total number of evidence chunks analyzed.
        decision_reasons: List of explanatory factors supporting the trust decision.
        relevance_details: Full RelevanceAssessment.
        coverage_details: Full CoverageAssessment.
        groundedness_details: Full GroundednessAssessment.
        confidence_details: Full ConfidenceAssessment.
        provenance_details: Full ProvenanceReport.
    """

    query: str
    decision: TrustDecision
    relevance_score: float
    coverage_score: float
    groundedness_score: float
    confidence_score: float
    provenance_valid: bool
    evidence_count: int
    decision_reasons: List[str]
    relevance_details: RelevanceAssessment
    coverage_details: CoverageAssessment
    groundedness_details: GroundednessAssessment
    confidence_details: ConfidenceAssessment
    provenance_details: ProvenanceReport

    def to_dict(self) -> Dict[str, Any]:
        """Convert TrustAssessment to dictionary format."""
        return {
            "query": self.query,
            "decision": self.decision.value,
            "relevance_score": self.relevance_score,
            "coverage_score": self.coverage_score,
            "groundedness_score": self.groundedness_score,
            "confidence_score": self.confidence_score,
            "provenance_valid": self.provenance_valid,
            "evidence_count": self.evidence_count,
            "decision_reasons": list(self.decision_reasons),
            "relevance_details": self.relevance_details.to_dict(),
            "coverage_details": self.coverage_details.to_dict(),
            "groundedness_details": self.groundedness_details.to_dict(),
            "confidence_details": self.confidence_details.to_dict(),
            "provenance_details": self.provenance_details.to_dict(),
        }


class TrustEngine:
    """Core evaluation engine that measures evidence grounding and issues trust decisions.

    The TrustEngine coordinates relevance aggregation, term coverage, provenance integrity,
    groundedness scoring, and confidence synthesis to determine whether evidence is sufficient
    to support downstream reasoning without hallucination.
    """

    def __init__(
        self,
        min_confidence_threshold: float = 0.50,
        min_groundedness_threshold: float = 0.45,
        min_coverage_threshold: float = 0.30,
        min_evidence_count: int = 1,
        require_valid_provenance: bool = True,
        weight_relevance: float = 0.55,
        weight_coverage: float = 0.35,
        weight_provenance: float = 0.10,
        weight_confidence_groundedness: float = 0.50,
        weight_confidence_top_relevance: float = 0.30,
        weight_confidence_coverage: float = 0.10,
        weight_confidence_volume: float = 0.10,
    ) -> None:
        """Initialize TrustEngine with configurable evaluation thresholds and weights.

        Args:
            min_confidence_threshold: Minimum confidence required for SUPPORTED decision.
            min_groundedness_threshold: Minimum groundedness score required for SUPPORTED.
            min_coverage_threshold: Minimum query term coverage ratio required for SUPPORTED.
            min_evidence_count: Minimum number of evidence chunks required.
            require_valid_provenance: If True, any provenance defect triggers INSUFFICIENT_EVIDENCE.
            weight_relevance: Relevance weight in groundedness score.
            weight_coverage: Coverage weight in groundedness score.
            weight_provenance: Provenance weight in groundedness score.
            weight_confidence_groundedness: Groundedness weight in confidence score.
            weight_confidence_top_relevance: Top-1 relevance weight in confidence score.
            weight_confidence_coverage: Coverage weight in confidence score.
            weight_confidence_volume: Volume sufficiency weight in confidence score.
        """
        for th_name, th_val in [
            ("min_confidence_threshold", min_confidence_threshold),
            ("min_groundedness_threshold", min_groundedness_threshold),
            ("min_coverage_threshold", min_coverage_threshold),
        ]:
            if not isinstance(th_val, (int, float)) or not math.isfinite(th_val) or th_val < 0.0 or th_val > 1.0:
                raise TrustEngineError(f"{th_name} must be a float bounded in [0.0, 1.0], got {th_val}.")

        if not isinstance(min_evidence_count, int) or min_evidence_count < 1:
            raise TrustEngineError(f"min_evidence_count must be an integer >= 1, got {min_evidence_count}.")

        self.min_confidence_threshold = float(min_confidence_threshold)
        self.min_groundedness_threshold = float(min_groundedness_threshold)
        self.min_coverage_threshold = float(min_coverage_threshold)
        self.min_evidence_count = min_evidence_count
        self.require_valid_provenance = bool(require_valid_provenance)

        self.weight_relevance = float(weight_relevance)
        self.weight_coverage = float(weight_coverage)
        self.weight_provenance = float(weight_provenance)

        self.weight_confidence_groundedness = float(weight_confidence_groundedness)
        self.weight_confidence_top_relevance = float(weight_confidence_top_relevance)
        self.weight_confidence_coverage = float(weight_confidence_coverage)
        self.weight_confidence_volume = float(weight_confidence_volume)

    def evaluate(
        self,
        query: str,
        evidence: Sequence[Union[TrustEvidence, RerankedChunk, Dict[str, Any]]],
    ) -> TrustAssessment:
        """Evaluate evidence grounding and produce a deterministic trust assessment.

        Args:
            query: User search query string.
            evidence: Sequence of retrieved & reranked evidence items.

        Returns:
            TrustAssessment containing scores, breakdown, and discrete decision.

        Raises:
            TypeError: If query or evidence types are invalid.
        """
        if not isinstance(query, str):
            raise TypeError(f"query must be a string, got {type(query).__name__}.")

        if not isinstance(evidence, (list, tuple)):
            raise TypeError(f"evidence must be a sequence, got {type(evidence).__name__}.")

        reasons: List[str] = []
        clean_query = query.strip()

        # Handle empty query edge-case
        if not clean_query:
            reasons.append("Query is empty or contains only whitespace.")
            g_assess = calculate_groundedness("", evidence)
            c_assess = calculate_confidence("", evidence, groundedness_assessment=g_assess)
            return TrustAssessment(
                query=query,
                decision=TrustDecision.INSUFFICIENT_EVIDENCE,
                relevance_score=0.0,
                coverage_score=0.0,
                groundedness_score=0.0,
                confidence_score=0.0,
                provenance_valid=True if len(evidence) == 0 else g_assess.provenance_details.is_valid,
                evidence_count=len(evidence),
                decision_reasons=reasons,
                relevance_details=g_assess.relevance_details,
                coverage_details=g_assess.coverage_details,
                groundedness_details=g_assess,
                confidence_details=c_assess,
                provenance_details=g_assess.provenance_details,
            )

        # Handle empty evidence corpus
        if len(evidence) == 0:
            reasons.append("No evidence chunks were provided.")
            g_assess = calculate_groundedness(clean_query, [])
            c_assess = calculate_confidence(clean_query, [], groundedness_assessment=g_assess)
            return TrustAssessment(
                query=query,
                decision=TrustDecision.INSUFFICIENT_EVIDENCE,
                relevance_score=0.0,
                coverage_score=0.0,
                groundedness_score=0.0,
                confidence_score=0.0,
                provenance_valid=True,
                evidence_count=0,
                decision_reasons=reasons,
                relevance_details=g_assess.relevance_details,
                coverage_details=g_assess.coverage_details,
                groundedness_details=g_assess,
                confidence_details=c_assess,
                provenance_details=g_assess.provenance_details,
            )

        # 1. Compute multi-signal groundedness
        g_assess = calculate_groundedness(
            clean_query,
            evidence,
            weight_relevance=self.weight_relevance,
            weight_coverage=self.weight_coverage,
            weight_provenance=self.weight_provenance,
        )

        # 2. Compute confidence index
        c_assess = calculate_confidence(
            clean_query,
            evidence,
            min_supporting_chunks=self.min_evidence_count,
            weight_groundedness=self.weight_confidence_groundedness,
            weight_top_relevance=self.weight_confidence_top_relevance,
            weight_coverage=self.weight_confidence_coverage,
            weight_volume=self.weight_confidence_volume,
            groundedness_assessment=g_assess,
        )

        rel_score = g_assess.relevance_score
        cov_score = g_assess.coverage_score
        groundedness_score = g_assess.groundedness_score
        conf_score = c_assess.confidence_score
        prov_valid = g_assess.provenance_details.is_valid
        n_chunks = len(evidence)

        # Decision rule evaluation
        is_supported = True

        if self.require_valid_provenance and not prov_valid:
            is_supported = False
            reasons.append(
                f"Provenance validation failed with {len(g_assess.provenance_details.issues)} metadata defect(s)."
            )

        if n_chunks < self.min_evidence_count:
            is_supported = False
            reasons.append(
                f"Evidence count ({n_chunks}) is below minimum threshold ({self.min_evidence_count})."
            )

        if cov_score < self.min_coverage_threshold:
            is_supported = False
            reasons.append(
                f"Coverage ratio ({cov_score:.3f}) is below minimum threshold ({self.min_coverage_threshold:.3f})."
            )

        if groundedness_score < self.min_groundedness_threshold:
            is_supported = False
            reasons.append(
                f"Groundedness score ({groundedness_score:.3f}) is below minimum threshold ({self.min_groundedness_threshold:.3f})."
            )

        if conf_score < self.min_confidence_threshold:
            is_supported = False
            reasons.append(
                f"Confidence score ({conf_score:.3f}) is below minimum threshold ({self.min_confidence_threshold:.3f})."
            )

        if is_supported:
            decision = TrustDecision.SUPPORTED
            reasons.append("Evidence meets all grounding, relevance, coverage, and confidence thresholds.")
        else:
            decision = TrustDecision.INSUFFICIENT_EVIDENCE

        return TrustAssessment(
            query=query,
            decision=decision,
            relevance_score=rel_score,
            coverage_score=cov_score,
            groundedness_score=groundedness_score,
            confidence_score=conf_score,
            provenance_valid=prov_valid,
            evidence_count=n_chunks,
            decision_reasons=reasons,
            relevance_details=g_assess.relevance_details,
            coverage_details=g_assess.coverage_details,
            groundedness_details=g_assess,
            confidence_details=c_assess,
            provenance_details=g_assess.provenance_details,
        )
