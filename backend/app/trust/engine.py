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
        config: Optional["TrustConfig"] = None,
        min_confidence_threshold: Optional[float] = None,
        min_groundedness_threshold: Optional[float] = None,
        min_coverage_threshold: Optional[float] = None,
        min_evidence_count: Optional[int] = None,
        require_valid_provenance: Optional[bool] = None,
        weight_relevance: Optional[float] = None,
        weight_coverage: Optional[float] = None,
        weight_provenance: Optional[float] = None,
        weight_confidence_groundedness: Optional[float] = None,
        weight_confidence_top_relevance: Optional[float] = None,
        weight_confidence_coverage: Optional[float] = None,
        weight_confidence_volume: Optional[float] = None,
    ) -> None:
        """Initialize TrustEngine with configurable evaluation thresholds and weights.

        Args:
            config: Optional TrustConfig instance.
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
        from backend.app.trust.config import TrustConfig, TrustConfigError

        cfg = config if config is not None else TrustConfig()

        c_min_conf = min_confidence_threshold if min_confidence_threshold is not None else cfg.min_confidence_threshold
        c_min_ground = min_groundedness_threshold if min_groundedness_threshold is not None else cfg.min_groundedness_threshold
        c_min_cov = min_coverage_threshold if min_coverage_threshold is not None else cfg.min_coverage_threshold
        c_min_ev = min_evidence_count if min_evidence_count is not None else cfg.min_evidence_count
        c_req_prov = require_valid_provenance if require_valid_provenance is not None else cfg.require_valid_provenance

        c_w_rel = weight_relevance if weight_relevance is not None else cfg.weight_relevance
        c_w_cov = weight_coverage if weight_coverage is not None else cfg.weight_coverage
        c_w_prov = weight_provenance if weight_provenance is not None else cfg.weight_provenance

        c_w_c_g = weight_confidence_groundedness if weight_confidence_groundedness is not None else cfg.weight_confidence_groundedness
        c_w_c_t = weight_confidence_top_relevance if weight_confidence_top_relevance is not None else cfg.weight_confidence_top_relevance
        c_w_c_c = weight_confidence_coverage if weight_confidence_coverage is not None else cfg.weight_confidence_coverage
        c_w_c_v = weight_confidence_volume if weight_confidence_volume is not None else cfg.weight_confidence_volume

        try:
            # Instantiate validated TrustConfig to ensure constraints
            self.config = TrustConfig(
                min_confidence_threshold=c_min_conf,
                min_groundedness_threshold=c_min_ground,
                min_coverage_threshold=c_min_cov,
                min_evidence_count=c_min_ev,
                require_valid_provenance=c_req_prov,
                weight_relevance=c_w_rel,
                weight_coverage=c_w_cov,
                weight_provenance=c_w_prov,
                weight_confidence_groundedness=c_w_c_g,
                weight_confidence_top_relevance=c_w_c_t,
                weight_confidence_coverage=c_w_c_c,
                weight_confidence_volume=c_w_c_v,
            )
        except TrustConfigError as exc:
            raise TrustEngineError(str(exc)) from exc

        self.min_confidence_threshold = self.config.min_confidence_threshold
        self.min_groundedness_threshold = self.config.min_groundedness_threshold
        self.min_coverage_threshold = self.config.min_coverage_threshold
        self.min_evidence_count = self.config.min_evidence_count
        self.require_valid_provenance = self.config.require_valid_provenance

        self.weight_relevance = self.config.weight_relevance
        self.weight_coverage = self.config.weight_coverage
        self.weight_provenance = self.config.weight_provenance

        self.weight_confidence_groundedness = self.config.weight_confidence_groundedness
        self.weight_confidence_top_relevance = self.config.weight_confidence_top_relevance
        self.weight_confidence_coverage = self.config.weight_confidence_coverage
        self.weight_confidence_volume = self.config.weight_confidence_volume

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
