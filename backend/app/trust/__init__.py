"""Trust Engine subsystem for TrustRAG.

Provides deterministic multi-signal evidence grounding evaluation, relevance
aggregation, lexical term coverage, provenance validation, groundedness scoring,
system confidence estimation, and discrete trust decisions (SUPPORTED vs INSUFFICIENT_EVIDENCE).
"""

from backend.app.trust.confidence import (
    ConfidenceAssessment,
    ConfidenceScoringError,
    calculate_confidence,
)
from backend.app.trust.config import (
    TrustConfig,
    TrustConfigError,
)
from backend.app.trust.coverage import (
    CoverageAssessment,
    CoverageScoringError,
    calculate_evidence_coverage,
    extract_query_terms,
)
from backend.app.trust.engine import (
    TrustAssessment,
    TrustDecision,
    TrustEngine,
    TrustEngineError,
)
from backend.app.trust.evidence import (
    EvidenceValidationError,
    TrustEvidence,
)
from backend.app.trust.groundedness import (
    GroundednessAssessment,
    GroundednessScoringError,
    calculate_groundedness,
)
from backend.app.trust.provenance import (
    ProvenanceIssue,
    ProvenanceReport,
    validate_evidence_provenance,
    validate_single_evidence_provenance,
)
from backend.app.trust.relevance import (
    RelevanceAssessment,
    RelevanceScoringError,
    aggregate_evidence_relevance,
    sigmoid,
)

__all__ = [
    "ConfidenceAssessment",
    "ConfidenceScoringError",
    "CoverageAssessment",
    "CoverageScoringError",
    "EvidenceValidationError",
    "GroundednessAssessment",
    "GroundednessScoringError",
    "ProvenanceIssue",
    "ProvenanceReport",
    "RelevanceAssessment",
    "RelevanceScoringError",
    "TrustAssessment",
    "TrustConfig",
    "TrustConfigError",
    "TrustDecision",
    "TrustEngine",
    "TrustEngineError",
    "TrustEvidence",
    "aggregate_evidence_relevance",
    "calculate_confidence",
    "calculate_evidence_coverage",
    "calculate_groundedness",
    "extract_query_terms",
    "sigmoid",
    "validate_evidence_provenance",
    "validate_single_evidence_provenance",
]
