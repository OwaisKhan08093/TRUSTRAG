"""Evidence coverage measurement and query term support for the Trust Engine."""

from dataclasses import asdict, dataclass
import re
from typing import Any, Dict, List, Optional, Sequence, Set, Union

from backend.app.reranking.schema import RerankedChunk
from backend.app.trust.evidence import TrustEvidence

# Standard English stop words for transparent query term extraction
STOP_WORDS: Set[str] = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
    "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both",
    "but", "by", "can", "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does",
    "doesn't", "doing", "don't", "down", "during", "each", "few", "for", "from", "further",
    "had", "hadn't", "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's",
    "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i",
    "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its",
    "itself", "let's", "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of",
    "off", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out",
    "over", "own", "same", "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't",
    "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them",
    "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're",
    "they've", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was",
    "wasn't", "we", "we'd", "we'll", "we're", "we've", "were", "weren't", "what", "what's",
    "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why",
    "why's", "with", "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've",
    "your", "yours", "yourself", "yourselves",
}


class CoverageScoringError(ValueError):
    """Raised when coverage evaluation encounters invalid inputs."""
    pass


def extract_query_terms(query: str, remove_stopwords: bool = True) -> List[str]:
    """Extract normalized alphanumeric terms from a query string.

    Args:
        query: Raw query text.
        remove_stopwords: Whether to filter out standard stop words.

    Returns:
        List of distinct lowercase alphanumeric terms preserving order of first appearance.
    """
    if not isinstance(query, str):
        raise TypeError(f"query must be a string, got {type(query).__name__}.")

    stripped = query.strip()
    if not stripped:
        return []

    tokens = re.findall(r"\b[a-zA-Z0-9_-]+\b", stripped.lower())

    seen: Set[str] = set()
    terms: List[str] = []
    for token in tokens:
        if remove_stopwords and token in STOP_WORDS:
            continue
        if len(token) < 2:  # ignore single characters
            continue
        if token not in seen:
            seen.add(token)
            terms.append(token)

    # Fallback: if all tokens were stopwords (e.g. "who what where"), keep all tokens >= 2 chars
    if not terms and tokens:
        for token in tokens:
            if len(token) >= 2 and token not in seen:
                seen.add(token)
                terms.append(token)

    return terms


@dataclass(frozen=True)
class CoverageAssessment:
    """Immutable assessment of lexical evidence coverage for a query.

    NOTE ON LIMITATIONS:
    Lexical term coverage measures the proportion of query terms present across retrieved evidence.
    It is a transparent, deterministic heuristic and does NOT constitute deep semantic entailment
    or logical reasoning.

    Attributes:
        query_terms: Distinct informative terms extracted from the query.
        covered_terms: Terms found in at least one retrieved evidence passage.
        uncovered_terms: Query terms absent from all retrieved evidence.
        coverage_ratio: Proportion of query terms covered (bounded in [0.0, 1.0]).
        chunk_coverage_ratios: Per-chunk coverage ratio for each evidence item.
        evidence_count: Total number of evidence chunks analyzed.
    """

    query_terms: List[str]
    covered_terms: List[str]
    uncovered_terms: List[str]
    coverage_ratio: float
    chunk_coverage_ratios: List[float]
    evidence_count: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert assessment to dictionary."""
        return {
            "query_terms": list(self.query_terms),
            "covered_terms": list(self.covered_terms),
            "uncovered_terms": list(self.uncovered_terms),
            "coverage_ratio": self.coverage_ratio,
            "chunk_coverage_ratios": list(self.chunk_coverage_ratios),
            "evidence_count": self.evidence_count,
        }


def calculate_evidence_coverage(
    query: str,
    evidence: Sequence[Union[TrustEvidence, RerankedChunk, Dict[str, Any]]],
    remove_stopwords: bool = True,
) -> CoverageAssessment:
    """Calculate deterministic lexical query term coverage across retrieved evidence chunks.

    Args:
        query: Query string to evaluate against evidence.
        evidence: Sequence of evidence items (TrustEvidence, RerankedChunk, or dict with 'text').
        remove_stopwords: Whether to remove common English stopwords from query terms.

    Returns:
        CoverageAssessment instance with coverage ratio and detailed term breakdowns.

    Raises:
        TypeError: If query or evidence has invalid type.
        CoverageScoringError: If query is empty.
    """
    if not isinstance(query, str):
        raise TypeError(f"query must be a string, got {type(query).__name__}.")

    stripped_query = query.strip()
    if not stripped_query:
        raise CoverageScoringError("query cannot be empty or contain only whitespace.")

    if not isinstance(evidence, (list, tuple)):
        raise TypeError(f"evidence must be a sequence, got {type(evidence).__name__}.")

    query_terms = extract_query_terms(stripped_query, remove_stopwords=remove_stopwords)

    if not query_terms:
        # If query has no valid terms (e.g. symbols only)
        return CoverageAssessment(
            query_terms=[],
            covered_terms=[],
            uncovered_terms=[],
            coverage_ratio=0.0,
            chunk_coverage_ratios=[0.0] * len(evidence),
            evidence_count=len(evidence),
        )

    if len(evidence) == 0:
        return CoverageAssessment(
            query_terms=query_terms,
            covered_terms=[],
            uncovered_terms=query_terms,
            coverage_ratio=0.0,
            chunk_coverage_ratios=[],
            evidence_count=0,
        )

    # Extract text and token sets from each evidence item
    chunk_token_sets: List[Set[str]] = []
    for idx, item in enumerate(evidence):
        if isinstance(item, (TrustEvidence, RerankedChunk)):
            text = item.text
        elif isinstance(item, dict):
            if "text" not in item or not isinstance(item["text"], str):
                raise CoverageScoringError(f"Evidence dict at index {idx} missing valid 'text' key.")
            text = item["text"]
        else:
            raise TypeError(
                f"Evidence item at index {idx} must be TrustEvidence, RerankedChunk, or dict, got {type(item).__name__}."
            )

        tokens = set(re.findall(r"\b[a-zA-Z0-9_-]+\b", text.lower()))
        chunk_token_sets.append(tokens)

    # Union of all tokens in evidence corpus
    all_evidence_tokens: Set[str] = set().union(*chunk_token_sets) if chunk_token_sets else set()

    covered_terms: List[str] = [t for t in query_terms if t in all_evidence_tokens]
    uncovered_terms: List[str] = [t for t in query_terms if t not in all_evidence_tokens]

    coverage_ratio = len(covered_terms) / len(query_terms) if query_terms else 0.0
    coverage_ratio = max(0.0, min(1.0, coverage_ratio))

    chunk_ratios: List[float] = []
    for chunk_tokens in chunk_token_sets:
        c_covered = sum(1 for t in query_terms if t in chunk_tokens)
        c_ratio = c_covered / len(query_terms) if query_terms else 0.0
        chunk_ratios.append(max(0.0, min(1.0, c_ratio)))

    return CoverageAssessment(
        query_terms=query_terms,
        covered_terms=covered_terms,
        uncovered_terms=uncovered_terms,
        coverage_ratio=coverage_ratio,
        chunk_coverage_ratios=chunk_ratios,
        evidence_count=len(evidence),
    )
