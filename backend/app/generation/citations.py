"""Provenance-derived citation reference system for evidence-grounded answers."""

from dataclasses import asdict, dataclass
import re
from typing import Any, Dict, List, Optional, Sequence, Set, Union

from backend.app.reranking.schema import RerankedChunk
from backend.app.trust.evidence import TrustEvidence


class CitationError(ValueError):
    """Raised when citation generation fails due to missing or invalid provenance."""
    pass


@dataclass(frozen=True)
class Citation:
    """Immutable, verifiable citation reference derived directly from chunk provenance.

    Attributes:
        index: 1-based sequential citation marker (e.g. 1 corresponding to [1]).
        chunk_id: Identifier of the supporting evidence chunk.
        document_id: Unique source document identifier.
        document_name: Human-readable file name of the document.
        page_start: Starting 1-based page number.
        page_end: Ending 1-based page number.
        text_snippet: Brief representative snippet of the passage content.
        formatted_reference: Standardized citation string (e.g. '[1] doc.pdf, page 5').
    """

    index: int
    chunk_id: str
    document_id: str
    document_name: str
    page_start: int
    page_end: int
    text_snippet: str
    formatted_reference: str

    def __post_init__(self) -> None:
        """Validate citation fields."""
        if not isinstance(self.index, int) or isinstance(self.index, bool) or self.index < 1:
            raise CitationError(f"Citation index must be a positive integer >= 1, got {self.index}.")

        if not isinstance(self.chunk_id, str) or not self.chunk_id.strip():
            raise CitationError("Citation chunk_id must be a non-empty string.")

        if not isinstance(self.document_id, str) or not self.document_id.strip():
            raise CitationError("Citation document_id must be a non-empty string.")

        if not isinstance(self.document_name, str) or not self.document_name.strip():
            raise CitationError("Citation document_name must be a non-empty string.")

        if not isinstance(self.page_start, int) or isinstance(self.page_start, bool) or self.page_start < 1:
            raise CitationError(f"Citation page_start must be integer >= 1, got {self.page_start}.")

        if not isinstance(self.page_end, int) or isinstance(self.page_end, bool) or self.page_end < self.page_start:
            raise CitationError(
                f"Citation page_end ({self.page_end}) cannot be less than page_start ({self.page_start})."
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert citation to dictionary representation."""
        return {
            "index": self.index,
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "document_name": self.document_name,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "text_snippet": self.text_snippet,
            "formatted_reference": self.formatted_reference,
        }


def format_citation_reference_string(
    index: int,
    document_name: str,
    page_start: int,
    page_end: int,
) -> str:
    """Construct a clean, standardized citation string.

    Args:
        index: 1-based citation index.
        document_name: Name of the source file.
        page_start: Starting page number.
        page_end: Ending page number.

    Returns:
        Formatted citation string like '[1] doc.pdf, page 5' or '[2] doc.pdf, pages 5-6'.
    """
    if page_start == page_end:
        pages_label = f"page {page_start}"
    else:
        pages_label = f"pages {page_start}-{page_end}"

    return f"[{index}] {document_name}, {pages_label}"


def build_citation(
    index: int,
    item: Union[TrustEvidence, RerankedChunk, Dict[str, Any]],
    snippet_max_chars: int = 150,
) -> Citation:
    """Build a strongly-typed Citation instance from a single evidence item.

    Args:
        index: 1-based sequential index.
        item: Evidence item (TrustEvidence, RerankedChunk, or dict).
        snippet_max_chars: Maximum character length for preview snippet.

    Returns:
        Validated Citation instance.

    Raises:
        CitationError: If required metadata is missing or invalid.
        TypeError: If item has an unsupported type.
    """
    if isinstance(item, (TrustEvidence, RerankedChunk)):
        chunk_id = item.chunk_id
        doc_id = item.document_id
        doc_name = item.document_name
        p_start = item.page_start
        p_end = item.page_end
        raw_text = item.text
    elif isinstance(item, dict):
        chunk_id = item.get("chunk_id")
        doc_id = item.get("document_id")
        doc_name = item.get("document_name")
        p_start = item.get("page_start")
        p_end = item.get("page_end", p_start)
        raw_text = item.get("text", "")
    else:
        raise TypeError(f"Unsupported evidence type: {type(item).__name__}")

    if not chunk_id or not isinstance(chunk_id, str):
        raise CitationError("Missing or invalid chunk_id in evidence item.")
    if not doc_id or not isinstance(doc_id, str):
        raise CitationError("Missing or invalid document_id in evidence item.")
    if not doc_name or not isinstance(doc_name, str):
        raise CitationError("Missing or invalid document_name in evidence item.")
    if not isinstance(p_start, int) or p_start < 1:
        raise CitationError(f"Invalid page_start ({p_start}) in evidence item.")
    if not isinstance(p_end, int) or p_end < p_start:
        raise CitationError(f"Invalid page_end ({p_end}) < page_start ({p_start}).")

    clean_text = raw_text.strip() if isinstance(raw_text, str) else ""
    snippet = (clean_text[:snippet_max_chars] + "...") if len(clean_text) > snippet_max_chars else clean_text

    formatted_ref = format_citation_reference_string(
        index=index,
        document_name=doc_name,
        page_start=p_start,
        page_end=p_end,
    )

    return Citation(
        index=index,
        chunk_id=chunk_id,
        document_id=doc_id,
        document_name=doc_name,
        page_start=p_start,
        page_end=p_end,
        text_snippet=snippet,
        formatted_reference=formatted_ref,
    )


def build_citation_references(
    evidence: Sequence[Union[TrustEvidence, RerankedChunk, Dict[str, Any]]],
    deduplicate_by_chunk_id: bool = True,
) -> List[Citation]:
    """Generate ordered, non-hallucinated citation objects from an evidence sequence.

    Args:
        evidence: Sequence of evidence chunks.
        deduplicate_by_chunk_id: If True, duplicate chunks will only receive one citation.

    Returns:
        List of Citation objects with 1-based sequential indices.
    """
    if not isinstance(evidence, (list, tuple)):
        raise TypeError(f"evidence must be a sequence, got {type(evidence).__name__}.")

    citations: List[Citation] = []
    seen_chunk_ids: Set[str] = set()

    for item in evidence:
        if deduplicate_by_chunk_id:
            c_id = item.chunk_id if hasattr(item, "chunk_id") else item.get("chunk_id") if isinstance(item, dict) else None
            if c_id and c_id in seen_chunk_ids:
                continue
            if c_id:
                seen_chunk_ids.add(c_id)

        next_idx = len(citations) + 1
        citation = build_citation(index=next_idx, item=item)
        citations.append(citation)

    return citations


def extract_cited_indices(text: str) -> List[int]:
    """Extract distinct bracketed citation indices (e.g. [1], [2]) from text.

    Args:
        text: Response text string.

    Returns:
        Sorted list of integer indices cited in the text.
    """
    if not isinstance(text, str):
        return []

    matches = re.findall(r"\[(\d+)\]", text)
    indices = sorted(list({int(m) for m in matches}))
    return indices


def format_citations_markdown(citations: Sequence[Citation]) -> str:
    """Format a list of Citation objects into a clean Markdown references block.

    Args:
        citations: Sequence of Citation objects.

    Returns:
        Markdown string of formatted references.
    """
    if not citations:
        return ""

    lines = ["### References", ""]
    for c in citations:
        lines.append(f"- **{c.formatted_reference}**")
        if c.text_snippet:
            lines.append(f"  > *\"{c.text_snippet}\"*")

    return "\n".join(lines)
