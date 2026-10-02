"""Provenance integrity verification for retrieved evidence."""

from dataclasses import dataclass, field
import math
from typing import Any, Dict, List, Optional, Sequence, Set, Union

from backend.app.reranking.schema import RerankedChunk
from backend.app.trust.evidence import EvidenceValidationError, TrustEvidence


@dataclass(frozen=True)
class ProvenanceIssue:
    """Represents a specific provenance integrity defect in an evidence chunk."""

    chunk_index: int
    chunk_id: Optional[str]
    issue_type: str
    message: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert issue to dictionary."""
        return {
            "chunk_index": self.chunk_index,
            "chunk_id": self.chunk_id,
            "issue_type": self.issue_type,
            "message": self.message,
        }


@dataclass(frozen=True)
class ProvenanceReport:
    """Summary of provenance validation across a sequence of evidence items."""

    is_valid: bool
    total_chunks: int
    valid_chunks_count: int
    invalid_chunks_count: int
    issues: List[ProvenanceIssue] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary."""
        return {
            "is_valid": self.is_valid,
            "total_chunks": self.total_chunks,
            "valid_chunks_count": self.valid_chunks_count,
            "invalid_chunks_count": self.invalid_chunks_count,
            "issues": [issue.to_dict() for issue in self.issues],
        }


def validate_single_evidence_provenance(
    item: Union[TrustEvidence, RerankedChunk, Dict[str, Any]],
    index: int = 0,
    seen_chunk_ids: Optional[Set[str]] = None,
) -> List[ProvenanceIssue]:
    """Validate provenance metadata of a single evidence chunk.

    Args:
        item: Evidence chunk (TrustEvidence, RerankedChunk, or dict).
        index: Position of chunk in the evidence list.
        seen_chunk_ids: Set of previously observed chunk IDs for uniqueness check.

    Returns:
        List of ProvenanceIssue items found (empty if completely valid).
    """
    issues: List[ProvenanceIssue] = []

    if isinstance(item, TrustEvidence):
        chunk_id = item.chunk_id
        doc_id = item.document_id
        doc_name = item.document_name
        page_start = item.page_start
        page_end = item.page_end
        text = item.text
    elif isinstance(item, RerankedChunk):
        chunk_id = item.chunk_id
        doc_id = item.document_id
        doc_name = item.document_name
        page_start = item.page_start
        page_end = item.page_end
        text = item.text
    elif isinstance(item, dict):
        chunk_id = item.get("chunk_id")
        doc_id = item.get("document_id")
        doc_name = item.get("document_name")
        page_start = item.get("page_start")
        page_end = item.get("page_end")
        text = item.get("text")
    else:
        issues.append(
            ProvenanceIssue(
                chunk_index=index,
                chunk_id=None,
                issue_type="invalid_type",
                message=f"Evidence item at index {index} has unsupported type {type(item).__name__}.",
            )
        )
        return issues

    # Validate chunk_id
    if not isinstance(chunk_id, str) or not chunk_id.strip():
        issues.append(
            ProvenanceIssue(
                chunk_index=index,
                chunk_id=str(chunk_id) if chunk_id is not None else None,
                issue_type="invalid_chunk_id",
                message=f"Chunk at index {index} has missing or empty chunk_id.",
            )
        )
    elif seen_chunk_ids is not None:
        if chunk_id in seen_chunk_ids:
            issues.append(
                ProvenanceIssue(
                    chunk_index=index,
                    chunk_id=chunk_id,
                    issue_type="duplicate_chunk_id",
                    message=f"Duplicate chunk_id '{chunk_id}' detected at index {index}.",
                )
            )
        else:
            seen_chunk_ids.add(chunk_id)

    # Validate document_id
    if not isinstance(doc_id, str) or not doc_id.strip():
        issues.append(
            ProvenanceIssue(
                chunk_index=index,
                chunk_id=chunk_id if isinstance(chunk_id, str) else None,
                issue_type="invalid_document_id",
                message=f"Chunk at index {index} has missing or empty document_id.",
            )
        )

    # Validate document_name
    if not isinstance(doc_name, str) or not doc_name.strip():
        issues.append(
            ProvenanceIssue(
                chunk_index=index,
                chunk_id=chunk_id if isinstance(chunk_id, str) else None,
                issue_type="invalid_document_name",
                message=f"Chunk at index {index} has missing or empty document_name.",
            )
        )

    # Validate page numbers
    if not isinstance(page_start, int) or isinstance(page_start, bool) or page_start < 1:
        issues.append(
            ProvenanceIssue(
                chunk_index=index,
                chunk_id=chunk_id if isinstance(chunk_id, str) else None,
                issue_type="invalid_page_start",
                message=f"Chunk at index {index} has invalid page_start ({page_start}). Must be integer >= 1.",
            )
        )
    if not isinstance(page_end, int) or isinstance(page_end, bool):
        issues.append(
            ProvenanceIssue(
                chunk_index=index,
                chunk_id=chunk_id if isinstance(chunk_id, str) else None,
                issue_type="invalid_page_end",
                message=f"Chunk at index {index} has non-integer page_end ({page_end}).",
            )
        )
    elif isinstance(page_start, int) and not isinstance(page_start, bool) and page_end < page_start:
        issues.append(
            ProvenanceIssue(
                chunk_index=index,
                chunk_id=chunk_id if isinstance(chunk_id, str) else None,
                issue_type="invalid_page_range",
                message=f"Chunk at index {index} has page_end ({page_end}) < page_start ({page_start}).",
            )
        )

    # Validate text content
    if not isinstance(text, str) or not text.strip():
        issues.append(
            ProvenanceIssue(
                chunk_index=index,
                chunk_id=chunk_id if isinstance(chunk_id, str) else None,
                issue_type="empty_text",
                message=f"Chunk at index {index} has missing or empty text content.",
            )
        )

    return issues


def validate_evidence_provenance(
    evidence: Sequence[Union[TrustEvidence, RerankedChunk, Dict[str, Any]]],
) -> ProvenanceReport:
    """Validate provenance integrity across all evidence chunks.

    Args:
        evidence: Sequence of evidence items.

    Returns:
        ProvenanceReport containing validity summary and detailed issue list.
    """
    if not isinstance(evidence, (list, tuple)):
        raise TypeError(f"evidence must be a sequence (list or tuple), got {type(evidence).__name__}.")

    if len(evidence) == 0:
        return ProvenanceReport(
            is_valid=True,
            total_chunks=0,
            valid_chunks_count=0,
            invalid_chunks_count=0,
            issues=[],
        )

    all_issues: List[ProvenanceIssue] = []
    seen_ids: Set[str] = set()
    invalid_indices: Set[int] = set()

    for idx, item in enumerate(evidence):
        item_issues = validate_single_evidence_provenance(item, index=idx, seen_chunk_ids=seen_ids)
        if item_issues:
            invalid_indices.add(idx)
            all_issues.extend(item_issues)

    invalid_count = len(invalid_indices)
    valid_count = len(evidence) - invalid_count

    return ProvenanceReport(
        is_valid=(invalid_count == 0),
        total_chunks=len(evidence),
        valid_chunks_count=valid_count,
        invalid_chunks_count=invalid_count,
        issues=all_issues,
    )
