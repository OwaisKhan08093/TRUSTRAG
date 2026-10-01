"""Structured evidence representation and validation for the Trust Engine."""

from dataclasses import asdict, dataclass
import math
from typing import Any, Dict

from backend.app.reranking.schema import RerankedChunk


class EvidenceValidationError(ValueError):
    """Raised when evidence fields fail schema or type validation."""
    pass


@dataclass(frozen=True)
class TrustEvidence:
    """Immutable, strongly-typed representation of an evidence chunk evaluated by the Trust Engine.

    Attributes:
        chunk_id: Unique identifier for the chunk.
        document_id: Identifier of the source document.
        document_name: Human-readable file name of the document.
        page_start: Starting 1-based page number.
        page_end: Ending 1-based page number.
        text: Extracted plain text content of the chunk.
        retrieval_rank: 1-based ranking from the initial retrieval / hybrid stage.
        retrieval_score: Score assigned by the initial retriever (e.g. RRF score).
        rerank_score: Neural relevance score from the cross-encoder reranker.
    """

    chunk_id: str
    document_id: str
    document_name: str
    page_start: int
    page_end: int
    text: str
    retrieval_rank: int
    retrieval_score: float
    rerank_score: float

    def __post_init__(self) -> None:
        """Validate all fields upon instantiation."""
        if not isinstance(self.chunk_id, str) or not self.chunk_id.strip():
            raise EvidenceValidationError("chunk_id must be a non-empty string.")

        if not isinstance(self.document_id, str) or not self.document_id.strip():
            raise EvidenceValidationError("document_id must be a non-empty string.")

        if not isinstance(self.document_name, str) or not self.document_name.strip():
            raise EvidenceValidationError("document_name must be a non-empty string.")

        if not isinstance(self.page_start, int) or self.page_start < 1:
            raise EvidenceValidationError(
                f"page_start must be a positive integer >= 1, got {self.page_start}."
            )

        if not isinstance(self.page_end, int) or self.page_end < self.page_start:
            raise EvidenceValidationError(
                f"page_end ({self.page_end}) must be an integer >= page_start ({self.page_start})."
            )

        if not isinstance(self.text, str) or not self.text.strip():
            raise EvidenceValidationError("text must be a non-empty string.")

        if not isinstance(self.retrieval_rank, int) or self.retrieval_rank < 1:
            raise EvidenceValidationError(
                f"retrieval_rank must be a positive integer >= 1, got {self.retrieval_rank}."
            )

        if not isinstance(self.retrieval_score, (int, float)) or not math.isfinite(self.retrieval_score):
            raise EvidenceValidationError(
                f"retrieval_score must be a finite float, got {self.retrieval_score}."
            )

        if not isinstance(self.rerank_score, (int, float)) or not math.isfinite(self.rerank_score):
            raise EvidenceValidationError(
                f"rerank_score must be a finite float, got {self.rerank_score}."
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert evidence item to dictionary."""
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "document_name": self.document_name,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "text": self.text,
            "retrieval_rank": self.retrieval_rank,
            "retrieval_score": float(self.retrieval_score),
            "rerank_score": float(self.rerank_score),
        }

    @classmethod
    def from_reranked_chunk(cls, chunk: RerankedChunk) -> "TrustEvidence":
        """Instantiate TrustEvidence from a RerankedChunk object.

        Args:
            chunk: RerankedChunk object from the reranking stage.

        Returns:
            Validated TrustEvidence instance.
        """
        if not isinstance(chunk, RerankedChunk):
            raise EvidenceValidationError(
                f"Expected RerankedChunk instance, got {type(chunk).__name__}."
            )

        return cls(
            chunk_id=chunk.chunk_id,
            document_id=chunk.document_id,
            document_name=chunk.document_name,
            page_start=chunk.page_start,
            page_end=chunk.page_end,
            text=chunk.text,
            retrieval_rank=chunk.original_rank,
            retrieval_score=chunk.original_score,
            rerank_score=chunk.rerank_score,
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrustEvidence":
        """Instantiate TrustEvidence from a dictionary.

        Args:
            data: Dictionary containing all required evidence fields.

        Returns:
            Validated TrustEvidence instance.
        """
        if not isinstance(data, dict):
            raise EvidenceValidationError(f"Expected dictionary, got {type(data).__name__}.")

        required_keys = [
            "chunk_id",
            "document_id",
            "document_name",
            "page_start",
            "page_end",
            "text",
            "retrieval_rank",
            "retrieval_score",
            "rerank_score",
        ]
        for key in required_keys:
            if key not in data:
                raise EvidenceValidationError(f"Missing required key '{key}' in evidence data.")

        return cls(
            chunk_id=str(data["chunk_id"]),
            document_id=str(data["document_id"]),
            document_name=str(data["document_name"]),
            page_start=int(data["page_start"]),
            page_end=int(data["page_end"]),
            text=str(data["text"]),
            retrieval_rank=int(data["retrieval_rank"]),
            retrieval_score=float(data["retrieval_score"]),
            rerank_score=float(data["rerank_score"]),
        )
