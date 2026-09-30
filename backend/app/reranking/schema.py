"""Data structures and schemas for neural reranking results."""

from dataclasses import asdict, dataclass
import math
from typing import Any, Dict


class SchemaValidationError(Exception):
    """Raised when data fails schema validation or invariant checks."""
    pass


@dataclass(frozen=True)
class RerankedChunk:
    """Represents a document chunk after neural cross-encoder reranking.

    Preserves full retrieval provenance alongside reranked relevance signals.
    """

    chunk_id: str
    document_id: str
    document_name: str
    page_start: int
    page_end: int
    text: str
    original_rank: int
    original_score: float
    rerank_score: float
    final_rank: int

    def __post_init__(self) -> None:
        """Validate fields upon initialization."""
        if not isinstance(self.chunk_id, str) or not self.chunk_id.strip():
            raise SchemaValidationError("chunk_id must be a non-empty string.")

        if not isinstance(self.document_id, str) or not self.document_id.strip():
            raise SchemaValidationError("document_id must be a non-empty string.")

        if not isinstance(self.document_name, str) or not self.document_name.strip():
            raise SchemaValidationError("document_name must be a non-empty string.")

        if not isinstance(self.page_start, int) or self.page_start < 0:
            raise SchemaValidationError(f"page_start must be a non-negative integer, got {self.page_start}.")

        if not isinstance(self.page_end, int) or self.page_end < self.page_start:
            raise SchemaValidationError(
                f"page_end ({self.page_end}) must be an integer >= page_start ({self.page_start})."
            )

        if not isinstance(self.text, str) or not self.text.strip():
            raise SchemaValidationError("text must be a non-empty string.")

        if not isinstance(self.original_rank, int) or self.original_rank < 1:
            raise SchemaValidationError(
                f"original_rank must be a positive integer >= 1, got {self.original_rank}."
            )

        if not isinstance(self.final_rank, int) or self.final_rank < 1:
            raise SchemaValidationError(
                f"final_rank must be a positive integer >= 1, got {self.final_rank}."
            )

        if not isinstance(self.original_score, (int, float)) or math.isnan(self.original_score):
            raise SchemaValidationError(f"original_score must be a finite float, got {self.original_score}.")

        if not isinstance(self.rerank_score, (int, float)) or math.isnan(self.rerank_score):
            raise SchemaValidationError(f"rerank_score must be a finite float, got {self.rerank_score}.")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a standardized dictionary representation."""
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "document_name": self.document_name,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "text": self.text,
            "original_rank": self.original_rank,
            "original_score": float(self.original_score),
            "rerank_score": float(self.rerank_score),
            "final_rank": self.final_rank,
        }

    @classmethod
    def from_retrieval_result(
        cls,
        result: Dict[str, Any],
        rerank_score: float,
        final_rank: int,
    ) -> "RerankedChunk":
        """Construct RerankedChunk by enriching a hybrid or vector retrieval result dictionary.

        Args:
            result: Dictionary with chunk_id, document_id, document_name, page_start, page_end, text, rank, score.
            rerank_score: New cross-encoder relevance score (float).
            final_rank: New 1-based reranked integer position.

        Returns:
            Instantiated and validated RerankedChunk.

        Raises:
            SchemaValidationError: If required keys are missing or invalid.
        """
        if not isinstance(result, dict):
            raise SchemaValidationError(f"Retrieval result must be a dictionary, got {type(result).__name__}.")

        required_keys = ["chunk_id", "document_id", "document_name", "page_start", "page_end", "text"]
        for key in required_keys:
            if key not in result:
                raise SchemaValidationError(f"Missing required key '{key}' in retrieval result.")

        orig_rank = result.get("rank", 1)
        orig_score = result.get("score", 0.0)

        return cls(
            chunk_id=str(result["chunk_id"]),
            document_id=str(result["document_id"]),
            document_name=str(result["document_name"]),
            page_start=int(result["page_start"]),
            page_end=int(result["page_end"]),
            text=str(result["text"]),
            original_rank=int(orig_rank),
            original_score=float(orig_score),
            rerank_score=float(rerank_score),
            final_rank=int(final_rank),
        )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RerankedChunk":
        """Construct RerankedChunk directly from a serialized dictionary."""
        if not isinstance(data, dict):
            raise SchemaValidationError(f"Data must be a dictionary, got {type(data).__name__}.")

        required_keys = [
            "chunk_id", "document_id", "document_name", "page_start", "page_end",
            "text", "original_rank", "original_score", "rerank_score", "final_rank",
        ]
        for key in required_keys:
            if key not in data:
                raise SchemaValidationError(f"Missing required key '{key}' in dictionary.")

        return cls(
            chunk_id=str(data["chunk_id"]),
            document_id=str(data["document_id"]),
            document_name=str(data["document_name"]),
            page_start=int(data["page_start"]),
            page_end=int(data["page_end"]),
            text=str(data["text"]),
            original_rank=int(data["original_rank"]),
            original_score=float(data["original_score"]),
            rerank_score=float(data["rerank_score"]),
            final_rank=int(data["final_rank"]),
        )
