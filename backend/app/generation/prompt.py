"""Evidence-grounded prompt builder for TrustRAG local LLM generation."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Union

from backend.app.reranking.schema import RerankedChunk
from backend.app.trust.evidence import TrustEvidence

DEFAULT_SYSTEM_INSTRUCTION: str = (
    "You are an evidence-grounded AI assistant for TrustRAG.\n"
    "Answer the user query using ONLY the supplied evidence passages below.\n"
    "Do not invent facts or extrapolate beyond what is explicitly stated in the evidence.\n"
    "If the evidence does not contain sufficient information, do not make unsupported claims.\n"
    "Reference evidence passages using citation indices (e.g. [1], [2]) when forming your answer."
)


class PromptBuildingError(ValueError):
    """Raised when prompt construction encounters invalid inputs or empty queries."""
    pass


@dataclass(frozen=True)
class GroundedPrompt:
    """Immutable structured container holding all prompt components.

    Attributes:
        query: Evaluated user query.
        system_instruction: System directive governing strictly grounded behavior.
        evidence_context: Formatted block of all trusted evidence passages.
        user_prompt: Final user prompt combining evidence context and query.
        full_prompt: Complete monolithic prompt representation.
        evidence_count: Number of evidence chunks included in the context.
        evidence_ids: List of chunk IDs in order of prompt appearance.
    """

    query: str
    system_instruction: str
    evidence_context: str
    user_prompt: str
    full_prompt: str
    evidence_count: int
    evidence_ids: List[str]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize GroundedPrompt to dictionary format."""
        return {
            "query": self.query,
            "system_instruction": self.system_instruction,
            "evidence_context": self.evidence_context,
            "user_prompt": self.user_prompt,
            "full_prompt": self.full_prompt,
            "evidence_count": self.evidence_count,
            "evidence_ids": list(self.evidence_ids),
        }


def format_evidence_block(
    evidence: Sequence[Union[TrustEvidence, RerankedChunk, Dict[str, Any]]],
) -> tuple[str, List[str]]:
    """Format a sequence of evidence items into a structured text block.

    Args:
        evidence: Sequence of evidence items.

    Returns:
        Tuple containing (formatted_evidence_text, list_of_chunk_ids).
    """
    if not isinstance(evidence, (list, tuple)):
        raise TypeError(f"evidence must be a sequence (list or tuple), got {type(evidence).__name__}.")

    if len(evidence) == 0:
        return "No supporting evidence provided.", []

    blocks: List[str] = []
    chunk_ids: List[str] = []

    for idx, item in enumerate(evidence, start=1):
        if isinstance(item, (TrustEvidence, RerankedChunk)):
            chunk_id = item.chunk_id
            doc_name = item.document_name
            p_start = item.page_start
            p_end = item.page_end
            text = item.text.strip()
        elif isinstance(item, dict):
            chunk_id = str(item.get("chunk_id", f"chunk_{idx}"))
            doc_name = str(item.get("document_name", "Unknown Document"))
            p_start = int(item.get("page_start", 1))
            p_end = int(item.get("page_end", p_start))
            text = str(item.get("text", "")).strip()
        else:
            raise TypeError(
                f"Evidence item at index {idx - 1} has invalid type {type(item).__name__}."
            )

        pages_str = f"Page {p_start}" if p_start == p_end else f"Pages {p_start}-{p_end}"

        block = (
            f"[Evidence {idx}]\n"
            f"Document: {doc_name}\n"
            f"{pages_str}\n"
            f"Content: {text}"
        )
        blocks.append(block)
        chunk_ids.append(chunk_id)

    formatted_text = "\n\n".join(blocks)
    return formatted_text, chunk_ids


def build_grounded_prompt(
    query: str,
    evidence: Sequence[Union[TrustEvidence, RerankedChunk, Dict[str, Any]]],
    system_instruction: Optional[str] = None,
) -> GroundedPrompt:
    """Build a deterministic evidence-grounded prompt for local LLM generation.

    Args:
        query: User search query string.
        evidence: Sequence of verified trusted evidence items.
        system_instruction: Optional override for system directive.

    Returns:
        GroundedPrompt instance containing structured prompt components.

    Raises:
        TypeError: If query or evidence types are invalid.
        PromptBuildingError: If query is empty or whitespace.
    """
    if not isinstance(query, str):
        raise TypeError(f"query must be a string, got {type(query).__name__}.")

    stripped_query = query.strip()
    if not stripped_query:
        raise PromptBuildingError("query cannot be empty or whitespace only.")

    sys_instruction = system_instruction or DEFAULT_SYSTEM_INSTRUCTION
    evidence_text, chunk_ids = format_evidence_block(evidence)

    user_prompt = (
        f"SUPPLIED EVIDENCE:\n"
        f"-----------------\n"
        f"{evidence_text}\n"
        f"-----------------\n\n"
        f"USER QUERY: {stripped_query}\n\n"
        f"GROUNDED ANSWER (cite sources with [1], [2], etc.):"
    )

    full_prompt = (
        f"SYSTEM INSTRUCTION:\n{sys_instruction}\n\n"
        f"{user_prompt}"
    )

    return GroundedPrompt(
        query=stripped_query,
        system_instruction=sys_instruction,
        evidence_context=evidence_text,
        user_prompt=user_prompt,
        full_prompt=full_prompt,
        evidence_count=len(chunk_ids),
        evidence_ids=chunk_ids,
    )
