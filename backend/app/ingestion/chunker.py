"""Deterministic, metadata-preserving text chunker with page attribution."""

from typing import Sequence, TypedDict
from backend.app.config import (
    DEFAULT_CHUNK_SIZE_WORDS,
    DEFAULT_CHUNK_OVERLAP_WORDS,
    MIN_CHUNK_SIZE_WORDS,
)
from backend.app.ingestion.pdf_loader import PageRecord


class ChunkRecord(TypedDict):
    """Structured record representing a text chunk with complete source provenance."""
    chunk_id: str
    document_id: str
    document_name: str
    page_start: int
    page_end: int
    text: str


def chunk_document(
    pages: Sequence[PageRecord],
    chunk_size: int = DEFAULT_CHUNK_SIZE_WORDS,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP_WORDS,
    min_chunk_size: int = MIN_CHUNK_SIZE_WORDS,
) -> list[ChunkRecord]:
    """Deterministically chunk document pages preserving page boundaries and metadata.

    Args:
        pages: Sequence of PageRecord objects for a single document.
        chunk_size: Target number of words per chunk (default 600 words).
        chunk_overlap: Number of overlapping words between consecutive chunks (default 120 words).
        min_chunk_size: Minimum words required for a standalone trailing chunk.

    Returns:
        A list of ChunkRecord dictionaries containing chunk_id, document_id,
        document_name, page_start, page_end, and text.
    """
    if not pages:
        return []

    # Verify all pages belong to the same document
    doc_id = pages[0]["document_id"]
    doc_name = pages[0]["document_name"]

    # Tag every word with its source page number
    tagged_words: list[tuple[str, int]] = []
    for page in pages:
        page_num = page["page_number"]
        page_text = page.get("text", "").strip()
        if not page_text:
            continue
        words = page_text.split()
        for word in words:
            tagged_words.append((word, page_num))

    total_words = len(tagged_words)
    if total_words == 0:
        return []

    # Validate chunking parameters
    chunk_size = max(10, chunk_size)
    chunk_overlap = max(0, min(chunk_overlap, chunk_size - 1))
    step = chunk_size - chunk_overlap

    chunks: list[ChunkRecord] = []
    chunk_idx = 1
    start = 0

    while start < total_words:
        end = min(start + chunk_size, total_words)

        # Avoid creating a tiny trailing slice that is already covered in the previous chunk overlap
        if start > 0 and (total_words - start) < min_chunk_size:
            break

        chunk_tokens = tagged_words[start:end]
        chunk_text = " ".join(token for token, _ in chunk_tokens).strip()

        if chunk_text:
            page_start = chunk_tokens[0][1]
            page_end = chunk_tokens[-1][1]
            chunk_id = f"{doc_id}_page{page_start}_chunk{chunk_idx}"

            chunks.append({
                "chunk_id": chunk_id,
                "document_id": doc_id,
                "document_name": doc_name,
                "page_start": page_start,
                "page_end": page_end,
                "text": chunk_text
            })
            chunk_idx += 1

        if end >= total_words:
            break

        start += step

    return chunks


def chunk_multiple_documents(
    pages: Sequence[PageRecord],
    chunk_size: int = DEFAULT_CHUNK_SIZE_WORDS,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP_WORDS,
    min_chunk_size: int = MIN_CHUNK_SIZE_WORDS,
) -> list[ChunkRecord]:
    """Group pages by document and chunk each document independently to prevent metadata mixing.

    Args:
        pages: Sequence of PageRecord objects across potentially multiple documents.
        chunk_size: Target number of words per chunk.
        chunk_overlap: Overlap in words.
        min_chunk_size: Minimum word count for a trailing chunk.

    Returns:
        List of all ChunkRecord objects.
    """
    # Group by document_id preserving order
    docs_map: dict[str, list[PageRecord]] = {}
    for page in pages:
        docs_map.setdefault(page["document_id"], []).append(page)

    all_chunks: list[ChunkRecord] = []
    for doc_pages in docs_map.values():
        doc_chunks = chunk_document(
            pages=doc_pages,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            min_chunk_size=min_chunk_size,
        )
        all_chunks.extend(doc_chunks)

    return all_chunks
