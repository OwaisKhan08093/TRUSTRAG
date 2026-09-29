"""Conservative text cleaner for preserving semantic integrity and legal numbering."""

import re
from typing import Sequence
from backend.app.ingestion.pdf_loader import PageRecord


def clean_text(text: str) -> str:
    """Conservatively clean extracted text while strictly preserving structure and legal numbering.

    Operations:
    1. Normalize line endings to standard Unix newline (\\n).
    2. Remove non-printable control characters.
    3. Collapse multiple consecutive horizontal spaces/tabs on each line to a single space.
    4. Strip trailing and leading whitespace per line.
    5. Reduce 3+ consecutive newlines down to 2 newlines (preserves paragraph breaks).
    6. Maintain all statutory/legal markers (e.g., 'Section 1', '(a)', '(b)', '1.', '2.').

    Args:
        text: Raw text string extracted from a document page.

    Returns:
        Cleaned text string.
    """
    if not text:
        return ""

    # Normalize line endings
    normalized = text.replace('\r\n', '\n').replace('\r', '\n')

    # Remove non-printable ASCII control characters (keep newline and tab)
    normalized = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', normalized)

    # Process line by line to preserve formatting and legal numbering
    cleaned_lines: list[str] = []
    for line in normalized.split('\n'):
        # Collapse multiple horizontal whitespaces into one and trim line ends
        cleaned_line = re.sub(r'[ \t]+', ' ', line).strip()
        cleaned_lines.append(cleaned_line)

    reconstructed = '\n'.join(cleaned_lines)

    # Collapse 3 or more consecutive newlines into 2 (one blank line between paragraphs)
    collapsed = re.sub(r'\n{3,}', '\n\n', reconstructed)

    return collapsed.strip()


def clean_page_records(pages: Sequence[PageRecord]) -> list[PageRecord]:
    """Apply conservative cleaning across a collection of PageRecord objects.

    Args:
        pages: Sequence of raw PageRecord dictionaries.

    Returns:
        New list of PageRecord dictionaries with cleaned text.
    """
    cleaned_records: list[PageRecord] = []
    for page in pages:
        cleaned_records.append({
            "document_id": page["document_id"],
            "document_name": page["document_name"],
            "page_number": page["page_number"],
            "text": clean_text(page.get("text", ""))
        })
    return cleaned_records
