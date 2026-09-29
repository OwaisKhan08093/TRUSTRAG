"""PDF loader module using PyMuPDF (fitz) to extract text with page metadata."""

from pathlib import Path
import re
from typing import TypedDict
import pymupdf as fitz


class PageRecord(TypedDict):
    """Structured record representing a single page from a document."""
    document_id: str
    document_name: str
    page_number: int
    text: str


class PDFLoaderError(Exception):
    """Base exception for PDF loading errors."""
    pass


class PDFNotFoundError(PDFLoaderError, FileNotFoundError):
    """Raised when the specified PDF file cannot be found."""
    pass


class PDFEncryptedError(PDFLoaderError):
    """Raised when the PDF file is password protected or encrypted."""
    pass


class PDFInvalidError(PDFLoaderError):
    """Raised when the PDF file is corrupted or not a valid PDF."""
    pass


def generate_document_id(file_name: str) -> str:
    """Generate a clean, deterministic document identifier from a filename.

    Example: 'DPDP_Act_2023.pdf' -> 'DPDP_Act_2023'
    """
    stem = Path(file_name).stem
    # Replace non-alphanumeric characters with underscores, keeping it clean
    cleaned = re.sub(r'[^a-zA-Z0-9_-]', '_', stem)
    return cleaned.strip('_') or "doc"


def load_pdf(file_path: str | Path, document_id: str | None = None) -> list[PageRecord]:
    """Safely open a PDF file, iterate through every page, and extract text with metadata.

    Args:
        file_path: Path to the PDF file.
        document_id: Optional custom identifier for the document. If omitted, derived from filename.

    Returns:
        A list of PageRecord dictionaries containing document_id, document_name,
        page_number (1-indexed), and extracted text.

    Raises:
        PDFNotFoundError: If the file does not exist.
        PDFEncryptedError: If the file is password-protected.
        PDFInvalidError: If the file is not a valid PDF or is corrupted.
    """
    path = Path(file_path)

    if not path.is_file():
        raise PDFNotFoundError(f"PDF file not found at path: '{path.resolve()}'")

    document_name = path.name
    doc_id = document_id if document_id is not None else generate_document_id(document_name)

    try:
        doc = fitz.open(str(path))
    except Exception as exc:
        raise PDFInvalidError(f"Failed to open PDF file '{document_name}': {exc}") from exc

    try:
        if doc.is_encrypted or doc.needs_pass:
            raise PDFEncryptedError(
                f"PDF file '{document_name}' is password-protected or encrypted."
            )

        pages: list[PageRecord] = []
        for page_num in range(1, len(doc) + 1):
            page = doc[page_num - 1]
            try:
                page_text = page.get_text("text")
            except Exception as exc:
                page_text = ""

            pages.append({
                "document_id": doc_id,
                "document_name": document_name,
                "page_number": page_num,
                "text": page_text
            })

        return pages

    finally:
        doc.close()
