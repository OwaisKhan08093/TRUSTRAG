"""Unit tests for the PDF loader and conservative text cleaner."""

from pathlib import Path
import pytest
import pymupdf as fitz

from backend.app.ingestion.pdf_loader import (
    load_pdf,
    PDFNotFoundError,
    PDFEncryptedError,
    PDFInvalidError,
    generate_document_id,
)
from backend.app.ingestion.cleaner import clean_text, clean_page_records


@pytest.fixture
def sample_pdf(tmp_path: Path) -> Path:
    """Create a temporary 3-page PDF document for testing."""
    pdf_path = tmp_path / "test_doc.pdf"
    doc = fitz.open()

    # Page 1: Standard legal text
    p1 = doc.new_page()
    p1.insert_text(
        (50, 50),
        "Section 1. Short Title and Commencement.\n(1) This Act may be called the Test Act, 2024.\n(2) It shall come into force at once."
    )

    # Page 2: Empty page
    doc.new_page()

    # Page 3: Additional clauses
    p3 = doc.new_page()
    p3.insert_text(
        (50, 50),
        "Section 2. Definitions.\nIn this Act, unless the context otherwise requires,-\n(a) 'data' means a representation of information;\n(b) 'data principal' means the individual."
    )

    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


@pytest.fixture
def encrypted_pdf(tmp_path: Path) -> Path:
    """Create a temporary password-protected PDF."""
    pdf_path = tmp_path / "protected_doc.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 50), "Confidential legal agreement.")
    
    # Save with user password
    perm = int(
        fitz.PDF_PERM_ACCESSIBILITY
        | fitz.PDF_PERM_PRINT
        | fitz.PDF_PERM_COPY
        | fitz.PDF_PERM_ANNOTATE
    )
    doc.save(
        str(pdf_path),
        encryption=fitz.PDF_ENCRYPT_AES_256,
        user_pw="password123",
        owner_pw="owner123",
        permissions=perm,
    )
    doc.close()
    return pdf_path


@pytest.fixture
def corrupt_pdf(tmp_path: Path) -> Path:
    """Create an invalid/corrupt PDF file."""
    bad_path = tmp_path / "corrupt.pdf"
    bad_path.write_bytes(b"This is not a valid PDF file content.")
    return bad_path


def test_document_id_generation():
    """Verify document IDs are generated predictably and safely."""
    assert generate_document_id("DPDP_Act_2023.pdf") == "DPDP_Act_2023"
    assert generate_document_id("My Report (Final) v1.0.pdf") == "My_Report__Final__v1_0"


def test_pdf_can_be_opened_and_extracted(sample_pdf: Path):
    """Test 1 & 2: PDF can be opened and pages are extracted."""
    pages = load_pdf(sample_pdf)
    assert len(pages) == 3


def test_page_numbers_start_correctly(sample_pdf: Path):
    """Test 3: Page numbers start at 1 and are sequential."""
    pages = load_pdf(sample_pdf)
    assert pages[0]["page_number"] == 1
    assert pages[1]["page_number"] == 2
    assert pages[2]["page_number"] == 3


def test_document_name_and_id_preserved(sample_pdf: Path):
    """Test 4: Document name and document ID are preserved across all pages."""
    pages = load_pdf(sample_pdf)
    for page in pages:
        assert page["document_name"] == "test_doc.pdf"
        assert page["document_id"] == "test_doc"


def test_empty_pages_handled_safely(sample_pdf: Path):
    """Test 5: Empty pages produce records with empty string without crashing."""
    pages = load_pdf(sample_pdf)
    assert pages[1]["page_number"] == 2
    assert pages[1]["text"].strip() == ""


def test_file_not_found_raises_error(tmp_path: Path):
    """Verify non-existent file raises PDFNotFoundError."""
    missing_path = tmp_path / "non_existent.pdf"
    with pytest.raises(PDFNotFoundError):
        load_pdf(missing_path)


def test_encrypted_pdf_raises_error(encrypted_pdf: Path):
    """Verify password protected PDF raises PDFEncryptedError."""
    with pytest.raises(PDFEncryptedError):
        load_pdf(encrypted_pdf)


def test_corrupt_pdf_raises_error(corrupt_pdf: Path):
    """Verify corrupted PDF raises PDFInvalidError."""
    with pytest.raises(PDFInvalidError):
        load_pdf(corrupt_pdf)


def test_cleaner_preserves_legal_numbering():
    """Test 6: Cleaning does not destroy legal structure or numbering."""
    raw = """
    Section 1. Short Title.
    
    
    (a) first clause with   extra   spaces
    (b) second clause.
    1. Sub-item one
    2. Sub-item two
    """
    cleaned = clean_text(raw)
    assert "Section 1. Short Title." in cleaned
    assert "(a) first clause with extra spaces" in cleaned
    assert "(b) second clause." in cleaned
    assert "1. Sub-item one" in cleaned
    assert "2. Sub-item two" in cleaned
    # Ensure no more than 2 consecutive newlines
    assert "\n\n\n" not in cleaned


def test_clean_page_records(sample_pdf: Path):
    """Verify cleaning across page records maintains metadata."""
    pages = load_pdf(sample_pdf)
    cleaned = clean_page_records(pages)
    assert len(cleaned) == len(pages)
    assert cleaned[0]["document_id"] == pages[0]["document_id"]
    assert "Section 1" in cleaned[0]["text"]
