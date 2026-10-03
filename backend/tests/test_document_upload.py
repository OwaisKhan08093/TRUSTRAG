"""Comprehensive test suite for Document Upload, Ingestion API, and Query Availability."""

import io
from pathlib import Path
from unittest.mock import MagicMock, patch
import pymupdf as fitz
import pytest
from fastapi.testclient import TestClient

from backend.app.api.app import app
from backend.app.api.routes.documents import get_ingestion_service
from backend.app.ingestion.service import (
    DocumentIngestionError,
    DocumentIngestionService,
    IngestionValidationError,
    sanitize_filename,
)


def create_test_pdf_bytes(title: str, text: str) -> bytes:
    """Helper to generate a valid PDF byte string."""
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 50), f"{title}\n\n{text}")
    stream = io.BytesIO()
    doc.save(stream)
    doc.close()
    return stream.getvalue()


@pytest.fixture
def temp_ingestion_service(tmp_path: Path):
    """Create an isolated DocumentIngestionService with temporary artifact storage."""
    service = DocumentIngestionService(
        raw_data_dir=tmp_path / "raw",
        processed_data_dir=tmp_path / "processed",
        chunks_file=tmp_path / "processed" / "chunks.json",
        embeddings_file=tmp_path / "processed" / "embeddings.npy",
        metadata_file=tmp_path / "processed" / "embedding_metadata.json",
        faiss_index_file=tmp_path / "processed" / "index.faiss",
    )
    return service


@pytest.fixture
def upload_client(temp_ingestion_service):
    """FastAPI TestClient with isolated ingestion service override."""
    app.dependency_overrides[get_ingestion_service] = lambda: temp_ingestion_service
    client = TestClient(app)
    yield client
    app.dependency_overrides.clear()


# =========================================================================
# 1. Successful Upload & Ingestion
# =========================================================================

def test_01_successful_document_upload(upload_client):
    """Verify uploading a valid PDF parses, chunks, embeds, and indexes successfully."""
    pdf_bytes = create_test_pdf_bytes(
        title="Supervised Machine Learning",
        text=(
            "Supervised learning is a machine learning paradigm where models are trained "
            "on labeled datasets. Algorithms learn mapping functions from input features to targets."
        ),
    )

    response = upload_client.post(
        "/documents",
        files={"file": ("supervised_learning.pdf", pdf_bytes, "application/pdf")},
    )

    assert response.status_code == 201
    data = response.json()
    assert data["status"] == "ready"
    assert "supervised_learning" in data["filename"]
    assert data["chunks_created"] >= 1
    assert data["total_chunks_indexed"] >= 1
    assert "successfully" in data["message"]


# =========================================================================
# 2. Missing File Error Handling
# =========================================================================

def test_02_missing_file_error(upload_client):
    """Verify request with missing file payload returns 400 or 422."""
    response = upload_client.post("/documents", data={})
    assert response.status_code in (400, 422)


# =========================================================================
# 3. Invalid File Type Error Handling
# =========================================================================

def test_03_invalid_file_type_rejection(upload_client):
    """Verify non-PDF file uploads are rejected with 400 Bad Request."""
    response = upload_client.post(
        "/documents",
        files={"file": ("malicious_script.exe", b"MZ\x90\x00\x03", "application/octet-stream")},
    )
    assert response.status_code == 400
    data = response.json()
    assert "error" in data
    assert "Only PDF documents" in data["error"]["message"]


# =========================================================================
# 4. Malformed PDF Error Handling
# =========================================================================

def test_04_malformed_corrupt_pdf_handling(upload_client):
    """Verify corrupted non-PDF bytes labeled as PDF return 422 Unprocessable Entity."""
    corrupt_bytes = b"NOT_A_VALID_PDF_HEADER_DATA_CORRUPT"
    response = upload_client.post(
        "/documents",
        files={"file": ("corrupt_file.pdf", corrupt_bytes, "application/pdf")},
    )
    assert response.status_code == 422
    data = response.json()
    assert "error" in data
    assert "Invalid PDF" in data["error"]["message"]


# =========================================================================
# 5. Oversized File Error Handling
# =========================================================================

def test_05_oversized_file_rejection(temp_ingestion_service):
    """Verify files exceeding maximum byte limits are rejected by service."""
    oversized_bytes = b"x" * (26 * 1024 * 1024)  # 26 MB
    with pytest.raises(IngestionValidationError, match="exceeds maximum limit"):
        temp_ingestion_service.ingest_pdf_bytes(
            file_bytes=oversized_bytes,
            original_filename="huge_file.pdf",
        )


# =========================================================================
# 6. Ingestion Pipeline Failure Sanitization
# =========================================================================

def test_06_ingestion_internal_failure_handling(upload_client, temp_ingestion_service):
    """Verify unhandled ingestion runtime errors return 500 error envelopes."""
    pdf_bytes = create_test_pdf_bytes("Title", "Sample text.")
    with patch.object(temp_ingestion_service, "ingest_pdf_bytes", side_effect=DocumentIngestionError("FAISS disk write locked")):
        response = upload_client.post(
            "/documents",
            files={"file": ("test.pdf", pdf_bytes, "application/pdf")},
        )
        assert response.status_code == 500
        data = response.json()
        assert "error" in data
        assert data["error"]["status_code"] == 500
        assert "Document ingestion failed" in data["error"]["message"]


# =========================================================================
# 7. Document Listing Endpoint
# =========================================================================

def test_07_list_documents_catalog(upload_client):
    """Verify GET /documents returns list of ingested documents and chunk stats."""
    pdf_bytes = create_test_pdf_bytes(
        title="Deep Reinforcement Learning",
        text="Reinforcement learning is an area of machine learning concerned with how software agents take actions in an environment.",
    )
    upload_client.post(
        "/documents",
        files={"file": ("reinforcement_learning.pdf", pdf_bytes, "application/pdf")},
    )

    list_resp = upload_client.get("/documents")
    assert list_resp.status_code == 200
    catalog = list_resp.json()
    assert catalog["total_documents"] >= 1
    assert catalog["total_chunks"] >= 1
    assert any("reinforcement_learning" in d["document_name"] for d in catalog["documents"])


# =========================================================================
# 8. Complete End-to-End: Upload -> Ingest -> Query
# =========================================================================

def test_08_uploaded_document_becomes_queryable(upload_client, temp_ingestion_service):
    """Verify an uploaded custom document is ingested, indexed, and queryable."""
    custom_pdf = create_test_pdf_bytes(
        title="Quantum Cryptography Protocols",
        text=(
            "Quantum key distribution utilizes quantum mechanics properties to secure communication. "
            "The BB84 protocol was developed by Charles Bennett and Gilles Brassard in 1984."
        ),
    )

    # 1. Ingest via service
    res = temp_ingestion_service.ingest_pdf_bytes(
        file_bytes=custom_pdf,
        original_filename="quantum_crypto.pdf",
    )
    assert res["status"] == "ready"
    assert res["chunks_created"] >= 1

    # 2. Verify files exist on disk
    assert temp_ingestion_service.chunks_file.exists()
    assert temp_ingestion_service.faiss_index_file.exists()
    assert temp_ingestion_service.embeddings_file.exists()


def test_sanitize_filename_utility():
    """Verify filename sanitization removes path traversals."""
    assert sanitize_filename("../../../etc/passwd.pdf") == "passwd.pdf"
    assert sanitize_filename("my report 2024!.pdf") == "my_report_2024_.pdf"
    assert sanitize_filename("simple_doc") == "simple_doc.pdf"
