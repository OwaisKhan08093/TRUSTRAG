"""FastAPI route handlers for document upload, ingestion, and cataloging."""

import logging
from typing import Optional
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from backend.app.api.schemas import DocumentListResponse, DocumentUploadResponse
from backend.app.ingestion.service import (
    DocumentIngestionError,
    DocumentIngestionService,
    IngestionValidationError,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["Documents"])

_ingestion_service: Optional[DocumentIngestionService] = None


def get_ingestion_service() -> DocumentIngestionService:
    """Dependency provider returning DocumentIngestionService instance."""
    global _ingestion_service
    if _ingestion_service is None:
        _ingestion_service = DocumentIngestionService()
    return _ingestion_service


@router.post(
    "",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload and ingest a document",
    description="Upload a PDF document to parse, chunk, embed, and index into TrustRAG retrieval corpus.",
)
async def upload_document(
    file: UploadFile = File(..., description="PDF document file to upload"),
    service: DocumentIngestionService = Depends(get_ingestion_service),
) -> DocumentUploadResponse:
    """Handle multipart file upload, validate, and execute document ingestion."""
    if not file or not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No file provided in upload request.",
        )

    # Validate file extension
    filename = file.filename.strip()
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format '{filename}'. Only PDF documents (.pdf) are currently supported.",
        )

    try:
        content = await file.read()
    except Exception as exc:
        logger.error("Failed to read uploaded file payload: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to read uploaded file data.",
        ) from exc

    if not content or len(content) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty (0 bytes).",
        )

    try:
        result = service.ingest_pdf_bytes(
            file_bytes=content,
            original_filename=filename,
        )
        return DocumentUploadResponse.model_validate(result)

    except IngestionValidationError as exc:
        logger.warning("Document validation error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    except DocumentIngestionError as exc:
        logger.error("Document ingestion failure: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Document ingestion failed: {exc}",
        ) from exc
    except Exception as exc:
        logger.error("Unexpected error during document ingestion: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unexpected internal server error during document processing.",
        ) from exc


@router.get(
    "",
    response_model=DocumentListResponse,
    status_code=status.HTTP_200_OK,
    summary="List ingested documents",
    description="Retrieve all documents and their chunk statistics currently indexed in TrustRAG.",
)
async def list_documents(
    service: DocumentIngestionService = Depends(get_ingestion_service),
) -> DocumentListResponse:
    """Return summary of all ingested documents in the corpus."""
    try:
        docs = service.list_documents()
        total_chunks = sum(d["chunks_count"] for d in docs)
        return DocumentListResponse(
            documents=docs,
            total_documents=len(docs),
            total_chunks=total_chunks,
        )
    except Exception as exc:
        logger.error("Failed to list ingested documents: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve indexed documents catalog.",
        ) from exc
