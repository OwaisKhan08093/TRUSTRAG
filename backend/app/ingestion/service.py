"""Service boundary for document ingestion, embedding generation, and index updates."""

import json
import logging
import os
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from backend.app.config import (
    CHUNKS_OUTPUT_FILE,
    DEFAULT_CHUNK_OVERLAP_WORDS,
    DEFAULT_CHUNK_SIZE_WORDS,
    DEFAULT_EMBEDDING_BATCH_SIZE,
    DEFAULT_EMBEDDING_MODEL,
    EMBEDDING_METADATA_FILE,
    EMBEDDINGS_OUTPUT_FILE,
    FAISS_INDEX_FILE,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
)
from backend.app.embeddings.encoder import EmbeddingEncoder, EmbeddingModelError
from backend.app.embeddings.validator import validate_embeddings
from backend.app.ingestion.chunker import chunk_multiple_documents
from backend.app.ingestion.cleaner import clean_page_records
from backend.app.ingestion.pdf_loader import (
    PageRecord,
    PDFEncryptedError,
    PDFInvalidError,
    PDFLoaderError,
    PDFNotFoundError,
    load_pdf,
)
from backend.app.retrieval.faiss_index import FaissVectorIndex

logger = logging.getLogger(__name__)


class DocumentIngestionError(Exception):
    """Base exception for document ingestion failures."""
    pass


class IngestionValidationError(DocumentIngestionError):
    """Raised when file validation fails."""
    pass


def sanitize_filename(filename: str) -> str:
    """Sanitize uploaded filename to prevent directory traversal and invalid characters."""
    clean_name = Path(filename).name
    clean_name = re.sub(r"[^a-zA-Z0-9_.-]", "_", clean_name)
    if not clean_name or clean_name.startswith("."):
        clean_name = f"doc_{clean_name.lstrip('.')}"
    if not clean_name.lower().endswith(".pdf"):
        clean_name += ".pdf"
    return clean_name


class DocumentIngestionService:
    """Encapsulates document parsing, chunking, embedding, and vector indexing operations."""

    def __init__(
        self,
        raw_data_dir: Path = RAW_DATA_DIR,
        processed_data_dir: Path = PROCESSED_DATA_DIR,
        chunks_file: Path = CHUNKS_OUTPUT_FILE,
        embeddings_file: Path = EMBEDDINGS_OUTPUT_FILE,
        metadata_file: Path = EMBEDDING_METADATA_FILE,
        faiss_index_file: Path = FAISS_INDEX_FILE,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    ) -> None:
        self.raw_data_dir = Path(raw_data_dir)
        self.processed_data_dir = Path(processed_data_dir)
        self.chunks_file = Path(chunks_file)
        self.embeddings_file = Path(embeddings_file)
        self.metadata_file = Path(metadata_file)
        self.faiss_index_file = Path(faiss_index_file)
        self.embedding_model = embedding_model

        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)

    def list_documents(self) -> List[Dict[str, Any]]:
        """List all currently indexed documents and their chunk statistics."""
        if not self.chunks_file.exists():
            return []

        try:
            with open(self.chunks_file, "r", encoding="utf-8") as f:
                chunks: List[Dict[str, Any]] = json.load(f)
        except Exception as exc:
            logger.error("Failed to read chunks file: %s", exc)
            return []

        docs: Dict[str, Dict[str, Any]] = {}
        for c in chunks:
            doc_id = c.get("document_id", "unknown")
            if doc_id not in docs:
                docs[doc_id] = {
                    "document_id": doc_id,
                    "document_name": c.get("document_name", doc_id),
                    "chunks_count": 0,
                    "pages": set(),
                    "total_words": 0,
                    "status": "ready",
                }
            docs[doc_id]["chunks_count"] += 1
            docs[doc_id]["pages"].add(c.get("page_start", 1))
            docs[doc_id]["pages"].add(c.get("page_end", 1))
            docs[doc_id]["total_words"] += len(c.get("text", "").split())

        result = []
        for doc in docs.values():
            pages_list = sorted(list(doc["pages"]))
            result.append({
                "document_id": doc["document_id"],
                "document_name": doc["document_name"],
                "chunks_count": doc["chunks_count"],
                "pages_count": len(pages_list),
                "total_words": doc["total_words"],
                "status": doc["status"],
            })
        return result

    def ingest_pdf_bytes(
        self,
        file_bytes: bytes,
        original_filename: str,
        chunk_size: int = DEFAULT_CHUNK_SIZE_WORDS,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP_WORDS,
    ) -> Dict[str, Any]:
        """Ingest a PDF from raw bytes, chunk it, compute embeddings, and update indexes."""
        if not file_bytes or len(file_bytes) == 0:
            raise IngestionValidationError("Uploaded file is empty.")

        # Max file size: 25 MB
        max_bytes = 25 * 1024 * 1024
        if len(file_bytes) > max_bytes:
            raise IngestionValidationError(f"File size exceeds maximum limit of 25MB (got {len(file_bytes)} bytes).")

        filename = sanitize_filename(original_filename)
        dest_pdf_path = self.raw_data_dir / filename

        # Write raw PDF to disk
        try:
            with open(dest_pdf_path, "wb") as f:
                f.write(file_bytes)
        except Exception as exc:
            raise DocumentIngestionError(f"Failed to save uploaded document to disk: {exc}") from exc

        return self.ingest_pdf_file(
            pdf_path=dest_pdf_path,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def ingest_pdf_file(
        self,
        pdf_path: Path,
        chunk_size: int = DEFAULT_CHUNK_SIZE_WORDS,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP_WORDS,
    ) -> Dict[str, Any]:
        """Process a PDF file on disk through the full ingestion and indexing pipeline."""
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise PDFNotFoundError(f"PDF file not found at: {pdf_path}")

        # 1. PDF Extraction
        try:
            pages = load_pdf(pdf_path)
        except (PDFInvalidError, PDFEncryptedError, PDFNotFoundError) as exc:
            raise IngestionValidationError(f"Invalid PDF file: {exc}") from exc
        except Exception as exc:
            raise DocumentIngestionError(f"PDF loading failed: {exc}") from exc

        if not pages:
            raise IngestionValidationError("No extractable text found in PDF.")

        # 2. Text Cleaning
        cleaned_pages = clean_page_records(pages)

        # 3. Chunking
        new_chunks = chunk_multiple_documents(
            pages=cleaned_pages,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        if not new_chunks:
            raise IngestionValidationError("Document produced 0 text chunks. Ensure the document contains readable text.")

        doc_id = new_chunks[0]["document_id"]
        doc_name = new_chunks[0]["document_name"]

        # 4. Load existing chunks and merge (replacing any previous chunks of the same doc)
        existing_chunks: List[Dict[str, Any]] = []
        if self.chunks_file.exists():
            try:
                with open(self.chunks_file, "r", encoding="utf-8") as f:
                    existing_chunks = json.load(f)
            except Exception:
                existing_chunks = []

        # Filter out old chunks for this doc_id
        filtered_chunks = [c for c in existing_chunks if c.get("document_id") != doc_id]
        all_chunks = filtered_chunks + new_chunks

        # 5. Generate Embeddings for all chunks
        all_texts = [c.get("text", "") for c in all_chunks]
        try:
            encoder = EmbeddingEncoder(model_name=self.embedding_model)
            embeddings = encoder.encode_documents(
                texts=all_texts,
                batch_size=DEFAULT_EMBEDDING_BATCH_SIZE,
                normalize_embeddings=True,
            )
        except EmbeddingModelError as exc:
            raise DocumentIngestionError(f"Embedding generation failed: {exc}") from exc
        except Exception as exc:
            raise DocumentIngestionError(f"Unexpected embedding error: {exc}") from exc

        # 6. Validate Embeddings
        try:
            validate_embeddings(embeddings, expected_dim=encoder.embedding_dimension, check_normalized=True)
        except Exception as exc:
            raise DocumentIngestionError(f"Embedding validation failed: {exc}") from exc

        # 7. Persist Artifacts to Disk
        try:
            # Chunks JSON
            with open(self.chunks_file, "w", encoding="utf-8") as f:
                json.dump(all_chunks, f, indent=2, ensure_ascii=False)

            # Embeddings NPY
            np.save(str(self.embeddings_file), embeddings)

            # Metadata JSON
            metadata_mapping = [
                {
                    "embedding_index": idx,
                    "chunk_id": c.get("chunk_id", f"chunk_{idx}"),
                }
                for idx, c in enumerate(all_chunks)
            ]
            with open(self.metadata_file, "w", encoding="utf-8") as f:
                json.dump(metadata_mapping, f, indent=2, ensure_ascii=False)

            # Rebuild and save FAISS Index
            vector_index = FaissVectorIndex(dimension=encoder.embedding_dimension)
            vector_index.add_embeddings(embeddings, check_normalized=True)
            vector_index.save(self.faiss_index_file)

        except Exception as exc:
            raise DocumentIngestionError(f"Failed to persist indexing artifacts: {exc}") from exc

        # 8. Refresh active orchestrator in memory
        self._refresh_orchestrator()

        logger.info(
            "Successfully ingested document '%s' (%s): created %d chunks (total indexed: %d).",
            doc_name,
            doc_id,
            len(new_chunks),
            len(all_chunks),
        )

        return {
            "document_id": doc_id,
            "filename": doc_name,
            "status": "ready",
            "chunks_created": len(new_chunks),
            "total_chunks_indexed": len(all_chunks),
            "message": "Document processed and indexed successfully.",
        }

    def _refresh_orchestrator(self) -> None:
        """Reload dependencies and recreate orchestrator instance so new index is active."""
        try:
            from backend.app.api.dependencies import get_orchestrator, set_orchestrator
            from backend.app.agents.orchestrator import TrustRAGOrchestrator

            # Reset orchestrator instance to re-initialize with fresh indexes
            set_orchestrator(TrustRAGOrchestrator())
            logger.info("TrustRAG orchestrator refreshed with updated indexes.")
        except Exception as exc:
            logger.warning("Could not refresh in-memory orchestrator: %s", exc)
