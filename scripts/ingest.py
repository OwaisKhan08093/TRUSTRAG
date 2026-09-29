"""Ingestion CLI for TrustRAG: Ingests PDFs, extracts text, cleans, chunks, and writes chunks.json."""

import argparse
import json
import sys
from pathlib import Path

# Add project root to sys.path so imports work seamlessly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.config import (
    CHUNKS_OUTPUT_FILE,
    DEFAULT_CHUNK_OVERLAP_WORDS,
    DEFAULT_CHUNK_SIZE_WORDS,
    RAW_DATA_DIR,
)
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


def ingest_file(pdf_path: Path, chunk_size: int, chunk_overlap: int) -> tuple[list[PageRecord], list[dict]]:
    """Load, clean, and chunk a single PDF file."""
    pages = load_pdf(pdf_path)
    cleaned_pages = clean_page_records(pages)
    chunks = chunk_multiple_documents(
        pages=cleaned_pages,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return cleaned_pages, chunks


def main() -> int:
    parser = argparse.ArgumentParser(
        description="TrustRAG PDF Ingestion Pipeline (Step 1)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "pdf_path",
        type=str,
        nargs="?",
        default=None,
        help="Path to the PDF file to ingest (e.g., data/raw/DPDP_Act_2023.pdf).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(CHUNKS_OUTPUT_FILE),
        help="Path to save the output chunks JSON file.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE_WORDS,
        help="Target chunk size in words.",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=DEFAULT_CHUNK_OVERLAP_WORDS,
        help="Overlap size between chunks in words.",
    )

    args = parser.parse_args()

    # Determine files to process
    if args.pdf_path:
        target_path = Path(args.pdf_path)
        if not target_path.is_absolute():
            target_path = (PROJECT_ROOT / target_path).resolve()
        pdf_files = [target_path]
    else:
        # Look in data/raw for any PDF files
        pdf_files = sorted(RAW_DATA_DIR.glob("*.pdf"))
        if not pdf_files:
            print(f"Error: No PDF path provided and no PDF files found in '{RAW_DATA_DIR}'.", file=sys.stderr)
            print("Usage: python scripts/ingest.py data/raw/<document_name>.pdf", file=sys.stderr)
            return 1

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = (PROJECT_ROOT / output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_chunks: list[dict] = []
    has_errors = False

    for pdf_file in pdf_files:
        try:
            pages, chunks = ingest_file(pdf_file, args.chunk_size, args.chunk_overlap)
            all_chunks.extend(chunks)

            total_pages = len(pages)
            pages_with_text = sum(1 for p in pages if p["text"].strip())
            total_chunks = len(chunks)

            if total_chunks > 0:
                avg_chunk_size = sum(len(c["text"].split()) for c in chunks) / total_chunks
            else:
                avg_chunk_size = 0.0

            print(f"Document: {pdf_file.name}")
            print(f"Pages: {total_pages}")
            print(f"Pages with text: {pages_with_text}")
            print(f"Chunks created: {total_chunks}")
            print(f"Average chunk size: {avg_chunk_size:.1f} words")
            print("-" * 40)

            if pages_with_text == 0 and total_pages > 0:
                print(f"Warning: PDF '{pdf_file.name}' has pages but no extractable text (may be scanned images).", file=sys.stderr)

        except PDFNotFoundError as e:
            print(f"Error: File not found: {e}", file=sys.stderr)
            has_errors = True
        except PDFEncryptedError as e:
            print(f"Error: Password protected PDF: {e}", file=sys.stderr)
            has_errors = True
        except PDFInvalidError as e:
            print(f"Error: Invalid or corrupted PDF: {e}", file=sys.stderr)
            has_errors = True
        except PDFLoaderError as e:
            print(f"Error: PDF loader error: {e}", file=sys.stderr)
            has_errors = True
        except Exception as e:
            print(f"Unexpected error while processing '{pdf_file}': {e}", file=sys.stderr)
            has_errors = True

    if all_chunks or not has_errors:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(all_chunks, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(all_chunks)} chunks to: {output_path}")

    return 1 if has_errors else 0


if __name__ == "__main__":
    sys.exit(main())
