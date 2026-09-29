"""Embedding generation CLI for TrustRAG: Encodes chunks into dense vector embeddings."""

import argparse
import json
import sys
from pathlib import Path
import numpy as np

# Add project root to sys.path so backend imports work seamlessly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.config import (
    CHUNKS_OUTPUT_FILE,
    DEFAULT_EMBEDDING_BATCH_SIZE,
    DEFAULT_EMBEDDING_MODEL,
    EMBEDDING_METADATA_FILE,
    EMBEDDINGS_OUTPUT_FILE,
)
from backend.app.embeddings.encoder import EmbeddingEncoder, EmbeddingModelError


def generate_embeddings(
    chunks_file: Path = CHUNKS_OUTPUT_FILE,
    embeddings_file: Path = EMBEDDINGS_OUTPUT_FILE,
    metadata_file: Path = EMBEDDING_METADATA_FILE,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE,
    normalize: bool = True,
) -> int:
    """Load chunks, generate embeddings, and save binary and metadata artifacts."""
    if not chunks_file.exists():
        print(
            f"Error: Chunks file not found at '{chunks_file}'.\n"
            f"Please run the ingestion script first (e.g., python scripts/ingest.py data/raw/DPDP_Act_2023.pdf).",
            file=sys.stderr,
        )
        return 1

    try:
        with open(chunks_file, "r", encoding="utf-8") as f:
            chunks = json.load(f)
    except Exception as exc:
        print(f"Error reading chunks file '{chunks_file}': {exc}", file=sys.stderr)
        return 1

    if not isinstance(chunks, list) or len(chunks) == 0:
        print(f"Error: Chunks file '{chunks_file}' is empty or not a valid JSON list.", file=sys.stderr)
        return 1

    chunk_texts = [chunk.get("text", "") for chunk in chunks]
    num_chunks = len(chunk_texts)

    try:
        encoder = EmbeddingEncoder(model_name=model_name)
        embeddings = encoder.encode_documents(
            texts=chunk_texts,
            batch_size=batch_size,
            normalize_embeddings=normalize,
        )
    except EmbeddingModelError as exc:
        print(f"Embedding error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Unexpected error during embedding generation: {exc}", file=sys.stderr)
        return 1

    # Ensure output directories exist
    embeddings_file.parent.mkdir(parents=True, exist_ok=True)
    metadata_file.parent.mkdir(parents=True, exist_ok=True)

    # Save NumPy binary vectors
    np.save(str(embeddings_file), embeddings)

    # Save lightweight metadata mapping
    metadata_mapping = [
        {
            "embedding_index": idx,
            "chunk_id": chunk.get("chunk_id", f"chunk_{idx}"),
        }
        for idx, chunk in enumerate(chunks)
    ]

    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata_mapping, f, indent=2, ensure_ascii=False)

    rel_emb_path = embeddings_file.relative_to(PROJECT_ROOT) if embeddings_file.is_relative_to(PROJECT_ROOT) else embeddings_file
    rel_meta_path = metadata_file.relative_to(PROJECT_ROOT) if metadata_file.is_relative_to(PROJECT_ROOT) else metadata_file

    # Print summary format requested
    print("TrustRAG Embedding Generation\n")
    print(f"Model:\n{encoder.model_name}\n")
    print(f"Chunks:\n{num_chunks}\n")
    print(f"Embedding dimension:\n{encoder.embedding_dimension}\n")
    print(f"Embedding shape:\n{embeddings.shape}\n")
    print(f"Normalized:\n{'yes' if normalize else 'no'}\n")
    print(f"Saved:\n{rel_emb_path}\n{rel_meta_path}")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="TrustRAG Document Embedding Generation (Step 2.2)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=str,
        default=str(CHUNKS_OUTPUT_FILE),
        help="Path to input chunks.json file.",
    )
    parser.add_argument(
        "--output-embeddings",
        type=str,
        default=str(EMBEDDINGS_OUTPUT_FILE),
        help="Path to save output embeddings .npy file.",
    )
    parser.add_argument(
        "--output-metadata",
        type=str,
        default=str(EMBEDDING_METADATA_FILE),
        help="Path to save output embedding metadata mapping JSON file.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_EMBEDDING_MODEL,
        help="SentenceTransformer model name to use.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_EMBEDDING_BATCH_SIZE,
        help="Batch size for embedding generation.",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Disable L2 normalization of embedding vectors.",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.is_absolute():
        input_path = (PROJECT_ROOT / input_path).resolve()

    emb_path = Path(args.output_embeddings)
    if not emb_path.is_absolute():
        emb_path = (PROJECT_ROOT / emb_path).resolve()

    meta_path = Path(args.output_metadata)
    if not meta_path.is_absolute():
        meta_path = (PROJECT_ROOT / meta_path).resolve()

    return generate_embeddings(
        chunks_file=input_path,
        embeddings_file=emb_path,
        metadata_file=meta_path,
        model_name=args.model,
        batch_size=args.batch_size,
        normalize=not args.no_normalize,
    )


if __name__ == "__main__":
    sys.exit(main())
