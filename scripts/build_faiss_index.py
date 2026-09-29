"""CLI script to build and persist FAISS vector index from document embeddings."""

import argparse
import sys
from pathlib import Path
import numpy as np

# Add project root to sys.path so backend imports work seamlessly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.config import EMBEDDINGS_OUTPUT_FILE, FAISS_INDEX_FILE
from backend.app.embeddings.validator import validate_embeddings
from backend.app.retrieval.faiss_index import FaissIndexError, FaissVectorIndex


def build_index(
    embeddings_file: Path = EMBEDDINGS_OUTPUT_FILE,
    index_file: Path = FAISS_INDEX_FILE,
    expected_dim: int = 384,
    check_normalized: bool = True,
) -> int:
    """Load embeddings, validate, construct FAISS index, and persist to disk.

    Args:
        embeddings_file: Path to input embeddings .npy file.
        index_file: Path to destination .faiss file.
        expected_dim: Expected vector dimension (default: 384).
        check_normalized: Whether to require unit L2 normalization (default: True).

    Returns:
        0 on success, 1 on failure.
    """
    print("TrustRAG FAISS Index Builder\n")

    if not embeddings_file.exists():
        print(
            f"Error: Embeddings file not found at '{embeddings_file}'.\n"
            f"Please run the embedding generation script first (e.g., python scripts/embed_chunks.py).",
            file=sys.stderr,
        )
        return 1

    try:
        embeddings = np.load(str(embeddings_file))
    except Exception as exc:
        print(f"Error reading embeddings file '{embeddings_file}': {exc}", file=sys.stderr)
        return 1

    try:
        stats = validate_embeddings(
            embeddings,
            expected_dim=expected_dim,
            check_normalized=check_normalized,
        )
    except Exception as exc:
        print(f"Validation error for embeddings in '{embeddings_file}': {exc}", file=sys.stderr)
        return 1

    dimension = stats["dimension"]
    num_vectors = stats["num_vectors"]

    try:
        vector_index = FaissVectorIndex(dimension=dimension)
        vector_index.add_embeddings(embeddings, check_normalized=check_normalized)
        vector_index.save(index_file)
    except Exception as exc:
        print(f"Error building or saving FAISS index: {exc}", file=sys.stderr)
        return 1

    rel_index_path = (
        index_file.relative_to(PROJECT_ROOT)
        if index_file.is_relative_to(PROJECT_ROOT)
        else index_file
    )

    print(f"Vectors: {num_vectors}")
    print(f"Dimension: {dimension}")
    print("Metric: Inner Product (Cosine Similarity)")
    print(f"Index path: {rel_index_path}")
    print("Status: SUCCESS")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build FAISS vector index from document embeddings (Milestone 3)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--embeddings",
        type=str,
        default=str(EMBEDDINGS_OUTPUT_FILE),
        help="Path to input embeddings .npy file.",
    )
    parser.add_argument(
        "--output-index",
        type=str,
        default=str(FAISS_INDEX_FILE),
        help="Path to output .faiss index file.",
    )
    parser.add_argument(
        "--expected-dim",
        type=int,
        default=384,
        help="Expected vector dimension.",
    )
    parser.add_argument(
        "--no-normalize-check",
        action="store_true",
        help="Skip unit L2 normalization requirement.",
    )

    args = parser.parse_args()

    emb_path = Path(args.embeddings)
    if not emb_path.is_absolute():
        emb_path = (PROJECT_ROOT / emb_path).resolve()

    index_path = Path(args.output_index)
    if not index_path.is_absolute():
        index_path = (PROJECT_ROOT / index_path).resolve()

    return build_index(
        embeddings_file=emb_path,
        index_file=index_path,
        expected_dim=args.expected_dim,
        check_normalized=not args.no_normalize_check,
    )


if __name__ == "__main__":
    sys.exit(main())
