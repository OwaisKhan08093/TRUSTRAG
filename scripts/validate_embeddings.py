"""TrustRAG Embedding Validation CLI: Validates saved embeddings and mathematical invariants."""

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
    DEFAULT_EMBEDDING_MODEL,
    EMBEDDING_METADATA_FILE,
    EMBEDDINGS_OUTPUT_FILE,
)
from backend.app.embeddings.encoder import EmbeddingEncoder, EmbeddingModelError
from backend.app.embeddings.validator import (
    run_semantic_sanity_check,
    validate_embeddings,
)


def validate_saved_embeddings(
    embeddings_file: Path = EMBEDDINGS_OUTPUT_FILE,
    metadata_file: Path = EMBEDDING_METADATA_FILE,
    expected_dim: int = 384,
    check_normalized: bool = True,
    run_sanity_check: bool = True,
) -> int:
    """Validate saved embedding binary artifact, metadata alignment, and semantic sanity.

    Returns:
        0 on success, non-zero on validation failure or error.
    """
    print("TrustRAG Embedding Validation\n")

    # 1. Check file existence
    if not embeddings_file.exists():
        print(f"Error: Embeddings file not found at '{embeddings_file}'.", file=sys.stderr)
        return 1

    if not metadata_file.exists():
        print(f"Error: Metadata file not found at '{metadata_file}'.", file=sys.stderr)
        return 1

    # 2. Load embeddings
    try:
        embeddings = np.load(str(embeddings_file))
    except Exception as exc:
        print(f"Error loading embeddings file '{embeddings_file}': {exc}", file=sys.stderr)
        return 1

    # 3. Load metadata
    try:
        with open(metadata_file, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    except Exception as exc:
        print(f"Error reading metadata file '{metadata_file}': {exc}", file=sys.stderr)
        return 1

    # 4. Validate embedding matrix
    try:
        stats = validate_embeddings(
            embeddings,
            expected_dim=expected_dim,
            check_normalized=check_normalized,
        )
    except Exception as exc:
        print(f"Validation Error: {exc}", file=sys.stderr)
        return 1

    # 5. Verify embedding count matches metadata count
    if not isinstance(metadata, list):
        print(f"Error: Metadata file must contain a JSON list, got {type(metadata).__name__}.", file=sys.stderr)
        return 1

    meta_count = len(metadata)
    emb_count = stats["num_vectors"]
    metadata_aligned = emb_count == meta_count

    if not metadata_aligned:
        print(
            f"Error: Count mismatch between embeddings ({emb_count}) and metadata entries ({meta_count}).",
            file=sys.stderr,
        )
        return 1

    # Print validation report matching the expected format
    print(f"Vectors: {stats['num_vectors']}")
    print(f"Dimension: {stats['dimension']}")
    print(f"Dtype: {stats['dtype']}\n")

    print("Finite values: PASS")
    print(f"NaN values: {stats['nan_count']}")
    print(f"Infinite values: {stats['inf_count']}")
    print(f"Zero vectors: {stats['zero_vector_count']}\n")

    print("L2 norm:")
    print(f"min: {stats['min_norm']:.6f}")
    print(f"max: {stats['max_norm']:.6f}")
    print(f"mean: {stats['mean_norm']:.6f}\n")

    print(f"Normalization: {'PASS' if stats['is_normalized'] else 'FAIL'}\n")
    print(f"Metadata alignment: {'PASS' if metadata_aligned else 'FAIL'}\n")

    # 6. Semantic sanity check
    sanity_passed = True
    if run_sanity_check:
        try:
            sanity_result = run_semantic_sanity_check()
            print("Semantic sanity check:")
            print(f"Similar pair: {'PASS' if sanity_result['similar_passed'] else 'FAIL'}")
            print(f"Different pair: {'PASS' if sanity_result['different_passed'] else 'FAIL'}\n")
            sanity_passed = sanity_result["passed"]
        except Exception as exc:
            print(f"Semantic sanity check failed with exception: {exc}", file=sys.stderr)
            sanity_passed = False

    if stats["is_normalized"] and metadata_aligned and sanity_passed:
        print("Overall validation: PASS")
        return 0
    else:
        print("Overall validation: FAIL", file=sys.stderr)
        return 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="TrustRAG Embedding Validation CLI (Step 2.3)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--embeddings",
        type=str,
        default=str(EMBEDDINGS_OUTPUT_FILE),
        help="Path to embeddings .npy file.",
    )
    parser.add_argument(
        "--metadata",
        type=str,
        default=str(EMBEDDING_METADATA_FILE),
        help="Path to embedding metadata JSON file.",
    )
    parser.add_argument(
        "--expected-dim",
        type=int,
        default=384,
        help="Expected vector dimensionality.",
    )
    parser.add_argument(
        "--no-normalize-check",
        action="store_true",
        help="Skip unit L2 normalization validation.",
    )
    parser.add_argument(
        "--skip-sanity-check",
        action="store_true",
        help="Skip semantic sanity check.",
    )

    args = parser.parse_args()

    emb_path = Path(args.embeddings)
    if not emb_path.is_absolute():
        emb_path = (PROJECT_ROOT / emb_path).resolve()

    meta_path = Path(args.metadata)
    if not meta_path.is_absolute():
        meta_path = (PROJECT_ROOT / meta_path).resolve()

    return validate_saved_embeddings(
        embeddings_file=emb_path,
        metadata_file=meta_path,
        expected_dim=args.expected_dim,
        check_normalized=not args.no_normalize_check,
        run_sanity_check=not args.skip_sanity_check,
    )


if __name__ == "__main__":
    sys.exit(main())
