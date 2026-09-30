"""Configuration settings for TrustRAG."""

from pathlib import Path

# Base Paths (relative to workspace root)
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
CHUNKS_OUTPUT_FILE = PROCESSED_DATA_DIR / "chunks.json"

# Ingestion & Chunking Defaults
DEFAULT_CHUNK_SIZE_WORDS = 600       # Target chunk size: 500-800 words
DEFAULT_CHUNK_OVERLAP_WORDS = 120    # Target chunk overlap: 100-150 words
MIN_CHUNK_SIZE_WORDS = 20            # Minimum words to form a valid chunk

# Embedding Defaults
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_EMBEDDING_BATCH_SIZE = 32
EMBEDDINGS_OUTPUT_FILE = PROCESSED_DATA_DIR / "embeddings.npy"
EMBEDDING_METADATA_FILE = PROCESSED_DATA_DIR / "embedding_metadata.json"

# Retrieval Defaults
FAISS_INDEX_FILE = PROCESSED_DATA_DIR / "index.faiss"
DEFAULT_TOP_K = 5
MAX_TOP_K = 20
DEFAULT_SIMILARITY_THRESHOLD = 0.0

# Hybrid Retrieval & Fusion Defaults
DEFAULT_DENSE_TOP_K = 5
DEFAULT_SPARSE_TOP_K = 5
DEFAULT_FINAL_TOP_K = 5
DEFAULT_RRF_K = 60
MIN_RRF_K = 1
MAX_RRF_K = 1000

# Cross-Encoder Reranker Defaults
DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_RERANKER_TOP_K = 3
DEFAULT_RERANKER_CANDIDATE_K = 5
DEFAULT_RERANKER_BATCH_SIZE = 16
MAX_RERANKER_TOP_K = 20
MIN_RERANKER_TOP_K = 1


def validate_reranker_config(
    candidate_k: int = DEFAULT_RERANKER_CANDIDATE_K,
    final_k: int = DEFAULT_RERANKER_TOP_K,
    batch_size: int = DEFAULT_RERANKER_BATCH_SIZE,
    model_name: str = DEFAULT_RERANKER_MODEL,
    max_k: int = MAX_RERANKER_TOP_K,
) -> None:
    """Validate cross-encoder reranker configuration parameters.

    Raises:
        TypeError: If types are invalid.
        ValueError: If bounds or invariants are violated.
    """
    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError("model_name must be a non-empty string.")

    if not isinstance(candidate_k, int) or candidate_k < 1:
        raise ValueError(f"candidate_k must be a positive integer >= 1, got {candidate_k}.")

    if not isinstance(final_k, int) or final_k < 1:
        raise ValueError(f"final_k must be a positive integer >= 1, got {final_k}.")

    if final_k > candidate_k:
        raise ValueError(f"final_k ({final_k}) cannot exceed candidate_k ({candidate_k}).")

    if candidate_k > max_k or final_k > max_k:
        raise ValueError(f"k values cannot exceed maximum limit of {max_k}.")

    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError(f"batch_size must be a positive integer >= 1, got {batch_size}.")


# Ensure necessary directories exist
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
