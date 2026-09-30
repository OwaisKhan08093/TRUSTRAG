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

# Ensure necessary directories exist
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
