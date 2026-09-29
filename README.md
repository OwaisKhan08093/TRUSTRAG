# TrustRAG

TrustRAG is a solo-developed Confidence-Aware Retrieval-Augmented Generation (RAG) system engineered to ingest trusted documents, retrieve grounded evidence, generate verifiable answers with exact citations, evaluate answer fidelity, compute reliability signals, and abstain whenever supporting evidence is insufficient.

## Current Status

**Phase 2 — Complete Semantic Vector Retrieval Engine implemented.**

* [x] **Step 0**: Repository structure, configuration, test suite, and virtual environment setup.
* [x] **Step 1**: Robust PyMuPDF PDF ingestion, conservative legal text cleaning, deterministic sliding-window chunking, and metadata-preserving JSON persistence.
* [x] **Step 2.1**: Embedding model foundation with `EmbeddingEncoder` wrapper (`sentence-transformers/all-MiniLM-L6-v2`, 384 dimensions).
* [x] **Step 2.2**: Local document chunk embedding generation with batching and L2 normalization.
* [x] **Step 2.3**: Embedding validation suite and mathematical cosine-similarity sanity checks.
* [x] **Phase 2 Retreival Engine**:
  * [x] **FAISS Index Foundation & Persistence**: `FaissVectorIndex` (`IndexFlatIP`) with native `.faiss` serialization.
  * [x] **FAISS Index Builder**: CLI script `scripts/build_faiss_index.py`.
  * [x] **Query Embedding**: Dedicated `encode_query` function producing unit-normalized 384-d vectors.
  * [x] **FAISS Similarity Search**: Top-K nearest neighbor search with descending score ranking.
  * [x] **Chunk Metadata Resolver**: High-speed lookup `ChunkMetadataResolver` with schema and reference validation.
  * [x] **Semantic Vector Retriever**: High-level `VectorRetriever` orchestrating query embedding, FAISS search, and metadata resolution.
  * [x] **Retrieval Configuration & Filtering**: Configurable `top_k`, `max_top_k`, and `score_threshold` filtering.
  * [x] **Retrieval Evaluation**: Benchmarking utility `scripts/evaluate_retrieval.py` calculating Hit@1, Hit@3, Hit@5, and MRR.

## Complete Retrieval Pipeline

```
PDF Document
 ↓
PyMuPDF Text Extraction
 ↓
Conservative Text Cleaning (Preserving legal structure & numbering)
 ↓
Deterministic Word-level Chunking (data/processed/chunks.json)
 ↓
Local SentenceTransformers Encoding (384-d, L2-normalized)
 ↓
Vector & Metadata Artifacts (embeddings.npy & embedding_metadata.json)
 ↓
FAISS Vector Index (data/processed/index.faiss)
 ↓
User Query → Query Embedding (384-d)
 ↓
FAISS Similarity Search (IndexFlatIP / Cosine Inner Product)
 ↓
Top-K Vector Indices & Similarity Scores
 ↓
Chunk Metadata Resolution
 ↓
Ranked Structured Retrieval Response
```

## Installation

1. Clone the repository and navigate to the project root:
   ```bash
   cd TrustRAG
   ```

2. Create and activate a Python 3.11+ virtual environment:
   ```bash
   # Windows
   py -3.11 -m venv .venv
   .venv\Scripts\activate

   # Linux / macOS
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 1. Ingest PDF Document
Extract, clean, and chunk raw documents:

```bash
.venv\Scripts\python scripts/ingest.py data/raw/DPDP_Act_2023.pdf
```

### 2. Generate Document Embeddings
Encode chunk text into 384-dimensional dense vectors:

```bash
.venv\Scripts\python scripts/embed_chunks.py
```

### 3. Build FAISS Vector Index
Construct and save native FAISS index from generated embeddings:

```bash
.venv\Scripts\python scripts/build_faiss_index.py
```

### 4. Validate Embeddings Pipeline
Verify the integrity of saved embeddings, invariants, and metadata mapping:

```bash
.venv\Scripts\python scripts/validate_embeddings.py
```

### 5. Evaluate Retrieval Performance
Measure Hit@1, Hit@3, Hit@5, and MRR metrics on evaluation benchmark queries:

```bash
.venv\Scripts\python scripts/evaluate_retrieval.py
```

### 6. Run Automated Test Suite
Execute the full pytest suite:

```bash
.venv\Scripts\pytest
```

## Output Artifacts

* **`data/processed/chunks.json`**: Preserves full document provenance and page attribution:
  ```json
  [
    {
      "chunk_id": "DPDP_Act_2023_page1_chunk1",
      "document_id": "DPDP_Act_2023",
      "document_name": "DPDP_Act_2023.pdf",
      "page_start": 1,
      "page_end": 1,
      "text": "THE DIGITAL PERSONAL DATA PROTECTION ACT, 2023..."
    }
  ]
  ```

* **`data/processed/embeddings.npy`**: NumPy binary array containing 384-dimensional, L2-normalized dense embeddings of shape `(number_of_chunks, 384)`.

* **`data/processed/embedding_metadata.json`**: Lightweight index-to-chunk provenance mapping:
  ```json
  [
    {
      "embedding_index": 0,
      "chunk_id": "DPDP_Act_2023_page1_chunk1"
    }
  ]
  ```

* **`data/processed/index.faiss`**: Serialized binary FAISS `IndexFlatIP` vector index.

* **`data/evaluation/retrieval_queries.json`**: Evaluation dataset with grounded benchmark queries and expected chunk IDs.

## Next Phase

**Upcoming Phase: Hybrid Retrieval (BM25 + RRF Reciprocal Rank Fusion) and Reranking.**


