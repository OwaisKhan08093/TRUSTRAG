# TrustRAG

TrustRAG is a solo-developed Confidence-Aware Retrieval-Augmented Generation (RAG) system engineered to ingest trusted documents, retrieve grounded evidence, generate verifiable answers with exact citations, evaluate answer fidelity, compute reliability signals, and abstain whenever supporting evidence is insufficient.

## Current Status

**Step 2.3 — Embedding validation completed.**

* [x] **Step 0**: Repository structure, configuration, test suite, and virtual environment setup.
* [x] **Step 1**: Robust PyMuPDF PDF ingestion, conservative legal text cleaning, deterministic sliding-window chunking, and metadata-preserving JSON persistence.
* [x] **Step 2.1**: Embedding model foundation with `EmbeddingEncoder` wrapper.
  * Model: `sentence-transformers/all-MiniLM-L6-v2` (dimension: 384).
* [x] **Step 2.2**: Local document chunk embedding generation with batching and L2 normalization.
* [x] **Step 2.3**: Embedding validation suite and mathematical cosine-similarity sanity checks.
* [ ] **Step 2.4 (Upcoming)**: FAISS vector indexing & similarity search.

## Current Pipeline

```
PDF
 ↓
PyMuPDF Text Extraction
 ↓
Conservative Text Cleaning (Preserving legal structure & numbering)
 ↓
Deterministic Word-level Chunking with Overlap & Page Attribution
 ↓
Metadata-Preserving JSON (data/processed/chunks.json)
 ↓
Local SentenceTransformers Encoding (384-d, L2-normalized)
 ↓
Vector Artifacts (data/processed/embeddings.npy & embedding_metadata.json)
 ↓
Embedding Validation & Alignment Checks (Dimensionality, Norms, Sanitized Cosine)
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
Place your raw PDF documents into `data/raw/` and execute the ingestion script:

```bash
python scripts/ingest.py data/raw/DPDP_Act_2023.pdf
```

You can customize chunk size and overlap parameters if needed:
```bash
python scripts/ingest.py data/raw/DPDP_Act_2023.pdf --chunk-size 600 --chunk-overlap 120
```

### 2. Generate Document Embeddings
Encode chunk text into 384-dimensional dense embeddings locally:

```bash
python scripts/embed_chunks.py
```

### 3. Validate Embeddings Pipeline
Verify the integrity of saved embeddings and metadata mapping:

```bash
python scripts/validate_embeddings.py
```

The validation suite rigorously verifies:
* **Dimensionality**: Strictly 2D matrix matching expected dimensions (384).
* **Finite Values**: Verifies all values are finite real numbers.
* **NaN / Inf Detection**: Ensures zero NaN and infinite entries.
* **Zero-Vector Detection**: Rejects empty/zero norm vectors.
* **L2 Normalization**: Confirms vectors have unit norm ($||v|| \approx 1.0$).
* **Metadata Alignment**: Verifies 1-to-1 correspondence between vector rows and chunk metadata entries.
* **Semantic Sanity Check**: Mathematical cosine similarity check comparing similar and distinct concept pairs.

### 4. Run Automated Tests
Execute the full test suite across ingestion, chunking, embeddings, and validation:

```bash
pytest
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

## Next Step

**Next milestone: FAISS vector indexing & similarity search.**

