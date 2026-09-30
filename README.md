# TrustRAG

TrustRAG is a solo-developed Confidence-Aware Retrieval-Augmented Generation (RAG) system engineered to ingest trusted documents, retrieve grounded evidence, generate verifiable answers with exact citations, evaluate answer fidelity, compute reliability signals, and abstain whenever supporting evidence is insufficient.

---

## Current Status

**Phase 3 — Hybrid Retrieval Engine (Dense FAISS + Sparse BM25 + Reciprocal Rank Fusion) is COMPLETE.**

* [x] **Phase 0 & 1**: Document ingestion (PyMuPDF), legal cleaning, sliding-window chunking, metadata persistence.
* [x] **Phase 2**: Dense semantic vector retrieval engine:
  * [x] SentenceTransformer embedding model wrapper (`sentence-transformers/all-MiniLM-L6-v2`, 384 dimensions).
  * [x] Document chunk embedding generation with L2 normalization.
  * [x] FAISS IndexFlatIP vector index with native persistence (`index.faiss`).
  * [x] Fast query embedding and nearest-neighbor search.
  * [x] Chunk metadata resolution and provenance mapping.
  * [x] `VectorRetriever` pipeline with score thresholding.
* [x] **Phase 3**: Hybrid Retrieval Subsystem:
  * [x] `BM25Index`: Sparse lexical index built on `rank_bm25` with tokenizer and invariant validation.
  * [x] `BM25Retriever`: Independent sparse keyword retriever returning standard structured chunk schemas.
  * [x] `reciprocal_rank_fusion`: Standalone Reciprocal Rank Fusion ($1 / (k + \text{rank})$) combining candidate rankings with tie-breaking and deduplication.
  * [x] `HybridRetriever`: Unified multi-path orchestrator uniting dense semantic search, sparse lexical search, and RRF.
  * [x] Hybrid configuration settings in `backend/app/config.py` with strict bounds validation.
  * [x] Comparative evaluation suite evaluating Dense, Sparse, and Hybrid performance on local benchmark queries.
  * [x] End-to-end integration test coverage.

---

## Architecture

```text
                    User Query
                        │
             ┌──────────┴──────────┐
             ▼                     ▼
       Dense Retrieval        Sparse Retrieval
   (SentenceTransformers)       (BM25Okapi)
            +                     +
          FAISS              Token Index
             │                     │
             └──────────┬──────────┘
                        ▼
                Reciprocal Rank
                  Fusion (RRF)
                        │
                        ▼
                 Hybrid Results
                        │
                        ▼
                 Ranked Evidence
```

### Retrieval Paradigm Comparison

| Subsystem | Paradigm | Implementation | Purpose |
|---|---|---|---|
| **Dense Path** | Semantic similarity | `SentenceTransformers` + `FAISS` | Captures contextual meaning, synonyms, and semantic paraphrasing. |
| **Sparse Path** | Lexical / keyword matching | `rank_bm25` (`BM25Okapi`) | Captures exact keyword hits, legal section numbers, acronyms, and rare terminology. |
| **Fusion Layer** | Multi-retriever combination | Reciprocal Rank Fusion (`RRF`) | Merges heterogeneous score distributions without manual score calibration or weighting. |

$$\text{RRF Score}(d) = \sum_{r \in \text{retrievers}} \frac{1}{k + \text{rank}(r, d)}$$

---

## Important Architectural Notes & Boundaries

> [!WARNING]
> * **Hybrid retrieval does NOT generate answers.**
> * **Hybrid retrieval does NOT prove factual correctness.**
> * **Hybrid retrieval does NOT prevent hallucinations.**
>
> Retrieval solely provides ranked candidate passages as grounded evidence for downstream generation and verification layers.

---

## Installation

1. Clone the repository and navigate to the project root:
   ```bash
   git clone https://github.com/OwaisKhan08093/TRUSTRAG.git
   cd TRUSTRAG
   ```

2. Create and activate a Python 3.11 virtual environment:
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

---

## Usage & Commands

### 1. Ingest PDF Document
Extract, clean, and chunk raw documents:
```bash
.venv\Scripts\python scripts/ingest.py data/raw/DPDP_Act_2023.pdf
```

### 2. Generate Document Embeddings
Encode chunk texts into 384-dimensional dense vectors:
```bash
.venv\Scripts\python scripts/embed_chunks.py
```

### 3. Build FAISS Vector Index
Construct and serialize binary FAISS vector index:
```bash
.venv\Scripts\python scripts/build_faiss_index.py
```

### 4. Evaluate Retrieval Performance (Dense vs Sparse vs Hybrid)
Run comparative benchmark evaluation across all retrieval paths:
```bash
.venv\Scripts\python scripts/evaluate_retrieval.py
```

### 5. Run Full Test Suite
Execute all unit, component, and integration tests:
```bash
.venv\Scripts\pytest
```

---

## Python API Usage

```python
from backend.app.retrieval import HybridRetriever, VectorRetriever, BM25Retriever

# 1. Unified Hybrid Retrieval (Dense + Sparse + RRF)
hybrid_retriever = HybridRetriever()
results = hybrid_retriever.retrieve(
    query="obligations of data fiduciary and notice requirements",
    top_k=5,
    dense_top_k=5,
    sparse_top_k=5,
    rrf_k=60,
)

for item in results:
    print(f"Rank {item['rank']} | RRF Score: {item['score']:.4f} | Chunk: {item['chunk_id']}")
    print(f"Text: {item['text'][:120]}...\n")

# 2. Independent Dense Retrieval
dense_retriever = VectorRetriever()
dense_results = dense_retriever.retrieve("rights of data principal", top_k=3)

# 3. Independent Sparse BM25 Retrieval
sparse_retriever = BM25Retriever()
sparse_results = sparse_retriever.retrieve("grievance redressal", top_k=3)
```

---

## Configuration Reference

Key constants configured in `backend/app/config.py`:

* `DEFAULT_CHUNK_SIZE_WORDS`: Target chunk size (600 words).
* `DEFAULT_CHUNK_OVERLAP_WORDS`: Overlap window (120 words).
* `DEFAULT_EMBEDDING_MODEL`: `sentence-transformers/all-MiniLM-L6-v2` (384-d).
* `FAISS_INDEX_FILE`: Path to `data/processed/index.faiss`.
* `DEFAULT_DENSE_TOP_K`: Default dense candidates (5).
* `DEFAULT_SPARSE_TOP_K`: Default sparse candidates (5).
* `DEFAULT_FINAL_TOP_K`: Default final fused chunks (5).
* `DEFAULT_RRF_K`: RRF smoothing constant (60, bounds: `1` to `1000`).

---

## Output Artifacts

* `data/processed/chunks.json`: Processed document chunks with provenance metadata.
* `data/processed/embeddings.npy`: Binary NumPy array of 384-d L2-normalized embeddings.
* `data/processed/embedding_metadata.json`: FAISS integer index to chunk ID mappings.
* `data/processed/index.faiss`: Native serialized FAISS vector index (`IndexFlatIP`).
* `data/evaluation/retrieval_queries.json`: Grounded benchmark queries with target chunks.

---

## Out of Scope (Planned for Subsequent Phases)

The following components are **NOT** implemented in Phase 3 and will be developed in future phases:
* Cross-Encoder Reranking
* Local LLM Inference (Ollama / vLLM)
* Answer Generation Engine
* Exact Citation & Provenance Generator
* Confidence Scoring Engine
* Answer Fidelity & Groundedness Evaluator
* Abstention Logic & Safety Fallbacks
* Multi-Agent Orchestration
* FastAPI Backend & Interactive UI
