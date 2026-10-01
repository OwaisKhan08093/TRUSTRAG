# TrustRAG

TrustRAG is a solo-developed Confidence-Aware Retrieval-Augmented Generation (RAG) system engineered to ingest trusted documents, retrieve grounded evidence, generate verifiable answers with exact citations, evaluate answer fidelity, compute reliability signals, and abstain whenever supporting evidence is insufficient.

---

## Current Status

**Phase 4 — Neural Cross-Encoder Reranking Engine is COMPLETE.**

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
  * [x] `reciprocal_rank_fusion`: Standalone Reciprocal Rank Fusion ($1 / (k + \text{rank})$) combining candidate rankings.
  * [x] `HybridRetriever`: Unified multi-path orchestrator uniting dense semantic search, sparse lexical search, and RRF.
  * [x] Hybrid configuration settings in `backend/app/config.py` with strict bounds validation.
  * [x] Comparative evaluation suite evaluating Dense, Sparse, and Hybrid performance on local benchmark queries.
* [x] **Phase 4**: Neural Cross-Encoder Reranking Subsystem:
  * [x] `CrossEncoderReranker`: Local Hugging Face cross-encoder model wrapper (`cross-encoder/ms-marco-MiniLM-L-6-v2`).
  * [x] Pair scoring with batch inference, score verification, and order preservation.
  * [x] `RerankedChunk`: Standardized output schema with complete provenance tracking and rank/score history.
  * [x] `ResultReranker`: Neural reranker re-ordering candidate pools with deterministic tie-breaking.
  * [x] `RerankingPipeline`: End-to-end orchestrator connecting HybridRetriever to CrossEncoder scoring.
  * [x] `RerankerConfig` + `validate_reranker_config`: Strict configuration models and bounds validation.
  * [x] Benchmark evaluation script (`scripts/evaluate_reranking.py`) with Hit@1, Hit@3, Hit@5, and MRR metrics.
  * [x] End-to-end integration and edge case coverage.

---

## Architecture

```text
                           User Query
                               │
                ┌──────────────┴──────────────┐
                ▼                             ▼
         Dense Retrieval               Sparse Retrieval
     (SentenceTransformers)              (BM25Okapi)
              +                               +
            FAISS                        Token Index
                │                             │
                └──────────────┬──────────────┘
                               ▼
                        Reciprocal Rank
                          Fusion (RRF)
                               │
                               ▼
                     Candidate Evidence Pool
                         (candidate_k)
                               │
                               ▼
                     Cross-Encoder Reranker
                 (ms-marco-MiniLM-L-6-v2)
                               │
                               ▼
                       Re-ranked Evidence
                           (final_k)
```

### Retrieval & Reranking Paradigm Comparison

| Subsystem | Paradigm | Implementation | Purpose |
|---|---|---|---|
| **Dense Path** | Bi-encoder semantic similarity | `SentenceTransformers` + `FAISS` | Fast retrieval capturing contextual meaning, synonyms, and semantic paraphrasing. |
| **Sparse Path** | Lexical / keyword matching | `rank_bm25` (`BM25Okapi`) | Fast retrieval capturing exact keyword hits, section numbers, acronyms, and terminology. |
| **Fusion Layer** | Multi-retriever combination | Reciprocal Rank Fusion (`RRF`) | Merges heterogeneous score distributions without manual score calibration. |
| **Cross-Encoder Reranker** | Full cross-attention joint scoring | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Deep query-document token interaction for high-precision final ranking of candidates. |

$$\text{RRF Score}(d) = \sum_{r \in \text{retrievers}} \frac{1}{k + \text{rank}(r, d)}$$

---

## Important Architectural Notes & Boundaries

> [!WARNING]
> * **Cross-encoder reranking does NOT generate answers.**
> * **Cross-encoder reranking does NOT prove factual correctness.**
> * **Cross-encoder reranking does NOT synthesize text or prevent hallucinations.**
> * **Cross-encoder reranking does NOT replace verification, citations, or confidence scoring.**
>
> Reranking solely provides re-ordered, high-precision candidate passages as grounded evidence for downstream generation, citation, and verification layers.

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

### 5. Evaluate Neural Reranking Performance (Hybrid vs Hybrid + Reranker)
Run comparative benchmark evaluation evaluating reranking precision:
```bash
.venv\Scripts\python scripts/evaluate_reranking.py
```

### 6. Run Full Test Suite
Execute all unit, component, and integration tests:
```bash
.venv\Scripts\pytest
```

---

## Python API Usage

### End-to-End Reranking Pipeline

```python
from backend.app.reranking import RerankingPipeline

# Initialize end-to-end pipeline (HybridRetriever + CrossEncoder)
pipeline = RerankingPipeline(
    default_candidate_k=5,
    default_final_k=3,
    default_batch_size=16,
    model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
)

results = pipeline.retrieve(
    query="What are the grounds for processing personal data and obligations of notice?",
    top_k=3,
    candidate_k=5,
)

for item in results:
    print(f"Final Rank: {item.final_rank} (was Hybrid Rank: {item.original_rank})")
    print(f"Rerank Score: {item.rerank_score:.4f} | Original Score: {item.original_score:.4f}")
    print(f"Doc: {item.document_name} | Pages: {item.page_start}-{item.page_end}")
    print(f"Text: {item.text[:120]}...\n")
```

### Standalone Component Usage

```python
from backend.app.retrieval import HybridRetriever, VectorRetriever, BM25Retriever
from backend.app.reranking import ResultReranker, CrossEncoderReranker

# 1. Hybrid Retrieval (Dense + Sparse + RRF)
hybrid_retriever = HybridRetriever()
candidates = hybrid_retriever.retrieve(
    query="obligations of data fiduciary",
    top_k=5,
)

# 2. Standalone Neural Reranking
reranker = ResultReranker()
reranked = reranker.rerank(
    query="obligations of data fiduciary",
    results=candidates,
    top_k=3,
)
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
* `DEFAULT_RERANKER_MODEL`: `cross-encoder/ms-marco-MiniLM-L-6-v2`.
* `DEFAULT_RERANKER_TOP_K`: Default final reranked passages (3).
* `DEFAULT_RERANKER_CANDIDATE_K`: Default candidate pool size for reranker (5).
* `DEFAULT_RERANKER_BATCH_SIZE`: Default inference batch size (16).
* `MAX_RERANKER_TOP_K`: Maximum allowed top-K limit (20).

---

## Output Artifacts

* `data/processed/chunks.json`: Processed document chunks with provenance metadata.
* `data/processed/embeddings.npy`: Binary NumPy array of 384-d L2-normalized embeddings.
* `data/processed/embedding_metadata.json`: FAISS integer index to chunk ID mappings.
* `data/processed/index.faiss`: Native serialized FAISS vector index (`IndexFlatIP`).
* `data/evaluation/retrieval_queries.json`: Grounded benchmark queries with target chunks.

---

## Out of Scope (Planned for Subsequent Phases)

The following components are **NOT** implemented in Phase 4 and will be developed in future phases:
* Local LLM Inference (Ollama / vLLM)
* Answer Generation Engine
* Exact Citation & Provenance Generator
* Confidence Scoring Engine
* Answer Fidelity & Groundedness Evaluator
* Abstention Logic & Safety Fallbacks
* Multi-Agent Orchestration
* FastAPI Backend & Interactive UI
