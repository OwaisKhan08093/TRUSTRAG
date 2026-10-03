# TrustRAG — Trustworthy Retrieval-Augmented Generation

TrustRAG is a production-grade, confidence-aware Retrieval-Augmented Generation (RAG) system engineered to ingest trusted PDF documents, retrieve grounded evidence through hybrid search, re-rank passages using neural cross-encoders, evaluate factual support deterministically via a 4-dimensional Trust Engine, generate verifiable answers with exact page citations, and abstain whenever supporting evidence is insufficient.

---

## Architecture Overview

```
                                  USER INTERFACE
                          (React + TypeScript + Vite)
                                       │
                         Document      │       Question
                         Upload        │       Query
                            │          ▼
                            │    FASTAPI BACKEND
                            │  (Routes: /query, /documents, /health)
                            │          │
         ┌──────────────────┘          ▼
         ▼                    TRUSTRAG ORCHESTRATOR
 ┌────────────────┐                    │
 │ DOCUMENT       │        ┌───────────┼───────────┐
 │ INGESTION      │        │           │           │
 │ SERVICE        │        ▼           ▼           ▼
 │                │   RETRIEVAL    EVIDENCE      TRUST
 │ 1. PDF Parser  │     AGENT        AGENT       AGENT
 │ 2. Cleaner     │   (FAISS+BM25)  (CrossEnc)  (4-Signal)
 │ 3. Chunker     │        │           │           │
 │ 4. Embeddings  │        └───────────┼───────────┘
 │ 5. Index Build │                    │
 └────────────────┘              TRUST DECISION
                                 /            \
                                /              \
                         SUPPORTED        INSUFFICIENT
                             │                  │
                             ▼                  ▼
                         GENERATION     STRUCTURED ABSTENTION
                           AGENT         (Gated without LLM)
                        (Local LLM)
                             │
                             ▼
                       CITATION AGENT
                   (Exact Page & Snippets)
                             │
                             ▼
                     VERIFIABLE ANSWER
```

---

## Core System Capabilities

### 1. Document Upload & Ingestion Pipeline
- **Multipart PDF Upload (`POST /documents`)**: Accepts custom PDF documents with file type verification and size limits (up to 25MB).
- **Page-Preserving Chunker**: Sliding-window chunking (600 words, 120-word overlap) preserving exact page boundaries and document IDs.
- **Dense Vector & Sparse Lexical Indexing**: Automatically updates dense FAISS embeddings (`sentence-transformers/all-MiniLM-L6-v2`) and sparse BM25 inverted indices upon ingestion.
- **Instant Hot-Reload**: Re-indexes in-memory retrievers dynamically so newly uploaded documents are immediately queryable.

### 2. Hybrid Retrieval Subsystem
- **Dense Semantic Retrieval**: FAISS `IndexFlatIP` vector index with L2-normalized embeddings for conceptual similarity.
- **Sparse Lexical Retrieval**: BM25Okapi scoring for exact legal citations and keyword matching.
- **Reciprocal Rank Fusion (RRF)**: Merges rank positions using $RRF(d) = \sum \frac{1}{k + r(d)}$, outperforming individual retrieval methods.

### 3. Neural Cross-Encoder Reranking
- **High-Precision Relevance Scoring**: Evaluates query-document pairs using `cross-encoder/ms-marco-MiniLM-L-6-v2`.
- **Calibrated Logits**: Converts raw transformer logits to calibrated sigmoid probabilities.

### 4. Deterministic Trust Engine
Evaluates retrieved evidence across four orthogonal safety dimensions:
- **Relevance Score**: Sigmoid probability of neural reranker scores with rank discounting.
- **Coverage Score**: Lexical query term representation in candidate evidence.
- **Provenance Validity**: Verification of document IDs, non-empty chunks, and valid page ranges ($1 \le \text{page\_start} \le \text{page\_end}$).
- **Groundedness & Confidence**: Composite score gating the generation stage. If confidence $< 0.50$ or groundedness $< 0.45$, the query is classified as `INSUFFICIENT_EVIDENCE`.

### 5. Grounded Generation & Citation Agent
- **Gated Generation**: Local LLM (`Qwen/Qwen2.5-3B-Instruct`) is invoked **only** when evidence is verified as `SUPPORTED`.
- **Zero Hallucination Guarantee**: If evidence is insufficient, generation is skipped and a structured refusal is returned in < 250ms.
- **Verifiable Citations**: Every citation marker (e.g. `[1]`) maps directly to source document name, page number, and evidence snippet.

---

## Quick Start & UI Demo Walkthrough

### 1. Start the FastAPI Backend
```powershell
# In repository root:
.\.venv\Scripts\Activate.ps1
python -m uvicorn backend.app.api.app:app --host 127.0.0.1 --port 8000
```
- **Backend API**: `http://127.0.0.1:8000`
- **Interactive Swagger Docs**: `http://127.0.0.1:8000/docs`

### 2. Start the React Frontend
```powershell
# In frontend directory:
cd frontend
npm run dev
```
- **Frontend Web UI**: `http://127.0.0.1:5173`

---

## Live UI Demonstration Flow

Open `http://127.0.0.1:5173` in your browser:

### Step 1: Ingest Documents
1. In the **"Ingest Documents"** section, drag & drop any PDF document (or browse from disk).
2. The UI displays the upload status:
   `✓ Document "DPDP_Act_2023.pdf" processed successfully (4 chunks created). Ready for questions!`
3. The catalog chip displays the document name, chunk count, and green `✓ Ready` badge.

### Step 2: Ask a Supported Question
1. Click the sample question:
   `"What notice must a Data Fiduciary give before requesting consent?"`
2. Click **"Ask TrustRAG"**.
3. Observe:
   - **Pipeline Flow**: Shows all 5 stages active (`RETRIEVAL` → `EVIDENCE` → `TRUST` → `GENERATION` → `CITATIONS`).
   - **Trust Decision**: Green `SUPPORTED` badge with high Confidence (94%), Groundedness (97%), Relevance, and Coverage.
   - **Grounded Answer**: Direct factual prose referencing `[1]`.
   - **Evidence & Provenance Panel**: Shows document name, page numbers, and preview snippet.
   - **Interactive Highlighting**: Clicking citation marker `[1]` highlights the backing evidence card.

### Step 3: Test Abstention & Safety Gating
1. Enter an unsupported or out-of-domain query:
   `"What is the best recipe for baking chocolate cake?"`
2. Click **"Ask TrustRAG"**.
3. Observe:
   - **Trust Decision**: Amber `INSUFFICIENT EVIDENCE` badge.
   - **Grounded Refusal**: Structured refusal statement explaining that the indexed documents lack factual support.
   - **Pipeline Visualization**: Shows `RETRIEVAL` (Completed), `EVIDENCE` (Completed), `TRUST` (Completed), while `GENERATION` and `CITATIONS` are explicitly marked as **`Gated/Skipped`**.
   - **Execution Latency**: Responds in < 280ms without invoking the heavy LLM.

---

## Production Benchmarks & Profiling

Detailed performance and resource profiling results are available in the [`docs/`](docs/) directory:

- [**Benchmark Latency Report (`docs/BENCHMARK_REPORT.md`)**](docs/BENCHMARK_REPORT.md)
  - Hybrid Retrieval Latency: **~13.7 ms**
  - Neural Reranking Latency: **~218.4 ms**
  - Trust Engine Evaluation Latency: **~0.5 ms**
  - Citation Assembly Latency: **~0.03 ms**
  - Warm API Latency (Refusal): **~209 ms**
- [**Resource & Memory Profile (`docs/RESOURCE_PROFILE.md`)**](docs/RESOURCE_PROFILE.md)
  - Lazy model loading verified (0 MB allocated before first generation)
  - Clean model unloading via `unload_model()`
  - Stable memory footprint over repeated query cycles (< 1 MB net drift across 15 full queries)
  - Frontend production distribution bundle: **~261 KB**
- [**Final Evaluation Report (`docs/EVALUATION_REPORT.md`)**](docs/EVALUATION_REPORT.md)
  - **100.0%** overall decision accuracy across 6 test categories (Supported, Partial, Unsupported, Multi-Doc, Corrupt Provenance, Adversarial)
  - **100.0%** refusal precision on unsupported and misleading queries
- [**Deployment Guide (`docs/DEPLOYMENT.md`)**](docs/DEPLOYMENT.md)
  - Docker & Docker-Compose multi-stage containers
  - CPU vs GPU hardware recommendations

---

## Running the Complete Test Suite

### Backend Test Suite (383 tests):
```powershell
.\.venv\Scripts\pytest.exe -v
```

### Frontend Test Suite (21 tests):
```powershell
npm --prefix frontend test
```

### Production Build:
```powershell
npm --prefix frontend run build
```

**Total Active Tests Passing**: **404 / 404 tests (100% pass rate)**.
