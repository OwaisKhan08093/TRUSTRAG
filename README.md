# TrustRAG

TrustRAG is a solo-developed Confidence-Aware Retrieval-Augmented Generation (RAG) system engineered to ingest trusted documents, retrieve grounded evidence, generate verifiable answers with exact citations, evaluate answer fidelity, compute reliability signals, and abstain whenever supporting evidence is insufficient.

---

## Current Status

**Phase 7 — Multi-Agent TrustRAG Orchestration is COMPLETE.**

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
* [x] **Phase 5**: Trust Engine & Evidence Grounding Subsystem:
  * [x] `TrustEvidence`: Strongly-typed evidence model preserving chunk ID, document provenance, page ranges, retrieval rank, retrieval score, and neural rerank score.
  * [x] `aggregate_evidence_relevance`: Deterministic relevance aggregation mapping cross-encoder logits to calibrated sigmoid probabilities with rank discounting.
  * [x] `calculate_evidence_coverage`: Transparent lexical query term coverage measurement tracking covered vs uncovered terms.
  * [x] `validate_evidence_provenance`: Provenance integrity verification checking for missing metadata, invalid page ranges, duplicate chunk IDs, and empty texts.
  * [x] `calculate_groundedness`: Multi-signal composite groundedness score synthesized from relevance, coverage, and provenance validity.
  * [x] `calculate_confidence`: System confidence index derived from groundedness, top-1 neural relevance, query term coverage, and volume sufficiency.
  * [x] `TrustEngine` & `TrustAssessment`: Core decision engine outputting deterministic trust decisions (`SUPPORTED` vs `INSUFFICIENT_EVIDENCE`) with full explainability.
  * [x] `TrustConfig`: Centralized configuration managing thresholds and weights with strict boundary validation.
  * [x] Evaluation harness (`scripts/evaluate_trust.py`) and end-to-end integration tests (`backend/tests/test_trust_integration.py`).
* [x] **Phase 6**: Grounded LLM Answer Generation & Citations:
  * [x] Local LLM wrapper (`LocalLLM`) supporting `Qwen/Qwen2.5-3B-Instruct` via Hugging Face Transformers with lazy loading and low-memory options.
  * [x] `build_grounded_prompt`: Structured prompt builder enforcing strict evidence constraints and formatted passage provenance.
  * [x] `GenerationRequest` & `GenerationResult`: Structured models with parameter validation, metadata tracking, and explicit refusal flags.
  * [x] `GroundedGenerator`: Evidence-constrained generator strictly gated by `TrustAssessment` (abstains immediately on `INSUFFICIENT_EVIDENCE` without LLM invocation).
  * [x] `build_citation_references` & `Citation`: Provenance-derived citation system with exact document and page mapping (no invented citations).
  * [x] `GroundedAnswer` & `assemble_grounded_answer`: Final response container combining generated prose, citation markers, and preserved trust metrics.
  * [x] Generation safeguards & parameter validation (`test_generation_safeguards.py`).
  * [x] Development evaluation harness (`scripts/evaluate_generation.py`) and end-to-end integration test suite (`backend/tests/test_generation_integration.py`).
* [x] **Phase 7**: Multi-Agent TrustRAG Orchestration:
  * [x] `BaseAgent` & `AgentResult`: Typed foundation base class and standardized execution wrappers with runtime latency tracking.
  * [x] `RetrievalAgent`: Specialized hybrid dense (FAISS) + sparse (BM25) search coordinator.
  * [x] `EvidenceAgent`: Neural cross-encoder evidence reranking and relevance filtering coordinator.
  * [x] `TrustAgent`: Deterministic multi-signal trust gating coordinator issuing `SUPPORTED` vs `INSUFFICIENT_EVIDENCE` decisions.
  * [x] `GenerationAgent`: Grounded answer generator strictly abstaining from LLM execution on insufficient evidence.
  * [x] `CitationAgent`: Provenance validation and structured citation formatting agent.
  * [x] `TrustRAGOrchestrator`: Master coordinator executing deterministic linear pipeline with trust safeguards.
  * [x] `AgentState` & `ExecutionTrace`: Full state tracking and event lineage with explicit `SKIPPED` markers on refusal paths.
  * [x] Multi-agent evaluation suite (`scripts/evaluate_agents.py`) verifying all 5 core scenarios.

---

## Architecture & Orchestration Pipeline

```text
                                  USER QUERY
                                      │
                                      ▼
                               RetrievalAgent
                        (FAISS + BM25Okapi + RRF)
                                      │
                                      ▼
                                EvidenceAgent
                        (Neural Cross-Encoder Rerank)
                                      │
                                      ▼
                                 TrustAgent
                   (Multi-Signal TrustEngine Verification)
                                      │
                      ┌───────────────┴───────────────┐
                      ▼                               ▼
                  SUPPORTED                 INSUFFICIENT_EVIDENCE
                      │                               │
                      ▼                               ▼
               GenerationAgent                   STOP / REFUSAL
             (Local LLM Qwen2.5)             (GenerationAgent SKIPPED)
                      │                      (CitationAgent SKIPPED)
                      ▼                               │
                CitationAgent                         │
             (Verified Provenance)                    │
                      │                               │
                      └───────────────┬───────────────┘
                                      ▼
                           FINAL ORCHESTRATED RESULT
```

---

## Multi-Agent Subsystem Details

### Specialized Agents

| Agent | Responsibility | Subsystem Reused |
|---|---|---|
| **`RetrievalAgent`** | Dense vector search + BM25 keyword search with RRF fusion | `HybridRetriever` |
| **`EvidenceAgent`** | Neural cross-encoder joint query-document cross-attention scoring | `ResultReranker` |
| **`TrustAgent`** | Deterministic evidence grounding, relevance, coverage, and provenance verification | `TrustEngine` |
| **`GenerationAgent`** | Evidence-constrained generation with strict refusal safeguards | `GroundedGenerator` |
| **`CitationAgent`** | Provenance verification and Markdown reference compilation | `Citation` / `GroundedAnswer` |
| **`TrustRAGOrchestrator`** | Master pipeline coordinator enforcing gating and event lineage | Full Agent Suite |

### Trust Safeguard Invariant
When `TrustAgent` determines evidence is `INSUFFICIENT_EVIDENCE`:
1. The pipeline terminates downstream synthesis immediately.
2. `GenerationAgent` is **NEVER** called with the LLM.
3. `CitationAgent` is **SKIPPED**.
4. The structured `ExecutionTrace` records `GenerationAgent: SKIPPED` and `CitationAgent: SKIPPED`.
5. A structured transparent refusal notification is returned.

---

## Subsystem Functionality Overview

### What TrustRAG Currently Does:
1. **Document Ingestion**: Extracts and cleans PDF text while preserving document and page numbers.
2. **Dense Semantic Retrieval**: Vector search over 384-dimensional embeddings via FAISS.
3. **Sparse Lexical Retrieval**: BM25 keyword matching with tokenization and invariant checks.
4. **Hybrid Rank Fusion**: Combines rankings using Reciprocal Rank Fusion ($k=60$).
5. **Neural Cross-Encoder Reranking**: Re-orders top passages via `ms-marco-MiniLM-L-6-v2`.
6. **Provenance & Quality Verification**: Enforces 1-based page numbers, monotonic page ranges, ID uniqueness, and metadata completeness.
7. **Deterministic Groundedness & Confidence**: Multi-signal heuristic scoring based on calibrated sigmoid probabilities, lexical coverage, and volume saturation.
8. **Hard Trust Gating**: Emits discrete verdicts (`SUPPORTED` vs `INSUFFICIENT_EVIDENCE`).
9. **Constrained LLM Generation**: Prompts `Qwen/Qwen2.5-3B-Instruct` using only verified evidence passages. If evidence is `INSUFFICIENT_EVIDENCE`, the LLM is **never called** and an abstention response is returned.
10. **Provenance-Derived Citations**: Builds citation references directly mapped to source document pages (e.g. `[1] DPDP_Act_2023.pdf, page 5`).
11. **Multi-Agent Orchestration**: End-to-end specialized agents with complete `AgentState` management and observable `ExecutionTrace`.

### What TrustRAG Does NOT Do Yet:
* **REST API & Web UI**: FastAPI endpoints and web frontend interfaces are reserved for subsequent phases.
* **Autonomous Multi-Turn Loops**: Agents follow deterministic, predictable directed acyclic pipeline flow rather than uncontrolled autonomous loops.

---

## Important Disclaimers

> [!IMPORTANT]
> * **TrustRAG does not guarantee factual correctness.**
> * **The TrustEngine is a deterministic evidence-based gating layer.**
> * **The evaluation dataset is a development dataset, not a statistically validated benchmark.**

---

## Hardware & Model Details

* **LLM**: `Qwen/Qwen2.5-3B-Instruct` (Hugging Face Transformers, local inference only).
* **Cross-Encoder**: `cross-encoder/ms-marco-MiniLM-L-6-v2`.
* **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2`.
* **RAM / Memory Conscious**: Weights are loaded lazily upon first generation request; unit tests utilize stubs and mocks to maintain sub-second execution without downloading multi-gigabyte models.
* **Decoding**: Greedy decoding (`temperature=0.0`) by default to maximize deterministic, factual grounding.

---

## Installation

1. Clone the repository:
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

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage & Commands

### 1. Ingest PDF Document
```bash
.venv\Scripts\python scripts/ingest.py data/raw/DPDP_Act_2023.pdf
```

### 2. Generate Document Embeddings
```bash
.venv\Scripts\python scripts/embed_chunks.py
```

### 3. Build FAISS Vector Index
```bash
.venv\Scripts\python scripts/build_faiss_index.py
```

### 4. Evaluate Retrieval Performance
```bash
.venv\Scripts\python scripts/evaluate_retrieval.py
```

### 5. Evaluate Neural Reranking Performance
```bash
.venv\Scripts\python scripts/evaluate_reranking.py
```

### 6. Evaluate Trust Engine Grounding
```bash
.venv\Scripts\python scripts/evaluate_trust.py
```

### 7. Evaluate Grounded Generation & Citations
```bash
.venv\Scripts\python scripts/evaluate_generation.py
```

### 8. Evaluate Multi-Agent Orchestration
```bash
.venv\Scripts\python scripts/evaluate_agents.py
```

### 9. Run Full Test Suite
```bash
.venv\Scripts\pytest
```

---

## Python API Usage

### End-to-End Multi-Agent Orchestration

```python
from backend.app.agents import TrustRAGOrchestrator, TrustDecision

# 1. Initialize Master Orchestrator
orchestrator = TrustRAGOrchestrator()

query = "What notice must a Data Fiduciary give to a Data Principal before collecting personal data?"

# 2. Execute Full Multi-Agent Pipeline
result = orchestrator.execute(query)

# 3. Inspect Response and Execution Trace
print(result.formatted_response)
print(f"Decision: {result.decision.value}")
print(f"Confidence Score: {result.confidence_score:.3f}")
print(f"Latency: {result.latency_seconds:.3f}s")

if result.trace:
    print(result.trace.format_trace_summary())
```

---

## Configuration Reference

Key configuration constants:

* `DEFAULT_GENERATION_MODEL`: `Qwen/Qwen2.5-3B-Instruct`.
* `DEFAULT_MAX_NEW_TOKENS`: `512`.
* `DEFAULT_TEMPERATURE`: `0.0` (greedy decoding).
* `DEFAULT_TOP_P`: `0.9`.
* `DEFAULT_REPETITION_PENALTY`: `1.1`.
* `TrustConfig.min_confidence_threshold`: `0.50`.
* `TrustConfig.min_groundedness_threshold`: `0.45`.
* `TrustConfig.min_coverage_threshold`: `0.30`.
* `TrustConfig.require_valid_provenance`: `True`.

---

## Out of Scope (Planned for Subsequent Phases)

* FastAPI backend REST service (Phase 8)
* React frontend user interface
* Authentication and multi-user sessions
* Cloud APIs and remote deployment
