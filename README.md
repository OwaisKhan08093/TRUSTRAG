# TrustRAG

TrustRAG is a solo-developed Confidence-Aware Retrieval-Augmented Generation (RAG) system engineered to ingest trusted documents, retrieve grounded evidence, generate verifiable answers with exact citations, evaluate answer fidelity, compute reliability signals, and abstain whenever supporting evidence is insufficient.

---

## Current Status

**Phase 5 — Trust Engine & Evidence Grounding is COMPLETE.**

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
  * [x] `calculate_confidence`: System confidence index derived from groundedness, top-1 relevance strength, query coverage, volume sufficiency, and provenance.
  * [x] `TrustEngine` & `TrustAssessment`: Core decision engine outputting deterministic trust decisions (`SUPPORTED` vs `INSUFFICIENT_EVIDENCE`) with full explainability.
  * [x] `TrustConfig`: Centralized configuration managing thresholds and weights with strict boundary validation.
  * [x] Evaluation harness (`scripts/evaluate_trust.py`) and end-to-end integration tests (`backend/tests/test_trust_integration.py`).

---

## Architecture

```text
                                USER QUERY
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
                                    │
                                    ▼
                          Cross-Encoder Reranker
                      (ms-marco-MiniLM-L-6-v2)
                                    │
                                    ▼
                             Ranked Evidence
                                    │
              ┌─────────────────────┼─────────────────────┐
              ▼                     ▼                     ▼
          Relevance              Coverage             Provenance
         Aggregation            Measurement           Validation
              │                     │                     │
              └─────────────────────┼─────────────────────┘
                                    ▼
                               Groundedness
                                    │
                                    ▼
                                Confidence
                                    │
                                    ▼
                               TrustEngine
                                    │
                      ┌─────────────┴─────────────┐
                      ▼                           ▼
                  SUPPORTED              INSUFFICIENT_EVIDENCE
```

---

## Subsystem Functionality Overview

### What TrustRAG Currently Does:
1. **Document Processing**: Ingests, normalizes, and splits PDF documents into semantically coherent overlapping chunks.
2. **Dense Semantic Retrieval**: Embeds chunks and performs nearest-neighbor vector search in FAISS.
3. **Sparse Lexical Retrieval**: Indexes chunks and performs BM25 keyword searches with stopword tokenization.
4. **Hybrid Rank Fusion**: Fuses diverse retrieval rankings using Reciprocal Rank Fusion ($k=60$).
5. **Neural Cross-Encoder Reranking**: Re-scores top hybrid candidates using deep query-passage cross-attention (`ms-marco-MiniLM-L-6-v2`).
6. **Evidence Quality Validation**: Enforces strongly typed schemas, finite numerical bounds, and non-empty metadata across evidence items.
7. **Provenance Integrity Checking**: Validates 1-based page numbers, monotonic page ranges, non-empty identifiers, and absence of duplicates.
8. **Deterministic Evidence Grounding**: Measures lexical query term coverage and sigmoid-calibrated neural relevance.
9. **Confidence Index Estimation**: Computes bounded $[0.0, 1.0]$ confidence index based on evidence groundedness, top-1 neural relevance, query term coverage, and volume sufficiency.
10. **Trust Decision Making**: Emits auditable discrete verdicts (`SUPPORTED` vs `INSUFFICIENT_EVIDENCE`) before answer synthesis.

### What TrustRAG Does NOT Do Yet:
* **LLM Answer Generation**: No generative LLM (Ollama / vLLM / OpenAI) is integrated yet.
* **Final Natural Language Citations**: Citation markers in generated prose are not yet synthesized.
* **Abstention Response Synthesis**: Abstention responses are handled at the structured decision level, not via natural language generation.
* **Agent Orchestration**: Multi-step query planning and tool-calling agents are planned for subsequent phases.
* **REST API & Web UI**: FastAPI routes and interactive frontend interfaces will be implemented in subsequent phases.

---

## Trust Scoring Methodology

The Trust Engine operates purely as a deterministic heuristic evaluation layer:

1. **Relevance Aggregation ($R$)**:
   $$\text{prob}_i = \text{sigmoid}(s_i) = \frac{1}{1 + e^{-s_i}}$$
   $$R = \frac{\sum_{i=0}^{N-1} \text{prob}_i \cdot \gamma^i}{\sum_{i=0}^{N-1} \gamma^i}, \quad \gamma = 0.5$$

2. **Lexical Term Coverage ($C$)**:
   $$C = \frac{|\text{Query Terms} \cap \text{Evidence Tokens}|}{|\text{Query Terms}|}$$

3. **Provenance Validity ($P$)**:
   $$P = \frac{\text{Valid Chunks Count}}{\text{Total Chunks Count}} \in [0.0, 1.0]$$

4. **Groundedness Score ($G$)**:
   $$G = w_r \cdot R + w_c \cdot C + w_p \cdot P \quad (w_r=0.55, w_c=0.35, w_p=0.10)$$

5. **Confidence Index**:
   $$\text{Confidence} = (w_g \cdot G + w_t \cdot \text{prob}_0 + w_c \cdot C + w_v \cdot S_{\text{volume}}) \times P$$
   where $S_{\text{volume}} = \min(1.0, N / N_{\text{min}})$.

6. **Trust Decision**:
   $$\text{Decision} = \begin{cases} \text{SUPPORTED} & \text{if } G \ge \theta_G, \; \text{Conf} \ge \theta_{\text{Conf}}, \; C \ge \theta_C, \; P = 1.0, \; N \ge N_{\text{min}} \\ \text{INSUFFICIENT\_EVIDENCE} & \text{otherwise} \end{cases}$$

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

### 6. Evaluate Trust Engine Grounding & Decisions
Run the Trust Engine development evaluation dataset:
```bash
.venv\Scripts\python scripts/evaluate_trust.py
```

### 7. Run Full Test Suite
Execute all unit, component, and integration tests:
```bash
.venv\Scripts\pytest
```

---

## Python API Usage

### End-to-End Retrieval, Reranking, and Trust Decision

```python
from backend.app.reranking import RerankingPipeline
from backend.app.trust import TrustEngine, TrustEvidence, TrustDecision

# 1. Initialize Reranking Pipeline and Trust Engine
pipeline = RerankingPipeline(default_candidate_k=5, default_final_k=3)
engine = TrustEngine()

query = "What are the obligations of a Data Fiduciary before collecting personal data?"

# 2. Retrieve and Rerank Evidence
reranked_chunks = pipeline.retrieve(query, top_k=3, candidate_k=5)
evidence = [TrustEvidence.from_reranked_chunk(c) for c in reranked_chunks]

# 3. Evaluate Grounding and Issue Trust Decision
assessment = engine.evaluate(query, evidence)

print(f"Query: {assessment.query}")
print(f"Decision: {assessment.decision.value}")
print(f"Groundedness Score: {assessment.groundedness_score:.3f}")
print(f"Confidence Score: {assessment.confidence_score:.3f}")
print(f"Coverage Ratio: {assessment.coverage_score:.3f}")
print(f"Provenance Valid: {assessment.provenance_valid}")
print(f"Reasons: {assessment.decision_reasons}")

if assessment.decision == TrustDecision.SUPPORTED:
    print("-> Evidence is strongly grounded. Safe to proceed to generation.")
else:
    print("-> Insufficient evidence. Abstain from generation to prevent hallucination.")
```

---

## Configuration Reference

Key constants configured in `backend/app/config.py` and `backend/app/trust/config.py`:

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
* `TrustConfig.min_confidence_threshold`: Minimum confidence required for `SUPPORTED` (0.50).
* `TrustConfig.min_groundedness_threshold`: Minimum groundedness required for `SUPPORTED` (0.45).
* `TrustConfig.min_coverage_threshold`: Minimum lexical term coverage required for `SUPPORTED` (0.30).
* `TrustConfig.min_evidence_count`: Minimum required supporting evidence chunks (1).
* `TrustConfig.require_valid_provenance`: Mandatory provenance validity check (`True`).

---

## Output Artifacts

* `data/processed/chunks.json`: Processed document chunks with provenance metadata.
* `data/processed/embeddings.npy`: Binary NumPy array of 384-d L2-normalized embeddings.
* `data/processed/embedding_metadata.json`: FAISS integer index to chunk ID mappings.
* `data/processed/index.faiss`: Native serialized FAISS vector index (`IndexFlatIP`).
* `data/evaluation/retrieval_queries.json`: Grounded benchmark queries with target chunks.

---

## Out of Scope (Planned for Subsequent Phases)

The following components are **NOT** implemented in Phase 5 and will be developed in future phases:
* Local LLM Inference (Ollama / vLLM)
* Answer Generation Engine
* Exact Citation & In-text Provenance Generator
* Abstention Natural Language Response Generator
* Multi-Agent Orchestration
* FastAPI Backend & Interactive UI
