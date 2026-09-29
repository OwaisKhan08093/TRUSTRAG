# TrustRAG

TrustRAG is a solo-developed Confidence-Aware Retrieval-Augmented Generation (RAG) system engineered to ingest trusted documents, retrieve grounded evidence, generate verifiable answers with exact citations, evaluate answer fidelity, compute reliability signals, and abstain whenever supporting evidence is insufficient.

## Current Status

**Step 0 + Step 1 completed.**

* [x] **Step 0**: Repository structure, configuration, test suite, and virtual environment setup.
* [x] **Step 1**: Robust PyMuPDF PDF ingestion, conservative legal text cleaning, deterministic sliding-window chunking, and metadata-preserving JSON persistence.
* [ ] **Step 2 (Upcoming)**: Embedding generation + FAISS indexing.

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

3. Install the minimal dependencies required for Step 0 and Step 1:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Place your raw PDF documents into `data/raw/` and execute the ingestion script:

```bash
python scripts/ingest.py data/raw/DPDP_Act_2023.pdf
```

You can customize chunk size and overlap parameters if needed:
```bash
python scripts/ingest.py data/raw/DPDP_Act_2023.pdf --chunk-size 600 --chunk-overlap 120
```

To run the automated test suite:
```bash
pytest
```

## Output

Processed chunks are saved to `data/processed/chunks.json`. Each chunk preserves full document provenance and page attribution:

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

## Next Step

**Next milestone: Hugging Face embedding generation + FAISS vector indexing.**
