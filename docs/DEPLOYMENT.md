# TrustRAG Deployment Guide

**Phase 10 — Production Containerization, Deployment & Hardware Specifications**

---

## 1. System Requirements & Hardware Guidelines

TrustRAG is engineered with confidence-aware local generation and deterministic trust gating. Compute requirements vary depending on target deployment mode:

| Environment | Minimum RAM | Recommended Hardware | Compute Device | Notes |
|:---|:---|:---|:---|:---|
| **Local Development** | 8 GB RAM | 16 GB RAM, 8-core CPU | CPU | Retrieval, reranking, and trust gating execute in < 250ms. |
| **CPU Server Deployment** | 16 GB RAM | 32 GB RAM, 16-core CPU | PyTorch CPU | Suitable for factual document retrieval & gating; greedy decoding. |
| **GPU Server Deployment** | 16 GB RAM | NVIDIA RTX 3080/4090 or A10G (8GB+ VRAM) | CUDA | Sub-second token generation latency using float16/bfloat16. |
| **Cloud Kubernetes / ECS** | 16 GB RAM | AWS `g5.xlarge` / GCP `g2-standard-4` | CUDA / CPU | Containerized with persistent volume for HuggingFace model cache. |

---

## 2. Local Development Workflow (Without Docker)

### Backend Service:
```bash
# Activate virtual environment
.\.venv\Scripts\Activate.ps1

# Run FastAPI backend
python -m uvicorn backend.app.api.app:app --host 127.0.0.1 --port 8000
```
Backend API will be accessible at: `http://127.0.0.1:8000`  
Swagger Documentation: `http://127.0.0.1:8000/docs`

### Frontend UI:
```bash
cd frontend
npm install
npm run dev
```
Frontend Web UI will be accessible at: `http://127.0.0.1:5173`

---

## 3. Containerized Deployment (Docker & Docker Compose)

### Single Command Launch:
```bash
docker-compose up -d --build
```

### Services Spawned:
- **`backend`**: FastAPI ASGI server on port `8000` with non-root security context and healthchecks.
- **`frontend`**: Nginx web server on port `80` serving optimized production Vite bundle.

### Volumes:
- `trustrag_data`: Persistent volume for PDF uploads, vector indexes (`index.faiss`), and chunk metadata (`chunks.json`).
- `hf_cache`: Persistent volume caching HuggingFace weights (`all-MiniLM-L6-v2`, `ms-marco-MiniLM-L-6-v2`, `Qwen2.5-3B-Instruct`).

---

## 4. Configurable Environment Variables

| Variable | Default Value | Description |
|:---|:---|:---|
| `PORT` | `8000` | Backend listening port |
| `HOST` | `0.0.0.0` | Backend bind host |
| `ALLOWED_ORIGINS` | `*` | Comma-separated CORS allowed origins |
| `TRUSTRAG_DATA_DIR` | `./data` | Filepath to on-disk index and corpus storage |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Dense embedding model |
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Neural cross-encoder reranker |
| `LLM_MODEL` | `Qwen/Qwen2.5-3B-Instruct` | Local generation model |
| `VITE_API_BASE_URL` | `http://localhost:8000` | Frontend API client backend URL |

---

## 5. Deployment Verification & Smoke Test

1. Verify backend health endpoint:
   ```bash
   curl -f http://localhost:8000/health
   ```
2. Verify document catalog:
   ```bash
   curl -f http://localhost:8000/documents
   ```
3. Test end-to-end query:
   ```bash
   curl -X POST http://localhost:8000/query \
     -H "Content-Type: application/json" \
     -d '{"query": "What notice must a Data Fiduciary give before requesting consent?"}'
   ```
