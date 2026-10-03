# =============================================================================
# TrustRAG Backend Dockerfile (Multi-Stage Production Container)
# =============================================================================

FROM python:3.11-slim as base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PORT=8000 \
    HOST=0.0.0.0

WORKDIR /app

# Install system runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch and backend dependencies
COPY pyproject.toml poetry.lock* requirements.txt* ./
RUN pip install --upgrade pip setuptools wheel && \
    pip install torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install transformers sentence-transformers faiss-cpu rank-bm25 pymupdf fastapi uvicorn pydantic python-multipart httpx pytest

# Copy application source code and corpus data
COPY backend/ ./backend/
COPY data/ ./data/
COPY scripts/ ./scripts/

# Create non-root user for security hardening
RUN useradd -m -u 1001 trustrag && \
    chown -R trustrag:trustrag /app
USER trustrag

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "backend.app.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
