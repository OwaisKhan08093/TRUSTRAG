"""FastAPI backend API package for TrustRAG."""

from backend.app.api.app import app, create_app
from backend.app.api.dependencies import get_orchestrator, set_orchestrator
from backend.app.api.schemas import (
    CitationItem,
    QueryRequest,
    QueryResponse,
    TraceEventItem,
    TrustMetrics,
)

__all__ = [
    "CitationItem",
    "QueryRequest",
    "QueryResponse",
    "TraceEventItem",
    "TrustMetrics",
    "app",
    "create_app",
    "get_orchestrator",
    "set_orchestrator",
]
