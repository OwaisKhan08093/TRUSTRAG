"""Health check route for TrustRAG API."""

from typing import Any, Dict
from fastapi import APIRouter

router = APIRouter(tags=["Health"])


@router.get(
    "/health",
    summary="Service Health Check",
    response_description="Service status and metadata",
)
def health_check() -> Dict[str, Any]:
    """Lightweight endpoint returning the service status without loading expensive ML models."""
    return {
        "status": "ok",
        "service": "trustrag",
        "version": "1.0.0",
    }
