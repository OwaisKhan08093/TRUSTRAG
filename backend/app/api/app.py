"""FastAPI application factory and foundation for TrustRAG."""

import logging
from typing import Any, Dict, Optional
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api.errors import register_error_handlers
from backend.app.api.lifecycle import app_lifespan
from backend.app.api.routes.health import router as health_router
from backend.app.api.routes.query import router as query_router

logger = logging.getLogger(__name__)


def create_app(
    title: str = "TrustRAG API",
    version: str = "1.0.0",
    description: str = "Confidence-Aware Grounded RAG with Deterministic Trust Gating",
    debug: bool = False,
) -> FastAPI:
    """Create and configure a new FastAPI application instance.

    Args:
        title: API title for OpenAPI docs.
        version: API semantic version.
        description: API description.
        debug: Enable debug mode.

    Returns:
        Configured FastAPI application instance.
    """
    app = FastAPI(
        title=title,
        version=version,
        description=description,
        debug=debug,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=app_lifespan,
    )

    # Configure permissive CORS for API clients
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register centralized exception handlers
    register_error_handlers(app)

    # Register Routers
    app.include_router(health_router)
    app.include_router(query_router)

    return app


# Default singleton instance for standard ASGI servers
app = create_app()
