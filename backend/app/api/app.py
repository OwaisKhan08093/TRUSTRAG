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

    # Configure environment-aware CORS
    import os
    from starlette.middleware.base import BaseHTTPMiddleware

    env_origins = os.getenv("ALLOWED_ORIGINS", "*")
    if env_origins.strip() == "*":
        origins = ["*"]
    else:
        origins = [o.strip() for o in env_origins.split(",") if o.strip()]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True if origins != ["*"] else False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

    # Security Headers Middleware
    class SecurityHeadersMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request, call_next):
            response = await call_next(request)
            response.headers["X-Content-Type-Options"] = "nosniff"
            response.headers["X-Frame-Options"] = "DENY"
            response.headers["X-XSS-Protection"] = "1; mode=block"
            response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
            return response

    app.add_middleware(SecurityHeadersMiddleware)

    # Register centralized exception handlers
    register_error_handlers(app)

    # Register Routers
    app.include_router(health_router)
    app.include_router(query_router)

    return app


# Default singleton instance for standard ASGI servers
app = create_app()
