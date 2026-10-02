"""Application lifecycle and lifespan management for TrustRAG FastAPI."""

from contextlib import asynccontextmanager
import logging
from typing import AsyncGenerator
from fastapi import FastAPI

from backend.app.api.dependencies import cleanup_dependencies, initialize_dependencies

logger = logging.getLogger(__name__)


@asynccontextmanager
async def app_lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application startup and shutdown lifecycle events.

    Startup:
        - Prepares lightweight dependencies and verifies orchestrator readiness without eager GPU loading.
    Shutdown:
        - Performs clean teardown of connection pools and caches.
    """
    logger.info("Initializing TrustRAG API service dependencies...")
    initialize_dependencies()
    logger.info("TrustRAG API service initialized successfully (lazy model loading preserved).")

    yield

    logger.info("Shutting down TrustRAG API service...")
    cleanup_dependencies()
    logger.info("TrustRAG API service shutdown complete.")
