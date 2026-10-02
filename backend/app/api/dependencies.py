"""FastAPI dependency injection providers for TrustRAG."""

import logging
from typing import Optional
from backend.app.agents.orchestrator import TrustRAGOrchestrator

logger = logging.getLogger(__name__)

_orchestrator_instance: Optional[TrustRAGOrchestrator] = None


def get_orchestrator() -> TrustRAGOrchestrator:
    """Dependency provider returning singleton TrustRAGOrchestrator instance.

    Maintains lazy model loading semantics without redundant initialization.
    """
    global _orchestrator_instance
    if _orchestrator_instance is None:
        _orchestrator_instance = TrustRAGOrchestrator()
    return _orchestrator_instance


def set_orchestrator(orchestrator: Optional[TrustRAGOrchestrator]) -> None:
    """Set or override global orchestrator instance (useful for testing)."""
    global _orchestrator_instance
    _orchestrator_instance = orchestrator


def initialize_dependencies() -> None:
    """Initialize singleton instances during application startup."""
    global _orchestrator_instance
    if _orchestrator_instance is None:
        _orchestrator_instance = TrustRAGOrchestrator()
    logger.debug("TrustRAG dependencies initialized.")


def cleanup_dependencies() -> None:
    """Teardown singleton instances during application shutdown."""
    global _orchestrator_instance
    _orchestrator_instance = None
    logger.debug("TrustRAG dependencies cleaned up.")
