"""FastAPI dependency injection providers for TrustRAG."""

from typing import Optional
from backend.app.agents.orchestrator import TrustRAGOrchestrator

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
