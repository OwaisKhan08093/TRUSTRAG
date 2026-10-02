"""Unit tests for API dependency and lifecycle management (Phase 8 Milestone 6)."""

import pytest
from fastapi.testclient import TestClient

from backend.app.agents.orchestrator import TrustRAGOrchestrator
from backend.app.api.app import app, create_app
from backend.app.api.dependencies import (
    cleanup_dependencies,
    get_orchestrator,
    initialize_dependencies,
    set_orchestrator,
)
from backend.app.api.lifecycle import app_lifespan


def test_singleton_orchestrator_dependency():
    """Verify get_orchestrator returns the same singleton instance on consecutive calls."""
    cleanup_dependencies()
    orch1 = get_orchestrator()
    orch2 = get_orchestrator()

    assert isinstance(orch1, TrustRAGOrchestrator)
    assert orch1 is orch2
    cleanup_dependencies()


def test_dependency_override_and_reset():
    """Verify set_orchestrator allows custom injection and cleanup."""
    custom_orch = TrustRAGOrchestrator()
    set_orchestrator(custom_orch)
    assert get_orchestrator() is custom_orch

    cleanup_dependencies()
    assert get_orchestrator() is not custom_orch
    cleanup_dependencies()


@pytest.mark.anyio
async def test_app_lifespan_context():
    """Verify app_lifespan executes startup initialization and shutdown teardown."""
    test_app = create_app()
    async with app_lifespan(test_app):
        orch = get_orchestrator()
        assert isinstance(orch, TrustRAGOrchestrator)

    # After exit, cleanup was executed
    # TestClient lifecycle integration:
    with TestClient(test_app) as client:
        resp = client.get("/health")
        assert resp.status_code == 200
