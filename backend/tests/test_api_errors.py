"""Unit tests for centralized API error handling (Phase 8 Milestone 5)."""

from unittest.mock import MagicMock
import pytest
from fastapi.testclient import TestClient

from backend.app.agents.base import AgentExecutionError, AgentInputError
from backend.app.agents.orchestrator import TrustRAGOrchestrator
from backend.app.api.app import app
from backend.app.api.dependencies import get_orchestrator


@pytest.fixture
def mock_orchestrator():
    """Fixture providing a mocked TrustRAGOrchestrator."""
    return MagicMock(spec=TrustRAGOrchestrator)


@pytest.fixture
def client(mock_orchestrator):
    """Fixture providing a TestClient with overridden get_orchestrator dependency."""
    app.dependency_overrides[get_orchestrator] = lambda: mock_orchestrator
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_validation_error_response_format(client):
    """Verify malformed JSON payload returns standardized 422 error structure."""
    response = client.post("/query", json={"invalid_field": 123})
    assert response.status_code == 422
    data = response.json()
    assert "error" in data
    assert data["error"]["status_code"] == 422
    assert data["error"]["type"] == "ValidationError"
    assert "details" in data["error"]


def test_agent_input_error_handling(client, mock_orchestrator):
    """Verify AgentInputError is mapped to HTTP 422 error structure."""
    mock_orchestrator.execute.side_effect = AgentInputError("Invalid query parameter format")

    response = client.post("/query", json={"query": "valid length query"})
    assert response.status_code == 422
    data = response.json()
    assert data["error"]["type"] in ["AgentInputError", "HTTPException"]
    assert "Invalid query parameter format" in data["error"]["message"]


def test_agent_execution_error_handling(client, mock_orchestrator):
    """Verify internal AgentExecutionError returns 500 without raw traceback leak."""
    mock_orchestrator.execute.side_effect = AgentExecutionError("FAISS vector retrieval failed")

    response = client.post("/query", json={"query": "What are notice rules?"})
    assert response.status_code == 500
    data = response.json()
    assert data["error"]["status_code"] == 500
    assert "Internal orchestration failure" in data["error"]["message"] or "FAISS" in data["error"]["message"]
    assert "Traceback" not in response.text


def test_not_found_endpoint(client):
    """Verify unmapped route returns standard 404 response."""
    response = client.get("/non_existent_route")
    assert response.status_code == 404
    data = response.json()
    assert "error" in data
    assert data["error"]["status_code"] == 404
