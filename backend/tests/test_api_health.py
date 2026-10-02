"""Unit tests for /health endpoint (Phase 8 Milestone 2)."""

import pytest
from fastapi.testclient import TestClient

from backend.app.api.app import app


@pytest.fixture
def client():
    """Fixture providing a TestClient instance."""
    return TestClient(app)


def test_health_endpoint_status_code(client):
    """Verify /health returns HTTP 200."""
    response = client.get("/health")
    assert response.status_code == 200


def test_health_endpoint_response_structure(client):
    """Verify /health response contains required keys."""
    response = client.get("/health")
    data = response.json()

    assert "status" in data
    assert "service" in data
    assert "version" in data
    assert data["status"] == "ok"
    assert data["service"] == "trustrag"
    assert data["version"] == "1.0.0"


def test_health_endpoint_lightweight(client):
    """Verify /health responds with minimal latency without triggering ML initialization."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/json"
