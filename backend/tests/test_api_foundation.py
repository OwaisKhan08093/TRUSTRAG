"""Unit tests for FastAPI backend foundation (Phase 8 Milestone 1)."""

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.app.api.app import app, create_app


def test_app_instance_creation():
    """Verify FastAPI application instance initializes with correct metadata."""
    custom_app = create_app(
        title="Custom TrustRAG",
        version="2.0.0",
        description="Custom Description",
    )
    assert isinstance(custom_app, FastAPI)
    assert custom_app.title == "Custom TrustRAG"
    assert custom_app.version == "2.0.0"
    assert custom_app.description == "Custom Description"


def test_default_app_instance():
    """Verify default singleton app has correct configuration."""
    assert isinstance(app, FastAPI)
    assert app.title == "TrustRAG API"
    assert app.docs_url == "/docs"
    assert app.openapi_url == "/openapi.json"


def test_openapi_schema_endpoint():
    """Verify OpenAPI JSON schema is generated and accessible via TestClient."""
    client = TestClient(app)
    response = client.get("/openapi.json")
    assert response.status_code == 200
    data = response.json()
    assert "openapi" in data
    assert data["info"]["title"] == "TrustRAG API"
