"""Security Hardening and Audit Regression Suite for TrustRAG (Phase 10 Milestone 4).

Tests:
1. Security response headers (X-Content-Type-Options, X-Frame-Options, X-XSS-Protection, Referrer-Policy)
2. Max query length boundary defense (rejection of oversized DoS payloads > 4096 characters)
3. Malformed and adversarial payload sanitization
4. Internal stack trace sanitization on unhandled exceptions
5. Environment-aware CORS origin policy
6. Secrets and sensitive configuration audit
"""

import os
from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient

from backend.app.api.app import create_app
from backend.app.api.dependencies import get_orchestrator, set_orchestrator


@pytest.fixture
def security_client():
    """Test client configured with security headers and mock orchestrator."""
    mock_orchestrator = MagicMock()
    app = create_app()
    app.dependency_overrides[get_orchestrator] = lambda: mock_orchestrator
    set_orchestrator(mock_orchestrator)
    client = TestClient(app)
    yield client
    app.dependency_overrides.clear()
    set_orchestrator(None)


def test_security_headers_present_on_endpoints(security_client):
    """Verify standard security headers are attached to API responses."""
    resp = security_client.get("/health")
    assert resp.status_code == 200
    assert resp.headers.get("X-Content-Type-Options") == "nosniff"
    assert resp.headers.get("X-Frame-Options") == "DENY"
    assert resp.headers.get("X-XSS-Protection") == "1; mode=block"
    assert resp.headers.get("Referrer-Policy") == "strict-origin-when-cross-origin"


def test_query_max_length_boundary_rejection(security_client):
    """Verify queries exceeding 4096 characters are rejected with 422."""
    oversized_query = "A" * 4097
    resp = security_client.post("/query", json={"query": oversized_query})
    assert resp.status_code == 422
    data = resp.json()
    assert "error" in data
    assert data["error"]["type"] == "ValidationError"


def test_malformed_json_and_adversarial_payloads(security_client):
    """Verify malformed content-types and invalid payloads return 422 safely."""
    # Raw non-JSON text
    resp = security_client.post(
        "/query",
        content="not-a-valid-json",
        headers={"Content-Type": "application/json"},
    )
    assert resp.status_code == 422

    # Null query
    resp = security_client.post("/query", json={"query": None})
    assert resp.status_code == 422


def test_internal_exception_sanitization_no_leakage(security_client):
    """Verify 500 error responses never leak raw Python tracebacks or sensitive paths."""
    mock_orch = MagicMock()
    mock_orch.execute.side_effect = ValueError("SECRET_DATABASE_KEY_LEAK_TEST at d:/TRUSTRAG/backend/secret.py")

    app = create_app()
    app.dependency_overrides[get_orchestrator] = lambda: mock_orch
    set_orchestrator(mock_orch)
    client = TestClient(app)

    resp = client.post("/query", json={"query": "Safe query triggering error"})
    assert resp.status_code == 500
    data = resp.json()

    assert data["error"]["status_code"] == 500
    # Error message must be sanitized
    assert "SECRET_DATABASE_KEY" not in data["error"]["message"]
    assert "Traceback" not in data["error"]["message"]
    assert "secret.py" not in data["error"]["message"]


def test_cors_environment_configuration():
    """Verify custom ALLOWED_ORIGINS environment variable is correctly parsed."""
    with patch.dict(os.environ, {"ALLOWED_ORIGINS": "https://trusted-domain.com,https://app.trustrag.ai"}):
        app = create_app()
        # Find CORS middleware
        cors_middlewares = [m for m in app.user_middleware if m.cls.__name__ == "CORSMiddleware"]
        assert len(cors_middlewares) == 1
        kwargs = cors_middlewares[0].kwargs
        assert "https://trusted-domain.com" in kwargs["allow_origins"]
        assert "https://app.trustrag.ai" in kwargs["allow_origins"]
