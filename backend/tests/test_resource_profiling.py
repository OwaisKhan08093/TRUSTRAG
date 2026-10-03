"""Regression tests for resource profiling and memory safeguards."""

import pytest
from backend.app.generation.llm import LocalLLM
from scripts.profile_resources import (
    profile_lazy_loading_and_unloading,
    profile_repeated_query_stability,
    profile_retrieval_artifacts,
)


def test_retrieval_artifacts_profile():
    """Verify retrieval artifact profiling finds index files."""
    res = profile_retrieval_artifacts()
    assert "artifacts" in res
    assert "total_disk_footprint_mb" in res
    assert res["total_disk_footprint_mb"] > 0
    assert res["artifacts"]["faiss_index"]["exists"] is True


def test_lazy_loading_and_unloading():
    """Verify LocalLLM does not load weights eagerly on init and unloads cleanly."""
    res = profile_lazy_loading_and_unloading()
    assert res["lazy_is_resident"] is False
    assert res["unloaded_is_resident"] is False
    assert res["lazy_loading_functional"] is True
    assert res["unloading_functional"] is True


def test_repeated_query_memory_stability():
    """Verify repeated queries do not cause runaway memory growth."""
    res = profile_repeated_query_stability(iterations=5)
    assert res["is_memory_stable"] is True
    assert res["iterations"] == 5
