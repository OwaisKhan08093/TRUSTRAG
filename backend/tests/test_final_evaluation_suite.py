"""Regression tests for the final evaluation suite."""

import pytest
from scripts.evaluate_final_suite import run_evaluation_suite


def test_final_evaluation_suite_execution():
    """Verify evaluation suite runs across all categories and meets accuracy benchmarks."""
    summary = run_evaluation_suite()
    assert summary["total_samples"] >= 6
    assert summary["overall_decision_accuracy_pct"] >= 85.0
    assert summary["abstention_accuracy_pct"] == 100.0
    assert summary["citation_validity_rate_pct"] == 100.0
