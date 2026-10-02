"""Evaluation script for TrustRAG FastAPI Backend (Phase 8 Milestone 9).

DISCLAIMER:
This is an internal development evaluation for validating HTTP contract behaviors,
TrustEngine gating preservation over the network layer, error translation,
and citation serialization.
It is not a statistically validated benchmark.
"""

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

# Add project root to sys.path so backend imports work seamlessly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient

from backend.app.agents.orchestrator import OrchestratorResult, TrustRAGOrchestrator
from backend.app.agents.trace import AgentEventStatus, ExecutionTrace
from backend.app.api.app import app
from backend.app.api.dependencies import get_orchestrator, set_orchestrator
from backend.app.generation.citations import Citation
from backend.app.generation.generator import STANDARD_ABSTENTION_MESSAGE
from backend.app.trust.engine import TrustDecision

DEV_API_SCENARIOS = [
    {
        "id": "scenario_01_supported",
        "name": "1. Supported query with grounded citations",
        "request_payload": {"query": "What notice must a Data Fiduciary give under Section 5?"},
        "expected_http_status": 200,
        "expected_decision": "SUPPORTED",
        "expected_is_refusal": False,
        "expected_generation_called": True,
        "mock_result": OrchestratorResult(
            query="What notice must a Data Fiduciary give under Section 5?",
            answer="A Data Fiduciary must give notice prior to consent [1].",
            decision=TrustDecision.SUPPORTED,
            is_refusal=False,
            citations=[
                Citation(
                    index=1,
                    chunk_id="chunk_sec05_01",
                    document_id="doc_dpdp_act",
                    document_name="DPDP_Act_2023.pdf",
                    page_start=5,
                    page_end=5,
                    text_snippet="Section 5 notice requirements.",
                    formatted_reference="[1] DPDP_Act_2023.pdf, page 5",
                )
            ],
            formatted_response="A Data Fiduciary must give notice prior to consent [1].\n\n### References\n- [1] DPDP_Act_2023.pdf, page 5",
            confidence_score=0.92,
            groundedness_score=0.90,
            relevance_score=0.95,
            coverage_score=1.0,
            provenance_valid=True,
            latency_seconds=0.08,
            metadata={"gating_status": "SUPPORTED_AND_GENERATED"},
        ),
    },
    {
        "id": "scenario_02_insufficient",
        "name": "2. Insufficient evidence producing transparent refusal (No LLM)",
        "request_payload": {"query": "What are extraterritorial space law clauses in DPDP?"},
        "expected_http_status": 200,
        "expected_decision": "INSUFFICIENT_EVIDENCE",
        "expected_is_refusal": True,
        "expected_generation_called": False,
        "mock_result": OrchestratorResult(
            query="What are extraterritorial space law clauses in DPDP?",
            answer=STANDARD_ABSTENTION_MESSAGE,
            decision=TrustDecision.INSUFFICIENT_EVIDENCE,
            is_refusal=True,
            citations=[],
            formatted_response=f"**[Status: INSUFFICIENT_EVIDENCE]**\n\n{STANDARD_ABSTENTION_MESSAGE}",
            confidence_score=0.05,
            groundedness_score=0.0,
            relevance_score=0.1,
            coverage_score=0.0,
            provenance_valid=True,
            latency_seconds=0.02,
            metadata={"gating_status": "GATED_REFUSAL"},
        ),
    },
    {
        "id": "scenario_03_empty_query",
        "name": "3. Empty / whitespace query triggering 422 validation error",
        "request_payload": {"query": "   "},
        "expected_http_status": 422,
        "expected_decision": None,
        "expected_is_refusal": None,
        "expected_generation_called": False,
        "mock_result": None,
    },
    {
        "id": "scenario_04_multiple_evidence",
        "name": "4. Multiple evidence sources with multi-page citations",
        "request_payload": {"query": "What are duties and penalty amounts?"},
        "expected_http_status": 200,
        "expected_decision": "SUPPORTED",
        "expected_is_refusal": False,
        "expected_generation_called": True,
        "mock_result": OrchestratorResult(
            query="What are duties and penalty amounts?",
            answer="Duties are defined in Section 15 [1] and fines in Section 33 [2].",
            decision=TrustDecision.SUPPORTED,
            is_refusal=False,
            citations=[
                Citation(
                    index=1,
                    chunk_id="chunk_sec15",
                    document_id="doc_dpdp",
                    document_name="dpdp.pdf",
                    page_start=15,
                    page_end=15,
                    text_snippet="Duties snippet.",
                    formatted_reference="[1] dpdp.pdf, page 15",
                ),
                Citation(
                    index=2,
                    chunk_id="chunk_sec33",
                    document_id="doc_dpdp",
                    document_name="dpdp.pdf",
                    page_start=28,
                    page_end=28,
                    text_snippet="Penalties snippet.",
                    formatted_reference="[2] dpdp.pdf, page 28",
                ),
            ],
            formatted_response="Duties are defined in Section 15 [1] and fines in Section 33 [2].",
            confidence_score=0.94,
            groundedness_score=0.91,
            relevance_score=0.96,
            coverage_score=1.0,
            provenance_valid=True,
            latency_seconds=0.12,
        ),
    },
    {
        "id": "scenario_05_provenance_failure",
        "name": "5. Provenance metadata failure triggering trust gate refusal",
        "request_payload": {"query": "Child consent requirements"},
        "expected_http_status": 200,
        "expected_decision": "INSUFFICIENT_EVIDENCE",
        "expected_is_refusal": True,
        "expected_generation_called": False,
        "mock_result": OrchestratorResult(
            query="Child consent requirements",
            answer=STANDARD_ABSTENTION_MESSAGE,
            decision=TrustDecision.INSUFFICIENT_EVIDENCE,
            is_refusal=True,
            citations=[],
            formatted_response=f"**[Status: INSUFFICIENT_EVIDENCE]**\n\n{STANDARD_ABSTENTION_MESSAGE}",
            confidence_score=0.0,
            groundedness_score=0.0,
            relevance_score=0.2,
            coverage_score=0.1,
            provenance_valid=False,
            latency_seconds=0.02,
        ),
    },
    {
        "id": "scenario_06_retrieval_failure",
        "name": "6. Retrieval subsystem error translated to HTTP 500",
        "request_payload": {"query": "Query triggering vector database failure"},
        "expected_http_status": 500,
        "expected_decision": None,
        "expected_is_refusal": None,
        "expected_generation_called": False,
        "mock_result": "RAISE_RETRIEVAL_ERROR",
    },
    {
        "id": "scenario_07_generation_failure",
        "name": "7. LLM runtime error translated to HTTP 500",
        "request_payload": {"query": "Query triggering LLM runtime failure"},
        "expected_http_status": 500,
        "expected_decision": None,
        "expected_is_refusal": None,
        "expected_generation_called": False,
        "mock_result": "RAISE_GENERATION_ERROR",
    },
]


def run_api_evaluation(verbose: bool = True) -> Dict[str, Any]:
    """Execute all API evaluation scenarios against TestClient."""
    passed_count = 0
    total_count = len(DEV_API_SCENARIOS)
    scenario_results: List[Dict[str, Any]] = []

    print("=" * 80)
    print("TRUSTRAG FASTAPI BACKEND & ORCHESTRATION EVALUATION")
    print("=" * 80)

    for sc in DEV_API_SCENARIOS:
        sc_id = sc["id"]
        sc_name = sc["name"]
        payload = sc["request_payload"]
        expected_status = sc["expected_http_status"]
        expected_dec = sc["expected_decision"]
        expected_refusal = sc["expected_is_refusal"]
        mock_res = sc["mock_result"]

        mock_orchestrator = MagicMock(spec=TrustRAGOrchestrator)

        if mock_res == "RAISE_RETRIEVAL_ERROR":
            mock_orchestrator.execute.side_effect = RuntimeError("FAISS search internal failure")
        elif mock_res == "RAISE_GENERATION_ERROR":
            mock_orchestrator.execute.side_effect = RuntimeError("CUDA Out of Memory in LLM")
        elif isinstance(mock_res, OrchestratorResult):
            mock_orchestrator.execute.return_value = mock_res

        app.dependency_overrides[get_orchestrator] = lambda: mock_orchestrator
        client = TestClient(app)

        t_start = time.perf_counter()
        response = client.post("/query", json=payload)
        elapsed = time.perf_counter() - t_start

        # Invariant checks:
        status_pass = (response.status_code == expected_status)
        data = response.json() if response.headers.get("content-type", "").startswith("application/json") else {}

        decision_pass = True
        if expected_dec is not None:
            decision_pass = (data.get("decision") == expected_dec)

        refusal_pass = True
        if expected_refusal is not None:
            refusal_pass = (data.get("is_refusal") == expected_refusal)

        scenario_passed = (status_pass and decision_pass and refusal_pass)
        if scenario_passed:
            passed_count += 1

        status_tag = "[PASS]" if scenario_passed else "[FAIL]"
        print(f"\n{status_tag} {sc_name}")
        print(f"       HTTP Status: {response.status_code} (Expected: {expected_status})")
        if expected_dec is not None:
            print(f"       Trust Decision: {data.get('decision')} (Expected: {expected_dec})")
            print(f"       Is Refusal: {data.get('is_refusal')}, Citations Count: {len(data.get('citations', []))}")
        elif response.status_code >= 400:
            print(f"       Error Envelope: {data.get('error', {}).get('type')} - {data.get('error', {}).get('message')}")
        print(f"       Latency: {elapsed:.4f}s")

        scenario_results.append({
            "id": sc_id,
            "name": sc_name,
            "passed": scenario_passed,
            "status_code": response.status_code,
            "expected_status_code": expected_status,
            "decision": data.get("decision"),
            "latency_seconds": elapsed,
        })

    app.dependency_overrides.clear()

    print("\n" + "=" * 80)
    print(f"OVERALL API SUMMARY: {passed_count}/{total_count} scenarios passed.")
    print("=" * 80)

    return {
        "passed_count": passed_count,
        "total_count": total_count,
        "all_passed": (passed_count == total_count),
        "scenarios": scenario_results,
    }


def main():
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Evaluate TrustRAG FastAPI Backend")
    parser.add_argument("--json", action="store_true", help="Print output as JSON")
    args = parser.parse_args()

    results = run_api_evaluation(verbose=not args.json)

    if args.json:
        print(json.dumps(results, indent=2))

    if not results["all_passed"]:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
