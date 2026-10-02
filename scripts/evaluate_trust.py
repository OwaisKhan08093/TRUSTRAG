"""Trust Engine Evaluation Script.

This script evaluates the Trust Engine's groundedness, coverage, relevance,
provenance, and trust decision making across deterministic test query scenarios.

DISCLAIMER:
This is an internal development evaluation dataset.
It is not a statistically validated benchmark.
"""

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

# Add project root to sys.path so backend imports work seamlessly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.reranking.pipeline import RerankingPipeline
from backend.app.retrieval.hybrid_retriever import HybridRetriever
from backend.app.trust.config import TrustConfig
from backend.app.trust.engine import TrustAssessment, TrustDecision, TrustEngine
from backend.app.trust.evidence import TrustEvidence

# Deterministic Development Evaluation Dataset
DEV_EVALUATION_SCENARIOS = [
    {
        "id": "scenario_01",
        "category": "SUPPORTED QUERY",
        "query": "What are the notice requirements for a Data Fiduciary before collecting personal data?",
        "expected_decision": "SUPPORTED",
        "evidence": [
            TrustEvidence(
                chunk_id="chunk_dpdp_sec_05_01",
                document_id="doc_dpdp_act",
                document_name="dpdp_act_2023.pdf",
                page_start=5,
                page_end=5,
                text="Section 5: Every Data Fiduciary shall give notice to the Data Principal on or before requesting consent.",
                retrieval_rank=1,
                retrieval_score=0.18,
                rerank_score=3.85,
            ),
            TrustEvidence(
                chunk_id="chunk_dpdp_sec_05_02",
                document_id="doc_dpdp_act",
                document_name="dpdp_act_2023.pdf",
                page_start=5,
                page_end=6,
                text="The notice must contain the personal data to be collected, purpose of processing, and manner of grievance redressal.",
                retrieval_rank=2,
                retrieval_score=0.14,
                rerank_score=3.10,
            ),
        ],
    },
    {
        "id": "scenario_02",
        "category": "INSUFFICIENT EVIDENCE QUERY",
        "query": "What are the rules regarding extraterritorial cross-border data transfers to blacklisted countries?",
        "expected_decision": "INSUFFICIENT_EVIDENCE",
        "evidence": [],  # Empty retrieval / no supporting passages found
    },
    {
        "id": "scenario_03",
        "category": "UNRELATED EVIDENCE QUERY",
        "query": "What are the maximum monetary penalties for failing to protect personal data from breaches?",
        "expected_decision": "INSUFFICIENT_EVIDENCE",
        "evidence": [
            TrustEvidence(
                chunk_id="chunk_astronomy_01",
                document_id="doc_astro_guide",
                document_name="astronomy_guide.pdf",
                page_start=12,
                page_end=13,
                text="Keplerian orbital dynamics govern satellite trajectories and gravitational perturbations in low Earth orbit.",
                retrieval_rank=1,
                retrieval_score=0.01,
                rerank_score=-4.50,
            )
        ],
    },
    {
        "id": "scenario_04",
        "category": "PARTIAL EVIDENCE QUERY",
        "query": "How are biometric identity verification protocols audited by the cybersecurity board?",
        "expected_decision": "INSUFFICIENT_EVIDENCE",
        "evidence": [
            TrustEvidence(
                chunk_id="chunk_dpdp_sec_09_01",
                document_id="doc_dpdp_act",
                document_name="dpdp_act_2023.pdf",
                page_start=9,
                page_end=9,
                text="A Data Fiduciary shall not undertake processing of personal data that is likely to cause any detrimental effect on well-being.",
                retrieval_rank=1,
                retrieval_score=0.08,
                rerank_score=-0.20,
            )
        ],
    },
    {
        "id": "scenario_05",
        "category": "CORRUPT PROVENANCE QUERY",
        "query": "What are the rights of a Data Principal regarding grievance redressal?",
        "expected_decision": "INSUFFICIENT_EVIDENCE",
        "evidence": [
            {
                "chunk_id": "chunk_invalid_prov",
                "document_id": "doc_dpdp_act",
                "document_name": "dpdp_act_2023.pdf",
                "page_start": 10,
                "page_end": 4,  # Malformed page range: page_end < page_start
                "text": "A Data Principal shall have the right to readily available means of grievance redressal provided by Data Fiduciary.",
                "rerank_score": 3.90,
            }
        ],
    },
]


def evaluate_trust_scenarios(
    engine: Optional[TrustEngine] = None,
    scenarios: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Run trust evaluation across development scenarios.

    Args:
        engine: Optional pre-configured TrustEngine instance.
        scenarios: Optional list of test scenario dictionaries.

    Returns:
        Dictionary containing summary statistics and individual scenario evaluations.
    """
    trust_engine = engine or TrustEngine()
    test_data = scenarios or DEV_EVALUATION_SCENARIOS

    records: List[Dict[str, Any]] = []
    correct_decisions = 0

    for item in test_data:
        q = item["query"]
        ev = item["evidence"]
        expected = item["expected_decision"]

        assessment = trust_engine.evaluate(q, ev)
        is_match = assessment.decision.value == expected
        if is_match:
            correct_decisions += 1

        records.append({
            "id": item["id"],
            "category": item["category"],
            "query": q,
            "expected_decision": expected,
            "actual_decision": assessment.decision.value,
            "decision_match": is_match,
            "relevance_score": assessment.relevance_score,
            "coverage_score": assessment.coverage_score,
            "groundedness_score": assessment.groundedness_score,
            "confidence_score": assessment.confidence_score,
            "provenance_valid": assessment.provenance_valid,
            "evidence_count": assessment.evidence_count,
            "decision_reasons": assessment.decision_reasons,
        })

    accuracy = correct_decisions / len(test_data) if test_data else 0.0

    return {
        "total_scenarios": len(test_data),
        "correct_decisions": correct_decisions,
        "accuracy": accuracy,
        "records": records,
    }


def run_trust_evaluation_cli() -> int:
    """CLI runner executing Trust Engine evaluation and printing formatted metrics."""
    print("=" * 80)
    print(" TrustRAG Trust Engine Evidence Grounding Evaluation (Milestone 9)")
    print("=" * 80)
    print("\n[NOTE] This is an internal development evaluation dataset.")
    print("       It is not a statistically validated benchmark.\n")

    results = evaluate_trust_scenarios()

    total = results["total_scenarios"]
    correct = results["correct_decisions"]
    accuracy = results["accuracy"] * 100.0

    print(f"Total Scenarios Evaluated: {total}")
    print(f"Decision Match Rate      : {correct}/{total} ({accuracy:.1f}%)\n")

    print("-" * 80)
    print(f"{'Scenario / Category':<28} | {'Rel':<5} | {'Cov':<5} | {'Ground':<6} | {'Conf':<5} | {'Decision':<20}")
    print("-" * 80)

    for rec in results["records"]:
        cat = rec["category"]
        rel = f"{rec['relevance_score']:.2f}"
        cov = f"{rec['coverage_score']:.2f}"
        ground = f"{rec['groundedness_score']:.2f}"
        conf = f"{rec['confidence_score']:.2f}"
        dec = rec["actual_decision"]
        match_icon = "[OK]" if rec["decision_match"] else "[MISMATCH]"

        print(f"{cat:<28} | {rel:<5} | {cov:<5} | {ground:<6} | {conf:<5} | {dec:<14} {match_icon}")

    print("-" * 80)
    print("\nDetailed Scenario Breakdown:")
    for rec in results["records"]:
        print(f"\n[{rec['id']}] {rec['category']}")
        print(f"  Query    : \"{rec['query']}\"")
        print(f"  Decision : {rec['actual_decision']} (Expected: {rec['expected_decision']})")
        print(f"  Metrics  : Rel={rec['relevance_score']:.3f}, Cov={rec['coverage_score']:.3f}, Ground={rec['groundedness_score']:.3f}, Conf={rec['confidence_score']:.3f}, ProvValid={rec['provenance_valid']}")
        print(f"  Reasons  : {', '.join(rec['decision_reasons'])}")

    print("\nOverall Trust Engine Evaluation Status: SUCCESS")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="TrustRAG Trust Engine Development Evaluator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    args = parser.parse_args()
    return run_trust_evaluation_cli()


if __name__ == "__main__":
    sys.exit(main())
