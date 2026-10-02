"""Evaluation script for Multi-Agent TrustRAG Orchestration (Phase 7 Milestone 9).

DISCLAIMER:
This is an internal development evaluation validating multi-agent coordination,
safeguard gating, execution traces, and provenance preservation.
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

from backend.app.agents.citation_agent import CitationAgent, CitationAgentResult
from backend.app.agents.evidence_agent import EvidenceAgent, EvidenceAgentResult
from backend.app.agents.generation_agent import GenerationAgent, GenerationAgentResult
from backend.app.agents.orchestrator import OrchestratorResult, TrustRAGOrchestrator
from backend.app.agents.retrieval_agent import RetrievalAgent, RetrievalAgentResult
from backend.app.agents.state import AgentState, PipelineStatus
from backend.app.agents.trace import AgentEventStatus, ExecutionTrace
from backend.app.agents.trust_agent import TrustAgent, TrustAgentResult
from backend.app.generation.citations import Citation
from backend.app.generation.generator import GroundedGenerator, STANDARD_ABSTENTION_MESSAGE
from backend.app.generation.models import GenerationResult
from backend.app.generation.response import GroundedAnswer
from backend.app.reranking.schema import RerankedChunk
from backend.app.trust.engine import TrustDecision, TrustEngine
from backend.app.trust.evidence import TrustEvidence

DEV_AGENT_EVAL_SCENARIOS = [
    {
        "id": "scenario_01_supported",
        "name": "1. Supported query with high relevance evidence",
        "query": "What notice must a Data Fiduciary give to a Data Principal before seeking consent?",
        "expected_decision": TrustDecision.SUPPORTED,
        "expected_generation_called": True,
        "retrieval_candidates": [
            {
                "chunk_id": "chunk_sec05_01",
                "document_id": "doc_dpdp_act",
                "document_name": "DPDP_Act_2023.pdf",
                "page_start": 5,
                "page_end": 5,
                "text": "Section 5: Every Data Fiduciary shall give notice to the Data Principal on or before requesting consent.",
                "rank": 1,
                "score": 0.18,
            }
        ],
        "evidence_chunks": [
            RerankedChunk(
                chunk_id="chunk_sec05_01",
                document_id="doc_dpdp_act",
                document_name="DPDP_Act_2023.pdf",
                page_start=5,
                page_end=5,
                text="Section 5: Every Data Fiduciary shall give notice to the Data Principal on or before requesting consent.",
                original_rank=1,
                original_score=0.18,
                rerank_score=3.85,
                final_rank=1,
            )
        ],
    },
    {
        "id": "scenario_02_insufficient",
        "name": "2. Insufficient evidence / out-of-domain query",
        "query": "What are the rules regarding extraterritorial cross-border data transfers to restricted jurisdictions?",
        "expected_decision": TrustDecision.INSUFFICIENT_EVIDENCE,
        "expected_generation_called": False,
        "retrieval_candidates": [
            {
                "chunk_id": "chunk_unrelated",
                "document_id": "doc_other",
                "document_name": "general_definitions.pdf",
                "page_start": 1,
                "page_end": 1,
                "text": "This document defines general words and terms used across administrative circulars.",
                "rank": 1,
                "score": 0.01,
            }
        ],
        "evidence_chunks": [
            RerankedChunk(
                chunk_id="chunk_unrelated",
                document_id="doc_other",
                document_name="general_definitions.pdf",
                page_start=1,
                page_end=1,
                text="This document defines general words and terms used across administrative circulars.",
                original_rank=1,
                original_score=0.01,
                rerank_score=-2.5,
                final_rank=1,
            )
        ],
    },
    {
        "id": "scenario_03_multiple_evidence",
        "name": "3. Multiple evidence passages with comprehensive coverage",
        "query": "What are the duties of a Data Principal and what penalties apply for breach?",
        "expected_decision": TrustDecision.SUPPORTED,
        "expected_generation_called": True,
        "retrieval_candidates": [
            {
                "chunk_id": "chunk_sec15_01",
                "document_id": "doc_dpdp_act",
                "document_name": "DPDP_Act_2023.pdf",
                "page_start": 15,
                "page_end": 15,
                "text": "Section 15: Duties of Data Principal include complying with applicable laws and not registering false grievances.",
                "rank": 1,
                "score": 0.22,
            },
            {
                "chunk_id": "chunk_sec33_01",
                "document_id": "doc_dpdp_act",
                "document_name": "DPDP_Act_2023.pdf",
                "page_start": 28,
                "page_end": 28,
                "text": "Section 33: Schedule of penalties specifies fine up to ten thousand rupees for breach of Data Principal duties.",
                "rank": 2,
                "score": 0.19,
            },
        ],
        "evidence_chunks": [
            RerankedChunk(
                chunk_id="chunk_sec15_01",
                document_id="doc_dpdp_act",
                document_name="DPDP_Act_2023.pdf",
                page_start=15,
                page_end=15,
                text="Section 15: Duties of Data Principal include complying with applicable laws and not registering false grievances.",
                original_rank=1,
                original_score=0.22,
                rerank_score=3.5,
                final_rank=1,
            ),
            RerankedChunk(
                chunk_id="chunk_sec33_01",
                document_id="doc_dpdp_act",
                document_name="DPDP_Act_2023.pdf",
                page_start=28,
                page_end=28,
                text="Section 33: Schedule of penalties specifies fine up to ten thousand rupees for breach of Data Principal duties.",
                original_rank=2,
                original_score=0.19,
                rerank_score=3.1,
                final_rank=2,
            ),
        ],
    },
    {
        "id": "scenario_04_invalid_provenance",
        "name": "4. Invalid provenance metadata triggering trust gating",
        "query": "What are child data consent regulations?",
        "expected_decision": TrustDecision.INSUFFICIENT_EVIDENCE,
        "expected_generation_called": False,
        "retrieval_candidates": [
            {
                "chunk_id": "chunk_corrupt",
                "document_id": "doc_dpdp_act",
                "document_name": "dpdp_act_2023.pdf",
                "page_start": 10,
                "page_end": 4,  # Malformed: page_end < page_start
                "text": "Processing personal data of a child requires verifiable parental consent.",
                "rank": 1,
                "score": 0.15,
            }
        ],
        "evidence_chunks": [
            {
                "chunk_id": "chunk_corrupt",
                "document_id": "doc_dpdp_act",
                "document_name": "dpdp_act_2023.pdf",
                "page_start": 10,
                "page_end": 4,  # Malformed page range
                "text": "Processing personal data of a child requires verifiable parental consent.",
                "rerank_score": 3.9,
            }
        ],
    },
    {
        "id": "scenario_05_empty_retrieval",
        "name": "5. Empty retrieval results",
        "query": "Query producing zero retrieval matches",
        "expected_decision": TrustDecision.INSUFFICIENT_EVIDENCE,
        "expected_generation_called": False,
        "retrieval_candidates": [],
        "evidence_chunks": [],
    },
]


def run_evaluation(verbose: bool = True) -> Dict[str, Any]:
    """Execute all test scenarios and verify invariants."""
    passed_count = 0
    total_count = len(DEV_AGENT_EVAL_SCENARIOS)
    scenario_results: List[Dict[str, Any]] = []

    print("=" * 80)
    print("TRUSTRAG MULTI-AGENT ORCHESTRATION EVALUATION")
    print("=" * 80)

    for sc in DEV_AGENT_EVAL_SCENARIOS:
        sc_id = sc["id"]
        sc_name = sc["name"]
        query = sc["query"]
        expected_dec = sc["expected_decision"]
        expected_gen_called = sc["expected_generation_called"]

        # Setup mock retrieval agent
        retrieval_agent = MagicMock(spec=RetrievalAgent)
        retrieval_agent.execute.return_value = RetrievalAgentResult(
            query=query,
            results=sc["retrieval_candidates"],
            candidate_count=len(sc["retrieval_candidates"]),
            top_k=5,
        )

        # Setup mock evidence agent
        evidence_agent = MagicMock(spec=EvidenceAgent)
        evidence_agent.execute.return_value = EvidenceAgentResult(
            query=query,
            evidence=sc["evidence_chunks"],
            evidence_count=len(sc["evidence_chunks"]),
            top_k=5,
        )

        # Real TrustAgent to verify actual TrustEngine gating rules
        trust_agent = TrustAgent()

        # Mock generation agent
        generation_agent = MagicMock(spec=GenerationAgent)
        gen_answer = f"Generated answer for query with citation [1]."
        gen_res = GenerationResult(
            query=query,
            answer=gen_answer,
            model_name="qwen-eval",
            evidence_ids=[
                c.chunk_id if hasattr(c, "chunk_id") else c.get("chunk_id", "c1")
                for c in sc["evidence_chunks"]
            ],
            is_refusal=False,
        )
        generation_agent.execute.return_value = GenerationAgentResult(
            query=query,
            answer_text=gen_answer,
            is_refusal=False,
            is_supported=True,
            generation_result=gen_res,
        )

        # Mock citation agent
        citation_agent = MagicMock(spec=CitationAgent)
        cits = [
            Citation(
                index=idx + 1,
                chunk_id=c.chunk_id if hasattr(c, "chunk_id") else c.get("chunk_id", f"c_{idx}"),
                document_id=c.document_id if hasattr(c, "document_id") else c.get("document_id", "doc"),
                document_name=c.document_name if hasattr(c, "document_name") else c.get("document_name", "doc.pdf"),
                page_start=c.page_start if hasattr(c, "page_start") else c.get("page_start", 1),
                page_end=c.page_end if hasattr(c, "page_end") else c.get("page_end", 1),
                text_snippet="Snippet",
                formatted_reference=f"[{idx+1}] doc.pdf, page 1",
            )
            for idx, c in enumerate(sc["evidence_chunks"])
            if (isinstance(c, dict) and c.get("page_start", 1) <= c.get("page_end", 1)) or hasattr(c, "page_start")
        ]
        citation_agent.execute.return_value = CitationAgentResult(
            query=query,
            citations=cits[:1],
            citation_count=len(cits[:1]),
            formatted_citations="### References\n- [1] doc.pdf",
        )

        orchestrator = TrustRAGOrchestrator(
            retrieval_agent=retrieval_agent,
            evidence_agent=evidence_agent,
            trust_agent=trust_agent,
            generation_agent=generation_agent,
            citation_agent=citation_agent,
        )

        t_start = time.perf_counter()
        result = orchestrator.execute(query)
        elapsed = time.perf_counter() - t_start

        # Verification Invariants:
        # 1. Trust Decision matches expected
        decision_match = (result.decision == expected_dec)

        # 2. GenerationAgent invocation matches expected_generation_called
        gen_called_actual = generation_agent.execute.called
        gen_invariant_pass = (gen_called_actual == expected_gen_called)

        # 3. CitationAgent invocation matches expected_generation_called
        cit_called_actual = citation_agent.execute.called
        cit_invariant_pass = (cit_called_actual == expected_gen_called)

        # 4. Refusal flags
        refusal_pass = (result.is_refusal == (expected_dec == TrustDecision.INSUFFICIENT_EVIDENCE))

        # 5. Trace verification
        trace_events = {e.agent_name: e for e in result.trace.events} if result.trace else {}
        if expected_dec == TrustDecision.INSUFFICIENT_EVIDENCE:
            trace_pass = (
                trace_events.get("GenerationAgent") is not None
                and trace_events["GenerationAgent"].status == AgentEventStatus.SKIPPED
                and trace_events.get("CitationAgent") is not None
                and trace_events["CitationAgent"].status == AgentEventStatus.SKIPPED
            )
        else:
            trace_pass = (
                trace_events.get("GenerationAgent") is not None
                and trace_events["GenerationAgent"].status == AgentEventStatus.COMPLETED
                and trace_events.get("CitationAgent") is not None
                and trace_events["CitationAgent"].status == AgentEventStatus.COMPLETED
            )

        scenario_passed = (
            decision_match
            and gen_invariant_pass
            and cit_invariant_pass
            and refusal_pass
            and trace_pass
        )

        if scenario_passed:
            passed_count += 1

        status_tag = "[PASS]" if scenario_passed else "[FAIL]"
        print(f"\n{status_tag} {sc_name}")
        print(f"       Decision: {result.decision.value} (Expected: {expected_dec.value})")
        print(f"       GenerationAgent Called: {gen_called_actual} (Expected: {expected_gen_called})")
        print(f"       CitationAgent Called:   {cit_called_actual} (Expected: {expected_gen_called})")
        print(f"       Is Refusal: {result.is_refusal}, Confidence: {result.confidence_score:.3f}")
        print(f"       Latency: {elapsed:.4f}s")

        if verbose and result.trace:
            print("       Trace Summary:")
            for evt in result.trace.events:
                r = f" ({evt.reason})" if evt.reason else ""
                print(f"         * {evt.agent_name}: {evt.status.value}{r}")

        scenario_results.append({
            "id": sc_id,
            "name": sc_name,
            "passed": scenario_passed,
            "decision": result.decision.value,
            "expected_decision": expected_dec.value,
            "generation_called": gen_called_actual,
            "expected_generation_called": expected_gen_called,
            "latency_seconds": elapsed,
        })

    print("\n" + "=" * 80)
    print(f"OVERALL SUMMARY: {passed_count}/{total_count} scenarios passed.")
    print("=" * 80)

    return {
        "passed_count": passed_count,
        "total_count": total_count,
        "all_passed": (passed_count == total_count),
        "scenarios": scenario_results,
    }


def main():
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(description="Evaluate Multi-Agent TrustRAG Orchestration")
    parser.add_argument("--json", action="store_true", help="Print output as JSON")
    args = parser.parse_args()

    results = run_evaluation(verbose=not args.json)

    if args.json:
        print(json.dumps(results, indent=2))

    if not results["all_passed"]:
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
