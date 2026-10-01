"""CLI and evaluation utility comparing Hybrid Retrieval against Hybrid + Cross-Encoder Reranking."""

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Union

# Add project root to sys.path so backend imports work seamlessly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.config import (
    DEFAULT_RERANKER_CANDIDATE_K,
    DEFAULT_RERANKER_TOP_K,
)
from backend.app.reranking.pipeline import RerankingPipeline
from backend.app.retrieval.hybrid_retriever import HybridRetriever
from scripts.evaluate_retrieval import DEFAULT_EVALUATION_FILE, compute_metrics


def evaluate_reranked_pipeline(
    queries_file: Union[str, Path] = DEFAULT_EVALUATION_FILE,
    pipeline: Optional[RerankingPipeline] = None,
    hybrid_retriever: Optional[HybridRetriever] = None,
    candidate_k: int = DEFAULT_RERANKER_CANDIDATE_K,
    final_k: int = DEFAULT_RERANKER_TOP_K,
) -> Dict[str, Any]:
    """Run comparative evaluation of Hybrid Retrieval vs. Hybrid + Cross-Encoder Reranking.

    Args:
        queries_file: Path to evaluation queries JSON file.
        pipeline: Optional pre-configured RerankingPipeline.
        hybrid_retriever: Optional pre-configured HybridRetriever.
        candidate_k: Number of hybrid candidates to retrieve for reranker.
        final_k: Number of top reranked chunks to evaluate.

    Returns:
        Dictionary with query count, metrics, and records for both systems.
    """
    path = Path(queries_file)
    if not path.exists():
        raise FileNotFoundError(f"Evaluation queries file not found at '{path}'.")

    with open(path, "r", encoding="utf-8") as f:
        queries = json.load(f)

    if not isinstance(queries, list) or len(queries) == 0:
        raise ValueError(f"Evaluation file '{path}' is empty or not a valid JSON list.")

    hybrid = hybrid_retriever or HybridRetriever()
    pipe = pipeline or RerankingPipeline(retriever=hybrid)

    hybrid_records: List[Dict[str, Any]] = []
    rerank_records: List[Dict[str, Any]] = []

    for item in queries:
        query_text = item["query"]
        expected_chunk_id = item.get("expected_chunk_id")
        expected_page = item.get("expected_page")

        # 1. Evaluate pure Hybrid Retrieval
        h_results = hybrid.retrieve(query_text, top_k=final_k)
        h_rank: Optional[int] = None
        for res in h_results:
            match = False
            if expected_chunk_id and res["chunk_id"] == expected_chunk_id:
                match = True
            elif expected_page and res["page_start"] <= expected_page <= res["page_end"]:
                match = True

            if match:
                h_rank = res["rank"]
                break

        hybrid_records.append({
            "query": query_text,
            "expected_chunk_id": expected_chunk_id,
            "expected_page": expected_page,
            "rank": h_rank,
            "top_match": h_results[0]["chunk_id"] if h_results else None,
            "top_score": h_results[0]["score"] if h_results else None,
        })

        # 2. Evaluate Hybrid + Cross-Encoder Reranking
        r_results = pipe.retrieve(query_text, top_k=final_k, candidate_k=candidate_k)
        r_rank: Optional[int] = None
        for res in r_results:
            match = False
            if expected_chunk_id and res.chunk_id == expected_chunk_id:
                match = True
            elif expected_page and res.page_start <= expected_page <= res.page_end:
                match = True

            if match:
                r_rank = res.final_rank
                break

        rerank_records.append({
            "query": query_text,
            "expected_chunk_id": expected_chunk_id,
            "expected_page": expected_page,
            "rank": r_rank,
            "top_match": r_results[0].chunk_id if r_results else None,
            "top_score": r_results[0].rerank_score if r_results else None,
        })

    h_metrics = compute_metrics(hybrid_records)
    r_metrics = compute_metrics(rerank_records)

    return {
        "num_queries": len(queries),
        "candidate_k": candidate_k,
        "final_k": final_k,
        "hybrid": {
            "metrics": h_metrics,
            "records": hybrid_records,
        },
        "reranked": {
            "metrics": r_metrics,
            "records": rerank_records,
        },
    }


def run_reranking_evaluation_cli(
    queries_file: Path = DEFAULT_EVALUATION_FILE,
    candidate_k: int = DEFAULT_RERANKER_CANDIDATE_K,
    final_k: int = DEFAULT_RERANKER_TOP_K,
) -> int:
    """CLI runner executing reranking benchmark evaluation and printing formatted metrics."""
    print("=" * 70)
    print(" TrustRAG Cross-Encoder Reranking Evaluation (Milestone 8)")
    print("=" * 70)
    print("\n[NOTE] Evaluated on local development benchmark dataset.")
    print("       Dataset is intentionally small (unit benchmark). Not a production claim.\n")

    try:
        results = evaluate_reranked_pipeline(
            queries_file=queries_file,
            candidate_k=candidate_k,
            final_k=final_k,
        )
    except Exception as exc:
        print(f"Error during evaluation: {exc}", file=sys.stderr)
        return 1

    num_queries = results["num_queries"]
    print(f"Total Queries Evaluated: {num_queries}")
    print(f"Candidate K (Hybrid Pool): {candidate_k}")
    print(f"Final K (Reranked Top-K) : {final_k}\n")

    print("-" * 70)
    print(f"{'Metric':<12} | {'Hybrid (RRF)':<24} | {'Hybrid + Reranker':<24}")
    print("-" * 70)

    h_m = results["hybrid"]["metrics"]
    r_m = results["reranked"]["metrics"]

    print(f"{'Hit@1':<12} | {h_m['hit_at_1'] * 100:>22.1f}% | {r_m['hit_at_1'] * 100:>22.1f}%")
    print(f"{'Hit@3':<12} | {h_m['hit_at_3'] * 100:>22.1f}% | {r_m['hit_at_3'] * 100:>22.1f}%")
    print(f"{'Hit@5':<12} | {h_m['hit_at_5'] * 100:>22.1f}% | {r_m['hit_at_5'] * 100:>22.1f}%")
    print(f"{'MRR':<12} | {h_m['mrr']:>23.4f} | {r_m['mrr']:>23.4f}")
    print("-" * 70)

    print("\nQuery Breakdown (Rank Comparison: Hybrid vs. Hybrid + Reranker):")
    for idx in range(num_queries):
        h_rec = results["hybrid"]["records"][idx]
        r_rec = results["reranked"]["records"][idx]

        q = h_rec["query"]
        h_rank = f"Rank {h_rec['rank']}" if h_rec['rank'] else "Miss"
        r_rank = f"Rank {r_rec['rank']}" if r_rec['rank'] else "Miss"

        print(f"  [{idx + 1}] \"{q}\"")
        print(f"       Hybrid: {h_rank:<12} | Hybrid + Reranker: {r_rank:<12}")

    print("\nOverall Status: SUCCESS")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="TrustRAG Cross-Encoder Reranking Benchmark Evaluator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--queries",
        type=str,
        default=str(DEFAULT_EVALUATION_FILE),
        help="Path to evaluation queries JSON file.",
    )
    parser.add_argument(
        "--candidate-k",
        type=int,
        default=DEFAULT_RERANKER_CANDIDATE_K,
        help="Number of candidates retrieved from hybrid stage.",
    )
    parser.add_argument(
        "--final-k",
        type=int,
        default=DEFAULT_RERANKER_TOP_K,
        help="Number of final reranked passages to evaluate.",
    )

    args = parser.parse_args()
    q_path = Path(args.queries)
    if not q_path.is_absolute():
        q_path = (PROJECT_ROOT / q_path).resolve()

    return run_reranking_evaluation_cli(
        queries_file=q_path,
        candidate_k=args.candidate_k,
        final_k=args.final_k,
    )


if __name__ == "__main__":
    sys.exit(main())
