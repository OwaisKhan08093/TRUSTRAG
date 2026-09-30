"""CLI and evaluation utility for measuring dense, sparse, and hybrid retrieval performance."""

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
    CHUNKS_OUTPUT_FILE,
    EMBEDDING_METADATA_FILE,
    FAISS_INDEX_FILE,
)
from backend.app.retrieval.bm25_retriever import BM25Retriever
from backend.app.retrieval.hybrid_retriever import HybridRetriever
from backend.app.retrieval.retriever import VectorRetriever

DEFAULT_EVALUATION_FILE = PROJECT_ROOT / "data" / "evaluation" / "retrieval_queries.json"


def compute_metrics(
    eval_records: List[Dict[str, Any]],
) -> Dict[str, float]:
    """Calculate retrieval metrics (Hit@1, Hit@3, Hit@5, MRR) across evaluated queries.

    Args:
        eval_records: List of dictionaries containing 'rank' (1-indexed int or None if not found).

    Returns:
        Dictionary of calculated metrics as floats in range [0.0, 1.0].
    """
    if not eval_records:
        return {
            "hit_at_1": 0.0,
            "hit_at_3": 0.0,
            "hit_at_5": 0.0,
            "mrr": 0.0,
        }

    total_queries = len(eval_records)
    hits_at_1 = 0
    hits_at_3 = 0
    hits_at_5 = 0
    reciprocal_ranks: List[float] = []

    for record in eval_records:
        rank = record.get("rank")
        if rank is not None and isinstance(rank, int) and rank > 0:
            if rank <= 1:
                hits_at_1 += 1
            if rank <= 3:
                hits_at_3 += 1
            if rank <= 5:
                hits_at_5 += 1
            reciprocal_ranks.append(1.0 / float(rank))
        else:
            reciprocal_ranks.append(0.0)

    return {
        "hit_at_1": hits_at_1 / total_queries,
        "hit_at_3": hits_at_3 / total_queries,
        "hit_at_5": hits_at_5 / total_queries,
        "mrr": sum(reciprocal_ranks) / total_queries,
    }


def evaluate_retriever(
    retriever: Any,
    queries: List[Dict[str, Any]],
    top_k: int = 5,
) -> Dict[str, Any]:
    """Evaluate a single retriever against a loaded query list.

    Args:
        retriever: VectorRetriever, BM25Retriever, or HybridRetriever.
        queries: List of query dictionaries with 'query' and expected targets.
        top_k: Top-K chunks to retrieve per query.

    Returns:
        Dictionary with metrics and individual query records.
    """
    eval_records: List[Dict[str, Any]] = []

    for item in queries:
        query_text = item["query"]
        expected_chunk_id = item.get("expected_chunk_id")
        expected_page = item.get("expected_page")

        results = retriever.retrieve(query_text, top_k=top_k)

        found_rank: Optional[int] = None
        for res in results:
            match = False
            if expected_chunk_id and res["chunk_id"] == expected_chunk_id:
                match = True
            elif expected_page and res["page_start"] <= expected_page <= res["page_end"]:
                match = True

            if match:
                found_rank = res["rank"]
                break

        eval_records.append({
            "query": query_text,
            "expected_chunk_id": expected_chunk_id,
            "expected_page": expected_page,
            "rank": found_rank,
            "top_match": results[0]["chunk_id"] if results else None,
            "top_score": results[0]["score"] if results else None,
        })

    metrics = compute_metrics(eval_records)
    return {
        "metrics": metrics,
        "records": eval_records,
    }


def evaluate_retrieval(
    queries_file: Union[str, Path] = DEFAULT_EVALUATION_FILE,
    retriever: Optional[Any] = None,
    top_k: int = 5,
) -> Dict[str, Any]:
    """Run retrieval evaluation against a query dataset and compute metrics.

    Args:
        queries_file: Path to evaluation JSON file.
        retriever: Optional retriever instance (defaults to VectorRetriever).
        top_k: Number of chunks to retrieve per query (default: 5).

    Returns:
        Dictionary containing evaluation summary, query details, and metrics.
    """
    path = Path(queries_file)
    if not path.exists():
        raise FileNotFoundError(f"Evaluation queries file not found at '{path}'.")

    with open(path, "r", encoding="utf-8") as f:
        queries = json.load(f)

    if not isinstance(queries, list) or len(queries) == 0:
        raise ValueError(f"Evaluation file '{path}' is empty or not a valid JSON list.")

    if retriever is None:
        retriever = VectorRetriever()

    res = evaluate_retriever(retriever, queries, top_k=top_k)

    return {
        "num_queries": len(queries),
        "top_k": top_k,
        "metrics": res["metrics"],
        "records": res["records"],
    }


def evaluate_hybrid_pipeline(
    queries_file: Union[str, Path] = DEFAULT_EVALUATION_FILE,
    top_k: int = 5,
    dense_retriever: Optional[VectorRetriever] = None,
    sparse_retriever: Optional[BM25Retriever] = None,
    hybrid_retriever: Optional[HybridRetriever] = None,
) -> Dict[str, Any]:
    """Run comprehensive comparative evaluation across Dense, Sparse (BM25), and Hybrid retrieval.

    Args:
        queries_file: Path to evaluation JSON file.
        top_k: Number of chunks to retrieve per query.
        dense_retriever: Optional pre-configured VectorRetriever.
        sparse_retriever: Optional pre-configured BM25Retriever.
        hybrid_retriever: Optional pre-configured HybridRetriever.

    Returns:
        Dictionary with num_queries, top_k, and separate evaluations for dense, sparse, and hybrid.
    """
    path = Path(queries_file)
    if not path.exists():
        raise FileNotFoundError(f"Evaluation queries file not found at '{path}'.")

    with open(path, "r", encoding="utf-8") as f:
        queries = json.load(f)

    if not isinstance(queries, list) or len(queries) == 0:
        raise ValueError(f"Evaluation file '{path}' is empty or not a valid JSON list.")

    dense = dense_retriever or VectorRetriever()
    sparse = sparse_retriever or BM25Retriever()
    hybrid = hybrid_retriever or HybridRetriever(dense_retriever=dense, sparse_retriever=sparse)

    dense_eval = evaluate_retriever(dense, queries, top_k=top_k)
    sparse_eval = evaluate_retriever(sparse, queries, top_k=top_k)
    hybrid_eval = evaluate_retriever(hybrid, queries, top_k=top_k)

    return {
        "num_queries": len(queries),
        "top_k": top_k,
        "dense": dense_eval,
        "sparse": sparse_eval,
        "hybrid": hybrid_eval,
    }


def run_evaluation_cli(
    queries_file: Path = DEFAULT_EVALUATION_FILE,
    top_k: int = 5,
) -> int:
    """CLI runner executing comparative retrieval evaluation across Dense, Sparse, and Hybrid."""
    print("=" * 70)
    print(" TrustRAG Hybrid Retrieval Engine Evaluation (Milestone 8)")
    print("=" * 70)
    print("\n[NOTE] Evaluated on local development benchmark dataset.")
    print("       Dataset is intentionally small (unit benchmark). Not a production claim.\n")

    try:
        results = evaluate_hybrid_pipeline(queries_file=queries_file, top_k=top_k)
    except Exception as exc:
        print(f"Error during evaluation: {exc}", file=sys.stderr)
        return 1

    num_queries = results["num_queries"]
    print(f"Total Queries Evaluated: {num_queries}")
    print(f"Top-K Limit: {top_k}\n")

    print("-" * 70)
    print(f"{'Metric':<12} | {'Dense (FAISS)':<16} | {'Sparse (BM25)':<16} | {'Hybrid (RRF)':<16}")
    print("-" * 70)

    d_m = results["dense"]["metrics"]
    s_m = results["sparse"]["metrics"]
    h_m = results["hybrid"]["metrics"]

    print(f"{'Hit@1':<12} | {d_m['hit_at_1'] * 100:>14.1f}% | {s_m['hit_at_1'] * 100:>14.1f}% | {h_m['hit_at_1'] * 100:>14.1f}%")
    print(f"{'Hit@3':<12} | {d_m['hit_at_3'] * 100:>14.1f}% | {s_m['hit_at_3'] * 100:>14.1f}% | {h_m['hit_at_3'] * 100:>14.1f}%")
    print(f"{'Hit@5':<12} | {d_m['hit_at_5'] * 100:>14.1f}% | {s_m['hit_at_5'] * 100:>14.1f}% | {h_m['hit_at_5'] * 100:>14.1f}%")
    print(f"{'MRR':<12} | {d_m['mrr']:>15.4f} | {s_m['mrr']:>15.4f} | {h_m['mrr']:>15.4f}")
    print("-" * 70)

    print("\nQuery Breakdown (Rank Comparison: Dense / Sparse / Hybrid):")
    for idx in range(num_queries):
        d_rec = results["dense"]["records"][idx]
        s_rec = results["sparse"]["records"][idx]
        h_rec = results["hybrid"]["records"][idx]

        q = d_rec["query"]
        d_rank = f"Rank {d_rec['rank']}" if d_rec['rank'] else "Miss"
        s_rank = f"Rank {s_rec['rank']}" if s_rec['rank'] else "Miss"
        h_rank = f"Rank {h_rec['rank']}" if h_rec['rank'] else "Miss"

        print(f"  [{idx + 1}] \"{q}\"")
        print(f"       Dense: {d_rank:<10} | Sparse: {s_rank:<10} | Hybrid: {h_rank:<10}")

    print("\nOverall Status: SUCCESS")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="TrustRAG Hybrid Retrieval Benchmark Evaluator (Milestone 8)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--queries",
        type=str,
        default=str(DEFAULT_EVALUATION_FILE),
        help="Path to evaluation queries JSON file.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Top-K results to retrieve per query.",
    )

    args = parser.parse_args()

    q_path = Path(args.queries)
    if not q_path.is_absolute():
        q_path = (PROJECT_ROOT / q_path).resolve()

    return run_evaluation_cli(queries_file=q_path, top_k=args.top_k)


if __name__ == "__main__":
    sys.exit(main())
