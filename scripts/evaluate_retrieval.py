"""CLI and evaluation utility for measuring vector retrieval performance on test queries."""

import argparse
import json
import sys
from pathlib import Path
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
from backend.app.embeddings.encoder import EmbeddingEncoder
from backend.app.retrieval.faiss_index import FaissVectorIndex
from backend.app.retrieval.metadata import ChunkMetadataResolver
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


def evaluate_retrieval(
    queries_file: Union[str, Path] = DEFAULT_EVALUATION_FILE,
    retriever: Optional[VectorRetriever] = None,
    top_k: int = 5,
) -> Dict[str, Any]:
    """Run retrieval evaluation against a query dataset and compute metrics.

    Args:
        queries_file: Path to evaluation JSON file.
        retriever: Optional pre-configured VectorRetriever instance.
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
        "num_queries": len(queries),
        "top_k": top_k,
        "metrics": metrics,
        "records": eval_records,
    }


def run_evaluation_cli(
    queries_file: Path = DEFAULT_EVALUATION_FILE,
    top_k: int = 5,
) -> int:
    """CLI runner executing retrieval evaluation and printing formatted metrics."""
    print("TrustRAG Semantic Retrieval Evaluation\n")
    print("Note: Evaluated on local development benchmark dataset (not a production claim).\n")

    try:
        results = evaluate_retrieval(queries_file=queries_file, top_k=top_k)
    except Exception as exc:
        print(f"Error during evaluation: {exc}", file=sys.stderr)
        return 1

    metrics = results["metrics"]
    num_queries = results["num_queries"]

    print(f"Queries Evaluated: {num_queries}")
    print(f"Top-K: {top_k}\n")
    print("Metrics:")
    print(f"  Hit@1 : {metrics['hit_at_1'] * 100:.1f}%")
    print(f"  Hit@3 : {metrics['hit_at_3'] * 100:.1f}%")
    print(f"  Hit@5 : {metrics['hit_at_5'] * 100:.1f}%")
    print(f"  MRR   : {metrics['mrr']:.4f}\n")

    print("Query Breakdown:")
    for idx, r in enumerate(results["records"], start=1):
        status = f"Hit (Rank {r['rank']})" if r['rank'] else "Miss"
        print(f"  [{idx}] \"{r['query']}\" -> {status} (Top Score: {r['top_score']:.4f})")

    print("\nOverall Status: SUCCESS")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="TrustRAG Retrieval Benchmark Evaluator (Milestone 9)",
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
