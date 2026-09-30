"""Reciprocal Rank Fusion (RRF) algorithm for combining multi-retriever rankings."""

from typing import Any, Dict, List, Optional, Sequence

from backend.app.config import DEFAULT_RRF_K, MAX_RRF_K, MIN_RRF_K


class RRFError(Exception):
    """Raised when Reciprocal Rank Fusion receives invalid arguments or malformed data."""
    pass


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[Dict[str, Any]]],
    k: int = DEFAULT_RRF_K,
    top_k: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Merge and re-rank multiple result lists using Reciprocal Rank Fusion (RRF).

    Formula:
        RRF_Score(d) = sum_{r in ranked_lists} (1 / (k + rank(r, d)))
        where rank(r, d) is the 1-based rank position of document d in list r.

    Args:
        ranked_lists: Sequence of ranked result lists, where each result is a dictionary
                      containing at least a 'chunk_id' key.
        k: Smoothing constant (positive integer, default: DEFAULT_RRF_K). Controls the balance
           between high-ranked and lower-ranked items.
        top_k: Optional limit on the number of returned fused results.

    Returns:
        List of fused and re-ranked dictionaries containing chunk metadata, updated
        1-based 'rank', and 'score' equal to the computed RRF score (sorted descending).

    Raises:
        TypeError: If ranked_lists is not a sequence, or elements are not dicts with str chunk_id.
        ValueError: If k is out of bounds [MIN_RRF_K, MAX_RRF_K] or top_k < 1.
    """
    if not isinstance(ranked_lists, (list, tuple)):
        raise TypeError(f"ranked_lists must be a sequence of lists, got {type(ranked_lists).__name__}.")

    if not isinstance(k, int) or k < MIN_RRF_K or k > MAX_RRF_K:
        raise ValueError(f"RRF constant k must be an integer between {MIN_RRF_K} and {MAX_RRF_K}, got {k}.")

    if top_k is not None:
        if not isinstance(top_k, int) or top_k < 1:
            raise ValueError(f"top_k must be a positive integer >= 1, got {top_k}.")

    fused_scores: Dict[str, float] = {}
    chunk_records: Dict[str, Dict[str, Any]] = {}

    for list_idx, result_list in enumerate(ranked_lists):
        if not isinstance(result_list, (list, tuple)):
            raise TypeError(
                f"Element at index {list_idx} in ranked_lists must be a sequence, got {type(result_list).__name__}."
            )

        seen_in_this_list = set()
        for pos, item in enumerate(result_list, start=1):
            if not isinstance(item, dict):
                raise TypeError(
                    f"Result item at position {pos} in list {list_idx} must be a dictionary, got {type(item).__name__}."
                )

            chunk_id = item.get("chunk_id")
            if not chunk_id or not isinstance(chunk_id, str):
                raise TypeError(
                    f"Result item at position {pos} in list {list_idx} missing valid string 'chunk_id'."
                )

            # Guard against internal duplicates within the same list
            if chunk_id in seen_in_this_list:
                continue
            seen_in_this_list.add(chunk_id)

            contribution = 1.0 / float(k + pos)
            fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + contribution

            # Preserve metadata from first occurrence
            if chunk_id not in chunk_records:
                record_copy = dict(item)
                chunk_records[chunk_id] = record_copy

    if not fused_scores:
        return []

    # Sort chunks: descending by fused score, ascending by chunk_id for tie-breaking
    sorted_chunk_ids = sorted(
        fused_scores.keys(),
        key=lambda cid: (-fused_scores[cid], cid),
    )

    if top_k is not None:
        sorted_chunk_ids = sorted_chunk_ids[:top_k]

    fused_results: List[Dict[str, Any]] = []
    for new_rank, cid in enumerate(sorted_chunk_ids, start=1):
        record = dict(chunk_records[cid])
        record["rank"] = new_rank
        record["score"] = float(fused_scores[cid])
        fused_results.append(record)

    return fused_results
