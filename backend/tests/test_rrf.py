"""Unit tests for Reciprocal Rank Fusion (RRF)."""

import pytest

from backend.app.retrieval.rrf import DEFAULT_RRF_K, reciprocal_rank_fusion


def test_rrf_basic_two_lists():
    """Verify RRF combines ranks from two lists and prioritizes items appearing in both."""
    list_dense = [
        {"chunk_id": "c1", "text": "Dense rank 1", "document_id": "docA"},
        {"chunk_id": "c2", "text": "Dense rank 2", "document_id": "docB"},
    ]
    list_sparse = [
        {"chunk_id": "c2", "text": "Dense rank 2", "document_id": "docB"},
        {"chunk_id": "c3", "text": "Sparse rank 2", "document_id": "docC"},
    ]

    # With k=60:
    # c2: 1/(60+2) + 1/(60+1) = 1/62 + 1/61 ≈ 0.032522
    # c1: 1/(60+1) = 1/61 ≈ 0.016393
    # c3: 1/(60+2) = 1/62 ≈ 0.016129
    fused = reciprocal_rank_fusion([list_dense, list_sparse], k=60)

    assert len(fused) == 3
    assert fused[0]["chunk_id"] == "c2"
    assert fused[0]["rank"] == 1
    assert fused[0]["score"] == pytest.approx(1 / 62 + 1 / 61)
    assert fused[0]["document_id"] == "docB"

    assert fused[1]["chunk_id"] == "c1"
    assert fused[1]["rank"] == 2
    assert fused[1]["score"] == pytest.approx(1 / 61)

    assert fused[2]["chunk_id"] == "c3"
    assert fused[2]["rank"] == 3
    assert fused[2]["score"] == pytest.approx(1 / 62)


def test_rrf_with_top_k():
    """Verify RRF limits output to top_k items when requested."""
    list1 = [{"chunk_id": f"c{i}"} for i in range(10)]
    fused = reciprocal_rank_fusion([list1], top_k=3)
    assert len(fused) == 3
    assert [item["rank"] for item in fused] == [1, 2, 3]


def test_rrf_custom_k():
    """Verify RRF computes correct scores with custom k."""
    list1 = [{"chunk_id": "c1"}]
    fused = reciprocal_rank_fusion([list1], k=10)
    assert fused[0]["score"] == pytest.approx(1 / 11)


def test_rrf_empty_inputs():
    """Verify RRF handles empty list sequences safely."""
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[]]) == []
    assert reciprocal_rank_fusion([[], []]) == []


def test_rrf_single_list():
    """Verify RRF on a single list preserves relative rankings."""
    items = [{"chunk_id": "c1"}, {"chunk_id": "c2"}, {"chunk_id": "c3"}]
    fused = reciprocal_rank_fusion([items])
    assert [x["chunk_id"] for x in fused] == ["c1", "c2", "c3"]


def test_rrf_duplicate_in_same_list():
    """Verify internal duplicates in the same list don't duplicate fusion score."""
    items = [{"chunk_id": "c1"}, {"chunk_id": "c1"}]
    fused = reciprocal_rank_fusion([items], k=60)
    assert len(fused) == 1
    assert fused[0]["score"] == pytest.approx(1 / 61)


def test_rrf_validation_errors():
    """Verify RRF raises appropriate exceptions on invalid inputs."""
    with pytest.raises(TypeError, match="must be a sequence"):
        reciprocal_rank_fusion("invalid")  # type: ignore

    with pytest.raises(TypeError, match="must be a sequence"):
        reciprocal_rank_fusion([123])  # type: ignore

    with pytest.raises(TypeError, match="must be a dictionary"):
        reciprocal_rank_fusion([["not a dict"]])  # type: ignore

    with pytest.raises(TypeError, match="missing valid string 'chunk_id'"):
        reciprocal_rank_fusion([[{"invalid": "no chunk id"}]])

    with pytest.raises(TypeError, match="missing valid string 'chunk_id'"):
        reciprocal_rank_fusion([[{"chunk_id": 123}]])

    with pytest.raises(ValueError, match="RRF constant k must be an integer between"):
        reciprocal_rank_fusion([[]], k=0)

    with pytest.raises(ValueError, match="RRF constant k must be an integer between"):
        reciprocal_rank_fusion([[]], k=-1)

    with pytest.raises(ValueError, match="RRF constant k must be an integer between"):
        reciprocal_rank_fusion([[]], k=5000)

    with pytest.raises(ValueError, match="top_k must be a positive integer"):
        reciprocal_rank_fusion([[{"chunk_id": "c1"}]], top_k=0)


def test_rrf_tie_breaking():
    """Verify RRF breaks score ties deterministically by chunk_id."""
    list1 = [{"chunk_id": "beta"}]
    list2 = [{"chunk_id": "alpha"}]
    # Both have rank 1 in single separate lists -> identical score 1/61
    fused = reciprocal_rank_fusion([list1, list2], k=60)
    assert len(fused) == 2
    assert fused[0]["score"] == fused[1]["score"]
    assert fused[0]["chunk_id"] == "alpha"
    assert fused[1]["chunk_id"] == "beta"
