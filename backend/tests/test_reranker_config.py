"""Unit tests for cross-encoder reranker configuration and parameter validation."""

import pytest

from backend.app.config import (
    DEFAULT_RERANKER_BATCH_SIZE,
    DEFAULT_RERANKER_CANDIDATE_K,
    DEFAULT_RERANKER_MODEL,
    DEFAULT_RERANKER_TOP_K,
    MAX_RERANKER_TOP_K,
    MIN_RERANKER_TOP_K,
    validate_reranker_config,
)


def test_reranker_config_defaults():
    """Verify default configuration constants exist and have expected types/values."""
    assert DEFAULT_RERANKER_MODEL == "cross-encoder/ms-marco-MiniLM-L-6-v2"
    assert DEFAULT_RERANKER_TOP_K == 3
    assert DEFAULT_RERANKER_CANDIDATE_K == 5
    assert DEFAULT_RERANKER_BATCH_SIZE == 16
    assert MAX_RERANKER_TOP_K == 20
    assert MIN_RERANKER_TOP_K == 1


def test_validate_reranker_config_success():
    """Verify validate_reranker_config succeeds with default and valid custom parameters."""
    validate_reranker_config()  # defaults
    validate_reranker_config(
        candidate_k=10,
        final_k=5,
        batch_size=32,
        model_name="custom-model",
    )


def test_validate_reranker_config_invalid_candidate_k():
    """Verify validation fails when candidate_k <= 0 or not an int."""
    with pytest.raises(ValueError, match="candidate_k must be a positive integer"):
        validate_reranker_config(candidate_k=0)

    with pytest.raises(ValueError, match="candidate_k must be a positive integer"):
        validate_reranker_config(candidate_k=-3)


def test_validate_reranker_config_invalid_final_k():
    """Verify validation fails when final_k <= 0 or final_k > candidate_k."""
    with pytest.raises(ValueError, match="final_k must be a positive integer"):
        validate_reranker_config(final_k=0)

    with pytest.raises(ValueError, match="cannot exceed candidate_k"):
        validate_reranker_config(candidate_k=5, final_k=6)


def test_validate_reranker_config_exceeds_max_k():
    """Verify validation fails when k exceeds MAX_RERANKER_TOP_K."""
    with pytest.raises(ValueError, match="cannot exceed maximum limit"):
        validate_reranker_config(candidate_k=25, final_k=10)


def test_validate_reranker_config_invalid_batch_size():
    """Verify validation fails when batch_size <= 0."""
    with pytest.raises(ValueError, match="batch_size must be a positive integer"):
        validate_reranker_config(batch_size=0)


def test_validate_reranker_config_invalid_model_name():
    """Verify validation fails when model_name is empty or whitespace."""
    with pytest.raises(ValueError, match="model_name must be a non-empty string"):
        validate_reranker_config(model_name="")

    with pytest.raises(ValueError, match="model_name must be a non-empty string"):
        validate_reranker_config(model_name="   ")


def test_reranker_config_dataclass():
    """Verify RerankerConfig creates valid objects and enforces invariants."""
    from backend.app.reranking.config import RerankerConfig, RerankerConfigurationError

    cfg = RerankerConfig(candidate_k=8, final_k=4, batch_size=8)
    assert cfg.candidate_k == 8
    assert cfg.final_k == 4
    assert cfg.to_dict()["candidate_k"] == 8

    # from_dict roundtrip
    restored = RerankerConfig.from_dict(cfg.to_dict())
    assert restored == cfg

    # Invariant failure: final_k > candidate_k
    with pytest.raises(RerankerConfigurationError, match="cannot exceed candidate_k"):
        RerankerConfig(candidate_k=2, final_k=4)

    # Non-dict in from_dict
    with pytest.raises(RerankerConfigurationError, match="must be a dictionary"):
        RerankerConfig.from_dict("not a dict")  # type: ignore
