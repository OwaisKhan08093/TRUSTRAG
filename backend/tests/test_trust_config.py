"""Unit tests for centralized Trust Engine Configuration (Phase 5 Milestone 8)."""

import pytest

from backend.app.trust.config import TrustConfig, TrustConfigError
from backend.app.trust.engine import TrustEngine


def test_trust_config_defaults():
    """Verify standard sensible defaults."""
    cfg = TrustConfig()
    assert cfg.min_confidence_threshold == 0.50
    assert cfg.min_groundedness_threshold == 0.45
    assert cfg.min_coverage_threshold == 0.30
    assert cfg.min_evidence_count == 1
    assert cfg.require_valid_provenance is True
    assert cfg.relevance_decay_factor == 0.50
    assert cfg.weight_relevance == 0.55
    assert cfg.weight_coverage == 0.35
    assert cfg.weight_provenance == 0.10


def test_trust_config_to_from_dict():
    """Verify dictionary serialization and deserialization."""
    cfg = TrustConfig(
        min_confidence_threshold=0.75,
        min_evidence_count=2,
    )
    d = cfg.to_dict()
    assert d["min_confidence_threshold"] == 0.75
    assert d["min_evidence_count"] == 2

    restored = TrustConfig.from_dict(d)
    assert restored.min_confidence_threshold == 0.75
    assert restored.min_evidence_count == 2
    assert restored == cfg


def test_trust_config_invalid_thresholds():
    """Verify out-of-range thresholds raise TrustConfigError."""
    with pytest.raises(TrustConfigError, match="must be a float bounded in"):
        TrustConfig(min_confidence_threshold=1.5)

    with pytest.raises(TrustConfigError, match="must be a float bounded in"):
        TrustConfig(min_groundedness_threshold=-0.1)

    with pytest.raises(TrustConfigError, match="min_evidence_count must be an integer >= 1"):
        TrustConfig(min_evidence_count=0)

    with pytest.raises(TrustConfigError, match="require_valid_provenance must be a boolean"):
        TrustConfig(require_valid_provenance="yes")  # type: ignore


def test_trust_config_invalid_weights():
    """Verify zero or negative weight combinations raise TrustConfigError."""
    with pytest.raises(TrustConfigError, match="must be a non-negative finite float"):
        TrustConfig(weight_relevance=-1.0)

    with pytest.raises(TrustConfigError, match="Sum of groundedness weights must be strictly positive"):
        TrustConfig(weight_relevance=0.0, weight_coverage=0.0, weight_provenance=0.0)

    with pytest.raises(TrustConfigError, match="Sum of confidence weights must be strictly positive"):
        TrustConfig(
            weight_confidence_groundedness=0.0,
            weight_confidence_top_relevance=0.0,
            weight_confidence_coverage=0.0,
            weight_confidence_volume=0.0,
        )


def test_trust_engine_integration_with_config():
    """Verify TrustEngine accepts and uses a TrustConfig instance."""
    cfg = TrustConfig(min_confidence_threshold=0.88, min_evidence_count=3)
    engine = TrustEngine(config=cfg)

    assert engine.min_confidence_threshold == 0.88
    assert engine.min_evidence_count == 3
    assert engine.config == cfg
