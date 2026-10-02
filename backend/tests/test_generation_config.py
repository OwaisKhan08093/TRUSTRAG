"""Unit tests for GenerationConfig (Phase 6 Milestone 1)."""

import pytest

from backend.app.generation.config import (
    DEFAULT_GENERATION_MODEL,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_REPETITION_PENALTY,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    GenerationConfig,
    GenerationConfigError,
)


def test_generation_config_defaults():
    """Verify default generation parameters match expected specifications."""
    cfg = GenerationConfig()
    assert cfg.model_name == "Qwen/Qwen2.5-3B-Instruct"
    assert cfg.model_name == DEFAULT_GENERATION_MODEL
    assert cfg.max_new_tokens == DEFAULT_MAX_NEW_TOKENS
    assert cfg.temperature == DEFAULT_TEMPERATURE
    assert cfg.top_p == DEFAULT_TOP_P
    assert cfg.repetition_penalty == DEFAULT_REPETITION_PENALTY


def test_generation_config_to_from_dict():
    """Verify dictionary serialization and deserialization."""
    cfg = GenerationConfig(
        model_name="Qwen/Qwen2.5-3B-Instruct",
        max_new_tokens=256,
        temperature=0.7,
        top_p=0.95,
        repetition_penalty=1.15,
    )
    d = cfg.to_dict()
    assert d["max_new_tokens"] == 256
    assert d["temperature"] == 0.7
    assert d["top_p"] == 0.95
    assert d["repetition_penalty"] == 1.15

    restored = GenerationConfig.from_dict(d)
    assert restored == cfg


def test_generation_config_validation_errors():
    """Verify improper configuration parameters trigger GenerationConfigError."""
    with pytest.raises(GenerationConfigError, match="model_name must be a non-empty string"):
        GenerationConfig(model_name="")

    with pytest.raises(GenerationConfigError, match="max_new_tokens must be a positive integer"):
        GenerationConfig(max_new_tokens=0)

    with pytest.raises(GenerationConfigError, match="temperature must be a non-negative finite float"):
        GenerationConfig(temperature=-0.5)

    with pytest.raises(GenerationConfigError, match="top_p must be a float in"):
        GenerationConfig(top_p=0.0)

    with pytest.raises(GenerationConfigError, match="top_p must be a float in"):
        GenerationConfig(top_p=1.5)

    with pytest.raises(GenerationConfigError, match="repetition_penalty must be a positive float"):
        GenerationConfig(repetition_penalty=0.0)


def test_generation_config_from_dict_invalid():
    """Verify invalid input data type to from_dict."""
    with pytest.raises(GenerationConfigError, match="data must be a dictionary"):
        GenerationConfig.from_dict("invalid")  # type: ignore
