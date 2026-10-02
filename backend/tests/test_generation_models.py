"""Unit tests for GenerationRequest and GenerationResult data models (Phase 6 Milestone 4)."""

import pytest

from backend.app.generation.models import (
    GenerationRequest,
    GenerationResult,
    ModelValidationError,
)


def test_generation_request_defaults():
    """Verify clean instantiation with standard defaults."""
    req = GenerationRequest(query="What is a Data Principal?")
    assert req.query == "What is a Data Principal?"
    assert req.model_name == "Qwen/Qwen2.5-3B-Instruct"
    assert req.evidence_ids == []
    assert req.temperature == 0.0
    assert req.max_new_tokens == 512


def test_generation_request_to_dict():
    """Verify serialization of GenerationRequest."""
    req = GenerationRequest(
        query="What is consent?",
        evidence_ids=["c1", "c2"],
        model_name="Qwen/Qwen2.5-3B-Instruct",
        max_new_tokens=200,
        temperature=0.2,
    )
    d = req.to_dict()
    assert d["query"] == "What is consent?"
    assert d["evidence_ids"] == ["c1", "c2"]
    assert d["max_new_tokens"] == 200
    assert d["temperature"] == 0.2


def test_generation_request_validation_errors():
    """Verify parameter bounds checking on GenerationRequest."""
    with pytest.raises(ModelValidationError, match="query must be a non-empty string"):
        GenerationRequest(query="   ")

    with pytest.raises(ModelValidationError, match="must be a list or tuple of strings"):
        GenerationRequest(query="valid", evidence_ids="not a list")  # type: ignore

    with pytest.raises(ModelValidationError, match="evidence_ids\\[0\\] must be a non-empty string"):
        GenerationRequest(query="valid", evidence_ids=[""])

    with pytest.raises(ModelValidationError, match="max_new_tokens must be an integer > 0"):
        GenerationRequest(query="valid", max_new_tokens=-5)

    with pytest.raises(ModelValidationError, match="temperature must be a finite float"):
        GenerationRequest(query="valid", temperature=-1.0)

    with pytest.raises(ModelValidationError, match="top_p must be a float in"):
        GenerationRequest(query="valid", top_p=1.5)


def test_generation_result_success():
    """Verify normal successful generation result model."""
    res = GenerationResult(
        query="What is a Data Fiduciary?",
        answer="A Data Fiduciary is an entity determining the purpose of processing [1].",
        model_name="Qwen/Qwen2.5-3B-Instruct",
        evidence_ids=["chunk_01"],
        generation_metadata={"latency_seconds": 1.25},
    )
    assert res.is_refusal is False
    assert res.refusal_reason is None
    assert res.generation_metadata["latency_seconds"] == 1.25

    d = res.to_dict()
    restored = GenerationResult.from_dict(d)
    assert restored == res


def test_generation_result_refusal():
    """Verify structured refusal generation result model."""
    res = GenerationResult(
        query="Unrelated quantum physics query",
        answer="I cannot answer because supporting evidence is insufficient.",
        model_name="Qwen/Qwen2.5-3B-Instruct",
        evidence_ids=[],
        is_refusal=True,
        refusal_reason="TrustEngine assessed INSUFFICIENT_EVIDENCE.",
    )
    assert res.is_refusal is True
    assert res.refusal_reason == "TrustEngine assessed INSUFFICIENT_EVIDENCE."


def test_generation_result_validation_errors():
    """Verify validation on GenerationResult fields."""
    with pytest.raises(ModelValidationError, match="query must be a non-empty string"):
        GenerationResult(query="", answer="ans", model_name="m", evidence_ids=[])

    with pytest.raises(ModelValidationError, match="answer must be a non-empty string"):
        GenerationResult(query="q", answer="  ", model_name="m", evidence_ids=[])

    with pytest.raises(ModelValidationError, match="model_name must be a non-empty string"):
        GenerationResult(query="q", answer="ans", model_name="", evidence_ids=[])

    with pytest.raises(ModelValidationError, match="data must be a dictionary"):
        GenerationResult.from_dict("not a dict")  # type: ignore
