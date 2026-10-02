"""Unit tests for Evidence-Grounded Prompt Builder (Phase 6 Milestone 3)."""

import pytest

from backend.app.generation.prompt import (
    DEFAULT_SYSTEM_INSTRUCTION,
    GroundedPrompt,
    PromptBuildingError,
    build_grounded_prompt,
    format_evidence_block,
)
from backend.app.trust.evidence import TrustEvidence


def test_format_evidence_block_valid_evidence():
    """Verify clean evidence formatting with document provenance and page ranges."""
    evidence = [
        TrustEvidence(
            chunk_id="chunk_01",
            document_id="doc_dpdp",
            document_name="dpdp_act_2023.pdf",
            page_start=5,
            page_end=5,
            text="Every Data Fiduciary shall give notice to the Data Principal.",
            retrieval_rank=1,
            retrieval_score=0.15,
            rerank_score=3.5,
        ),
        TrustEvidence(
            chunk_id="chunk_02",
            document_id="doc_dpdp",
            document_name="dpdp_act_2023.pdf",
            page_start=6,
            page_end=7,
            text="Notice must contain details of data collected.",
            retrieval_rank=2,
            retrieval_score=0.12,
            rerank_score=2.8,
        ),
    ]

    text, ids = format_evidence_block(evidence)
    assert "[Evidence 1]" in text
    assert "Document: dpdp_act_2023.pdf" in text
    assert "Page 5" in text
    assert "[Evidence 2]" in text
    assert "Pages 6-7" in text
    assert ids == ["chunk_01", "chunk_02"]


def test_format_evidence_block_empty():
    """Verify empty evidence formatting returns fallback string and empty ID list."""
    text, ids = format_evidence_block([])
    assert "No supporting evidence provided" in text
    assert ids == []


def test_build_grounded_prompt_structure():
    """Verify structured prompt building with system instruction and evidence block."""
    evidence = [
        TrustEvidence(
            chunk_id="c1",
            document_id="d1",
            document_name="dpdp.pdf",
            page_start=1,
            page_end=1,
            text="Grounded text content.",
            retrieval_rank=1,
            retrieval_score=0.1,
            rerank_score=2.0,
        )
    ]
    query = "What is the notice requirement?"
    prompt = build_grounded_prompt(query, evidence)

    assert isinstance(prompt, GroundedPrompt)
    assert prompt.query == query
    assert prompt.system_instruction == DEFAULT_SYSTEM_INSTRUCTION
    assert "[Evidence 1]" in prompt.evidence_context
    assert "USER QUERY: What is the notice requirement?" in prompt.user_prompt
    assert prompt.evidence_count == 1
    assert prompt.evidence_ids == ["c1"]
    assert "GROUNDED ANSWER" in prompt.user_prompt


def test_build_grounded_prompt_custom_system_instruction():
    """Verify overriding system instruction."""
    custom_instruction = "Custom strict instruction: only cite verified facts."
    prompt = build_grounded_prompt(
        "Some query",
        [],
        system_instruction=custom_instruction,
    )
    assert prompt.system_instruction == custom_instruction
    assert custom_instruction in prompt.full_prompt


def test_build_grounded_prompt_validation_errors():
    """Verify invalid inputs raise appropriate errors."""
    with pytest.raises(TypeError, match="query must be a string"):
        build_grounded_prompt(12345, [])  # type: ignore

    with pytest.raises(PromptBuildingError, match="query cannot be empty"):
        build_grounded_prompt("   ", [])

    with pytest.raises(TypeError, match="evidence must be a sequence"):
        build_grounded_prompt("valid query", "not a list")  # type: ignore


def test_grounded_prompt_to_dict():
    """Verify serialization to dictionary."""
    prompt = build_grounded_prompt("query", [])
    d = prompt.to_dict()
    assert d["query"] == "query"
    assert "evidence_context" in d
    assert "full_prompt" in d
    assert d["evidence_count"] == 0
