"""Grounded local LLM generation and citation subsystem for TrustRAG."""

from backend.app.generation.citations import (
    Citation,
    CitationError,
    build_citation,
    build_citation_references,
    extract_cited_indices,
    format_citation_reference_string,
    format_citations_markdown,
)
from backend.app.generation.config import (
    DEFAULT_GENERATION_MODEL,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_REPETITION_PENALTY,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    GenerationConfig,
    GenerationConfigError,
)
from backend.app.generation.generator import (
    STANDARD_ABSTENTION_MESSAGE,
    GeneratorError,
    GroundedGenerator,
)
from backend.app.generation.llm import (
    LLMError,
    LLMGenerationError,
    LLMLoadError,
    LocalLLM,
)
from backend.app.generation.models import (
    GenerationRequest,
    GenerationResult,
    ModelValidationError,
)
from backend.app.generation.prompt import (
    DEFAULT_SYSTEM_INSTRUCTION,
    GroundedPrompt,
    PromptBuildingError,
    build_grounded_prompt,
    format_evidence_block,
)
from backend.app.generation.response import (
    AnswerAssemblyError,
    GroundedAnswer,
    assemble_grounded_answer,
)

__all__ = [
    "AnswerAssemblyError",
    "Citation",
    "CitationError",
    "DEFAULT_GENERATION_MODEL",
    "DEFAULT_MAX_NEW_TOKENS",
    "DEFAULT_REPETITION_PENALTY",
    "DEFAULT_SYSTEM_INSTRUCTION",
    "DEFAULT_TEMPERATURE",
    "DEFAULT_TOP_P",
    "GenerationConfig",
    "GenerationConfigError",
    "GenerationRequest",
    "GenerationResult",
    "GeneratorError",
    "GroundedAnswer",
    "GroundedGenerator",
    "GroundedPrompt",
    "LLMError",
    "LLMGenerationError",
    "LLMLoadError",
    "LocalLLM",
    "ModelValidationError",
    "PromptBuildingError",
    "STANDARD_ABSTENTION_MESSAGE",
    "assemble_grounded_answer",
    "build_citation",
    "build_citation_references",
    "build_grounded_prompt",
    "extract_cited_indices",
    "format_citation_reference_string",
    "format_citations_markdown",
    "format_evidence_block",
]
