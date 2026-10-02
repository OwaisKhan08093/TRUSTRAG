"""Unit tests for LocalLLM wrapper (Phase 6 Milestone 2)."""

from unittest.mock import MagicMock, patch
import pytest

from backend.app.generation.config import GenerationConfig
from backend.app.generation.llm import (
    LLMError,
    LLMGenerationError,
    LLMLoadError,
    LocalLLM,
)


def test_local_llm_lazy_initialization():
    """Verify LocalLLM does not load weights at instantiation when lazy_load=True."""
    llm = LocalLLM(lazy_load=True)
    assert llm.is_loaded is False
    assert llm.model_name == "Qwen/Qwen2.5-3B-Instruct"


def test_local_llm_custom_config():
    """Verify custom GenerationConfig is properly assigned."""
    cfg = GenerationConfig(model_name="Custom/Model-7B", max_new_tokens=128)
    llm = LocalLLM(config=cfg, lazy_load=True)
    assert llm.model_name == "Custom/Model-7B"
    assert llm.config.max_new_tokens == 128


def test_local_llm_load_and_generate_mocked():
    """Verify tokenization and generation inference flow using mocked transformers."""
    mock_tokenizer = MagicMock()
    mock_tokenizer.pad_token = None
    mock_tokenizer.eos_token = "<|endoftext|>"
    mock_tokenizer.pad_token_id = 0
    mock_tokenizer.eos_token_id = 0
    mock_tokenizer.chat_template = None

    import torch
    input_ids = torch.tensor([[10, 20, 30]])
    mock_tokenizer.return_value = {
        "input_ids": input_ids,
        "attention_mask": torch.tensor([[1, 1, 1]]),
    }
    # Return prompt tokens + 2 generated tokens
    output_ids = torch.tensor([[10, 20, 30, 40, 50]])
    mock_model = MagicMock()
    mock_model.device = torch.device("cpu")
    mock_model.generate.return_value = output_ids
    mock_tokenizer.decode.return_value = "A Data Fiduciary is an entity."

    with patch("transformers.AutoTokenizer.from_pretrained", return_value=mock_tokenizer), \
         patch("transformers.AutoModelForCausalLM.from_pretrained", return_value=mock_model):

        llm = LocalLLM(lazy_load=True)
        result = llm.generate("What is a Data Fiduciary?")

        assert llm.is_loaded is True
        assert result == "A Data Fiduciary is an entity."
        assert mock_model.generate.called


def test_local_llm_input_validation():
    """Verify invalid prompts raise TypeError or ValueError."""
    llm = LocalLLM(lazy_load=True)

    with pytest.raises(TypeError, match="prompt must be a string"):
        llm.generate(12345)  # type: ignore

    with pytest.raises(ValueError, match="prompt cannot be empty"):
        llm.generate("   ")


def test_local_llm_load_failure():
    """Verify load errors raise LLMLoadError."""
    with patch("transformers.AutoTokenizer.from_pretrained", side_effect=RuntimeError("Download failed")):
        llm = LocalLLM(lazy_load=True)
        with pytest.raises(LLMLoadError, match="Failed to load Hugging Face model"):
            llm.load_model()
        assert llm.is_loaded is False


def test_local_llm_unload():
    """Verify unloading clears memory handles."""
    llm = LocalLLM(lazy_load=True)
    llm._tokenizer = MagicMock()
    llm._model = MagicMock()
    assert llm.is_loaded is True

    llm.unload_model()
    assert llm.is_loaded is False
