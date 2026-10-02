"""Local Hugging Face LLM generation wrapper with lazy model loading and memory safeguards."""

import logging
from typing import Any, Dict, List, Optional, Union

from backend.app.generation.config import GenerationConfig

logger = logging.getLogger(__name__)


class LLMError(RuntimeError):
    """Base exception for local LLM operations."""
    pass


class LLMLoadError(LLMError):
    """Raised when model weights or tokenizer fail to load."""
    pass


class LLMGenerationError(LLMError):
    """Raised when token generation encounters runtime errors."""
    pass


class LocalLLM:
    """Encapsulates local Hugging Face CausalLM inference with lazy initialization.

    Attributes:
        config: Active GenerationConfig instance.
        model_name: Identifier of the loaded or target Hugging Face model.
        is_loaded: Boolean indicating if weights and tokenizer are in memory.
    """

    def __init__(
        self,
        config: Optional[GenerationConfig] = None,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        lazy_load: bool = True,
    ) -> None:
        """Initialize LocalLLM wrapper.

        Args:
            config: Optional GenerationConfig instance.
            model_name: Optional override for model name.
            device: Optional device specification ('cpu', 'cuda', etc.).
            lazy_load: If True, delays loading weights until first generation call.
        """
        if config is not None:
            if model_name is not None:
                self.config = GenerationConfig(
                    model_name=model_name,
                    max_new_tokens=config.max_new_tokens,
                    temperature=config.temperature,
                    top_p=config.top_p,
                    repetition_penalty=config.repetition_penalty,
                    system_prompt_template=config.system_prompt_template,
                )
            else:
                self.config = config
        else:
            self.config = GenerationConfig(model_name=model_name) if model_name else GenerationConfig()

        self._device_override = device
        self._tokenizer: Optional[Any] = None
        self._model: Optional[Any] = None

        if not lazy_load:
            self.load_model()

    @property
    def model_name(self) -> str:
        """Return model name identifier."""
        return self.config.model_name

    @property
    def is_loaded(self) -> bool:
        """Return True if model and tokenizer are currently resident in memory."""
        return self._model is not None and self._tokenizer is not None

    @property
    def device(self) -> str:
        """Return current execution device string."""
        if self._model is not None and hasattr(self._model, "device"):
            return str(self._model.device)
        if self._device_override:
            return self._device_override
        return "cpu"

    def load_model(self) -> None:
        """Instantiate Hugging Face tokenizer and CausalLM model in memory.

        Raises:
            LLMLoadError: If dependencies or model weights cannot be retrieved/allocated.
        """
        if self.is_loaded:
            return

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise LLMLoadError(
                f"Required deep learning libraries not installed: {exc}. "
                "Ensure transformers and torch are installed."
            ) from exc

        target_device = self._device_override or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Loading local LLM '%s' onto device: %s", self.model_name, target_device)

        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
            )

            # Ensure pad_token is set
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token

            # Use float32 on CPU or bfloat16/float16 if GPU available
            torch_dtype = torch.float16 if target_device == "cuda" else torch.float32

            self._model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )
            self._model.to(target_device)
            self._model.eval()

        except Exception as exc:
            self._tokenizer = None
            self._model = None
            raise LLMLoadError(
                f"Failed to load Hugging Face model '{self.model_name}': {exc}"
            ) from exc

    def unload_model(self) -> None:
        """Release model weights and tokenizer from memory."""
        self._model = None
        self._tokenizer = None
        try:
            import gc
            import torch
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
    ) -> str:
        """Generate text completion from input prompt using the local LLM.

        Args:
            prompt: Formatted user query / evidence prompt.
            system_prompt: Optional system directive instructions.
            max_new_tokens: Maximum tokens to generate (overrides config).
            temperature: Sampling temperature (overrides config).
            top_p: Top-P nucleus sampling (overrides config).
            repetition_penalty: Repetition penalty factor (overrides config).

        Returns:
            Generated text string.

        Raises:
            TypeError: If prompt is not a string.
            ValueError: If prompt is empty or whitespace.
            LLMGenerationError: If inference fails during token generation.
        """
        if not isinstance(prompt, str):
            raise TypeError(f"prompt must be a string, got {type(prompt).__name__}.")

        stripped_prompt = prompt.strip()
        if not stripped_prompt:
            raise ValueError("prompt cannot be empty or whitespace only.")

        if not self.is_loaded:
            self.load_model()

        # Resolve parameters against defaults
        n_tokens = max_new_tokens if max_new_tokens is not None else self.config.max_new_tokens
        temp = temperature if temperature is not None else self.config.temperature
        p_val = top_p if top_p is not None else self.config.top_p
        rep_pen = repetition_penalty if repetition_penalty is not None else self.config.repetition_penalty
        sys_prompt = system_prompt or self.config.system_prompt_template

        try:
            import torch

            tokenizer = self._tokenizer
            model = self._model
            assert tokenizer is not None and model is not None

            # Build chat messages if supported
            if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
                messages = []
                if sys_prompt:
                    messages.append({"role": "system", "content": sys_prompt})
                messages.append({"role": "user", "content": stripped_prompt})

                input_text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            else:
                if sys_prompt:
                    input_text = f"System: {sys_prompt}\n\nUser: {stripped_prompt}\n\nAssistant:"
                else:
                    input_text = stripped_prompt

            inputs = tokenizer(input_text, return_tensors="pt")
            device = model.device
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs.get("attention_mask", None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)

            gen_kwargs: Dict[str, Any] = {
                "max_new_tokens": n_tokens,
                "repetition_penalty": rep_pen,
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
            }

            if temp > 0.0:
                gen_kwargs["do_sample"] = True
                gen_kwargs["temperature"] = temp
                gen_kwargs["top_p"] = p_val
            else:
                gen_kwargs["do_sample"] = False

            with torch.no_grad():
                output_ids = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    **gen_kwargs,
                )

            # Slice out input tokens to isolate newly generated tokens
            generated_tokens = output_ids[0][input_ids.shape[1]:]
            decoded_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

            return decoded_text

        except Exception as exc:
            raise LLMGenerationError(f"Generation failed during inference: {exc}") from exc
