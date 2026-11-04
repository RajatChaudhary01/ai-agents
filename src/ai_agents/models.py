"""LLM model construction with an optional automatic fallback model.

Set ``FALLBACK_MODEL_ID`` (e.g. a Groq or Gemini model) and requests that fail on the
primary model (Ollama host offline, connection refused, provider error) are retried
transparently on the fallback through a LiteLLM Router.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from smolagents import LiteLLMModel, LiteLLMRouterModel

DEFAULT_MODEL_ID = "ollama_chat/qwen2.5-coder:7b"
DEFAULT_OLLAMA_API_BASE = "http://127.0.0.1:11434"
DEFAULT_OLLAMA_NUM_CTX = 8192
_FLATTEN_PREFIXES = ("ollama", "groq", "cerebras")


@dataclass(frozen=True)
class ModelSpec:
    """Connection settings for one LiteLLM model."""

    model_id: str
    api_base: str | None = None
    api_key: str | None = None
    num_ctx: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def is_ollama(self) -> bool:
        return self.model_id.startswith("ollama")

    def litellm_params(self) -> dict[str, Any]:
        """Per-model parameters passed to ``litellm.completion``."""
        params: dict[str, Any] = {"model": self.model_id, **self.extra}
        if self.api_base:
            params["api_base"] = self.api_base
        if self.api_key:
            params["api_key"] = self.api_key
        # num_ctx is Ollama-only; cloud APIs like Groq reject it.
        if self.is_ollama and self.num_ctx:
            params["num_ctx"] = self.num_ctx
        return params


def _env_int(name: str) -> int | None:
    """Read an optional positive integer env var, failing with a clear message if invalid."""
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from None
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value}")
    return value


def spec_from_env(prefix: str, default_model_id: str | None = None) -> ModelSpec | None:
    """Read ``{prefix}ID``, ``{prefix}API_BASE``, ``{prefix}API_KEY`` and ``{prefix}NUM_CTX``."""
    model_id = (os.getenv(f"{prefix}ID") or default_model_id or "").strip()
    if not model_id:
        return None
    is_ollama = model_id.startswith("ollama")
    api_base = (os.getenv(f"{prefix}API_BASE") or "").strip() or (
        DEFAULT_OLLAMA_API_BASE if is_ollama else None
    )
    num_ctx = _env_int(f"{prefix}NUM_CTX")
    return ModelSpec(
        model_id=model_id,
        api_base=api_base,
        api_key=(os.getenv(f"{prefix}API_KEY") or "").strip() or None,
        num_ctx=num_ctx or (DEFAULT_OLLAMA_NUM_CTX if is_ollama else None),
    )


class FallbackRouterModel(LiteLLMRouterModel):
    """LiteLLM Router model that keeps each deployment's own api_base / api_key.

    smolagents always passes ``api_base`` and ``api_key`` to the completion call; with a
    Router those ``None`` values would override the per-model settings in ``model_list``
    (sending Ollama requests to localhost, for example), so they are dropped here.
    """

    def __init__(self, primary: ModelSpec, fallbacks: list[ModelSpec], **kwargs: Any) -> None:
        self.primary = primary
        self.fallbacks = fallbacks
        chain = [primary, *fallbacks]
        super().__init__(
            model_id=primary.model_id,
            model_list=[
                {"model_name": spec.model_id, "litellm_params": spec.litellm_params()}
                for spec in chain
            ],
            client_kwargs={
                "fallbacks": [{primary.model_id: [spec.model_id for spec in fallbacks]}],
                # Switch to the fallback immediately instead of retrying a dead primary.
                "num_retries": 0,
            },
            # Plain-text messages work with every provider; some (Ollama, Groq) need them.
            flatten_messages_as_text=any(s.model_id.startswith(_FLATTEN_PREFIXES) for s in chain),
            **kwargs,
        )

    def _prepare_completion_kwargs(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        completion_kwargs = super()._prepare_completion_kwargs(*args, **kwargs)
        for key in ("api_base", "api_key"):
            if completion_kwargs.get(key) is None:
                completion_kwargs.pop(key, None)
        return completion_kwargs


def build_model(primary: ModelSpec, fallback: ModelSpec | None = None) -> LiteLLMModel:
    """Return a plain LiteLLMModel, or a Router-backed model when a fallback is given."""
    if fallback is None or fallback.model_id == primary.model_id:
        extra = {
            k: v
            for k, v in primary.litellm_params().items()
            if k not in ("model", "api_base", "api_key")
        }
        return LiteLLMModel(
            model_id=primary.model_id,
            api_base=primary.api_base,
            api_key=primary.api_key,
            **extra,
        )
    return FallbackRouterModel(primary, [fallback])


def model_from_env() -> LiteLLMModel:
    """Build the agent's model from MODEL_* and optional FALLBACK_MODEL_* variables."""
    primary = spec_from_env("MODEL_", DEFAULT_MODEL_ID) or ModelSpec(DEFAULT_MODEL_ID)
    return build_model(primary, spec_from_env("FALLBACK_MODEL_"))


def describe_model(model: LiteLLMModel) -> str:
    """Human-readable model chain, e.g. 'ollama_chat/qwen2.5:14b -> groq/llama-3.3-70b'."""
    if isinstance(model, FallbackRouterModel):
        return " -> ".join(s.model_id for s in (model.primary, *model.fallbacks))
    return model.model_id or ""
