import pytest

pytest.importorskip("litellm")

from smolagents import ChatMessage, LiteLLMModel, MessageRole

from ai_agents.models import (
    DEFAULT_MODEL_ID,
    FallbackRouterModel,
    ModelSpec,
    build_model,
    describe_model,
    model_from_env,
    spec_from_env,
)

# Nothing listens on port 9 (discard), so connections are refused immediately.
DEAD_OLLAMA = "http://127.0.0.1:9"


def user_message(text):
    return [ChatMessage(role=MessageRole.USER, content=[{"type": "text", "text": text}])]


class TestModelSpec:
    def test_ollama_params_include_num_ctx(self):
        spec = ModelSpec("ollama_chat/qwen2.5:14b", api_base="http://h:11434", num_ctx=4096)
        assert spec.litellm_params() == {
            "model": "ollama_chat/qwen2.5:14b",
            "api_base": "http://h:11434",
            "num_ctx": 4096,
        }

    def test_cloud_params_never_include_num_ctx(self):
        spec = ModelSpec("groq/llama-3.3-70b-versatile", api_key="k", num_ctx=4096)
        assert spec.litellm_params() == {"model": "groq/llama-3.3-70b-versatile", "api_key": "k"}


class TestSpecFromEnv:
    def test_defaults_for_ollama(self):
        spec = spec_from_env("MODEL_", DEFAULT_MODEL_ID)
        assert spec == ModelSpec(DEFAULT_MODEL_ID, "http://127.0.0.1:11434", None, 8192)

    def test_reads_all_variables(self, monkeypatch):
        monkeypatch.setenv("FALLBACK_MODEL_ID", "ollama_chat/llama3.1:8b")
        monkeypatch.setenv("FALLBACK_MODEL_API_BASE", "http://192.168.1.5:11434")
        monkeypatch.setenv("FALLBACK_MODEL_API_KEY", "secret")
        monkeypatch.setenv("FALLBACK_MODEL_NUM_CTX", "2048")
        assert spec_from_env("FALLBACK_MODEL_") == ModelSpec(
            "ollama_chat/llama3.1:8b", "http://192.168.1.5:11434", "secret", 2048
        )

    def test_missing_fallback_returns_none(self):
        assert spec_from_env("FALLBACK_MODEL_") is None

    def test_cloud_model_gets_no_api_base_or_num_ctx(self, monkeypatch):
        monkeypatch.setenv("MODEL_ID", "groq/llama-3.3-70b-versatile")
        spec = spec_from_env("MODEL_")
        assert spec.api_base is None
        assert spec.num_ctx is None


class TestBuildModel:
    def test_without_fallback_is_plain_litellm_model(self):
        model = build_model(ModelSpec("ollama_chat/m", "http://h:11434", num_ctx=1024))
        assert type(model) is LiteLLMModel
        assert model.api_base == "http://h:11434"
        assert model.kwargs == {"num_ctx": 1024}

    def test_same_model_as_fallback_is_ignored(self):
        spec = ModelSpec("groq/x")
        assert type(build_model(spec, spec)) is LiteLLMModel

    def test_with_fallback_builds_router(self):
        primary = ModelSpec("ollama_chat/qwen2.5:14b", "http://h:11434", num_ctx=8192)
        fallback = ModelSpec("groq/llama-3.3-70b-versatile", api_key="k")
        model = build_model(primary, fallback)

        assert isinstance(model, FallbackRouterModel)
        assert model.model_id == "ollama_chat/qwen2.5:14b"
        assert model.flatten_messages_as_text is True
        assert model.client_kwargs["fallbacks"] == [
            {"ollama_chat/qwen2.5:14b": ["groq/llama-3.3-70b-versatile"]}
        ]
        params = [m["litellm_params"] for m in model.client_kwargs["model_list"]]
        assert params == [primary.litellm_params(), fallback.litellm_params()]
        assert describe_model(model) == "ollama_chat/qwen2.5:14b -> groq/llama-3.3-70b-versatile"

    def test_router_does_not_override_per_model_api_base(self):
        model = build_model(ModelSpec("ollama_chat/a", "http://h:1"), ModelSpec("groq/b"))
        kwargs = model._prepare_completion_kwargs(messages=user_message("hi"), model="x")
        assert "api_base" not in kwargs
        assert "api_key" not in kwargs

    def test_model_from_env(self, monkeypatch):
        monkeypatch.setenv("MODEL_ID", "ollama_chat/a")
        assert type(model_from_env()) is LiteLLMModel
        monkeypatch.setenv("FALLBACK_MODEL_ID", "groq/b")
        assert isinstance(model_from_env(), FallbackRouterModel)


class TestFallbackBehaviour:
    """End-to-end through the LiteLLM Router; no real LLM is contacted."""

    def test_primary_down_uses_fallback(self):
        primary = ModelSpec("ollama_chat/primary", api_base=DEAD_OLLAMA)
        fallback = ModelSpec("ollama_chat/backup", extra={"mock_response": "answer from backup"})
        model = build_model(primary, fallback)
        assert model.generate(user_message("hello")).content == "answer from backup"

    def test_primary_up_is_used(self):
        primary = ModelSpec("ollama_chat/primary", extra={"mock_response": "answer from primary"})
        fallback = ModelSpec("ollama_chat/backup", extra={"mock_response": "answer from backup"})
        model = build_model(primary, fallback)
        assert model.generate(user_message("hello")).content == "answer from primary"

    def test_both_down_raises(self):
        model = build_model(
            ModelSpec("ollama_chat/primary", api_base=DEAD_OLLAMA),
            ModelSpec("ollama_chat/backup", api_base=DEAD_OLLAMA),
        )
        with pytest.raises(Exception, match=r"(?i)connect|refused"):
            model.generate(user_message("hello"))
