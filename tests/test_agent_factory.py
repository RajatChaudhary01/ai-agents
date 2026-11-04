import pytest

pytest.importorskip("litellm")

from ai_agents.agents import create_first_agent
from ai_agents.agents.first_agent import load_prompt_templates
from ai_agents.models import FallbackRouterModel
from ai_agents.tools import ResilientWebSearchTool


def test_agent_has_expected_tools():
    agent = create_first_agent()
    assert {"calculate", "get_current_time_in_timezone", "web_search"} <= set(agent.tools)
    assert isinstance(agent.tools["web_search"], ResilientWebSearchTool)


def test_search_tool_reads_env(monkeypatch):
    monkeypatch.setenv("SEARCH_MAX_RESULTS", "3")
    monkeypatch.setenv("SEARCH_TIMEOUT", "4")
    monkeypatch.setenv("SEARCH_BACKENDS", "wikipedia, duckduckgo")
    tool = create_first_agent().tools["web_search"]
    assert tool.max_results == 3
    assert tool.timeout == 4.0
    assert tool.backends == ("wikipedia", "duckduckgo")


def test_ollama_model_gets_num_ctx(monkeypatch):
    monkeypatch.setenv("MODEL_ID", "ollama_chat/qwen2.5-coder:7b")
    monkeypatch.setenv("MODEL_NUM_CTX", "4096")
    assert create_first_agent().model.kwargs.get("num_ctx") == 4096


def test_cloud_model_has_no_num_ctx(monkeypatch):
    monkeypatch.setenv("MODEL_ID", "groq/llama-3.3-70b-versatile")
    assert "num_ctx" not in create_first_agent().model.kwargs


def test_fallback_model_from_env(monkeypatch):
    monkeypatch.setenv("FALLBACK_MODEL_ID", "groq/llama-3.3-70b-versatile")
    assert isinstance(create_first_agent().model, FallbackRouterModel)


class TestPromptTemplates:
    def test_packaged_prompts_load_from_any_directory(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)  # no prompts.yaml here
        agent = create_first_agent()
        assert agent.prompt_templates["system_prompt"] == load_prompt_templates()["system_prompt"]
        assert "web_search(query=" in agent.prompt_templates["system_prompt"]

    def test_prompts_path_override(self, tmp_path, monkeypatch):
        custom = load_prompt_templates()
        custom["system_prompt"] = "CUSTOM PROMPT {{ tools }}"
        path = tmp_path / "custom.yaml"
        import yaml

        path.write_text(yaml.safe_dump(custom), encoding="utf-8")
        monkeypatch.setenv("PROMPTS_PATH", str(path))
        assert load_prompt_templates()["system_prompt"].startswith("CUSTOM PROMPT")

    def test_missing_prompts_path_fails_loudly(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PROMPTS_PATH", str(tmp_path / "nope.yaml"))
        with pytest.raises(FileNotFoundError, match="PROMPTS_PATH"):
            load_prompt_templates()


@pytest.mark.parametrize(
    ("var", "value"), [("SEARCH_MAX_RESULTS", "five"), ("SEARCH_TIMEOUT", "10s")]
)
def test_invalid_search_env_has_clear_error(monkeypatch, var, value):
    monkeypatch.setenv(var, value)
    with pytest.raises(ValueError, match=var):
        create_first_agent()


def test_system_prompt_uses_agent_code_block_tags():
    agent = create_first_agent()
    opening, closing = agent.code_block_tags
    prompt = agent.system_prompt
    assert "<end_code>" not in prompt
    assert opening in prompt
    assert closing in prompt
    assert "{{" not in prompt
