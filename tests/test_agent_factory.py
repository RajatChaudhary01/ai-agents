import pytest

pytest.importorskip("litellm")

from ai_agents.agents import create_first_agent
from ai_agents.tools import ResilientWebSearchTool


def test_agent_has_expected_tools(monkeypatch):
    monkeypatch.setenv("MODEL_ID", "ollama_chat/qwen2.5-coder:7b")
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
