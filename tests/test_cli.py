import sys

import pytest

pytest.importorskip("litellm")

from smolagents.utils import AgentGenerationError

import ai_agents


class _Logger:
    def log_error(self, *args, **kwargs):
        pass


class _FailingAgent:
    model = type("M", (), {"model_id": "ollama_chat/test"})()

    def run(self, prompt):
        raise AgentGenerationError("connection refused", _Logger())


def test_cli_reports_model_errors_without_traceback(monkeypatch):
    monkeypatch.setattr(ai_agents, "create_first_agent", _FailingAgent)
    monkeypatch.setattr(sys, "argv", ["ai-agents", "hi"])
    with pytest.raises(SystemExit) as exc_info:
        ai_agents.main()
    message = str(exc_info.value.code)
    assert message.startswith("\n[Model error] ollama_chat/test")
    assert "FALLBACK_MODEL_ID" in message


def test_cli_reports_config_errors(monkeypatch):
    monkeypatch.setenv("AGENT_MAX_STEPS", "zero")
    monkeypatch.setattr(sys, "argv", ["ai-agents", "hi"])
    with pytest.raises(SystemExit, match=r"\[Config error\] AGENT_MAX_STEPS"):
        ai_agents.main()
