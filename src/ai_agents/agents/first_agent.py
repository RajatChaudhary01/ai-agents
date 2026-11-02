import os
from importlib import resources
from pathlib import Path

import yaml
from dotenv import load_dotenv
from smolagents import CodeAgent

from ai_agents.models import model_from_env
from ai_agents.tools import ResilientWebSearchTool, calculate, get_current_time_in_timezone

# Automatically load environment variables from .env if present
load_dotenv()


def load_prompt_templates() -> dict:
    """Load the agent's prompt templates.

    Uses ``PROMPTS_PATH`` if set, otherwise the ``prompts.yaml`` shipped inside the
    package, so the agent behaves the same whichever directory it is started from.
    """
    override = os.getenv("PROMPTS_PATH")
    if override:
        path = Path(override).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"PROMPTS_PATH points to a missing file: {path}")
        text = path.read_text(encoding="utf-8")
    else:
        text = resources.files("ai_agents").joinpath("prompts.yaml").read_text(encoding="utf-8")
    return yaml.safe_load(text)


def _build_search_tool() -> ResilientWebSearchTool:
    """Web search tool configured from SEARCH_* environment variables."""
    kwargs: dict = {}
    if max_results := os.getenv("SEARCH_MAX_RESULTS"):
        kwargs["max_results"] = int(max_results)
    if timeout := os.getenv("SEARCH_TIMEOUT"):
        kwargs["timeout"] = float(timeout)
    if backends := os.getenv("SEARCH_BACKENDS"):
        kwargs["backends"] = backends.split(",")
    return ResilientWebSearchTool(**kwargs)


def create_first_agent() -> CodeAgent:
    """Builds and returns our CodeAgent powered by any configured LLM."""
    # 1. The Brain: model from MODEL_* (+ optional FALLBACK_MODEL_*) in .env
    model = model_from_env()

    # 2. The Hands: The list of tools the agent can call
    tools = [
        calculate,
        get_current_time_in_timezone,
        _build_search_tool(),
    ]

    # 3. The Agent: Wire the brain, hands, and prompt templates together
    agent = CodeAgent(
        tools=tools,
        model=model,
        max_steps=6,
        verbosity_level=2,
        prompt_templates=load_prompt_templates(),
    )

    return agent
