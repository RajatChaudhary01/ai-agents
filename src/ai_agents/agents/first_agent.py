"""Builds the CodeAgent: the configured model, the three tools and the prompt templates."""

import os
from importlib import resources
from pathlib import Path

import yaml
from dotenv import load_dotenv
from smolagents import CodeAgent

from ai_agents.models import model_from_env
from ai_agents.tools import ResilientWebSearchTool, calculate, get_current_time_in_timezone

load_dotenv()  # settings may come from a .env file in the working directory

DEFAULT_MAX_STEPS = 6
DEFAULT_VERBOSITY = 2  # 0 = off, 1 = info, 2 = debug


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


def _env_number(name: str, cast: type[int] | type[float]) -> int | float | None:
    """Read an optional numeric env var, failing with a clear message if it is invalid."""
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return None
    try:
        return cast(raw)
    except ValueError:
        raise ValueError(f"{name} must be a {cast.__name__}, got {raw!r}") from None


def _max_steps_from_env() -> int:
    steps = _env_number("AGENT_MAX_STEPS", int)
    if steps is None:
        return DEFAULT_MAX_STEPS
    if steps < 1:
        raise ValueError(f"AGENT_MAX_STEPS must be at least 1, got {steps}")
    return steps


def _verbosity_from_env() -> int:
    level = _env_number("AGENT_VERBOSITY", int)
    if level is None:
        return DEFAULT_VERBOSITY
    if not 0 <= level <= 2:
        raise ValueError(f"AGENT_VERBOSITY must be 0, 1 or 2, got {level}")
    return level


def _build_search_tool() -> ResilientWebSearchTool:
    """Web search tool configured from SEARCH_* environment variables."""
    kwargs: dict = {}
    if (max_results := _env_number("SEARCH_MAX_RESULTS", int)) is not None:
        kwargs["max_results"] = max_results
    if (timeout := _env_number("SEARCH_TIMEOUT", float)) is not None:
        kwargs["timeout"] = timeout
    if backends := os.getenv("SEARCH_BACKENDS"):
        kwargs["backends"] = backends.split(",")
    return ResilientWebSearchTool(**kwargs)


def create_first_agent() -> CodeAgent:
    """Return a CodeAgent wired to the model chain, tools and prompts from the environment."""
    return CodeAgent(
        model=model_from_env(),
        tools=[calculate, get_current_time_in_timezone, _build_search_tool()],
        prompt_templates=load_prompt_templates(),
        max_steps=_max_steps_from_env(),
        verbosity_level=_verbosity_from_env(),
    )
