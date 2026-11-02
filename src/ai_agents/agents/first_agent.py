import os

import yaml
from dotenv import load_dotenv
from smolagents import CodeAgent, LiteLLMModel

from ai_agents.tools import ResilientWebSearchTool, calculate, get_current_time_in_timezone

# Automatically load environment variables from .env if present
load_dotenv()


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
    # 1. The Brain: Universal model configuration
    # Default to local Ollama; can be overridden via MODEL_ID in .env
    model_id = os.getenv("MODEL_ID", "ollama_chat/qwen2.5-coder:7b")

    # Only default api_base to localhost:11434 for Ollama models
    default_api_base = "http://127.0.0.1:11434" if model_id.startswith("ollama") else None
    api_base = os.getenv("MODEL_API_BASE", default_api_base)
    api_key = os.getenv("MODEL_API_KEY", None)

    # Extra arguments: only pass num_ctx to Ollama (cloud APIs like Groq reject it)
    extra_kwargs = {}
    if model_id.startswith("ollama"):
        extra_kwargs["num_ctx"] = int(os.getenv("MODEL_NUM_CTX", "8192"))

    model = LiteLLMModel(
        model_id=model_id,
        api_base=api_base,
        api_key=api_key,
        **extra_kwargs,
    )

    # 2. The Hands: The list of tools the agent can call
    tools = [
        calculate,
        get_current_time_in_timezone,
        _build_search_tool(),
    ]

    # 3. Load prompt templates from prompts.yaml if present
    prompt_templates = None
    if os.path.exists("prompts.yaml"):
        with open("prompts.yaml", encoding="utf-8") as stream:
            prompt_templates = yaml.safe_load(stream)

    # 4. The Agent: Wire the brain, hands, and prompt templates together
    agent = CodeAgent(
        tools=tools,
        model=model,
        max_steps=6,
        verbosity_level=2,
        prompt_templates=prompt_templates,
    )

    return agent
