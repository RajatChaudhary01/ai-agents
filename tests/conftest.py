import os

import pytest

_CONFIG_PREFIXES = (
    "MODEL_",
    "FALLBACK_MODEL_",
    "SEARCH_",
    "AGENT_",
    "GRADIO_",
    "APP_",
    "TURNSTILE_",
    "PROMPTS_PATH",
)


@pytest.fixture(autouse=True)
def _isolate_config_env(monkeypatch):
    """Keep a developer's local .env (loaded on import) from leaking into tests."""
    for key in list(os.environ):
        if key.startswith(_CONFIG_PREFIXES):
            monkeypatch.delenv(key, raising=False)
