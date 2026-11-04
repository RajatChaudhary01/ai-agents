import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("gradio")

APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def _load_app():
    spec = importlib.util.spec_from_file_location("app_under_test", APP_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(("env", "expected_share"), [(None, False), ("true", True)])
def test_web_ui_is_not_public_by_default(monkeypatch, env, expected_share):
    app = _load_app()
    launched = {}

    class FakeUI:
        def __init__(self, agent):
            pass

        def launch(self, **kwargs):
            launched.update(kwargs)

    monkeypatch.setattr(app, "GradioUI", FakeUI)
    monkeypatch.setattr(app, "create_first_agent", lambda: object())
    if env is not None:
        monkeypatch.setenv("GRADIO_SHARE", env)
    app.main()
    assert launched["share"] is expected_share
    assert launched["server_name"] == "127.0.0.1"
