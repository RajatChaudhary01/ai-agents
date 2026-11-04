import pytest

pytest.importorskip("gradio")

from smolagents import FinalAnswerStep
from smolagents.utils import AgentGenerationError

from ai_agents import web
from ai_agents.web import DirectTools, SessionChat, SessionStore, WebConfig, session_id


class FakeRequest:
    def __init__(self, session_hash):
        self.session_hash = session_hash


class _Logger:
    def log_error(self, *args, **kwargs):
        pass


class FakeAgent:
    """Remembers what it was told, like a real agent's memory."""

    stream_outputs = False

    def __init__(self):
        self.memory = []
        self.resets = 0

    def run(self, task, stream, reset):
        if reset:
            self.memory = []
            self.resets += 1
        self.memory.append(task)
        yield FinalAnswerStep(output=" | ".join(self.memory))


def final_text(messages):
    return str(messages[-1].content)


def chat_turn(chat, session, text, history=()):
    return final_text(list(chat.respond(text, list(history), FakeRequest(session)))[-1])


# ------------------------------------------------------------------ sessions
class TestSessionIsolation:
    def test_each_session_gets_its_own_agent(self):
        chat = SessionChat(SessionStore(FakeAgent))
        chat_turn(chat, "alice", "my name is Alice")
        out_bob = chat_turn(chat, "bob", "my name is Bob")
        out_alice = chat_turn(chat, "alice", "what is my name?", history=["prev"])
        assert "Alice" not in out_bob
        assert "Bob" not in out_alice
        assert "my name is Alice" in out_alice

    def test_empty_history_starts_a_fresh_conversation(self):
        store = SessionStore(FakeAgent)
        chat = SessionChat(store)
        chat_turn(chat, "s", "first")
        chat_turn(chat, "s", "second", history=["first"])
        out = chat_turn(chat, "s", "after clear")  # user pressed Clear: history is empty
        assert out.endswith("after clear")
        assert "first" not in out

    def test_closing_the_tab_drops_the_agent(self):
        store = SessionStore(FakeAgent)
        chat = SessionChat(store)
        chat_turn(chat, "s", "hi")
        assert "s" in store
        chat.end_session(FakeRequest("s"))
        assert "s" not in store

    def test_lru_cap_evicts_oldest_session(self):
        store = SessionStore(FakeAgent, max_sessions=2)
        a = store.get("a")
        store.get("b")
        store.get("a")  # a is now most recently used
        store.get("c")
        assert "b" not in store
        assert store.get("a") is a
        assert len(store) == 2

    def test_idle_sessions_expire(self):
        now = [0.0]
        store = SessionStore(FakeAgent, ttl_seconds=60, clock=lambda: now[0])
        first = store.get("a")
        now[0] = 30
        assert store.get("a") is first  # still fresh, and the use refreshes it
        now[0] = 200
        store.get("other")
        assert "a" not in store


# ------------------------------------------------------------------- messages
def test_long_messages_are_rejected_without_calling_the_agent():
    store = SessionStore(FakeAgent)
    chat = SessionChat(store, max_message_chars=10)
    assert "too long" in chat_turn(chat, "s", "x" * 11)
    assert len(store) == 0


def test_empty_message_gets_a_prompt():
    assert "type a question" in chat_turn(SessionChat(SessionStore(FakeAgent)), "s", "   ")


def test_model_errors_become_a_chat_message():
    class Broken(FakeAgent):
        def run(self, task, stream, reset):
            raise AgentGenerationError("connection refused", _Logger())
            yield

    out = chat_turn(SessionChat(SessionStore(Broken)), "s", "hi")
    assert "could not generate a response" in out
    assert "connection refused" in out

    hosted = chat_turn(SessionChat(SessionStore(Broken), show_error_details=False), "s", "hi")
    assert "unavailable right now" in hosted
    assert "connection refused" not in hosted  # internal details stay in the server log


def test_guard_refusal_skips_the_agent():
    store = SessionStore(FakeAgent)
    chat = SessionChat(store, guard=lambda request: "Rate limit reached")
    assert chat_turn(chat, "s", "hi") == "Rate limit reached"
    assert len(store) == 0


def test_sessions_are_namespaced_by_account():
    request = FakeRequest("tab1")
    request.username = "alice"
    assert session_id(request) == "alice:tab1"
    assert session_id(FakeRequest("tab1")) == "tab1"


def test_direct_tools_work_and_respect_guard():
    tools = DirectTools()
    assert tools.calculate("2 + 3", FakeRequest("s")) == "5"
    assert "Asia/Tokyo" in tools.world_clock("Asia/Tokyo", FakeRequest("s"))
    blocked = DirectTools(guard=lambda request: "slow down")
    assert blocked.calculate("2 + 3", FakeRequest("s")) == "slow down"


# --------------------------------------------------------------------- config
class TestConfig:
    def test_local_mode_defaults(self):
        config = WebConfig.from_env()
        assert config.host_mode is False
        assert config.server_name == "127.0.0.1"
        assert config.share is False

    def test_host_mode_defaults(self):
        config = WebConfig.from_env(host_mode=True)
        assert config.server_name == "0.0.0.0"
        assert config.concurrency == 4
        assert config.max_sessions == 50

    def test_share_is_refused_in_host_mode(self, monkeypatch):
        monkeypatch.setenv("GRADIO_SHARE", "true")
        with pytest.raises(ValueError, match="GRADIO_SHARE"):
            WebConfig.from_env(host_mode=True)

    def test_app_mode_env_selects_host_mode(self, monkeypatch):
        monkeypatch.setenv("APP_MODE", "host")
        assert WebConfig.from_env().host_mode is True

    def test_cli_flag_overrides_env(self, monkeypatch):
        monkeypatch.setenv("APP_MODE", "host")
        assert WebConfig.from_env(host_mode=False).host_mode is False

    @pytest.mark.parametrize(
        ("var", "value"),
        [("APP_MODE", "cloud"), ("APP_CONCURRENCY", "0"), ("GRADIO_SERVER_PORT", "http")],
    )
    def test_invalid_settings_name_the_variable(self, monkeypatch, var, value):
        monkeypatch.setenv(var, value)
        with pytest.raises(ValueError, match=var):
            WebConfig.from_env()


# --------------------------------------------------------------------- launch
def test_local_launch_is_private(monkeypatch):
    captured = {}

    class FakeDemo:
        def launch(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(web, "create_app", lambda factory, config: FakeDemo())
    monkeypatch.setattr("ai_agents.agents.create_first_agent", FakeAgent)
    web.main([])
    assert captured == {"server_name": "127.0.0.1", "server_port": 7860, "share": False}


def test_host_launch_uses_hosting_server(monkeypatch, tmp_path):
    served = {}
    monkeypatch.setenv("APP_DB_PATH", str(tmp_path / "app.db"))
    monkeypatch.setattr("ai_agents.agents.create_first_agent", FakeAgent)
    monkeypatch.setattr(
        "ai_agents.hosting.server.serve", lambda factory, config: served.update(config=config)
    )
    web.main(["--host"])
    assert served["config"].host_mode is True


def test_create_app_builds_a_gradio_app():
    demo = web.create_app(FakeAgent, WebConfig.from_env())
    assert isinstance(demo.session_store, SessionStore)
