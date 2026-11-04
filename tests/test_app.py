import pytest

pytest.importorskip("gradio")

from smolagents import FinalAnswerStep
from smolagents.utils import AgentGenerationError

from ai_agents import web
from ai_agents.web import SessionChat, SessionStore, WebConfig, parse_auth


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


# --------------------------------------------------------------------- config
class TestConfig:
    def test_local_mode_defaults(self):
        config = WebConfig.from_env()
        assert config.host_mode is False
        assert config.server_name == "127.0.0.1"
        assert config.share is False
        assert config.auth == []

    def test_host_mode_requires_login(self):
        with pytest.raises(ValueError, match="APP_AUTH"):
            WebConfig.from_env(host_mode=True)

    def test_host_mode_with_login(self, monkeypatch):
        monkeypatch.setenv("APP_AUTH", "alice:s3cret, bob:pa:ss")
        config = WebConfig.from_env(host_mode=True)
        assert config.server_name == "0.0.0.0"
        assert config.auth == [("alice", "s3cret"), ("bob", "pa:ss")]
        assert config.concurrency == 4
        assert config.max_sessions == 50

    def test_host_mode_anonymous_opt_in(self, monkeypatch):
        monkeypatch.setenv("APP_ALLOW_ANONYMOUS", "true")
        assert WebConfig.from_env(host_mode=True).auth == []

    def test_app_mode_env_selects_host_mode(self, monkeypatch):
        monkeypatch.setenv("APP_MODE", "host")
        monkeypatch.setenv("APP_AUTH", "a:b")
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

    @pytest.mark.parametrize("raw", ["alice", ":pw", "alice:"])
    def test_bad_auth_entries(self, raw):
        with pytest.raises(ValueError, match="user:password"):
            parse_auth(raw)


# --------------------------------------------------------------------- launch
@pytest.fixture
def launched(monkeypatch):
    captured = {}

    class FakeDemo:
        def launch(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(web, "create_app", lambda factory, config: FakeDemo())
    monkeypatch.setattr("ai_agents.agents.create_first_agent", FakeAgent)
    return captured


def test_local_launch_is_private(launched):
    web.main([])
    assert launched["share"] is False
    assert launched["server_name"] == "127.0.0.1"
    assert launched["auth"] is None


def test_host_launch_uses_login(launched, monkeypatch):
    monkeypatch.setenv("APP_AUTH", "alice:pw")
    web.main(["--host"])
    assert launched["server_name"] == "0.0.0.0"
    assert launched["auth"] == [("alice", "pw")]
    assert launched["show_error"] is False


def test_host_launch_without_login_exits(launched):
    with pytest.raises(SystemExit, match="APP_AUTH"):
        web.main(["--host"])
    assert launched == {}


def test_create_app_builds_a_gradio_app():
    demo = web.create_app(FakeAgent, WebConfig.from_env())
    assert isinstance(demo.session_store, SessionStore)
