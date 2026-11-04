"""Gradio web UI with one agent per browser session, plus an opt-in hosting mode.

Local mode (default) binds to 127.0.0.1 for a single user. Hosting mode (``--host`` or
``APP_MODE=host``) is meant for serving other people: it listens on all interfaces,
requires a login, runs several sessions in parallel and caps resources.

smolagents' own ``GradioUI`` drives a single agent for every visitor, so two people
chatting at once would share (and corrupt) one conversation memory. Here every browser
session gets its own agent, created on its first message and dropped when the tab is
closed, after it has been idle too long, or when too many sessions are open.
"""

import argparse
import os
import sys
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Generator, Sequence
from dataclasses import dataclass
from typing import Any

from smolagents import ActionStep, ChatMessageStreamDelta, FinalAnswerStep, PlanningStep
from smolagents.gradio_ui import agglomerate_stream_deltas, pull_messages_from_step
from smolagents.utils import AgentError

AgentFactory = Callable[[], Any]

LOCAL_HOST = "127.0.0.1"
PUBLIC_HOST = "0.0.0.0"  # hosting mode deliberately listens on all interfaces
DEFAULT_PORT = 7860


# --------------------------------------------------------------------------- config
def _env_flag(name: str) -> bool:
    return (os.getenv(name) or "").strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int, minimum: int = 1) -> int:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from None
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}, got {value}")
    return value


def parse_auth(raw: str | None) -> list[tuple[str, str]]:
    """Parse ``APP_AUTH`` (``user:password[,user2:password2]``) into Gradio's auth list."""
    users = []
    for entry in (raw or "").split(","):
        entry = entry.strip()
        if not entry:
            continue
        username, sep, password = entry.partition(":")
        if not sep or not username.strip() or not password:
            raise ValueError("APP_AUTH entries must look like 'user:password'")
        users.append((username.strip(), password))
    return users


@dataclass(frozen=True)
class WebConfig:
    host_mode: bool
    server_name: str
    server_port: int
    share: bool
    auth: list[tuple[str, str]]
    concurrency: int
    max_sessions: int
    session_ttl_seconds: int
    max_message_chars: int

    @classmethod
    def from_env(cls, host_mode: bool | None = None) -> "WebConfig":
        if host_mode is None:
            mode = (os.getenv("APP_MODE") or "local").strip().lower()
            if mode not in {"local", "host"}:
                raise ValueError(f"APP_MODE must be 'local' or 'host', got {mode!r}")
            host_mode = mode == "host"

        auth = parse_auth(os.getenv("APP_AUTH"))
        if host_mode and not auth and not _env_flag("APP_ALLOW_ANONYMOUS"):
            raise ValueError(
                "Hosting mode needs a login so strangers cannot spend your API quota. "
                "Set APP_AUTH=user:password (comma-separate several users), or set "
                "APP_ALLOW_ANONYMOUS=true to run without one."
            )

        default_host = PUBLIC_HOST if host_mode else LOCAL_HOST
        return cls(
            host_mode=host_mode,
            server_name=(os.getenv("GRADIO_SERVER_NAME") or "").strip() or default_host,
            server_port=_env_int("GRADIO_SERVER_PORT", DEFAULT_PORT),
            share=_env_flag("GRADIO_SHARE"),
            auth=auth,
            concurrency=_env_int("APP_CONCURRENCY", 4 if host_mode else 1),
            max_sessions=_env_int("APP_MAX_SESSIONS", 50 if host_mode else 5),
            session_ttl_seconds=60 * _env_int("APP_SESSION_TTL_MINUTES", 60),
            max_message_chars=_env_int("APP_MAX_MESSAGE_CHARS", 4000 if host_mode else 20000),
        )


# ------------------------------------------------------------------------ sessions
class SessionStore:
    """Thread-safe map of browser session -> agent, with idle expiry and an LRU cap."""

    def __init__(
        self,
        factory: AgentFactory,
        max_sessions: int = 50,
        ttl_seconds: float = 3600,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._factory = factory
        self.max_sessions = max(1, max_sessions)
        self.ttl_seconds = ttl_seconds
        self._clock = clock
        self._agents: OrderedDict[str, tuple[Any, float]] = OrderedDict()
        self._lock = threading.Lock()

    def __len__(self) -> int:
        with self._lock:
            return len(self._agents)

    def __contains__(self, session_id: object) -> bool:
        with self._lock:
            return session_id in self._agents

    def get(self, session_id: str) -> Any:
        """Return this session's agent, creating it on first use."""
        now = self._clock()
        with self._lock:
            self._expire(now)
            entry = self._agents.pop(session_id, None)
            agent = entry[0] if entry else None
        if agent is None:
            agent = self._factory()  # outside the lock: building a model can be slow
        with self._lock:
            self._agents[session_id] = (agent, now)
            while len(self._agents) > self.max_sessions:
                self._agents.popitem(last=False)  # evict the least recently used
        return agent

    def drop(self, session_id: str | None) -> None:
        if session_id is None:
            return
        with self._lock:
            self._agents.pop(session_id, None)

    def _expire(self, now: float) -> None:
        stale = [sid for sid, (_, seen) in self._agents.items() if now - seen > self.ttl_seconds]
        for sid in stale:
            del self._agents[sid]


# -------------------------------------------------------------------------- the UI
def _session_id(request: Any) -> str:
    session_hash = getattr(request, "session_hash", None)
    return session_hash or "default"


def _escape(text: str) -> str:
    return text.replace("<", r"\<").replace(">", r"\>")


class SessionChat:
    """Chat callback for ``gr.ChatInterface`` that routes each session to its own agent."""

    def __init__(self, store: SessionStore, max_message_chars: int = 20000) -> None:
        self.store = store
        self.max_message_chars = max_message_chars

    def respond(self, message: str, history: Sequence[Any], request: Any) -> Generator[list[Any]]:
        import gradio as gr

        text = message.get("text", "") if isinstance(message, dict) else str(message or "")
        if not text.strip():
            yield [gr.ChatMessage(role="assistant", content="Please type a question.")]
            return
        if len(text) > self.max_message_chars:
            yield [
                gr.ChatMessage(
                    role="assistant",
                    content=f"Message too long ({len(text)} characters, max "
                    f"{self.max_message_chars}). Please shorten it.",
                )
            ]
            return

        agent = self.store.get(_session_id(request))
        messages: list[Any] = []
        streaming_idx: int | None = None
        deltas: list[ChatMessageStreamDelta] = []
        try:
            # An empty history means a fresh chat (new tab or "clear"), so start a new memory.
            for event in agent.run(text, stream=True, reset=not history):
                if isinstance(event, ActionStep | PlanningStep | FinalAnswerStep):
                    if streaming_idx is not None:
                        messages.pop(streaming_idx)
                        streaming_idx = None
                    for msg in pull_messages_from_step(
                        event, skip_model_outputs=getattr(agent, "stream_outputs", False)
                    ):
                        messages.append(
                            gr.ChatMessage(
                                role=msg.role, content=msg.content, metadata=msg.metadata
                            )
                        )
                        yield messages
                    deltas = []
                elif isinstance(event, ChatMessageStreamDelta):
                    deltas.append(event)
                    msg = gr.ChatMessage(
                        role="assistant",
                        content=_escape(agglomerate_stream_deltas(deltas).render_as_markdown()),
                    )
                    if streaming_idx is None:
                        streaming_idx = len(messages)
                        messages.append(msg)
                    else:
                        messages[streaming_idx] = msg
                    yield messages
        except AgentError as exc:
            messages.append(
                gr.ChatMessage(
                    role="assistant",
                    content="⚠️ The model could not generate a response. Check that the model "
                    f"server is running or that your API key is set.\n\n`{_escape(str(exc))[:500]}`",
                )
            )
            yield messages

    def end_session(self, request: Any) -> None:
        self.store.drop(_session_id(request))


def create_app(factory: AgentFactory, config: WebConfig) -> Any:
    import gradio as gr

    store = SessionStore(factory, config.max_sessions, config.session_ttl_seconds)
    chat = SessionChat(store, config.max_message_chars)

    def respond(message: str, history: list, request: gr.Request):
        yield from chat.respond(message, history, request)

    def end_session(request: gr.Request) -> None:
        chat.end_session(request)

    title = "AI Agent" + (" (hosted)" if config.host_mode else "")
    with gr.Blocks(title=title) as demo:
        gr.ChatInterface(
            fn=respond,
            title=title,
            chatbot=gr.Chatbot(label="Agent"),
            # One conversation per tab: saved-history switching would desync agent memory.
            save_history=False,
        )
        demo.unload(end_session)
    demo.queue(default_concurrency_limit=config.concurrency)
    demo.session_store = store  # exposed for tests and diagnostics
    return demo


def launch(factory: AgentFactory, config: WebConfig) -> None:
    demo = create_app(factory, config)
    mode = "hosting" if config.host_mode else "local"
    print(
        f"🚀 Launching Agent Web UI ({mode} mode) on http://{config.server_name}:{config.server_port} ..."
    )
    if config.host_mode:
        login = f"{len(config.auth)} user(s)" if config.auth else "DISABLED (anonymous access)"
        print(
            f"   login: {login} | parallel requests: {config.concurrency} | "
            f"max sessions: {config.max_sessions}"
        )
    if config.share:
        print("⚠️  GRADIO_SHARE is enabled: the agent is also reachable via a public link.")
    demo.launch(
        server_name=config.server_name,
        server_port=config.server_port,
        share=config.share,
        auth=config.auth or None,
        auth_message="Log in to use the AI agent." if config.auth else None,
        show_error=not config.host_mode,  # don't leak tracebacks to visitors
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Launch the AI agent web UI.")
    parser.add_argument(
        "--host",
        action="store_true",
        default=None,
        help="hosting mode: listen on all interfaces, require login (APP_AUTH), serve several users",
    )
    args = parser.parse_args(argv)

    from ai_agents.agents import create_first_agent

    try:
        config = WebConfig.from_env(host_mode=args.host)
        create_first_agent()  # fail fast on bad MODEL_* / AGENT_* settings before serving
    except (ValueError, FileNotFoundError) as exc:
        sys.exit(f"[Config error] {exc}")
    launch(create_first_agent, config)


if __name__ == "__main__":
    main()
