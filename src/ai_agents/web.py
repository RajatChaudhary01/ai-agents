"""Gradio web UI with one agent per browser session, plus an opt-in hosting mode.

Local mode (default) binds to 127.0.0.1 for a single user with no login.

Hosting mode (``--host`` or ``APP_MODE=host``) serves other people: users sign up and log
in (see ``ai_agents.hosting``), requests are rate-limited per account *and* per client IP,
and the app is meant to run behind an HTTPS reverse proxy (see ``deploy/``).

smolagents' own ``GradioUI`` drives a single agent for every visitor, so two people
chatting at once would share (and corrupt) one conversation memory. Here every browser
session gets its own agent, created on its first message and dropped when the tab is
closed, after it has been idle too long, or when too many sessions are open.
"""

import argparse
import getpass
import html
import logging
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

logger = logging.getLogger(__name__)

AgentFactory = Callable[[], Any]
# Returns a refusal message to show the user, or None to let the request through.
Guard = Callable[[Any], str | None]

LOCAL_HOST = "127.0.0.1"
PUBLIC_HOST = "0.0.0.0"  # hosting mode deliberately listens on all interfaces
DEFAULT_PORT = 7860


# --------------------------------------------------------------------------- config
def env_flag(name: str, default: bool = False) -> bool:
    raw = (os.getenv(name) or "").strip().lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def env_int(name: str, default: int, minimum: int = 1) -> int:
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


@dataclass(frozen=True)
class WebConfig:
    host_mode: bool
    server_name: str
    server_port: int
    share: bool
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
        share = env_flag("GRADIO_SHARE")
        if host_mode and share:
            raise ValueError(
                "GRADIO_SHARE cannot be used in hosting mode; serve the app through the "
                "reverse proxy in deploy/ instead."
            )
        default_host = PUBLIC_HOST if host_mode else LOCAL_HOST
        return cls(
            host_mode=host_mode,
            server_name=(os.getenv("GRADIO_SERVER_NAME") or "").strip() or default_host,
            server_port=env_int("GRADIO_SERVER_PORT", DEFAULT_PORT),
            share=share,
            concurrency=env_int("APP_CONCURRENCY", 4 if host_mode else 1),
            max_sessions=env_int("APP_MAX_SESSIONS", 50 if host_mode else 5),
            session_ttl_seconds=60 * env_int("APP_SESSION_TTL_MINUTES", 60),
            max_message_chars=env_int("APP_MAX_MESSAGE_CHARS", 4000 if host_mode else 20000),
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
def session_id(request: Any) -> str:
    """Key for a browser session; prefixed with the account so users can never collide."""
    session_hash = getattr(request, "session_hash", None) or "default"
    username = getattr(request, "username", None)
    return f"{username}:{session_hash}" if username else session_hash


def _escape(text: str) -> str:
    return text.replace("<", r"\<").replace(">", r"\>")


def _reply(text: str) -> list[Any]:
    import gradio as gr

    return [gr.ChatMessage(role="assistant", content=text)]


class SessionChat:
    """Chat callback for ``gr.ChatInterface`` that routes each session to its own agent."""

    def __init__(
        self,
        store: SessionStore,
        max_message_chars: int = 20000,
        guard: Guard | None = None,
        show_error_details: bool = True,
    ) -> None:
        self.store = store
        self.max_message_chars = max_message_chars
        self.guard = guard
        # Hosting mode hides model errors (they can name internal hosts) from visitors.
        self.show_error_details = show_error_details

    def respond(self, message: str, history: Sequence[Any], request: Any) -> Generator[list[Any]]:
        import gradio as gr

        text = message.get("text", "") if isinstance(message, dict) else str(message or "")
        if not text.strip():
            yield _reply("Please type a question.")
            return
        if len(text) > self.max_message_chars:
            yield _reply(
                f"Message too long ({len(text)} characters, max "
                f"{self.max_message_chars}). Please shorten it."
            )
            return
        if self.guard is not None and (refusal := self.guard(request)):
            yield _reply(refusal)
            return

        agent = self.store.get(session_id(request))
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
            logger.warning("model error for session %s: %s", session_id(request), exc)
            if self.show_error_details:
                text = (
                    "⚠️ The model could not generate a response. Check that the model "
                    "server is running or that your API key is set.\n\n"
                    f"`{_escape(str(exc))[:500]}`"
                )
            else:
                text = "⚠️ The AI model is unavailable right now. Please try again in a minute."
            messages.append(gr.ChatMessage(role="assistant", content=text))
            yield messages

    def end_session(self, request: Any) -> None:
        self.store.drop(session_id(request))


class DirectTools:
    """The calculator and world clock, usable without the LLM (no model cost)."""

    def __init__(self, guard: Guard | None = None) -> None:
        self.guard = guard

    def _run(self, request: Any, call: Callable[[], str]) -> str:
        if self.guard is not None and (refusal := self.guard(request)):
            return refusal
        return call()

    def calculate(self, expression: str, request: Any) -> str:
        from ai_agents.tools import calculate

        return self._run(request, lambda: calculate(expression=expression or ""))

    def world_clock(self, timezone: str, request: Any) -> str:
        from ai_agents.tools import get_current_time_in_timezone

        return self._run(request, lambda: get_current_time_in_timezone(timezone=timezone or ""))


def build_blocks(
    factory: AgentFactory,
    config: WebConfig,
    chat_guard: Guard | None = None,
    tools_guard: Guard | None = None,
    logout_url: str | None = None,
    show_error_details: bool = True,
) -> Any:
    """The Gradio UI: a Chat tab with per-session agents and a direct Tools tab."""
    import gradio as gr

    store = SessionStore(factory, config.max_sessions, config.session_ttl_seconds)
    chat = SessionChat(
        store, config.max_message_chars, guard=chat_guard, show_error_details=show_error_details
    )
    tools = DirectTools(guard=tools_guard)

    def respond(message: str, history: list, request: gr.Request):
        yield from chat.respond(message, history, request)

    def end_session(request: gr.Request) -> None:
        chat.end_session(request)

    def run_calculator(expression: str, request: gr.Request) -> str:
        return tools.calculate(expression, request)

    def run_clock(timezone: str, request: gr.Request) -> str:
        return tools.world_clock(timezone, request)

    def greet(request: gr.Request) -> str:
        if not logout_url:
            return ""
        name = html.escape(getattr(request, "username", None) or "unknown")
        # Plain HTML link: Gradio opens Markdown links in a new tab, which would leave
        # this tab looking logged in.
        return (
            f'<div style="text-align:right">Signed in as <b>{name}</b> · '
            f'<a href="{html.escape(logout_url)}" target="_self">Log out</a></div>'
        )

    title = "AI Agent"
    with gr.Blocks(title=title) as demo:
        account = gr.HTML()
        with gr.Tab("Chat"):
            gr.ChatInterface(
                fn=respond,
                title=title,
                chatbot=gr.Chatbot(label="Agent"),
                # One conversation per tab: saved-history switching would desync agent memory.
                save_history=False,
            )
        with gr.Tab("Tools"):
            gr.Markdown("Use the tools directly, without the AI model.")
            with gr.Row():
                expression = gr.Textbox(label="Calculator", placeholder="e.g. sqrt(144) + 2**5")
                calc_out = gr.Textbox(label="Result", interactive=False)
            gr.Button("Calculate").click(run_calculator, expression, calc_out)
            expression.submit(run_calculator, expression, calc_out)
            with gr.Row():
                tz = gr.Textbox(label="World clock", placeholder="e.g. Asia/Tokyo")
                tz_out = gr.Textbox(label="Local time", interactive=False)
            gr.Button("Get time").click(run_clock, tz, tz_out)
            tz.submit(run_clock, tz, tz_out)
        demo.load(greet, None, account)
        demo.unload(end_session)
    demo.queue(default_concurrency_limit=config.concurrency)
    demo.session_store = store  # exposed for tests and diagnostics
    return demo


def create_app(factory: AgentFactory, config: WebConfig) -> Any:
    """The local (single-user, no login) app."""
    return build_blocks(factory, config)


def launch(factory: AgentFactory, config: WebConfig) -> None:
    if config.host_mode:
        from ai_agents.hosting.server import serve

        serve(factory, config)
        return
    demo = create_app(factory, config)
    print(
        f"🚀 Launching Agent Web UI (local mode) on http://{config.server_name}:{config.server_port} ..."
    )
    if config.share:
        print("⚠️  GRADIO_SHARE is enabled: the agent is also reachable via a public link.")
    demo.launch(server_name=config.server_name, server_port=config.server_port, share=config.share)


# ----------------------------------------------------------------------------- CLI
def _users_command(args: argparse.Namespace) -> None:
    from ai_agents.hosting.accounts import AccountError, AccountStore
    from ai_agents.hosting.server import HostingConfig

    store = AccountStore(HostingConfig.from_env().db_path)
    try:
        if args.action == "list":
            for user in store.list_users():
                status = "disabled" if user.disabled else "active"
                created = time.strftime("%Y-%m-%d", time.localtime(user.created_at))
                print(f"{user.display_name:<32} {status:<9} created {created}")
            return
        if args.action in {"add", "passwd"}:
            password = getpass.getpass(f"Password for {args.username}: ")
            if password != getpass.getpass("Repeat password: "):
                sys.exit("Passwords do not match.")
            if args.action == "add":
                store.create_user(args.username, password)
                print(f"Created user {args.username}.")
            elif store.set_password(args.username, password):
                print(f"Password changed for {args.username}; their sessions were logged out.")
            else:
                sys.exit(f"No such user: {args.username}")
            return
        if store.set_disabled(args.username, disabled=args.action == "disable"):
            print(f"User {args.username} {args.action}d.")
        else:
            sys.exit(f"No such user: {args.username}")
    except AccountError as exc:
        sys.exit(str(exc))


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Launch the AI agent web UI.")
    parser.add_argument(
        "--host",
        action="store_true",
        default=None,
        help="hosting mode: user accounts, rate limits, meant to run behind an HTTPS proxy",
    )
    sub = parser.add_subparsers(dest="command")
    users = sub.add_parser("users", help="manage hosting-mode user accounts")
    users.add_argument("action", choices=["list", "add", "passwd", "disable", "enable"])
    users.add_argument("username", nargs="?")
    args = parser.parse_args(argv)

    if args.command == "users":
        if args.action != "list" and not args.username:
            parser.error(f"users {args.action} needs a username")
        _users_command(args)
        return

    from ai_agents.agents import create_first_agent

    try:
        config = WebConfig.from_env(host_mode=args.host)
        create_first_agent()  # fail fast on bad MODEL_* / AGENT_* settings before serving
        if config.host_mode:
            from ai_agents.hosting.accounts import AccountStore
            from ai_agents.hosting.server import HostingConfig

            AccountStore(HostingConfig.from_env().db_path)  # fail fast if it is not writable
    except (ValueError, FileNotFoundError) as exc:
        sys.exit(f"[Config error] {exc}")
    launch(create_first_agent, config)


if __name__ == "__main__":
    main()
