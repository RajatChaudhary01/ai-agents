"""Hosting-mode server: sign-up/login pages, rate limits and the Gradio UI behind them.

Layout:

* ``/login``, ``/signup``, ``/logout``: small server-rendered pages (no JavaScript).
* ``/app/``: the Gradio UI. Gradio asks :func:`HostedApp.current_user` on every request,
  so nothing under ``/app`` (including its API routes) works without a valid session.

Abuse controls (all configurable, see ``HostingConfig``):

* Chat messages are limited per account **and** per client IP, plus a global daily cap
  that bounds total model spend. Dummy accounts made from one IP share that IP's budget,
  so mass sign-ups do not buy more usage.
* Sign-ups are limited per IP per day; login failures are limited per IP and per account.
* IPv6 clients are grouped by /64 so rotating addresses inside one allocation does not help.
"""

import html
import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

from ai_agents.hosting.accounts import AccountError, AccountStore
from ai_agents.hosting.ratelimit import RateLimiter, Rule, client_ip
from ai_agents.web import AgentFactory, WebConfig, build_blocks, env_flag, env_int

MINUTE, HOUR, DAY = 60, 3600, 86400
APP_PATH = "/app"
MAX_UPLOAD_BYTES = 1024 * 1024


@dataclass(frozen=True)
class HostingConfig:
    db_path: str
    signup_open: bool
    trust_proxy: bool
    cookie_secure: bool
    signups_per_ip_per_day: int
    signup_attempts_per_ip_per_hour: int
    login_failures_per_ip: int  # per 15 minutes
    login_failures_per_account: int  # per 15 minutes
    messages_per_account_per_hour: int
    messages_per_account_per_day: int
    messages_per_ip_per_hour: int
    messages_per_ip_per_day: int
    messages_global_per_day: int
    tool_calls_per_minute: int

    @classmethod
    def from_env(cls) -> "HostingConfig":
        signup = (os.getenv("APP_SIGNUP") or "open").strip().lower()
        if signup not in {"open", "closed"}:
            raise ValueError(f"APP_SIGNUP must be 'open' or 'closed', got {signup!r}")
        return cls(
            db_path=(os.getenv("APP_DB_PATH") or "").strip() or "data/app.db",
            signup_open=signup == "open",
            trust_proxy=env_flag("APP_TRUST_PROXY"),
            cookie_secure=env_flag("APP_COOKIE_SECURE", default=True),
            signups_per_ip_per_day=env_int("APP_SIGNUPS_PER_IP_PER_DAY", 3),
            signup_attempts_per_ip_per_hour=env_int("APP_SIGNUP_ATTEMPTS_PER_IP_PER_HOUR", 10),
            login_failures_per_ip=env_int("APP_LOGIN_FAILURES_PER_IP", 20),
            login_failures_per_account=env_int("APP_LOGIN_FAILURES_PER_ACCOUNT", 10),
            messages_per_account_per_hour=env_int("APP_MESSAGES_PER_ACCOUNT_PER_HOUR", 20),
            messages_per_account_per_day=env_int("APP_MESSAGES_PER_ACCOUNT_PER_DAY", 100),
            messages_per_ip_per_hour=env_int("APP_MESSAGES_PER_IP_PER_HOUR", 40),
            messages_per_ip_per_day=env_int("APP_MESSAGES_PER_IP_PER_DAY", 200),
            messages_global_per_day=env_int("APP_MESSAGES_GLOBAL_PER_DAY", 2000),
            tool_calls_per_minute=env_int("APP_TOOL_CALLS_PER_MINUTE", 60),
        )


# --------------------------------------------------------------------------- pages
_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title} · AI Agent</title>
<style>
:root {{ color-scheme: light dark; --bg:#f6f7f9; --card:#fff; --fg:#1d2330; --muted:#5d6675;
  --line:#d9dde3; --accent:#2f5bd3; --err-bg:#fdecec; --err:#a12626; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#14171c; --card:#1d2128; --fg:#e8ebf0;
  --muted:#9aa3b2; --line:#333a45; --accent:#7c9cf5; --err-bg:#3a1f22; --err:#f2a7a7; }} }}
body {{ margin:0; min-height:100vh; display:grid; place-items:center; background:var(--bg);
  color:var(--fg); font:16px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif; padding:16px; }}
main {{ width:100%; max-width:380px; background:var(--card); border:1px solid var(--line);
  border-radius:12px; padding:28px; box-sizing:border-box; }}
h1 {{ margin:0 0 4px; font-size:1.4rem; }} p.sub {{ margin:0 0 20px; color:var(--muted); }}
label {{ display:block; margin:14px 0 4px; font-weight:600; font-size:.9rem; }}
input {{ width:100%; box-sizing:border-box; padding:10px 12px; border:1px solid var(--line);
  border-radius:8px; background:transparent; color:inherit; font:inherit; }}
button {{ width:100%; margin-top:20px; padding:11px; border:0; border-radius:8px;
  background:var(--accent); color:#fff; font:inherit; font-weight:600; cursor:pointer; }}
.err {{ background:var(--err-bg); color:var(--err); padding:10px 12px; border-radius:8px; }}
.hint {{ color:var(--muted); font-size:.85rem; margin-top:4px; }}
.alt {{ margin-top:18px; text-align:center; color:var(--muted); }} a {{ color:var(--accent); }}
</style></head><body><main>{body}</main></body></html>"""


def _page(title: str, body: str, status: int = 200) -> Any:
    from fastapi.responses import HTMLResponse

    return HTMLResponse(
        _PAGE.format(title=html.escape(title), body=body),
        status_code=status,
        headers={
            "Content-Security-Policy": "default-src 'none'; style-src 'unsafe-inline'; "
            "form-action 'self'; frame-ancestors 'none'; base-uri 'none'",
            "Cache-Control": "no-store",
        },
    )


def _error_html(error: str | None) -> str:
    return f'<p class="err" role="alert">{html.escape(error)}</p>' if error else ""


def login_page(error: str | None = None, signup_open: bool = True, status: int = 200) -> Any:
    alt = '<p class="alt">No account? <a href="/signup">Sign up</a></p>' if signup_open else ""
    return _page(
        "Log in",
        f"""<h1>Log in</h1><p class="sub">to use the AI agent and its tools</p>
{_error_html(error)}
<form method="post" action="/login">
<label for="u">Username</label>
<input id="u" name="username" autocomplete="username" required maxlength="32" autofocus>
<label for="p">Password</label>
<input id="p" name="password" type="password" autocomplete="current-password" required maxlength="128">
<button type="submit">Log in</button></form>{alt}""",
        status,
    )


def signup_page(error: str | None = None, username: str = "", status: int = 200) -> Any:
    return _page(
        "Sign up",
        f"""<h1>Create an account</h1><p class="sub">free, with fair-use limits</p>
{_error_html(error)}
<form method="post" action="/signup">
<label for="u">Username</label>
<input id="u" name="username" autocomplete="username" required minlength="3" maxlength="32"
 pattern="[A-Za-z0-9][A-Za-z0-9_.\\-]{{2,31}}" value="{html.escape(username)}" autofocus>
<p class="hint">3-32 letters, digits, '.', '_' or '-'</p>
<label for="p">Password</label>
<input id="p" name="password" type="password" autocomplete="new-password" required minlength="10" maxlength="128">
<p class="hint">At least 10 characters</p>
<label for="c">Repeat password</label>
<input id="c" name="confirm" type="password" autocomplete="new-password" required maxlength="128">
<button type="submit">Sign up</button></form>
<p class="alt">Have an account? <a href="/login">Log in</a></p>""",
        status,
    )


def closed_page() -> Any:
    return _page(
        "Sign-up closed",
        '<h1>Sign-up is closed</h1><p class="sub">Ask the administrator for an account.</p>'
        '<p class="alt"><a href="/login">Log in</a></p>',
        403,
    )


# ----------------------------------------------------------------------------- app
class HostedApp:
    """Wires accounts, rate limits and the Gradio UI into one FastAPI app."""

    def __init__(
        self,
        factory: AgentFactory,
        web_config: WebConfig,
        config: HostingConfig,
        accounts: AccountStore | None = None,
        limiter: RateLimiter | None = None,
    ) -> None:
        self.web_config = web_config
        self.config = config
        self.accounts = accounts or AccountStore(config.db_path)
        self.limiter = limiter or RateLimiter()
        self.cookie_name = "__Host-agent_session" if config.cookie_secure else "agent_session"
        self.accounts.purge_expired_sessions()
        self.blocks = build_blocks(
            factory,
            web_config,
            chat_guard=self.chat_guard,
            tools_guard=self.tools_guard,
            logout_url="/logout",
            show_error_details=False,
        )
        self.app = self._build()

    # ---------------------------------------------------------------- helpers
    def ip_of(self, headers: Any, peer: str | None) -> str:
        return client_ip(headers, peer, self.config.trust_proxy)

    def _request_ip(self, request: Any) -> str:
        client = getattr(request, "client", None)
        return self.ip_of(request.headers, getattr(client, "host", None))

    def current_user(self, request: Any) -> str | None:
        """Gradio's auth hook: the logged-in username, or None (-> 401)."""
        return self.accounts.session_user(request.cookies.get(self.cookie_name))

    # ----------------------------------------------------------------- guards
    def chat_guard(self, request: Any) -> str | None:
        user = getattr(request, "username", None)
        if not user:
            return "Please log in again."
        ip = self._request_ip(request)
        c = self.config
        decision = self.limiter.hit(
            [
                Rule(
                    f"msg:acct:{user}", c.messages_per_account_per_hour, HOUR, "messages per hour"
                ),
                Rule(f"msg:acct:{user}", c.messages_per_account_per_day, DAY, "messages per day"),
                Rule(
                    f"msg:ip:{ip}",
                    c.messages_per_ip_per_hour,
                    HOUR,
                    "messages per hour from your network",
                ),
                Rule(
                    f"msg:ip:{ip}",
                    c.messages_per_ip_per_day,
                    DAY,
                    "messages per day from your network",
                ),
                Rule(
                    "msg:global",
                    c.messages_global_per_day,
                    DAY,
                    "messages per day for this service",
                ),
            ]
        )
        return None if decision.allowed else decision.message()

    def tools_guard(self, request: Any) -> str | None:
        user = getattr(request, "username", None) or "anonymous"
        ip = self._request_ip(request)
        limit = self.config.tool_calls_per_minute
        decision = self.limiter.hit(
            [
                Rule(f"tool:acct:{user}", limit, MINUTE, "tool uses per minute"),
                Rule(f"tool:ip:{ip}", limit * 2, MINUTE, "tool uses per minute from your network"),
            ]
        )
        return None if decision.allowed else decision.message()

    # ------------------------------------------------------------------ build
    def _set_session_cookie(self, response: Any, token: str) -> None:
        from ai_agents.hosting.accounts import SESSION_TTL_SECONDS

        response.set_cookie(
            self.cookie_name,
            token,
            max_age=SESSION_TTL_SECONDS,
            path="/",
            httponly=True,
            secure=self.config.cookie_secure,
            samesite="lax",
        )

    def _build(self) -> Any:
        import gradio as gr
        from fastapi import FastAPI, Request
        from fastapi.responses import RedirectResponse

        app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
        config = self.config

        def redirect(url: str) -> RedirectResponse:
            return RedirectResponse(url, status_code=303)

        @app.get("/", include_in_schema=False)
        def root(request: Request):
            return redirect(f"{APP_PATH}/" if self.current_user(request) else "/login")

        @app.get("/healthz", include_in_schema=False)
        def healthz():
            return {"ok": True}

        @app.get("/login", include_in_schema=False)
        def login_form(request: Request):
            if self.current_user(request):
                return redirect(f"{APP_PATH}/")
            return login_page(signup_open=config.signup_open)

        @app.post("/login", include_in_schema=False)
        async def login(request: Request):
            form = await request.form()
            username = str(form.get("username") or "")[:64]
            password = str(form.get("password") or "")[:256]
            ip = self._request_ip(request)
            window = 15 * MINUTE
            ip_key, acct_key = f"loginfail:ip:{ip}", f"loginfail:acct:{username.casefold()}"
            if (
                self.limiter.count(ip_key, window) >= config.login_failures_per_ip
                or self.limiter.count(acct_key, window) >= config.login_failures_per_account
            ):
                return login_page(
                    "Too many failed logins. Please wait 15 minutes and try again.",
                    config.signup_open,
                    429,
                )
            user = self.accounts.authenticate(username, password)
            if user is None:
                self.limiter.hit([Rule(ip_key, 10**9, window), Rule(acct_key, 10**9, window)])
                return login_page("Wrong username or password.", config.signup_open, 401)
            response = redirect(f"{APP_PATH}/")
            self._set_session_cookie(response, self.accounts.create_session(user.username))
            return response

        @app.get("/signup", include_in_schema=False)
        def signup_form(request: Request):
            if not config.signup_open:
                return closed_page()
            if self.current_user(request):
                return redirect(f"{APP_PATH}/")
            return signup_page()

        @app.post("/signup", include_in_schema=False)
        async def signup(request: Request):
            if not config.signup_open:
                return closed_page()
            form = await request.form()
            username = str(form.get("username") or "").strip()[:64]
            password = str(form.get("password") or "")[:256]
            confirm = str(form.get("confirm") or "")[:256]
            ip = self._request_ip(request)
            attempts = self.limiter.hit(
                [Rule(f"signup-try:ip:{ip}", config.signup_attempts_per_ip_per_hour, HOUR)]
            )
            if not attempts.allowed:
                return signup_page(
                    "Too many sign-up attempts. Please try again later.", username, 429
                )
            created_key = f"signup-ok:ip:{ip}"
            if self.limiter.count(created_key, DAY) >= config.signups_per_ip_per_day:
                return signup_page(
                    "Too many accounts were created from your network today. "
                    "Please try again tomorrow.",
                    username,
                    429,
                )
            if password != confirm:
                return signup_page("The passwords do not match.", username, 400)
            try:
                user = self.accounts.create_user(username, password, ip=ip)
            except AccountError as exc:
                return signup_page(str(exc), username, 400)
            self.limiter.hit([Rule(created_key, 10**9, DAY)])
            response = redirect(f"{APP_PATH}/")
            self._set_session_cookie(response, self.accounts.create_session(user.username))
            return response

        @app.api_route("/logout", methods=["GET", "POST"], include_in_schema=False)
        def logout(request: Request):
            self.accounts.delete_session(request.cookies.get(self.cookie_name))
            response = redirect("/login")
            response.delete_cookie(self.cookie_name, path="/", secure=config.cookie_secure)
            return response

        app = gr.mount_gradio_app(
            app,
            self.blocks,
            path=APP_PATH,
            auth_dependency=self.current_user,
            show_error=False,  # never send tracebacks to visitors
            ssr_mode=False,
            footer_links=[],  # no "Use via API" docs or Gradio links for visitors
            # The UI has no upload widget, but Gradio's upload route exists; keep it tiny.
            max_file_size=MAX_UPLOAD_BYTES,
        )
        return SecurityMiddleware(app, self)


class SecurityMiddleware:
    """Pure-ASGI middleware (safe for Gradio's streaming responses).

    * Adds security headers to every response.
    * Rejects cross-site form posts (Origin/Referer must match the Host).
    * Sends visitors without a session from ``/app/`` to the login page.
    """

    HEADERS = (
        (b"x-content-type-options", b"nosniff"),
        (b"x-frame-options", b"DENY"),
        (b"referrer-policy", b"strict-origin-when-cross-origin"),
        (b"permissions-policy", b"camera=(), microphone=(), geolocation=()"),
        (b"cross-origin-opener-policy", b"same-origin"),
    )

    def __init__(self, app: Callable, hosted: HostedApp) -> None:
        self.app = app
        self.hosted = hosted

    async def __call__(self, scope: dict, receive: Callable, send: Callable) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        from starlette.requests import Request
        from starlette.responses import PlainTextResponse, RedirectResponse

        request = Request(scope)
        path = scope.get("path", "")
        method = scope.get("method", "GET")

        if (
            method == "POST"
            and path in {"/login", "/signup", "/logout"}
            and not _same_origin(request)
        ):
            await PlainTextResponse("Cross-site request blocked.", status_code=403)(
                scope, receive, send
            )
            return
        if (
            method == "GET"
            and path in {APP_PATH, f"{APP_PATH}/"}
            and not self.hosted.current_user(request)
        ):
            await RedirectResponse("/login", status_code=303)(scope, receive, send)
            return

        async def send_with_headers(message: dict) -> None:
            if message["type"] == "http.response.start":
                existing = {k.lower() for k, _ in message.get("headers", [])}
                extra = [(k, v) for k, v in self.HEADERS if k not in existing]
                message["headers"] = [*message.get("headers", []), *extra]
            await send(message)

        await self.app(scope, receive, send_with_headers)


def _same_origin(request: Any) -> bool:
    """True unless the browser says the form was posted from another site."""
    host = (request.headers.get("host") or "").lower()
    for header in ("origin", "referer"):
        value = request.headers.get(header)
        if value:
            if value == "null":
                return False
            return urlsplit(value).netloc.lower() == host
    return True  # non-browser clients send neither; SameSite cookies still apply


def create_hosted_app(factory: AgentFactory, web_config: WebConfig | None = None) -> HostedApp:
    return HostedApp(
        factory, web_config or WebConfig.from_env(host_mode=True), HostingConfig.from_env()
    )


def serve(factory: AgentFactory, web_config: WebConfig) -> None:
    import uvicorn

    hosted = create_hosted_app(factory, web_config)
    c = hosted.config
    print(
        f"🚀 Launching Agent Web UI (hosting mode) on http://{web_config.server_name}:"
        f"{web_config.server_port} ..."
    )
    print(
        f"   sign-up: {'open' if c.signup_open else 'closed'} | accounts db: {c.db_path} | "
        f"trust proxy headers: {c.trust_proxy} | secure cookies: {c.cookie_secure}"
    )
    print(
        f"   limits: {c.messages_per_account_per_hour}/h & {c.messages_per_account_per_day}/day "
        f"per account, {c.messages_per_ip_per_hour}/h & {c.messages_per_ip_per_day}/day per IP, "
        f"{c.messages_global_per_day}/day total"
    )
    if not c.trust_proxy:
        print(
            "⚠️  APP_TRUST_PROXY is off: behind a reverse proxy every visitor shares the "
            "proxy's IP. Turn it on when the app is only reachable through the proxy."
        )
    uvicorn.run(
        hosted.app,
        host=web_config.server_name,
        port=web_config.server_port,
        proxy_headers=False,  # client IPs are resolved by APP_TRUST_PROXY instead
        server_header=False,
        log_level="info",
    )
