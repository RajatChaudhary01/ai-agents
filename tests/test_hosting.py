import pytest

pytest.importorskip("gradio")

from fastapi.testclient import TestClient

from ai_agents.hosting.accounts import AccountError, AccountStore, hash_password, verify_password
from ai_agents.hosting.ratelimit import RateLimiter, Rule, client_ip, ip_bucket
from ai_agents.hosting.server import HostedApp, HostingConfig
from ai_agents.web import WebConfig

PASSWORD = "correct horse battery"


class FakeClock:
    def __init__(self, now=1_000_000.0):
        self.now = now

    def __call__(self):
        return self.now


class FakeAgent:
    stream_outputs = False


# ---------------------------------------------------------------- rate limiter
class TestRateLimiter:
    def test_limit_and_window(self):
        clock = FakeClock()
        limiter = RateLimiter(clock)
        rule = Rule("k", 2, 60)
        assert limiter.hit([rule]).allowed
        assert limiter.hit([rule]).allowed
        blocked = limiter.hit([rule])
        assert not blocked.allowed
        assert blocked.retry_after == pytest.approx(60)
        clock.now += 61
        assert limiter.hit([rule]).allowed

    def test_refused_request_consumes_nothing(self):
        limiter = RateLimiter(FakeClock())
        account, ip = Rule("acct", 10, 60), Rule("ip", 1, 60)
        assert limiter.hit([account, ip]).allowed
        assert not limiter.hit([account, ip]).allowed  # blocked by the IP rule...
        assert limiter.count("acct", 60) == 1  # ...without spending the account budget

    def test_message_names_the_limit(self):
        limiter = RateLimiter(FakeClock())
        rule = Rule("k", 1, 3600, "messages per hour")
        limiter.hit([rule])
        message = limiter.hit([rule]).message()
        assert "1 messages per hour" in message
        assert "60 minutes" in message


class TestClientIp:
    def test_ipv6_grouped_by_64(self):
        assert ip_bucket("2001:db8:1:2:aaaa::1") == ip_bucket("2001:db8:1:2:ffff::9")
        assert ip_bucket("2001:db8:1:2::1") != ip_bucket("2001:db8:1:3::1")

    def test_ipv4_mapped_ipv6_is_ipv4(self):
        assert ip_bucket("::ffff:203.0.113.7") == "203.0.113.7"

    def test_forwarded_header_ignored_unless_trusted(self):
        headers = {"x-forwarded-for": "198.51.100.1"}
        assert client_ip(headers, "10.0.0.2", trust_proxy=False) == "10.0.0.2"
        assert client_ip(headers, "10.0.0.2", trust_proxy=True) == "198.51.100.1"

    def test_spoofed_hops_before_the_proxy_are_ignored(self):
        # The client sent "1.2.3.4"; the proxy appended the real address last.
        headers = {"x-forwarded-for": "1.2.3.4, 198.51.100.1"}
        assert client_ip(headers, "10.0.0.2", trust_proxy=True) == "198.51.100.1"


# -------------------------------------------------------------------- accounts
class TestAccounts:
    def test_password_hashing(self):
        stored = hash_password(PASSWORD)
        assert PASSWORD not in stored
        assert verify_password(PASSWORD, stored)
        assert not verify_password("wrong password!", stored)
        assert hash_password(PASSWORD) != stored  # random salt

    def test_create_and_authenticate(self):
        store = AccountStore(":memory:")
        store.create_user("Alice", PASSWORD)
        assert store.authenticate("alice", PASSWORD).display_name == "Alice"
        assert store.authenticate("alice", "wrong password") is None
        assert store.authenticate("nobody", PASSWORD) is None

    def test_usernames_are_case_insensitive(self):
        store = AccountStore(":memory:")
        store.create_user("Alice", PASSWORD)
        with pytest.raises(AccountError, match="taken"):
            store.create_user("ALICE", PASSWORD)

    @pytest.mark.parametrize(
        ("username", "password", "match"),
        [
            ("ab", PASSWORD, "Username"),
            ("bad name", PASSWORD, "Username"),
            ("<script>", PASSWORD, "Username"),
            ("alice", "short", "Password"),
            ("alice", "alice12345678", "username"),
        ],
    )
    def test_validation(self, username, password, match):
        with pytest.raises(AccountError, match=match):
            AccountStore(":memory:").create_user(username, password)

    def test_sessions(self):
        clock = FakeClock()
        store = AccountStore(":memory:", clock=clock)
        store.create_user("alice", PASSWORD)
        token = store.create_session("alice", ttl=60)
        assert store.session_user(token) == "alice"
        assert store.session_user("forged-token") is None
        clock.now += 61
        assert store.session_user(token) is None

    def test_disabling_or_password_change_logs_out(self):
        store = AccountStore(":memory:")
        store.create_user("alice", PASSWORD)
        token = store.create_session("alice")
        store.set_password("alice", "another long password")
        assert store.session_user(token) is None
        token = store.create_session("alice")
        store.set_disabled("alice")
        assert store.session_user(token) is None
        assert store.authenticate("alice", "another long password") is None

    def test_only_token_hashes_are_stored(self, tmp_path):
        store = AccountStore(tmp_path / "app.db")
        store.create_user("alice", PASSWORD)
        token = store.create_session("alice")
        raw = (tmp_path / "app.db").read_bytes() + (tmp_path / "app.db-wal").read_bytes()
        assert token.encode() not in raw
        assert PASSWORD.encode() not in raw


# ---------------------------------------------------------------------- server
def make_config(**overrides):
    values = {
        "db_path": ":memory:",
        "signup_open": True,
        "trust_proxy": True,
        "cookie_secure": False,  # TestClient talks plain http
        "signups_per_ip_per_day": 2,
        "signup_attempts_per_ip_per_hour": 10,
        "login_failures_per_ip": 5,
        "login_failures_per_account": 3,
        "messages_per_account_per_hour": 2,
        "messages_per_account_per_day": 100,
        "messages_per_ip_per_hour": 3,
        "messages_per_ip_per_day": 100,
        "messages_global_per_day": 1000,
        "tool_calls_per_minute": 60,
    }
    values.update(overrides)
    return HostingConfig(**values)


@pytest.fixture
def hosted():
    return HostedApp(
        FakeAgent,
        WebConfig.from_env(host_mode=True),
        make_config(),
        accounts=AccountStore(":memory:"),
    )


@pytest.fixture
def client(hosted):
    return TestClient(hosted.app, follow_redirects=False)


def signup(client, username, ip="198.51.100.1", password=PASSWORD, confirm=None):
    return client.post(
        "/signup",
        data={"username": username, "password": password, "confirm": confirm or password},
        headers={"x-forwarded-for": ip},
    )


def login(client, username, password=PASSWORD, ip="198.51.100.1"):
    return client.post(
        "/login",
        data={"username": username, "password": password},
        headers={"x-forwarded-for": ip},
    )


class TestPages:
    def test_anonymous_visitor_is_sent_to_login(self, client):
        assert client.get("/").headers["location"] == "/login"
        assert client.get("/app/").headers["location"] == "/login"

    def test_gradio_api_requires_login(self, client):
        assert client.get("/app/config").status_code == 401

    def test_signup_logs_in_and_opens_the_app(self, client):
        response = signup(client, "alice")
        assert response.status_code == 303
        assert response.headers["location"] == "/app/"
        assert client.get("/app/config").status_code == 200

    def test_session_cookie_is_httponly_and_samesite(self, client):
        cookie = signup(client, "alice").headers["set-cookie"].lower()
        assert "httponly" in cookie
        assert "samesite=lax" in cookie

    def test_secure_cookie_uses_host_prefix(self):
        hosted = HostedApp(
            FakeAgent,
            WebConfig.from_env(host_mode=True),
            make_config(cookie_secure=True),
            accounts=AccountStore(":memory:"),
        )
        cookie = signup(TestClient(hosted.app, follow_redirects=False), "alice").headers[
            "set-cookie"
        ]
        assert cookie.startswith("__Host-agent_session=")
        assert "secure" in cookie.lower()

    def test_login_and_logout(self, client):
        signup(client, "alice")
        client.cookies.clear()
        assert login(client, "alice").status_code == 303
        assert client.get("/app/config").status_code == 200
        client.get("/logout")
        client.cookies.clear()
        assert client.get("/app/config").status_code == 401

    def test_wrong_password(self, client):
        signup(client, "alice")
        client.cookies.clear()
        response = login(client, "alice", "wrong password")
        assert response.status_code == 401
        assert "Wrong username or password" in response.text

    def test_errors_are_html_escaped(self, client):
        response = signup(client, "<b>x</b>")
        assert "<b>x</b>" not in response.text
        assert "&lt;b&gt;x&lt;/b&gt;" in response.text

    def test_security_headers(self, client):
        response = client.get("/login")
        assert response.headers["x-frame-options"] == "DENY"
        assert response.headers["x-content-type-options"] == "nosniff"
        assert "frame-ancestors 'none'" in response.headers["content-security-policy"]

    def test_cross_site_form_post_is_blocked(self, client):
        response = client.post(
            "/login",
            data={"username": "a", "password": "b"},
            headers={"origin": "https://evil.example"},
        )
        assert response.status_code == 403

    def test_closed_signup(self):
        hosted = HostedApp(
            FakeAgent,
            WebConfig.from_env(host_mode=True),
            make_config(signup_open=False),
            accounts=AccountStore(":memory:"),
        )
        client = TestClient(hosted.app, follow_redirects=False)
        assert client.get("/signup").status_code == 403
        assert signup(client, "alice").status_code == 403
        assert "Sign up" not in client.get("/login").text


class TestAbuseLimits:
    def test_signups_per_ip_per_day(self, client):
        assert signup(client, "user1").status_code == 303
        assert signup(client, "user2").status_code == 303
        third = signup(client, "user3")
        assert third.status_code == 429
        assert "Too many accounts" in third.text
        assert signup(client, "user3", ip="203.0.113.9").status_code == 303  # other network

    def test_ipv6_rotation_does_not_bypass_signup_limit(self, client):
        assert signup(client, "user1", ip="2001:db8::1").status_code == 303
        assert signup(client, "user2", ip="2001:db8::2").status_code == 303
        assert signup(client, "user3", ip="2001:db8::ffff").status_code == 429

    def test_login_lockout_per_account(self, client):
        signup(client, "alice")
        client.cookies.clear()
        for n in range(3):
            login(client, "alice", "wrong password", ip=f"203.0.113.{n}")
        # Even the right password from a fresh IP waits out the lockout.
        response = login(client, "alice", ip="203.0.113.99")
        assert response.status_code == 429

    def test_login_lockout_per_ip(self, client):
        for n in range(5):
            login(client, f"guess{n}", "wrong password")
        assert login(client, "guess9", "wrong password").status_code == 429


class FakeGradioRequest:
    def __init__(self, username, ip):
        self.username = username
        self.headers = {"x-forwarded-for": ip}
        self.client = None


class TestChatLimits:
    def test_per_account_limit(self, hosted):
        request = FakeGradioRequest("alice", "198.51.100.1")
        assert hosted.chat_guard(request) is None
        assert hosted.chat_guard(request) is None
        refusal = hosted.chat_guard(request)
        assert refusal.startswith("Rate limit reached (2 messages per hour)")

    def test_dummy_accounts_share_the_ip_budget(self, hosted):
        # Three throwaway accounts from one IP get 3 messages in total, not 3 x 2.
        results = [
            hosted.chat_guard(FakeGradioRequest(f"dummy{n}", "198.51.100.1")) for n in range(6)
        ]
        assert results[:3] == [None, None, None]
        assert all("from your network" in r for r in results[3:])

    def test_one_user_on_many_ips_is_still_limited(self, hosted):
        results = [
            hosted.chat_guard(FakeGradioRequest("alice", f"203.0.113.{n}")) for n in range(3)
        ]
        assert results[:2] == [None, None]
        assert "messages per hour" in results[2]

    def test_global_daily_cap(self):
        hosted = HostedApp(
            FakeAgent,
            WebConfig.from_env(host_mode=True),
            make_config(messages_global_per_day=2),
            accounts=AccountStore(":memory:"),
        )
        assert hosted.chat_guard(FakeGradioRequest("a", "198.51.100.1")) is None
        assert hosted.chat_guard(FakeGradioRequest("b", "198.51.100.2")) is None
        assert "for this service" in hosted.chat_guard(FakeGradioRequest("c", "198.51.100.3"))

    def test_tools_are_limited_too(self):
        hosted = HostedApp(
            FakeAgent,
            WebConfig.from_env(host_mode=True),
            make_config(tool_calls_per_minute=1),
            accounts=AccountStore(":memory:"),
        )
        request = FakeGradioRequest("alice", "198.51.100.1")
        assert hosted.tools_guard(request) is None
        assert "tool uses per minute" in hosted.tools_guard(request)


def test_hosting_config_from_env(monkeypatch, tmp_path):
    monkeypatch.setenv("APP_SIGNUP", "closed")
    monkeypatch.setenv("APP_TRUST_PROXY", "true")
    monkeypatch.setenv("APP_MESSAGES_PER_IP_PER_DAY", "7")
    config = HostingConfig.from_env()
    assert config.signup_open is False
    assert config.trust_proxy is True
    assert config.cookie_secure is True
    assert config.messages_per_ip_per_day == 7
    monkeypatch.setenv("APP_SIGNUP", "maybe")
    with pytest.raises(ValueError, match="APP_SIGNUP"):
        HostingConfig.from_env()


def test_uploads_are_capped_for_logged_in_users(client):
    signup(client, "alice")
    big = b"x" * (2 * 1024 * 1024)
    response = client.post("/app/gradio_api/upload", files={"files": ("big.txt", big)})
    assert response.status_code == 413


def test_unwritable_database_gives_a_clear_error(tmp_path):
    from ai_agents.hosting.accounts import AccountStoreError

    blocker = tmp_path / "not-a-dir"
    blocker.write_text("")
    with pytest.raises(AccountStoreError, match="accounts database"):
        AccountStore(blocker / "app.db")


def test_host_launch_reports_unwritable_database(monkeypatch, tmp_path):
    from ai_agents import web

    blocker = tmp_path / "not-a-dir"
    blocker.write_text("")
    monkeypatch.setenv("APP_DB_PATH", str(blocker / "app.db"))
    monkeypatch.setattr("ai_agents.agents.create_first_agent", FakeAgent)
    with pytest.raises(SystemExit, match=r"\[Config error\] Cannot create"):
        web.main(["--host"])
