import json
from urllib.parse import parse_qs

import httpx
import pytest

pytest.importorskip("gradio")

from fastapi.testclient import TestClient

from ai_agents.hosting.accounts import AccountStore
from ai_agents.hosting.captcha import TOKEN_FIELD, VERIFY_URL, Turnstile
from ai_agents.hosting.server import HostedApp
from ai_agents.web import WebConfig
from tests.test_hosting import PASSWORD, FakeAgent, make_config

SECRET = "test-secret"
GOOD_TOKEN = "good-token"


class FakeSiteverify:
    """Stands in for Cloudflare's siteverify endpoint."""

    def __init__(self, status=200, body=None):
        self.calls = []
        self.status = status
        self.body = body

    def __call__(self, request: httpx.Request) -> httpx.Response:
        assert str(request.url) == VERIFY_URL
        form = {k: v[0] for k, v in parse_qs(request.content.decode()).items()}
        self.calls.append(form)
        if self.body is not None:
            return httpx.Response(self.status, content=self.body)
        ok = form.get("secret") == SECRET and form.get("response") == GOOD_TOKEN
        payload = {"success": ok, "error-codes": [] if ok else ["invalid-input-response"]}
        return httpx.Response(self.status, json=payload)


def turnstile(fake):
    return Turnstile("site-key-123", SECRET, transport=httpx.MockTransport(fake))


def make_client(fake, **overrides):
    hosted = HostedApp(
        FakeAgent,
        WebConfig.from_env(host_mode=True),
        make_config(turnstile=turnstile(fake), **overrides),
        accounts=AccountStore(":memory:"),
    )
    return TestClient(hosted.app, follow_redirects=False)


def signup(client, username="alice", token=GOOD_TOKEN, ip="198.51.100.1"):
    data = {"username": username, "password": PASSWORD, "confirm": PASSWORD}
    if token is not None:
        data[TOKEN_FIELD] = token
    return client.post("/signup", data=data, headers={"x-forwarded-for": ip})


def login(client, username="alice", token=GOOD_TOKEN, password=PASSWORD):
    data = {"username": username, "password": password}
    if token is not None:
        data[TOKEN_FIELD] = token
    return client.post("/login", data=data, headers={"x-forwarded-for": "198.51.100.1"})


# ------------------------------------------------------------------- the verifier
class TestVerify:
    async def _verify(self, fake, token, ip=None):
        return await turnstile(fake).verify(token, ip)

    def run(self, coro):
        import asyncio

        return asyncio.run(coro)

    def test_good_token_passes_and_sends_secret_and_ip(self):
        fake = FakeSiteverify()
        assert self.run(self._verify(fake, GOOD_TOKEN, "203.0.113.5")).ok
        assert fake.calls == [{"secret": SECRET, "response": GOOD_TOKEN, "remoteip": "203.0.113.5"}]

    def test_bad_token_fails_with_cloudflare_code(self):
        result = self.run(self._verify(FakeSiteverify(), "forged"))
        assert not result.ok
        assert result.reason == "invalid-input-response"

    def test_missing_or_oversized_token_skips_the_network(self):
        fake = FakeSiteverify()
        assert self.run(self._verify(fake, None)).reason == "missing"
        assert self.run(self._verify(fake, "x" * 5000)).reason == "invalid"
        assert fake.calls == []

    @pytest.mark.parametrize(
        "fake", [FakeSiteverify(status=500), FakeSiteverify(body=b"<html>not json")]
    )
    def test_fails_closed_when_cloudflare_misbehaves(self, fake):
        assert self.run(self._verify(fake, GOOD_TOKEN)).reason == "unavailable"

    def test_fails_closed_when_cloudflare_is_unreachable(self):
        def unreachable(request):
            raise httpx.ConnectError("no route to host")

        assert self.run(self._verify(unreachable, GOOD_TOKEN)).reason == "unavailable"

    def test_only_success_true_counts(self):
        fake = FakeSiteverify(body=json.dumps({"success": "yes"}).encode())
        assert not self.run(self._verify(fake, GOOD_TOKEN)).ok


# ------------------------------------------------------------------------ config
class TestConfig:
    def test_disabled_without_keys(self):
        assert Turnstile.from_env() is None

    def test_enabled_with_both_keys(self, monkeypatch):
        monkeypatch.setenv("TURNSTILE_SITE_KEY", "site")
        monkeypatch.setenv("TURNSTILE_SECRET_KEY", "secret")
        captcha = Turnstile.from_env()
        assert captcha.site_key == "site"
        assert "secret" not in repr(captcha)  # never logged by accident

    def test_one_key_alone_is_an_error(self, monkeypatch):
        monkeypatch.setenv("TURNSTILE_SITE_KEY", "site")
        with pytest.raises(ValueError, match="TURNSTILE_SECRET_KEY"):
            Turnstile.from_env()


# ------------------------------------------------------------------------- pages
class TestPages:
    def test_widget_and_csp_on_signup_page(self):
        response = make_client(FakeSiteverify()).get("/signup")
        assert 'class="cf-turnstile" data-sitekey="site-key-123"' in response.text
        assert "challenges.cloudflare.com/turnstile/v0/api.js" in response.text
        csp = response.headers["content-security-policy"]
        assert "script-src https://challenges.cloudflare.com" in csp
        assert "frame-src https://challenges.cloudflare.com" in csp

    def test_login_widget_can_be_turned_off(self):
        client = make_client(FakeSiteverify(), captcha_on_login=False)
        assert "cf-turnstile" not in client.get("/login").text
        assert "cf-turnstile" in client.get("/signup").text

    def test_no_widget_or_script_when_disabled(self):
        hosted = HostedApp(
            FakeAgent,
            WebConfig.from_env(host_mode=True),
            make_config(),
            accounts=AccountStore(":memory:"),
        )
        response = TestClient(hosted.app).get("/signup")
        assert "cf-turnstile" not in response.text
        assert "script-src" not in response.headers["content-security-policy"]


# ------------------------------------------------------------------------- flows
class TestSignup:
    def test_good_token_creates_account(self):
        fake = FakeSiteverify()
        client = make_client(fake)
        response = signup(client)
        assert response.status_code == 303
        assert fake.calls[0]["remoteip"] == "198.51.100.1"  # real IP, not the proxy's

    @pytest.mark.parametrize("token", [None, "forged"])
    def test_missing_or_bad_token_is_rejected(self, token):
        client = make_client(FakeSiteverify())
        response = signup(client, token=token)
        assert response.status_code == 400
        assert "verification check" in response.text
        assert signup(client).status_code == 303  # the name was not taken by the failure

    def test_captcha_runs_before_username_checks(self):
        client = make_client(FakeSiteverify())
        signup(client)
        client.cookies.clear()
        # A bot without a valid token cannot learn that "alice" already exists.
        response = signup(client, token="forged")
        assert "verification check" in response.text
        assert "taken" not in response.text

    def test_rate_limit_applies_before_calling_cloudflare(self):
        fake = FakeSiteverify()
        client = make_client(fake, signup_attempts_per_ip_per_hour=2)
        signup(client, "user1", token="forged")
        signup(client, "user2", token="forged")
        assert signup(client, "user3").status_code == 429
        assert len(fake.calls) == 2


class TestLogin:
    def _client_with_alice(self, fake, **overrides):
        client = make_client(fake, **overrides)
        signup(client)
        client.cookies.clear()
        return client

    def test_good_token_and_password_logs_in(self):
        client = self._client_with_alice(FakeSiteverify())
        assert login(client).status_code == 303

    def test_bad_token_is_rejected_even_with_right_password(self):
        client = self._client_with_alice(FakeSiteverify())
        response = login(client, token="forged")
        assert response.status_code == 400
        assert "set-cookie" not in response.headers

    def test_captcha_failures_do_not_lock_the_account(self):
        client = self._client_with_alice(FakeSiteverify())
        for _ in range(5):
            login(client, token=None)
        assert login(client).status_code == 303

    def test_login_without_captcha_when_turned_off(self):
        client = self._client_with_alice(FakeSiteverify(), captcha_on_login=False)
        assert login(client, token=None).status_code == 303
