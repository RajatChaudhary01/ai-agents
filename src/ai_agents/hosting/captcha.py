"""Cloudflare Turnstile CAPTCHA for the sign-up (and optionally login) forms.

Turnstile is free and usually invisible to real people: the widget runs a background
check and only shows a challenge when it is unsure. The browser puts a one-time token in
the form field ``cf-turnstile-response``; the server confirms it with Cloudflare's
siteverify API before creating an account or checking a password.

Enable it by setting both ``TURNSTILE_SITE_KEY`` and ``TURNSTILE_SECRET_KEY`` (create a
widget for your domain in the Cloudflare dashboard under Turnstile). Verification fails
closed: if Cloudflare cannot be reached, the form is rejected rather than let through.
"""

import html
import logging
import os
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

SCRIPT_URL = "https://challenges.cloudflare.com/turnstile/v0/api.js"
VERIFY_URL = "https://challenges.cloudflare.com/turnstile/v0/siteverify"
ORIGIN = "https://challenges.cloudflare.com"
TOKEN_FIELD = "cf-turnstile-response"
MAX_TOKEN_LENGTH = 2048  # Cloudflare tokens are far shorter; reject junk early


@dataclass(frozen=True)
class CaptchaResult:
    ok: bool
    reason: str = ""


@dataclass(frozen=True)
class Turnstile:
    site_key: str
    secret_key: str = field(repr=False)
    timeout: float = 5.0
    # Test hook: an ``httpx`` transport used instead of the network.
    transport: Any = field(default=None, repr=False, compare=False)

    @classmethod
    def from_env(cls) -> "Turnstile | None":
        site_key = (os.getenv("TURNSTILE_SITE_KEY") or "").strip()
        secret_key = (os.getenv("TURNSTILE_SECRET_KEY") or "").strip()
        if not site_key and not secret_key:
            return None
        if not (site_key and secret_key):
            raise ValueError(
                "Set both TURNSTILE_SITE_KEY and TURNSTILE_SECRET_KEY to enable the CAPTCHA "
                "(or neither to disable it)."
            )
        return cls(site_key, secret_key)

    def widget_html(self) -> str:
        return (
            f'<div class="cf-turnstile" data-sitekey="{html.escape(self.site_key)}" '
            'data-theme="auto" data-size="flexible"></div>'
        )

    @staticmethod
    def script_html() -> str:
        return f'<script src="{SCRIPT_URL}" async defer></script>'

    async def verify(self, token: str | None, remote_ip: str | None = None) -> CaptchaResult:
        if not token:
            return CaptchaResult(False, "missing")
        if len(token) > MAX_TOKEN_LENGTH:
            return CaptchaResult(False, "invalid")

        import httpx

        data = {"secret": self.secret_key, "response": token}
        if remote_ip:
            data["remoteip"] = remote_ip
        try:
            async with httpx.AsyncClient(timeout=self.timeout, transport=self.transport) as client:
                response = await client.post(VERIFY_URL, data=data)
                response.raise_for_status()
                outcome = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            logger.warning("Turnstile verification unavailable: %r", exc)
            return CaptchaResult(False, "unavailable")

        if outcome.get("success") is True:
            return CaptchaResult(True)
        codes = ",".join(outcome.get("error-codes") or []) or "rejected"
        logger.info("Turnstile rejected a token: %s", codes)
        return CaptchaResult(False, codes)
