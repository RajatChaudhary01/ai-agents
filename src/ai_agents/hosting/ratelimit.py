"""In-memory sliding-window rate limiting keyed by account, client IP or anything else.

Limits are checked as a group: a request is allowed only if *every* rule passes, and a
refused request consumes nothing, so being blocked by one limit never burns another.

State lives in process memory, which suits the single-process server this project runs.
A restart clears the counters.
"""

import ipaddress
import threading
import time
from collections import deque
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class Rule:
    """At most ``limit`` hits per ``window`` seconds for ``key``."""

    key: str
    limit: int
    window: float
    label: str = ""  # shown to the user when this rule blocks a request


@dataclass(frozen=True)
class Decision:
    allowed: bool
    retry_after: float = 0.0
    rule: Rule | None = None

    def message(self) -> str:
        if self.allowed or self.rule is None:
            return ""
        wait = _human_duration(self.retry_after)
        what = self.rule.label or "requests"
        return f"Rate limit reached ({self.rule.limit} {what}). Please try again in {wait}."


def _human_duration(seconds: float) -> str:
    seconds = max(1, int(seconds + 0.999))
    if seconds < 90:
        return f"{seconds} seconds"
    minutes = (seconds + 59) // 60
    if minutes < 90:
        return f"{minutes} minutes"
    return f"{(minutes + 59) // 60} hours"


class RateLimiter:
    """Thread-safe sliding-log rate limiter."""

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._hits: dict[tuple[str, float], deque[float]] = {}
        self._lock = threading.Lock()
        self._last_sweep = clock()

    def hit(self, rules: Iterable[Rule]) -> Decision:
        """Record one hit against every rule, or none if any rule is already at its limit."""
        rules = list(rules)
        now = self._clock()
        with self._lock:
            self._maybe_sweep(now)
            logs = []
            for rule in rules:
                log = self._hits.setdefault((rule.key, rule.window), deque())
                while log and now - log[0] >= rule.window:
                    log.popleft()
                if len(log) >= rule.limit:
                    return Decision(False, retry_after=rule.window - (now - log[0]), rule=rule)
                logs.append(log)
            for log in logs:
                log.append(now)
        return Decision(True)

    def count(self, key: str, window: float) -> int:
        now = self._clock()
        with self._lock:
            log = self._hits.get((key, window), deque())
            return sum(1 for t in log if now - t < window)

    def _maybe_sweep(self, now: float) -> None:
        """Drop empty or fully expired logs so memory stays bounded."""
        if now - self._last_sweep < 300:
            return
        self._last_sweep = now
        for key in [k for k, log in self._hits.items() if not log or now - log[-1] >= k[1]]:
            del self._hits[key]


# ----------------------------------------------------------------------- client IP
def ip_bucket(ip: str) -> str:
    """Group addresses one person can trivially rotate through.

    IPv4 addresses are used as-is. IPv6 users usually get a whole /64, so the bucket is
    the /64 prefix; otherwise one abuser could look like billions of different clients.
    """
    try:
        addr = ipaddress.ip_address(ip.strip())
    except ValueError:
        return ip.strip() or "unknown"
    if isinstance(addr, ipaddress.IPv6Address):
        if addr.ipv4_mapped:
            return str(addr.ipv4_mapped)
        return str(ipaddress.ip_network(f"{addr}/64", strict=False))
    return str(addr)


def client_ip(headers: Mapping[str, str], peer: str | None, trust_proxy: bool) -> str:
    """The real client address, as a rate-limit bucket.

    Only trust ``X-Forwarded-For`` when the app is reachable solely through a reverse
    proxy (``APP_TRUST_PROXY=true``); otherwise anyone could spoof the header. The last
    entry is the one the proxy itself appended, so it is the only one that can be trusted.
    """
    if trust_proxy:
        forwarded = headers.get("x-forwarded-for") or headers.get("X-Forwarded-For") or ""
        hops = [h.strip() for h in forwarded.split(",") if h.strip()]
        if hops:
            return ip_bucket(hops[-1])
    return ip_bucket(peer or "unknown")
