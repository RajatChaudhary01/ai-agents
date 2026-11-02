"""Resilient web search tool.

Wraps the ``ddgs`` metasearch client with:

* input validation and normalisation (empty / non-string / oversized queries),
* per-request timeouts,
* retries with exponential backoff for transient failures (timeouts, rate limits,
  connection errors),
* fallback across several search backends,
* an in-memory cache that is served when every backend is unavailable,
* a graceful text response instead of an exception, so a network failure never
  crashes the agent run.
"""

from __future__ import annotations

import logging
import re
import time
from collections import OrderedDict
from collections.abc import Callable, Sequence
from typing import Any

from smolagents import Tool

logger = logging.getLogger(__name__)

MAX_QUERY_LENGTH = 400
DEFAULT_BACKENDS: tuple[str, ...] = ("duckduckgo", "auto", "wikipedia")

_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f]")
_WHITESPACE = re.compile(r"\s+")
_TRANSIENT_HINTS = ("timeout", "timed out", "ratelimit", "rate limit", "connect", "temporar")


class SearchInputError(ValueError):
    """Raised internally when a search query is invalid."""


def normalize_query(query: Any, max_length: int = MAX_QUERY_LENGTH) -> str:
    """Validate and clean a search query.

    Strips control characters, collapses whitespace and truncates overly long
    queries on a word boundary.

    Raises:
        SearchInputError: if the query is not a string or is empty after cleaning.
    """
    if not isinstance(query, str):
        raise SearchInputError(f"query must be a string, got {type(query).__name__}")

    cleaned = _WHITESPACE.sub(" ", _CONTROL_CHARS.sub(" ", query)).strip()
    if not cleaned:
        raise SearchInputError("query is empty")

    if len(cleaned) > max_length:
        truncated = cleaned[:max_length]
        cleaned = truncated.rsplit(" ", 1)[0] if " " in truncated else truncated
    return cleaned


def is_transient_error(exc: BaseException) -> bool:
    """Return True for errors worth retrying on the same backend.

    ddgs wraps network failures in a generic ``DDGSException`` whose message carries
    the real cause (e.g. ``ConnectError``), so both the type name and the message
    are inspected.
    """
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return True
    text = f"{type(exc).__name__} {exc}".lower()
    return any(hint in text for hint in _TRANSIENT_HINTS)


def _default_client_factory(timeout: float) -> Any:
    from ddgs import DDGS

    return DDGS(timeout=int(max(1, timeout)))


def format_results(results: Sequence[dict[str, Any]]) -> str:
    lines = []
    for item in results:
        title = str(item.get("title") or "Untitled").strip()
        href = str(item.get("href") or item.get("url") or "").strip()
        body = str(item.get("body") or item.get("snippet") or "").strip()
        lines.append(f"[{title}]({href})\n{body}" if href else f"{title}\n{body}")
    return "## Search Results\n\n" + "\n\n".join(lines)


class ResilientWebSearchTool(Tool):
    """Drop-in replacement for ``smolagents.DuckDuckGoSearchTool`` with fallbacks."""

    name = "web_search"
    description = (
        "Performs a web search for the query and returns the top results "
        "(title, link and snippet). If search is unavailable it returns a message "
        "starting with 'Search unavailable' instead of failing."
    )
    inputs = {"query": {"type": "string", "description": "The search query to perform."}}  # noqa: RUF012
    output_type = "string"

    def __init__(
        self,
        max_results: int = 5,
        timeout: float = 10.0,
        backends: Sequence[str] = DEFAULT_BACKENDS,
        max_attempts_per_backend: int = 2,
        backoff_base: float = 1.0,
        total_time_budget: float = 45.0,
        cache_size: int = 64,
        client_factory: Callable[[float], Any] | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.max_results = max(1, min(int(max_results), 25))
        self.timeout = max(1.0, float(timeout))
        self.backends = tuple(b.strip() for b in backends if b and b.strip()) or DEFAULT_BACKENDS
        self.max_attempts_per_backend = max(1, int(max_attempts_per_backend))
        self.backoff_base = max(0.0, float(backoff_base))
        self.total_time_budget = max(self.timeout, float(total_time_budget))
        self.cache_size = max(0, int(cache_size))
        self._client_factory = client_factory or _default_client_factory
        self._sleep = sleep
        self._clock = clock
        self._cache: OrderedDict[str, str] = OrderedDict()

    # ------------------------------------------------------------------ cache
    def _cache_get(self, key: str) -> str | None:
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        return None

    def _cache_put(self, key: str, value: str) -> None:
        if self.cache_size == 0:
            return
        self._cache[key] = value
        self._cache.move_to_end(key)
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)

    # ----------------------------------------------------------------- search
    def _search_backend(self, query: str, backend: str) -> list[dict[str, Any]]:
        client = self._client_factory(self.timeout)
        results = client.text(query, max_results=self.max_results, backend=backend)
        return list(results or [])

    def forward(self, query: str) -> str:
        try:
            cleaned = normalize_query(query)
        except SearchInputError as exc:
            return (
                f"Invalid search query: {exc}. Provide a short, non-empty text query, "
                "e.g. web_search(query='population of Tokyo')."
            )

        cache_key = cleaned.casefold()
        start = self._clock()
        errors: list[str] = []

        for backend in self.backends:
            for attempt in range(1, self.max_attempts_per_backend + 1):
                if self._clock() - start > self.total_time_budget:
                    errors.append("time budget exhausted")
                    return self._fallback(cleaned, cache_key, errors)
                try:
                    results = self._search_backend(cleaned, backend)
                except Exception as exc:  # noqa: BLE001 - any client failure triggers fallback
                    errors.append(f"{backend}: {type(exc).__name__}: {exc}")
                    logger.warning(
                        "web_search backend=%s attempt=%d failed: %r", backend, attempt, exc
                    )
                    if is_transient_error(exc) and attempt < self.max_attempts_per_backend:
                        self._sleep(self.backoff_base * 2 ** (attempt - 1))
                        continue
                    break  # non-transient error or retries exhausted: next backend

                if results:
                    formatted = format_results(results)
                    self._cache_put(cache_key, formatted)
                    return formatted
                errors.append(f"{backend}: no results")
                break  # empty result set: try the next backend

        return self._fallback(cleaned, cache_key, errors)

    def _fallback(self, query: str, cache_key: str, errors: list[str]) -> str:
        cached = self._cache_get(cache_key)
        if cached is not None:
            return f"(Live search failed; showing cached results.)\n\n{cached}"

        if errors and all(e.endswith("no results") for e in errors):
            return f"No results found for '{query}'. Try a shorter or more general query."
        detail = "; ".join(errors[-3:]) or "unknown error"
        return (
            f"Search unavailable for '{query}' ({detail}). Do not retry the same query; "
            "answer from your own knowledge or other tools, and state that live "
            "search could not be used."
        )
