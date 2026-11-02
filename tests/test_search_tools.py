import pytest

from ai_agents.tools.search_tools import (
    MAX_QUERY_LENGTH,
    ResilientWebSearchTool,
    SearchInputError,
    format_results,
    is_transient_error,
    normalize_query,
)

RESULT = [{"title": "Tokyo", "href": "https://example.com/tokyo", "body": "Capital of Japan"}]


class RatelimitException(Exception):
    """Mimics ddgs.exceptions.RatelimitException (matched by name)."""


class FakeClient:
    """Fake ddgs client. ``script`` maps backend -> list of outcomes consumed in order.

    An outcome is either a list of results or an exception instance to raise.
    """

    def __init__(self, script):
        self.script = {k: list(v) for k, v in script.items()}
        self.calls = []

    def text(self, query, max_results, backend):
        self.calls.append((backend, query, max_results))
        outcomes = self.script.get(backend) or [[]]
        outcome = outcomes.pop(0) if len(outcomes) > 1 else outcomes[0]
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def make_tool(script, **kwargs):
    client = FakeClient(script)
    sleeps = []
    clock = kwargs.pop("clock", FakeClock())
    tool = ResilientWebSearchTool(
        client_factory=lambda timeout: client,
        sleep=sleeps.append,
        clock=clock,
        **kwargs,
    )
    return tool, client, sleeps


# --------------------------------------------------------------- input checks
class TestNormalizeQuery:
    def test_collapses_whitespace_and_control_chars(self):
        assert normalize_query("  tokyo\t\n  population\x00 ") == "tokyo population"

    @pytest.mark.parametrize("bad", ["", "   ", "\n\t", "\x00\x01"])
    def test_rejects_empty(self, bad):
        with pytest.raises(SearchInputError, match="empty"):
            normalize_query(bad)

    @pytest.mark.parametrize("bad", [None, 42, ["tokyo"], {"q": "tokyo"}])
    def test_rejects_non_string(self, bad):
        with pytest.raises(SearchInputError, match="must be a string"):
            normalize_query(bad)

    def test_truncates_long_query_on_word_boundary(self):
        query = "word " * 200
        cleaned = normalize_query(query)
        assert len(cleaned) <= MAX_QUERY_LENGTH
        assert cleaned.endswith("word")

    def test_truncates_long_query_without_spaces(self):
        assert len(normalize_query("x" * 1000)) == MAX_QUERY_LENGTH


@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        (TimeoutError(), True),
        (ConnectionError(), True),
        (RatelimitException(), True),
        (type("TimeoutException", (Exception,), {})(), True),
        (type("DDGSException", (Exception,), {})("ConnectError: connection refused"), True),
        (type("DDGSException", (Exception,), {})("operation timed out"), True),
        (type("DDGSException", (Exception,), {})("No results found."), False),
        (ValueError("bad backend"), False),
        (RuntimeError("No results found"), False),
    ],
)
def test_is_transient_error(exc, expected):
    assert is_transient_error(exc) is expected


def test_format_results_handles_missing_fields():
    out = format_results([{"title": "A", "href": "https://a", "body": "x"}, {"body": "only"}])
    assert out.startswith("## Search Results")
    assert "[A](https://a)\nx" in out
    assert "Untitled\nonly" in out


# ------------------------------------------------------------- tool behaviour
def test_invalid_query_does_not_hit_network():
    tool, client, _ = make_tool({"duckduckgo": [RESULT]})
    assert tool(query="   ").startswith("Invalid search query")
    assert client.calls == []


def test_success_on_first_backend():
    tool, client, sleeps = make_tool({"duckduckgo": [RESULT]})
    out = tool(query="Tokyo population")
    assert "[Tokyo](https://example.com/tokyo)" in out
    assert [c[0] for c in client.calls] == ["duckduckgo"]
    assert sleeps == []


def test_transient_error_is_retried_with_backoff():
    tool, client, sleeps = make_tool(
        {"duckduckgo": [TimeoutError("read timed out"), RESULT]},
        max_attempts_per_backend=3,
        backoff_base=0.5,
    )
    assert "Tokyo" in tool(query="tokyo")
    assert [c[0] for c in client.calls] == ["duckduckgo", "duckduckgo"]
    assert sleeps == [0.5]


def test_exponential_backoff_then_next_backend():
    tool, client, sleeps = make_tool(
        {"duckduckgo": [RatelimitException("202 ratelimit")], "auto": [RESULT]},
        max_attempts_per_backend=3,
        backoff_base=1.0,
    )
    assert "Tokyo" in tool(query="tokyo")
    assert [c[0] for c in client.calls] == ["duckduckgo"] * 3 + ["auto"]
    assert sleeps == [1.0, 2.0]


def test_non_transient_error_skips_to_next_backend_without_retry():
    tool, client, sleeps = make_tool(
        {"duckduckgo": [ValueError("unsupported backend")], "auto": [RESULT]},
        max_attempts_per_backend=3,
    )
    assert "Tokyo" in tool(query="tokyo")
    assert [c[0] for c in client.calls] == ["duckduckgo", "auto"]
    assert sleeps == []


def test_empty_results_fall_through_to_next_backend():
    tool, client, _ = make_tool({"duckduckgo": [[]], "auto": [[]], "wikipedia": [RESULT]})
    assert "Tokyo" in tool(query="tokyo")
    assert [c[0] for c in client.calls] == ["duckduckgo", "auto", "wikipedia"]


def test_no_results_anywhere():
    tool, _, _ = make_tool({})
    assert tool(query="qwertyuiop").startswith("No results found for 'qwertyuiop'")


def test_all_backends_fail_returns_graceful_message():
    tool, _, _ = make_tool(
        {b: [TimeoutError("timed out")] for b in ("duckduckgo", "auto", "wikipedia")}
    )
    out = tool(query="tokyo")
    assert out.startswith("Search unavailable for 'tokyo'")
    assert "TimeoutError" in out


def test_client_construction_failure_is_handled():
    def broken_factory(timeout):
        raise ConnectionError("no network")

    tool = ResilientWebSearchTool(client_factory=broken_factory, sleep=lambda s: None)
    assert tool(query="tokyo").startswith("Search unavailable")


def test_cached_results_served_when_search_later_fails():
    tool, client, _ = make_tool({"duckduckgo": [RESULT]})
    first = tool(query="Tokyo")
    client.script = {b: [TimeoutError("down")] for b in ("duckduckgo", "auto", "wikipedia")}
    second = tool(query="  tokyo ")  # same query after normalisation / case-folding
    assert second.startswith("(Live search failed; showing cached results.)")
    assert first in second


def test_cache_is_bounded():
    tool, _, _ = make_tool({"duckduckgo": [RESULT]}, cache_size=2)
    for q in ("a", "b", "c"):
        tool(query=q)
    assert list(tool._cache) == ["b", "c"]


def test_time_budget_stops_further_attempts():
    clock = FakeClock()

    def slow_failure(timeout):
        clock.now += 30
        raise TimeoutError("slow")

    tool = ResilientWebSearchTool(
        client_factory=slow_failure,
        sleep=lambda s: None,
        clock=clock,
        timeout=10,
        total_time_budget=45,
    )
    out = tool(query="tokyo")
    assert "time budget exhausted" in out
    assert clock.now == 60  # two attempts, then the budget check aborts


def test_constructor_clamps_bad_settings():
    tool = ResilientWebSearchTool(
        max_results=1000, timeout=0, backends=["", "  "], max_attempts_per_backend=0
    )
    assert tool.max_results == 25
    assert tool.timeout == 1.0
    assert tool.backends == ("duckduckgo", "auto", "wikipedia")
    assert tool.max_attempts_per_backend == 1


def test_tool_metadata_matches_smolagents_contract():
    tool = ResilientWebSearchTool()
    assert tool.name == "web_search"
    assert tool.output_type == "string"
    assert set(tool.inputs) == {"query"}
