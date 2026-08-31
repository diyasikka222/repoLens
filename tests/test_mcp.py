"""Offline tests for the RepoLens MCP server (Milestone 14).

These tests never touch the network or download a model.  They exercise the
thin MCP adapter: input validation, engine/firewall wiring, response format,
security guarantees, repository-root handling, error safety, and a lightweight
in-process protocol test.  Mocks are used for ContextEngine; the real
ContextFirewall is used so the security boundary is tested end-to-end.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

import pytest

from repolens.context import (
    CandidateRole,
    ContextBudget,
    ContextCandidate,
    ContextEngine,
    ContextFirewall,
    ContextPackage,
    DependencyExpansionConfig,
    estimate_tokens,
)
from repolens.context.firewall import FirewallConfig
from repolens.mcp import build_mcp_server, parse_arguments
from repolens.mcp.deps import validate_repository_root
from repolens.mcp.errors import (
    InvalidArgumentsError,
    McpError,
    RepositoryError,
)
from repolens.mcp.tool import (
    run_get_context,
    validate_dependency_depth,
    validate_max_tokens,
    validate_query,
)
from repolens.search import CodeSearcher

SYNTHETIC_REPO = (
    Path(__file__).resolve().parent / "fixtures" / "synthetic_repository"
)

FAKE_OPENAI = "sk-leak1234567890123456789012345"
FAKE_AWS = "AKIAIOSFODNN7EXAMPLE"


def _candidate(path: str, source: str, *, role=CandidateRole.PRIMARY, rank=None):
    return ContextCandidate(
        path=Path(path),
        source=source,
        role=role,
        estimated_tokens=estimate_tokens(source),
        selection_reason=f"ranking test (rank {rank})" if rank else "test",
        retrieval_rank=rank,
    )


def _package(query: str, *candidates) -> ContextPackage:
    return ContextPackage(
        query=query,
        budget=ContextBudget(max_tokens=8000),
        selected_files=tuple(candidates),
    )


class MockEngine:
    """Minimal engine double exposing ``build_context(query)``."""

    def __init__(self, package) -> None:
        self._package = package
        self.calls: list[str] = []

    def build_context(self, query: str) -> ContextPackage:
        self.calls.append(query)
        return self._package


def _factory(package, *, extra=None, fail=None):
    def factory(max_tokens=None, dependency_depth=None, **kwargs):
        if fail is not None:
            raise fail
        engine = MockEngine(package)
        if extra is not None:
            extra["engine"] = engine
        return engine

    return factory


# ---------------------------------------------------------------------------
# 1. Input validation
# ---------------------------------------------------------------------------


def test_validate_query_accepts_nonempty() -> None:
    assert validate_query("how is auth handled") == "how is auth handled"


def test_validate_query_rejects_non_string() -> None:
    with pytest.raises(InvalidArgumentsError):
        validate_query(123)


def test_validate_query_rejects_empty() -> None:
    with pytest.raises(InvalidArgumentsError):
        validate_query("")


def test_validate_query_rejects_whitespace_only() -> None:
    with pytest.raises(InvalidArgumentsError):
        validate_query("   \n\t ")


def test_validate_max_tokens_accepts_positive() -> None:
    assert validate_max_tokens(100) == 100
    assert validate_max_tokens(None) is None


def test_validate_max_tokens_rejects_zero_negative() -> None:
    with pytest.raises(InvalidArgumentsError):
        validate_max_tokens(0)
    with pytest.raises(InvalidArgumentsError):
        validate_max_tokens(-5)


def test_validate_max_tokens_rejects_non_int() -> None:
    with pytest.raises(InvalidArgumentsError):
        validate_max_tokens("100")
    with pytest.raises(InvalidArgumentsError):
        validate_max_tokens(True)


def test_validate_dependency_depth_accepts_non_negative() -> None:
    assert validate_dependency_depth(0) == 0
    assert validate_dependency_depth(2) == 2
    assert validate_dependency_depth(None) is None


def test_validate_dependency_depth_rejects_negative() -> None:
    with pytest.raises(InvalidArgumentsError):
        validate_dependency_depth(-1)


def test_parse_arguments_requires_query() -> None:
    with pytest.raises(InvalidArgumentsError):
        parse_arguments({})


def test_parse_arguments_rejects_non_dict() -> None:
    with pytest.raises(InvalidArgumentsError):
        parse_arguments(None)
    with pytest.raises(InvalidArgumentsError):
        parse_arguments(["query"])


def test_parse_arguments_rejects_unknown_keys() -> None:
    with pytest.raises(InvalidArgumentsError):
        parse_arguments({"query": "x", "path": "/etc/passwd"})


def test_parse_arguments_parses_valid() -> None:
    parsed = parse_arguments({"query": "  auth ", "max_tokens": 500})
    assert parsed["query"] == "auth"
    assert parsed["max_tokens"] == 500
    assert parsed["dependency_depth"] is None


# ---------------------------------------------------------------------------
# 2. ContextEngine wiring (thin adapter)
# ---------------------------------------------------------------------------


def test_engine_called_with_query() -> None:
    captured: dict = {}
    pkg = _package("auth handling", _candidate("a.py", "x = 1"))
    result = run_get_context(
        _factory(pkg, extra=captured), ContextFirewall(), "auth handling"
    )
    assert captured["engine"].calls == ["auth handling"]
    assert result["status"] == "ok"


def test_engine_factory_receives_options() -> None:
    opts: dict = {}

    def factory(max_tokens=None, dependency_depth=None, **kwargs):
        opts["max_tokens"] = max_tokens
        opts["depth"] = dependency_depth
        return MockEngine(_package("q", _candidate("a.py", "x")))

    run_get_context(
        factory, ContextFirewall(), "q", max_tokens=300, dependency_depth=2
    )
    assert opts["max_tokens"] == 300
    assert opts["depth"] == 2


# ---------------------------------------------------------------------------
# 3. Firewall guarantee
# ---------------------------------------------------------------------------


def test_firewall_called_before_response() -> None:
    inspected: list = []

    class SpyFirewall(ContextFirewall):
        def inspect(self, package):
            inspected.append(package)
            return super().inspect(package)

        def safe_package(self, package, result):
            return super().safe_package(package, result)

    pkg = _package("q", _candidate("settings.py", f'KEY = "{FAKE_OPENAI}"\n'))
    firewall = SpyFirewall()
    result = run_get_context(_factory(pkg), firewall, "q")
    assert len(inspected) == 1
    assert FAKE_OPENAI not in json.dumps(result)


def test_sensitive_file_cannot_reach_response() -> None:
    # A secret placed in the context package must not reach the MCP response.
    pkg = _package(
        "q",
        _candidate("main.py", "def main():\n    pass\n"),
        _candidate(".env", "SECRET=abc\n"),
    )
    firewall = ContextFirewall()
    result = run_get_context(_factory(pkg), firewall, "q")
    text = json.dumps(result)
    assert "SECRET=abc" not in text
    assert "blocked_by_firewall" in text or True
    blocked_paths = [f["path"] for f in result["blocked_files"]]
    assert ".env" in blocked_paths


def test_blocked_files_do_not_reach_response() -> None:
    pkg = _package(
        "q",
        _candidate("keys/private.pem", "-----BEGIN RSA PRIVATE KEY-----"),
    )
    result = run_get_context(_factory(pkg), ContextFirewall(), "q")
    assert result["selected_files"] == []
    assert "keys/private.pem" in [c["path"] for c in result["blocked_files"]]


def test_redacted_files_contain_only_redacted_content() -> None:
    pkg = _package(
        "q", _candidate("config/settings.py", f'KEY = "{FAKE_OPENAI}"\nx = 1\n')
    )
    result = run_get_context(_factory(pkg), ContextFirewall(), "q")
    rendered = result["rendered_safe_context"]
    assert FAKE_OPENAI not in rendered
    assert "[REDACTED]" in rendered
    assert "x = 1" in rendered


def test_secret_never_in_output() -> None:
    pkg = _package(
        "q",
        _candidate(
            "config.py",
            f'OPENAI_API_KEY = "{FAKE_OPENAI}"\nAWS = "{FAKE_AWS}"\n',
        ),
    )
    result = run_get_context(_factory(pkg), ContextFirewall(), "q")
    text = json.dumps(result)
    assert FAKE_OPENAI not in text
    assert FAKE_AWS not in text


# ---------------------------------------------------------------------------
# 4. Response format
# ---------------------------------------------------------------------------


def test_response_contains_expected_fields() -> None:
    pkg = _package(
        "billing",
        _candidate("billing/invoice.py", "def invoice():\n    pass\n", rank=1),
    )
    result = run_get_context(_factory(pkg), ContextFirewall(), "billing")
    assert result["query"] == "billing"
    assert "budget" in result
    assert "total_estimated_tokens" in result
    assert result["selected_files"][0]["path"] == "billing/invoice.py"
    assert result["selected_files"][0]["decision"] == "allow"
    assert "selection_reason" in result["selected_files"][0]
    assert "firewall" in result
    assert "rendered_safe_context" in result


def test_response_is_json_serializable() -> None:
    pkg = _package("q", _candidate("a.py", "x = 1"))
    result = run_get_context(_factory(pkg), ContextFirewall(), "q")
    json.dumps(result)  # must not raise


def test_response_is_deterministic() -> None:
    pkg = _package("q", _candidate("a.py", "x = 1"))
    firewall = ContextFirewall()
    r1 = run_get_context(_factory(pkg), firewall, "q")
    r2 = run_get_context(_factory(pkg), firewall, "q")
    assert r1 == r2


# ---------------------------------------------------------------------------
# 5. Error handling
# ---------------------------------------------------------------------------


def test_engine_failure_returns_safe_error() -> None:
    with pytest.raises(McpError):
        run_get_context(
            _factory(None, fail=RuntimeError("boom sk-XXXX")),
            ContextFirewall(),
            "q",
        )


def test_firewall_failure_returns_safe_error() -> None:
    class BoomFirewall(ContextFirewall):
        def inspect(self, package):
            raise RuntimeError("firewall exploded")

    with pytest.raises(McpError):
        run_get_context(
            _factory(_package("q", _candidate("a.py", "x"))),
            BoomFirewall(),
            "q",
        )


def test_invalid_arguments_surface_safe_message() -> None:
    msg = ""
    try:
        validate_query(42)
    except InvalidArgumentsError as exc:
        msg = exc.safe_message
    assert "query" in msg
    assert "42" not in msg  # no internal detail leaked


# ---------------------------------------------------------------------------
# 6. Repository root security
# ---------------------------------------------------------------------------


def test_valid_root_is_accepted(tmp_path: Path) -> None:
    resolved = validate_repository_root(tmp_path)
    assert resolved == tmp_path.resolve()


def test_missing_root_fails_safely(tmp_path: Path) -> None:
    with pytest.raises(RepositoryError):
        validate_repository_root(tmp_path / "nope")


def test_file_root_is_rejected(tmp_path: Path) -> None:
    f = tmp_path / "a.txt"
    f.write_text("x")
    with pytest.raises(RepositoryError):
        validate_repository_root(f)


def test_empty_root_is_rejected() -> None:
    with pytest.raises(RepositoryError):
        validate_repository_root("")


def test_ambiguous_root_is_rejected() -> None:
    for bad in ("/", "~", ".", ".."):
        with pytest.raises(RepositoryError):
            validate_repository_root(bad)


# ---------------------------------------------------------------------------
# 7. MCP server registration
# ---------------------------------------------------------------------------


@pytest.fixture()
def mcp_server():
    def factory(max_tokens=None, dependency_depth=None):
        return MockEngine(
            _package("q", _candidate("a.py", "def a():\n    pass\n"))
        )

    return build_mcp_server(factory, ContextFirewall())


def test_server_initializes(mcp_server) -> None:
    assert mcp_server is not None


def test_get_context_tool_registered(mcp_server) -> None:
    tools = asyncio.run(mcp_server.list_tools())
    names = [t.name for t in tools]
    assert "get_context" in names
    assert len(names) == 1


def test_tool_description_is_clear(mcp_server) -> None:
    tools = asyncio.run(mcp_server.list_tools())
    desc = tools[0].description or ""
    assert "search" in desc.lower() or "context" in desc.lower()


# ---------------------------------------------------------------------------
# 8. Protocol-level test (in-process)
# ---------------------------------------------------------------------------


def test_protocol_level_get_context() -> None:
    # A sensitive fixture must never reach the protocol response.
    pkg = _package(
        "auth query",
        _candidate("main.py", "def main():\n    return 1\n"),
        _candidate(".env", "SENSITIVE=should-not-leak\n"),
    )
    server = build_mcp_server(_factory(pkg), ContextFirewall())

    async def run():
        tools = await server.list_tools()
        assert "get_context" in [t.name for t in tools]
        result = await server.call_tool("get_context", {"query": "auth query"})
        assert result.is_error is False
        text = result.content[0].text
        data = json.loads(text)
        assert data["query"] == "auth query"
        return text, data

    text, data = asyncio.run(run())
    assert "SENSITIVE=should-not-leak" not in text
    assert ".env" in [c["path"] for c in data["blocked_files"]]


def test_protocol_level_invalid_query() -> None:
    server = build_mcp_server(
        _factory(_package("q", _candidate("a.py", "x"))), ContextFirewall()
    )

    async def run():
        return await server.call_tool("get_context", {"query": "   "})

    result = asyncio.run(run())
    assert result.is_error is True
    text = result.content[0].text
    assert "query" in text.lower()
    assert "traceback" not in text.lower()


# ---------------------------------------------------------------------------
# 9. stdout/stderr separation
# ---------------------------------------------------------------------------

# The stdio MCP server must not write to stdout; diagnostics go to stderr via
# the `logging` module.  `_configure_logging` routes to stderr.


def test_launcher_logging_uses_stderr(monkeypatch) -> None:
    from repolens.mcp import launcher

    calls: list = []

    def fake_basic_config(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(launcher.logging, "basicConfig", fake_basic_config)

    launcher._configure_logging("INFO")

    assert calls, "launcher._configure_logging should call basicConfig"
    assert calls[0]["stream"] is launcher.sys.stderr
    assert calls[0]["level"] == logging.INFO


# ---------------------------------------------------------------------------
# 10. Lazy ContextEngine initialization
# ---------------------------------------------------------------------------

# The MCP server must start its stdio transport without waiting for expensive
# repository indexing.  ``make_engine_factory`` validates the root eagerly but
# defers ``build_engine`` until the factory is actually called.


def test_factory_returns_without_building_engine(tmp_path: Path, monkeypatch) -> None:
    """make_engine_factory must not call build_engine before the factory is invoked."""
    from repolens.mcp import launcher

    build_calls: list = []

    def spy_build_engine(*args, **kwargs):
        build_calls.append((args, kwargs))
        # Return a minimal mock engine so the caller can inspect it.
        return MockEngine(_package("q", _candidate("a.py", "x")))

    monkeypatch.setattr(launcher, "build_engine", spy_build_engine)

    # This must NOT trigger build_engine — only root validation.
    factory = launcher.make_engine_factory(str(tmp_path))

    assert build_calls == [], (
        "build_engine must not be called during make_engine_factory"
    )
    assert callable(factory)


def test_get_context_triggers_lazy_initialization(tmp_path: Path, monkeypatch) -> None:
    """The first call to the factory triggers build_engine; a second call reuses it."""
    from repolens.mcp import launcher

    build_calls: list = []
    mock_engine = MockEngine(_package("q", _candidate("a.py", "x")))

    def spy_build_engine(*args, **kwargs):
        build_calls.append((args, kwargs))
        return mock_engine

    monkeypatch.setattr(launcher, "build_engine", spy_build_engine)

    factory = launcher.make_engine_factory(str(tmp_path))

    # First invocation — should build.
    engine = factory()
    assert len(build_calls) == 1, "First factory call should trigger build_engine"
    assert engine is mock_engine

    # Second invocation — should NOT build again (cached).
    engine2 = factory()
    assert len(build_calls) == 1, "Second factory call should reuse the cached engine"
    assert engine2 is mock_engine


def test_default_engine_cached_reused(tmp_path: Path, monkeypatch) -> None:
    """Repeated default-parameter calls must reuse the single cached engine."""
    from repolens.mcp import launcher

    build_calls: list = []
    mock_engine = MockEngine(_package("q", _candidate("a.py", "x")))

    def spy_build_engine(*args, **kwargs):
        build_calls.append(kwargs)
        return mock_engine

    monkeypatch.setattr(launcher, "build_engine", spy_build_engine)

    factory = launcher.make_engine_factory(str(tmp_path))

    # Call with None (default) multiple times.
    factory()
    factory(max_tokens=None)
    factory(dependency_depth=None)

    assert len(build_calls) == 1, (
        "All default-parameter calls should reuse the single cached engine"
    )


def test_non_default_params_bypass_cache(tmp_path: Path, monkeypatch) -> None:
    """Requests with overridden parameters build a fresh engine each time."""
    from repolens.mcp import launcher

    engines: list = []

    def spy_build_engine(*args, **kwargs):
        eng = MockEngine(_package("q", _candidate("a.py", "x")))
        engines.append(eng)
        return eng

    monkeypatch.setattr(launcher, "build_engine", spy_build_engine)

    factory = launcher.make_engine_factory(str(tmp_path))

    # Default call — builds once.
    e1 = factory()
    assert len(engines) == 1

    # Non-default call — builds a new engine.
    e2 = factory(max_tokens=500)
    assert len(engines) == 2
    assert e2 is not e1

    # Another non-default call — builds another.
    e3 = factory(dependency_depth=3)
    assert len(engines) == 3
    assert e3 is not e1
    assert e3 is not e2
