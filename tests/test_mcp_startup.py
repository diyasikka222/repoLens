"""MCP server startup / health behaviour (Milestone 20).

These tests pin the operating contract of the MCP server startup path:

- importing the MCP modules never builds a repository index (verified in a
  fresh subprocess so module-import side effects are observable);
- ``make_engine_factory`` starts fast: root validation is eager, engine
  construction is lazy;
- the first request triggers initialization exactly once; repeated requests
  reuse that state;
- initialization failures surface a *useful, safe* error instead of an opaque
  crash.

All tests are offline and deterministic.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from repolens.context import (
    CandidateRole,
    ContextBudget,
    ContextCandidate,
    ContextEngine,
    ContextFirewall,
    ContextPackage,
    estimate_tokens,
)
from repolens.mcp import launcher
from repolens.mcp.deps import build_engine
from repolens.mcp.errors import ConfigurationError, RepositoryError

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _candidate(path: str, source: str) -> ContextCandidate:
    return ContextCandidate(
        path=Path(path),
        source=source,
        role=CandidateRole.PRIMARY,
        estimated_tokens=estimate_tokens(source),
        selection_reason="test",
    )


def _package(query: str) -> ContextPackage:
    return ContextPackage(
        query=query,
        budget=ContextBudget(max_tokens=8000),
        selected_files=(_candidate("a.py", "x = 1"),),
    )


class MockEngine:
    def __init__(self) -> None:
        self.calls = 0

    def build_context(self, query: str) -> ContextPackage:
        self.calls += 1
        return _package(query)


# ---------------------------------------------------------------------------
# Import-time behaviour (fresh subprocess)
# ---------------------------------------------------------------------------


def test_importing_mcp_module_does_not_build_index(tmp_path: Path) -> None:
    """Importing ``repolens.mcp`` must not touch any cache directory."""
    cache_root = tmp_path / "cache"
    script = (
        "import os\n"
        "os.environ['REPOLENS_CACHE_DIR'] = "
        f"{str(cache_root)!r}\n"
        "import repolens.mcp.server\n"
        "import repolens.mcp.tool\n"
        "import repolens.mcp.launcher\n"
        "import repolens.incremental_index\n"
        "print(list(os.scandir() if False else []), end='')\n"
    )
    result = subprocess.run(
        [sys.executable, "-I", "-c", script],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    leftovers = sorted(str(p) for p in cache_root.rglob("*") if p.is_dir())
    assert leftovers == [], (
        "importing repolens.mcp must not create cache directories (got "
        f"{leftovers})"
    )


def test_mcp_server_startup_needs_no_index(tmp_path: Path, monkeypatch) -> None:
    """``make_engine_factory`` validates the root but builds no engine/index."""
    called: list[str] = []

    def spy_build_engine(*args, **kwargs):
        called.append("build_engine")
        return MockEngine()

    monkeypatch.setattr(launcher, "build_engine", spy_build_engine)
    factory = launcher.make_engine_factory(str(tmp_path))
    assert called == []
    assert callable(factory)


# ---------------------------------------------------------------------------
# Lazy initialization and reuse
# ---------------------------------------------------------------------------


def test_first_request_initializes_once(tmp_path: Path, monkeypatch) -> None:
    build_calls: list[str] = []
    engine = MockEngine()

    def counting_build_engine(*args, **kwargs):
        build_calls.append("build")
        return engine

    monkeypatch.setattr(launcher, "build_engine", counting_build_engine)
    factory = launcher.make_engine_factory(str(tmp_path))
    assert build_calls == []

    first = factory()
    second = factory()
    assert first is engine
    assert second is engine
    assert len(build_calls) == 1


def test_factory_failure_is_a_useful_mcp_error() -> None:
    with pytest.raises(RepositoryError):
        launcher.make_engine_factory("/definitely/does/not/exist")


def test_build_engine_initialization_failure_is_not_opaque() -> None:
    """A bad repository root raises a safe, useful configuration error."""
    from repolens.mcp.deps import validate_repository_root

    with pytest.raises(RepositoryError) as exc:
        validate_repository_root("/definitely/not/a/repo")
    assert exc.value.safe_message
    assert exc.value.diagnostic


def test_runtime_engine_failure_wraps_as_configuration_error(
    tmp_path: Path, monkeypatch
) -> None:
    """An index-build failure inside ``build_engine`` maps to a McpError."""
    from repolens.mcp import deps

    def broken_index(root):
        raise RuntimeError("index exploded")

    monkeypatch.setattr(deps, "_build_index", broken_index)
    with pytest.raises(ConfigurationError) as exc:
        build_engine(tmp_path)
    assert exc.value.safe_message
    assert exc.value.diagnostic


# ---------------------------------------------------------------------------
# End-to-end: real engine over a real repo, everything leveraged
# ---------------------------------------------------------------------------


def test_first_context_request_builds_real_engine(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "a.py").write_text("def alpha():\n    return 1\n", encoding="utf-8")
    factory = launcher.make_engine_factory(str(repo))
    engine = factory()  # triggers lazy initialization
    assert isinstance(engine, ContextEngine)
    package = engine.build_context("alpha function")
    assert package.selected_files
    # Repeated requests reuse the same cached engine instance.
    assert factory() is engine
    assert factory(max_tokens=None, dependency_depth=None) is engine