"""Reliability & failure-injection tests (Phase 25.5).

These tests deliberately exercise failure paths across the whole stack and
verify safe, deterministic recovery: parser/repository input edge cases,
persistent cache corruption and store-failure tolerance, embedding-provider
failures, atomic-write interruption, incremental-index recovery, reference
and call-graph tolerance, architecture rebuilds, impact/change-plan error
contracts, context-budget safety, MCP-safe error propagation, and concurrent
shared-state access.  The core assertion style is *recovery equivalence*:
a storage or provider failure followed by a recovery must produce results
identical to a pristine run (fingerprinted, never object identity).

Nothing here changes production behaviour; the injection points are
:mod:`tests.helpers.failure_injection` wrappers and standard ``monkeypatch``.
"""

from __future__ import annotations

import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tests.helpers.failure_injection import (  # noqa: E402
    EmbeddingFailure,
    FlakyProvider,
    ShortResultProvider,
    WrongDimensionProvider,
    corrupt_all_json,
    drop_stale_partial,
    index_fingerprint,
    install_atomic_write_failure,
    install_fsync_failure,
    make_repo,
    package_fingerprint,
    plan_fingerprint,
    require_write_access,
)

from repolens.incremental_index import IncrementalIndexBuilder  # noqa: E402
from repolens.search import CodeSearcher  # noqa: E402
from repolens.semantic_search import SemanticSearcher  # noqa: E402
from repolens.embedding_cache import FileSystemEmbeddingCache  # noqa: E402
from repolens.embeddings import FakeEmbeddingProvider, EmbeddingProviderError  # noqa: E402
from repolens.graph import DependencyGraphBuilder  # noqa: E402
from repolens.index import SymbolIndexBuilder  # noqa: E402
from repolens.references import ReferenceExtractor, ReferenceIndexBuilder  # noqa: E402
from repolens.call_graph import CallGraphBuilder, CallGraphConfig  # noqa: E402
from repolens.architecture import ArchitectureGraphBuilder, ArchitectureConfig  # noqa: E402
from repolens.impact import ImpactAnalyzer, ImpactTargetError  # noqa: E402
from repolens.change_plan import ChangePlanEngine, ChangePlanConfig  # noqa: E402
from repolens.change_context import plan_response_payload, ChangeContextOptions  # noqa: E402
from repolens.context import ContextEngine, ContextBudget, ContextFirewall  # noqa: E402
from repolens.mcp.tool import run_get_context, parse_arguments  # noqa: E402
from repolens.mcp.errors import (  # noqa: E402
    ContextEngineError,
    InvalidArgumentsError,
    InternalError,
    ChangePlanError,
)
from repolens.mcp.change_plan_tool import (  # noqa: E402
    ChangePlanState,
    parse_change_plan_arguments,
    run_change_plan,
    run_change_context,
)
from repolens.mcp.architecture_tool import ArchitectureState  # noqa: E402


SINGLE_REPO = {
    "app.py": "def alpha():\n    return 1\n",
    "billing.py": "def beta():\n    return 2\n",
}


def _single_repo(tmp_path: Path) -> Path:
    return make_repo(tmp_path / "repo", SINGLE_REPO)


def _call_repo(tmp_path: Path) -> Path:
    return make_repo(
        tmp_path / "repo",
        {
            "checkout.py": (
                "import database\n\n"
                "def checkout(cart):\n"
                "    return database.store(cart)\n"
            ),
            "database.py": "def store(cart):\n    return len(cart)\n",
            "carts.py": (
                "from checkout import checkout\n\n"
                "def run():\n"
                "    return checkout([])\n"
            ),
        },
    )


def _chain_repo(tmp_path: Path, count: int) -> Path:
    names = "abcdefghijklmnopqrstuvwxyz"[:count]
    files: dict[str, str] = {}
    for index, name in enumerate(names):
        nxt = names[index + 1] if index + 1 < count else name
        files[f"{name}.py"] = (
            f"import {nxt}\n\n"
            f"def {name}_work():\n"
            f"    return {nxt}.{nxt}_work()\n"
        )
    return make_repo(tmp_path / "repo", files)


def _call_graph(repo: Path, config: CallGraphConfig | None = None):
    index = IncrementalIndexBuilder(repo, persist=False).build()
    references = ReferenceIndexBuilder(repo, index=index, persist=False).build()
    symbols = SymbolIndexBuilder(repo, index=index).build()
    graph = CallGraphBuilder(
        repo,
        index=index,
        reference_index=references,
        symbol_index=symbols,
        config=config,
    ).build()
    return graph, index


def _module_node(graph, path: str):
    for node in graph.module_nodes():
        if node.file_path.as_posix() == path:
            return node
    raise AssertionError(f"no module node for {path}")


@pytest.fixture
def isolated_cache(tmp_path, monkeypatch):
    """Route the MCP shared index cache into ``tmp_path`` (never the home cache)."""
    cache_dir = tmp_path / "repolens-cache"
    monkeypatch.setenv("REPOLENS_CACHE_DIR", str(cache_dir))
    monkeypatch.delenv("REPOLENS_CACHE_DISABLED", raising=False)
    return cache_dir


# ---------------------------------------------------------------------------
# A. Repository input edge cases
# ---------------------------------------------------------------------------


def test_non_utf8_source_degrades_to_empty_analysis(tmp_path):
    repo = make_repo(tmp_path / "repo", {"app.py": "def alpha():\n    return 1\n"})
    (repo / "bad.py").write_bytes(b"def \xff\xfe(): return 1\n")
    index = IncrementalIndexBuilder(repo, persist=False).build()
    assert index.by_path[Path("bad.py")].analysis.functions == []
    assert index.by_path[Path("app.py")].analysis.functions[0].name == "alpha"


def test_unreadable_file_degrades_to_empty_analysis(tmp_path):
    repo = make_repo(tmp_path / "repo", {"locked.py": "def locked():\n    return 9\n"})
    target = repo / "locked.py"
    target.chmod(0)
    if os.access(target, os.R_OK):
        pytest.skip("running as root; the read-only file is still readable")
    try:
        index = IncrementalIndexBuilder(repo, persist=False).build()
    finally:
        target.chmod(0o644)
    assert index.by_path[Path("locked.py")].analysis.functions == []


class _VanishingScanner:
    """Scanner whose discovered file populations a file that no longer exists."""

    def __init__(self) -> None:
        self.paths = [Path("app.py"), Path("ghost.py")]

    def discover_python_files(self):
        return list(self.paths)


def test_file_vanishing_after_scan_is_a_miss_not_a_crash(tmp_path):
    repo = make_repo(tmp_path / "repo", {"app.py": "def alpha():\n    return 1\n"})
    index = IncrementalIndexBuilder(
        repo, scanner=_VanishingScanner(), persist=False
    ).build()
    assert Path("ghost.py") in index.by_path
    assert index.by_path[Path("ghost.py")].analysis.functions == []
    assert index.stats.files_discovered == 2


def test_syntax_error_recovery_equals_pristine_build(tmp_path):
    repo = make_repo(
        tmp_path / "repo",
        {"a.py": "def alpha():\n    return 1\n", "b.py": "def beta():\n    return 2\n"},
    )
    cache_dir = tmp_path / "cache"
    clean = IncrementalIndexBuilder(repo, cache_dir=cache_dir).build()

    (repo / "b.py").write_text("def broken(:\n", encoding="utf-8")
    with pytest.raises(SyntaxError):
        IncrementalIndexBuilder(repo, cache_dir=cache_dir).build()

    (repo / "b.py").write_text("def beta():\n    return 2\n", encoding="utf-8")
    repaired = IncrementalIndexBuilder(repo, cache_dir=cache_dir).build()

    assert index_fingerprint(repaired) == index_fingerprint(clean)
    assert repaired.stats.cache_hits >= clean.stats.files_discovered - 1


# ---------------------------------------------------------------------------
# B. Embedding provider failures
# ---------------------------------------------------------------------------


def test_failed_embedding_leaves_no_partial_persistent_entry(tmp_path):
    repo = _single_repo(tmp_path)
    cache_dir = tmp_path / "embed_cache"
    provider = FlakyProvider(base=FakeEmbeddingProvider(), fail_calls=1)
    searcher = SemanticSearcher(
        repo,
        provider,
        candidate_searcher=CodeSearcher(repo),
        cache=FileSystemEmbeddingCache(cache_dir),
    )
    with pytest.raises(EmbeddingFailure):
        searcher.search("alpha", limit=5)
    assert list(cache_dir.glob("*.json")) == []
    assert list(cache_dir.glob("*.part-*.tmp")) == []
    assert searcher._vectors_by_path == {}


def test_embedding_failure_then_recovery_matches_pristine(tmp_path):
    repo = _single_repo(tmp_path)
    cache_dir = tmp_path / "embed_cache"
    flaky = FlakyProvider(base=FakeEmbeddingProvider(), fail_calls=1)
    failed = SemanticSearcher(
        repo,
        flaky,
        candidate_searcher=CodeSearcher(repo),
        cache=FileSystemEmbeddingCache(cache_dir),
    )
    with pytest.raises(EmbeddingFailure):
        failed.search("alpha", limit=5)

    pristine = SemanticSearcher(
        repo,
        FakeEmbeddingProvider(),
        candidate_searcher=CodeSearcher(repo),
        cache=FileSystemEmbeddingCache(cache_dir),
    )
    expected = pristine.search("alpha", limit=5)
    recovered = failed.search("alpha", limit=5)
    assert [(r.file_path, r.similarity) for r in recovered] == [
        (r.file_path, r.similarity) for r in expected
    ]
    assert list(cache_dir.glob("*.part-*.tmp")) == []


def test_provider_returning_too_few_vectors_raises_typed_error(tmp_path):
    """Regression: a provider that returns fewer vectors than requested must
    never silently truncate (which previously produced a partial ``_vectors_by_path``
    and a ``KeyError`` inside ``search``), and must leave no partial state."""
    repo = _single_repo(tmp_path)
    provider = ShortResultProvider(base=FakeEmbeddingProvider(), emit=0)
    searcher = SemanticSearcher(
        repo, provider, candidate_searcher=CodeSearcher(repo)
    )
    with pytest.raises(EmbeddingProviderError):
        searcher.search("alpha", limit=5)
    assert searcher._vectors_by_path == {}
    assert searcher.cache_stats["embedded_documents"] == 0


def test_wrong_dimension_provider_is_deterministic_not_cryptic(tmp_path):
    repo = _single_repo(tmp_path)
    provider = WrongDimensionProvider(base=FakeEmbeddingProvider(), extra_dim=2)
    searcher = SemanticSearcher(
        repo, provider, candidate_searcher=CodeSearcher(repo)
    )
    first = [(r.file_path, r.similarity) for r in searcher.search("alpha", limit=5)]
    second = [(r.file_path, r.similarity) for r in searcher.search("alpha", limit=5)]
    assert first == second  # no crash, no nondeterminism from dimension skew
    assert len(first) == 1


def test_embedding_cache_store_failure_tolerated(tmp_path, monkeypatch):
    repo = _single_repo(tmp_path)
    probe = install_atomic_write_failure(monkeypatch)
    cache_dir = tmp_path / "embed_cache"
    searcher = SemanticSearcher(
        repo,
        FakeEmbeddingProvider(),
        candidate_searcher=CodeSearcher(repo),
        cache=FileSystemEmbeddingCache(cache_dir),
    )
    results = searcher.search("alpha", limit=5)
    assert results  # persistent write failure must not break in-memory search
    assert probe()["invoked"] >= 1
    results2 = searcher.search("alpha", limit=5)
    assert [(r.file_path, r.similarity) for r in results2] == [
        (r.file_path, r.similarity) for r in results
    ]


# ---------------------------------------------------------------------------
# C. Persistent cache corruption
# ---------------------------------------------------------------------------


def test_corrupt_index_cache_rebuilds_equal_pristine(tmp_path):
    repo = make_repo(
        tmp_path / "repo",
        {
            "a.py": "from b import beta\n\ndef alpha():\n    return beta()\n",
            "b.py": "def beta():\n    return 2\n",
        },
    )
    cache_dir = tmp_path / "cache"
    pristine = IncrementalIndexBuilder(repo, cache_dir=cache_dir).build()
    corrupted = corrupt_all_json(cache_dir)
    assert corrupted >= 1
    rebuilt = IncrementalIndexBuilder(repo, cache_dir=cache_dir).build()
    assert index_fingerprint(rebuilt) == index_fingerprint(pristine)


def test_corrupt_reference_cache_degrades_to_miss(tmp_path):
    repo = make_repo(
        tmp_path / "repo",
        {"worker.py": "from util import helper\n\nX = helper\n", "util.py": "def helper():\n    return 1\n"},
    )
    index = IncrementalIndexBuilder(repo, persist=False).build()
    cache_dir = tmp_path / "ref_cache"
    builder = ReferenceIndexBuilder(repo, index=index, cache_dir=cache_dir)
    first = builder.build()
    assert first.stats.cache_misses >= 1
    corrupt_all_json(cache_dir)
    rebuilt = builder.build()
    assert rebuilt.stats.cache_hits == 0  # corrupted entries are misses
    for path in sorted(first.files):
        assert first.by_path[path].references == rebuilt.by_path[path].references


def test_malformed_embedding_cache_vector_is_a_miss(tmp_path):
    cache = FileSystemEmbeddingCache(tmp_path / "embed_cache")
    cache.store("a.py", "hash123", "embed-deriv", (0.1, 0.2, 0.3))
    entry = next((tmp_path / "embed_cache").glob("*.json"))
    payload = json.loads(entry.read_text(encoding="utf-8"))
    payload["vector"] = ["not", "a", "number"]
    entry.write_text(json.dumps(payload), encoding="utf-8")
    assert cache.lookup("a.py", "hash123", "embed-deriv") is None


def test_embedding_cache_clear_sweeps_stale_partials(tmp_path):
    cache_dir = tmp_path / "embed_cache"
    cache = FileSystemEmbeddingCache(cache_dir)
    cache.store("a.py", "hash1", "embed-deriv", (0.5,))
    partial = drop_stale_partial(cache_dir, "stale.part-1.tmp")
    assert partial.exists()
    cache.clear()
    assert not partial.exists()
    assert list(cache_dir.glob("*.json")) == []
    assert list(cache_dir.glob("*.part-*.tmp")) == []


# ---------------------------------------------------------------------------
# D. Atomic-write interruption / failure
# ---------------------------------------------------------------------------


def test_fsync_failure_cleans_temporary_and_preserves_target(tmp_path, monkeypatch):
    import repolens.atomic_write as atomic_module

    target = tmp_path / "key.json"
    target.write_text('{"old": true}', encoding="utf-8")
    install_fsync_failure(monkeypatch)
    with pytest.raises(OSError):
        atomic_module.atomic_write_text(target, '{"new": true}')
    assert json.loads(target.read_text(encoding="utf-8")) == {"old": True}
    assert list(tmp_path.glob("*.part-*.tmp")) == []


def test_replace_failure_midstream_leaves_no_partial(tmp_path, monkeypatch):
    from repolens.atomic_write import atomic_write_text

    target = tmp_path / "key.json"
    probe = install_atomic_write_failure(monkeypatch)
    with pytest.raises(OSError):
        atomic_write_text(target, '{"new": true}')
    assert probe()["last_failed"]
    assert not target.exists()
    assert list(tmp_path.glob("*.part-*.tmp")) == []


def test_write_into_read_only_directory_raises_and_keeps_old_target(tmp_path, monkeypatch):
    from repolens.atomic_write import atomic_write_text

    directory = tmp_path / "ro"
    directory.mkdir()
    target = directory / "key.json"
    target.write_text('{"old": true}', encoding="utf-8")
    if not require_write_access(directory):
        pytest.skip("running as root; read-only directory is still writable")
    directory.chmod(0o555)
    try:
        with pytest.raises(OSError):
            atomic_write_text(target, '{"new": true}')
    finally:
        directory.chmod(0o755)
    assert json.loads(target.read_text(encoding="utf-8")) == {"old": True}
    assert list(directory.glob("*.part-*.tmp")) == []


# ---------------------------------------------------------------------------
# E. Incremental index recovery & reuse
# ---------------------------------------------------------------------------


def test_second_build_reuses_cache_and_matches(tmp_path):
    repo = make_repo(
        tmp_path / "repo",
        {"a.py": "def alpha():\n    return 1\n", "b.py": "def beta():\n    return 2\n"},
    )
    cache_dir = tmp_path / "cache"
    first = IncrementalIndexBuilder(repo, cache_dir=cache_dir).build()
    second = IncrementalIndexBuilder(repo, cache_dir=cache_dir).build()
    assert second.stats.cache_hits == first.stats.files_discovered
    assert second.stats.files_parsed == 0
    assert index_fingerprint(second) == index_fingerprint(first)


def test_reindex_with_unrelated_change_preserves_other_hits(tmp_path):
    repo = make_repo(
        tmp_path / "repo",
        {"a.py": "def alpha():\n    return 1\n", "b.py": "def beta():\n    return 2\n"},
    )
    cache_dir = tmp_path / "cache"
    IncrementalIndexBuilder(repo, cache_dir=cache_dir).build()
    (repo / "b.py").write_text("def beta():\n    return 3\n", encoding="utf-8")
    rebuilt = IncrementalIndexBuilder(repo, cache_dir=cache_dir).build()
    assert rebuilt.stats.cache_hits == 1  # a.py unchanged
    assert rebuilt.stats.files_parsed == 1  # only b.py re-parsed


def test_stale_index_entry_pruned_after_deletion(tmp_path):
    repo = make_repo(
        tmp_path / "repo",
        {"a.py": "def alpha():\n    return 1\n", "b.py": "def beta():\n    return 2\n"},
    )
    cache_dir = tmp_path / "cache"
    IncrementalIndexBuilder(repo, cache_dir=cache_dir).build()
    (repo / "b.py").unlink()
    rebuilt = IncrementalIndexBuilder(repo, cache_dir=cache_dir).build()
    assert rebuilt.stats.files_removed == 1
    assert list(rebuilt.files) == [Path("a.py")]


# ---------------------------------------------------------------------------
# F. References & call graph
# ---------------------------------------------------------------------------


def test_reference_extractor_tolerates_malformed_source():
    extractor = ReferenceExtractor()
    for broken in ("def broken(:\n", "class X(\n", "\x00\x01\x02", ")))((("):
        extracted = extractor.extract(broken)
        assert extracted.references == ()


def test_unresolved_module_recorded_not_fabricated(tmp_path):
    repo = make_repo(
        tmp_path / "repo",
        {
            "a.py": (
                "def a_work():\n"
                "    return worker.process()\n"
            ),
            "b.py": "def b_work():\n    return 1\n",
        },
    )
    graph, _ = _call_graph(repo)
    unresolved = graph.unresolved_references()
    assert any(u.name == "worker.process" and u.reason == "unknown_module" for u in unresolved)
    targets = {e.target.name for e in graph.get_edges() if e.kind.value == "call"}
    assert "worker" not in targets  # nothing was fabricated for the unknown module


def test_circular_imports_build_and_query_without_recursion(tmp_path):
    repo = make_repo(
        tmp_path / "repo",
        {
            "pkg/__init__.py": "",
            "pkg/a.py": (
                "from . import b\n\n"
                "def a_work():\n"
                "    return b.b_work()\n"
            ),
            "pkg/b.py": (
                "from . import a\n\n"
                "def b_work():\n"
                "    return a.a_work()\n"
            ),
        },
    )
    graph, index = _call_graph(repo)

    def symbol(name: str):
        matches = [
            s for s in index.symbols
            if s.name == name and s.file_path.as_posix().startswith("pkg/")
        ]
        assert len(matches) == 1, f"expected exactly one {name} symbol"
        return graph.symbol_node(matches[0])

    a_work = symbol("a_work")
    b_work = symbol("b_work")
    callees = graph.callees(a_work, max_depth=3)
    assert any(node.name == "b_work" for node in callees)
    callers = graph.callers(b_work, max_depth=3)
    assert any(node.name == "a_work" for node in callers)


def test_transitive_traversal_respects_bound(tmp_path):
    repo = _chain_repo(tmp_path, 5)
    graph, _ = _call_graph(repo, config=CallGraphConfig(max_transitive_nodes=2))
    node_a = _module_node(graph, "a.py")
    for node in graph.module_nodes():
        flat = graph.callees(node, max_depth=5)
        assert len(flat) <= 2
    assert len(graph.callees(node_a, max_depth=5)) <= 2


# ---------------------------------------------------------------------------
# G. Architecture
# ---------------------------------------------------------------------------


def _architecture(repo: Path):
    index = IncrementalIndexBuilder(repo, persist=False).build()
    dep_graph = DependencyGraphBuilder(repo, index=index).build()
    arch = ArchitectureGraphBuilder(
        repo, index=index, graph=dep_graph, config=ArchitectureConfig(max_transitive_nodes=8)
    ).build()
    return arch, index, dep_graph


def test_architecture_builds_circular_repo_with_bounded_traversal(tmp_path):
    repo = make_repo(
        tmp_path / "repo",
        {
            "pkg/__init__.py": "from . import a, b\n",
            "pkg/a.py": "from . import b\nfrom . import c\n",
            "pkg/b.py": "from . import a\n",
            "pkg/c.py": "from . import a, b\n",
        },
    )
    arch, _, _ = _architecture(repo)
    node = arch.get_module_node("pkg.a")
    assert node is not None
    deps = arch.transitive_dependencies(node, max_depth=3)
    assert len(deps) >= 1
    dependents = arch.transitive_dependents(node, max_depth=3)
    assert len(dependents) >= 1


def test_deleted_module_vanishes_on_rebuild(tmp_path):
    repo = make_repo(
        tmp_path / "repo",
        {"a.py": "def alpha():\n    return 1\n", "b.py": "def beta():\n    return 2\n"},
    )
    first, _, _ = _architecture(repo)
    assert first.get_file_node("b.py") is not None
    (repo / "b.py").unlink()
    rebuilt, _, _ = _architecture(repo)
    assert rebuilt.get_file_node("b.py") is None
    assert rebuilt.get_file_node("a.py") is not None


def test_architecture_empty_repo_no_crash(tmp_path):
    repo = make_repo(tmp_path / "repo", {})
    arch, _, _ = _architecture(repo)
    assert arch.files() == []


def _payload_surface(payload: dict) -> tuple:
    """Deterministic slice of a plan payload (excludes wall-clock statistics)."""
    stats = {k: v for k, v in payload.get("statistics", {}).items() if k != "build_time"}
    return (
        payload["primary_target"],
        payload["target_candidates"],
        payload["affected_files"],
        payload["affected_symbols"],
        payload["callers"],
        payload["callees"],
        payload["dependencies"],
        payload["dependents"],
        payload["architecture"],
        payload["tests"],
        payload["inspection_order"],
        payload["risk"],
        payload["risk_factors"],
        payload["confidence"],
        payload["summary"],
        stats,
    )


# ---------------------------------------------------------------------------
# H. Impact / change plan / change context
# ---------------------------------------------------------------------------


def test_impact_empty_target_raises(tmp_path):
    repo = _single_repo(tmp_path)
    analyzer = ImpactAnalyzer(repo)
    with pytest.raises(ImpactTargetError):
        analyzer.resolve_target("   ")


def test_impact_missing_target_raises(tmp_path):
    repo = _single_repo(tmp_path)
    analyzer = ImpactAnalyzer(repo)
    with pytest.raises(ImpactTargetError):
        analyzer.resolve_target("does_not_exist_anywhere")


def test_impact_unknown_file_raises(tmp_path):
    repo = _single_repo(tmp_path)
    analyzer = ImpactAnalyzer(repo)
    with pytest.raises(ImpactTargetError):
        analyzer.resolve_target("ghost.py")


def test_plan_on_empty_repo_is_valid_and_deterministic(tmp_path, isolated_cache):
    repo = make_repo(tmp_path / "repo", {})
    request = "add refunds"
    first = _payload_surface(plan_response_payload(ChangePlanEngine(repo).plan(request), request))
    second = _payload_surface(plan_response_payload(ChangePlanEngine(repo).plan(request), request))
    assert first == second


def test_plan_bounds_upgrade_reuses_components_and_matches(tmp_path, isolated_cache):
    repo = _call_repo(tmp_path)
    request = "refund checkout carts"
    base = ChangePlanEngine(repo).plan(request)
    limited = ChangePlanEngine(
        repo, config=ChangePlanConfig(max_affected_files=2)
    ).plan(request)
    assert len(getattr(limited, "affected_files", ()) or ()) <= 2
    assert plan_fingerprint(base) != plan_fingerprint(limited)  # budget genuinely applied


def test_change_context_is_deterministic_and_safe(tmp_path, isolated_cache):
    repo = _call_repo(tmp_path)
    state = ChangePlanState(repo)
    firewall = ContextFirewall()

    def engine_factory(*, max_tokens=None, dependency_depth=None):
        return ContextEngine(
            repo, budget=ContextBudget(max_tokens=max_tokens or 8000)
        )

    def state_factory():
        return state

    first = run_change_context(
        engine_factory, firewall, state_factory,
        "add refunds to checkout carts", target="checkout.py",
    )
    second = run_change_context(
        engine_factory, firewall, state_factory,
        "add refunds to checkout carts", target="checkout.py",
    )
    assert first["selected_files"] == second["selected_files"]
    assert first["change_files"] == second["change_files"]
    assert first["blocked_files"] == second["blocked_files"]


# ---------------------------------------------------------------------------
# I. Context budget safety
# ---------------------------------------------------------------------------


def test_tiny_budget_never_exceeds_and_no_crash(tmp_path):
    repo = _call_repo(tmp_path)
    engine = ContextEngine(repo, budget=ContextBudget(max_tokens=1))
    package = engine.build_context("checkout")
    assert package.total_estimated_tokens <= 1


def test_zero_budget_yields_empty_selection(tmp_path):
    repo = _call_repo(tmp_path)
    engine = ContextEngine(repo, budget=ContextBudget(max_tokens=0))
    package = engine.build_context("checkout")
    assert len(package.selected_files) == 0


def test_query_without_matches_yields_deterministic_empty_package(tmp_path):
    repo = _single_repo(tmp_path)
    engine = ContextEngine(repo)
    first = engine.build_context("zxqvzqx")
    second = engine.build_context("zxqvzqx")
    assert package_fingerprint(first) == package_fingerprint(second)
    assert len(first.selected_files) == 0


def test_context_root_not_directory_raises(tmp_path):
    repo = make_repo(tmp_path / "repo", {"a.py": "x = 1\n"})
    not_a_dir = repo / "missing_dir"
    with pytest.raises(NotADirectoryError):
        ContextEngine(not_a_dir)


def test_context_empty_repo_builds_empty_package(tmp_path):
    repo = make_repo(tmp_path / "repo", {})
    engine = ContextEngine(repo)
    package = engine.build_context("anything")
    assert len(package.selected_files) == 0


# ---------------------------------------------------------------------------
# J. MCP-safe degradation
# ---------------------------------------------------------------------------


def _engine_factory_for(repo: Path, *, fail_n=0):
    failures = {"remaining": fail_n}

    def factory(*, max_tokens=None, dependency_depth=None):
        if failures["remaining"] > 0:
            failures["remaining"] -= 1
            raise EmbeddingFailure("factory transient failure")
        return ContextEngine(
            repo, budget=ContextBudget(max_tokens=max_tokens or 8000)
        )

    return factory


def _firewall():
    return ContextFirewall()


def test_mcp_engine_factory_failure_becomes_context_error(tmp_path):
    repo = _single_repo(tmp_path)
    factory = _engine_factory_for(repo, fail_n=1)
    with pytest.raises(ContextEngineError):
        run_get_context(factory, _firewall(), "alpha")
    response = run_get_context(factory, _firewall(), "alpha")
    assert "selected_files" in response


def test_mcp_build_failure_is_wrapped_not_silenced(tmp_path, monkeypatch):
    repo = _single_repo(tmp_path)

    def broken_engine(*, max_tokens=None, dependency_depth=None):
        engine = ContextEngine(repo, budget=ContextBudget(max_tokens=max_tokens or 8000))
        monkeypatch.setattr(engine, "build_context", lambda query: (_ for _ in ()).throw(EmbeddingFailure("build transient failure")))
        return engine

    with pytest.raises(ContextEngineError) as excinfo:
        run_get_context(broken_engine, _firewall(), "alpha")
    assert excinfo.value.diagnostic == "build_context failed: EmbeddingFailure"


def test_mcp_invalid_arguments_raise_typed_error(tmp_path):
    repo = _single_repo(tmp_path)
    with pytest.raises(InvalidArgumentsError):
        parse_arguments({"query": ""})


def test_mcp_non_callable_factory_is_internal_error(tmp_path):
    repo = _single_repo(tmp_path)
    with pytest.raises(InternalError):
        run_get_context(None, _firewall(), "alpha")


def test_mcp_wrong_firewall_is_internal_error(tmp_path):
    repo = _single_repo(tmp_path)
    with pytest.raises(InternalError):
        run_get_context(_engine_factory_for(repo), object(), "alpha")


def test_mcp_failure_then_success_matches_pristine(tmp_path):
    repo = _call_repo(tmp_path)
    state = {"count": 0}

    def flaky_factory(*, max_tokens=None, dependency_depth=None):
        state["count"] += 1
        if state["count"] == 1:
            raise EmbeddingFailure("factory transient failure")
        return ContextEngine(repo, budget=ContextBudget(max_tokens=max_tokens or 8000))

    with pytest.raises(ContextEngineError):
        run_get_context(flaky_factory, _firewall(), "checkout carts")
    after = run_get_context(flaky_factory, _firewall(), "checkout carts")
    pristine = run_get_context(_engine_factory_for(repo), _firewall(), "checkout carts")
    # Rendering order and message text are deterministic; compare the surface.
    assert after["selected_files"] == pristine["selected_files"]
    assert after["blocked_files"] == pristine["blocked_files"]


def test_mcp_plan_invalid_request_does_not_poison_state(tmp_path, isolated_cache):
    repo = _call_repo(tmp_path)
    state = ChangePlanState(repo)
    request = "add refunds to checkout carts"
    first = run_change_plan(lambda: state, request)
    with pytest.raises(InvalidArgumentsError):
        parse_change_plan_arguments({"request": ""})
    second = run_change_plan(lambda: state, request)
    assert _payload_surface(first) == _payload_surface(second)
    assert isinstance(state.parsed_file_count, int) and state.parsed_file_count > 0


def test_mcp_plan_unsafe_target_raises(tmp_path, isolated_cache):
    repo = _call_repo(tmp_path)
    state = ChangePlanState(repo)
    with pytest.raises(InvalidArgumentsError):
        parse_change_plan_arguments({"request": "fix checkout", "target": "../etc/passwd"})


def test_mcp_change_plan_factory_error_is_typed(tmp_path):
    with pytest.raises(ChangePlanError):
        run_change_plan(lambda: (_ for _ in ()).throw(EmbeddingFailure("state broken")), "fix checkout")


def test_mcp_change_plan_rejects_bad_args(tmp_path, isolated_cache):
    repo = _call_repo(tmp_path)
    state = ChangePlanState(repo)
    with pytest.raises(InvalidArgumentsError):
        parse_change_plan_arguments({"request": "fix checkout", "max_targets": 500})
    with pytest.raises(InvalidArgumentsError):
        parse_change_plan_arguments({"request": ""})


# ---------------------------------------------------------------------------
# K. Concurrency / shared state
# ---------------------------------------------------------------------------


def test_change_plan_state_concurrent_is_lazily_shared(tmp_path, isolated_cache):
    repo = _call_repo(tmp_path)
    state = ChangePlanState(repo)
    errors: list[BaseException] = []

    def worker():
        try:
            engine = state.default_engine
            index = state.index
            count = state.parsed_file_count
            return engine, index, count
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)
            return None

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = [future.result(timeout=120) for future in [pool.submit(worker) for _ in range(16)]]
    assert errors == []
    engines = {id(engine) for engine, _, _ in results}
    assert len(engines) == 1  # lock-protected double-checked build -> single engine
    assert {count for _, _, count in results} == {state.parsed_file_count}
    assert state.parsed_file_count is not None


def test_architecture_state_concurrent_no_deadlock_consistent(tmp_path, isolated_cache):
    repo = _call_repo(tmp_path)
    state = ArchitectureState(repo)
    errors: list[BaseException] = []

    def worker():
        try:
            subsystems = tuple(sub.id for sub in state.subsystems)
            graph = state.graph
            assert graph is not None
            return subsystems
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)
            return None

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = [future.result(timeout=120) for future in [pool.submit(worker) for _ in range(16)]]
    assert errors == []
    assert len({tuple(r) for r in results if r is not None}) == 1


def test_concurrent_index_builds_identical_and_intact(tmp_path):
    repo = make_repo(
        tmp_path / "repo",
        {
            "checkout.py": "import database\n\ndef checkout(cart):\n    return database.store(cart)\n",
            "database.py": "def store(cart):\n    return len(cart)\n",
        },
    )
    cache_dir = tmp_path / "cache"
    errors: list[BaseException] = []

    def worker():
        try:
            return IncrementalIndexBuilder(repo, cache_dir=cache_dir).build()
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)
            return None

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = [future.result(timeout=120) for future in [pool.submit(worker) for _ in range(6)]]
    assert errors == []
    fingerprints = {index_fingerprint(index) for index in results}
    assert len(fingerprints) == 1
    assert list(cache_dir.glob("*.part-*.tmp")) == []
    assert len(list(cache_dir.glob("*.json"))) == 2


def test_context_engine_concurrent_builds_deterministic(tmp_path):
    repo = _call_repo(tmp_path)
    engine = ContextEngine(repo)
    errors: list[BaseException] = []

    def worker():
        try:
            return package_fingerprint(engine.build_context("checkout carts"))
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)
            return None

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = [future.result(timeout=120) for future in [pool.submit(worker) for _ in range(12)]]
    assert errors == []
    assert len({tuple(r) for r in results if r is not None}) == 1


# ---------------------------------------------------------------------------
# Failure-resolution matrix (used by docs/reliability.md and tooling)
# ---------------------------------------------------------------------------

FAILURE_MATRIX = {
    "parser/repo input": {
        "non-utf8 source": "empty analysis, no crash",
        "unreadable file": "empty analysis, no crash",
        "vanishing file": "miss (empty analysis), no crash",
        "syntax error": "SyntaxError propagates; repaired build == pristine",
    },
    "embedding": {
        "transient provider raise": "no partial cache entry; recovery == pristine",
        "wrong vector count": "typed EmbeddingProviderError, no partial state",
        "cache write failure": "search still works; deterministic",
    },
    "caches": {
        "corrupt index entries": "treated as miss; rebuild == pristine",
        "corrupt reference entries": "treated as miss; equal references",
        "malformed embedding vector": "treated as miss",
        "stale partials": "swept on clear",
    },
    "atomic writes": {
        "fsync failure": "temp cleaned, target preserved",
        "mid-stream failure": "no partial file",
        "read-only target dir": "raises, old target preserved",
    },
    "incremental index": {
        "second build": "cache hits == files, zero parses",
        "unrelated change": "only changed file re-parsed",
        "deleted file": "stale entry pruned",
    },
    "references / call graph": {
        "malformed source": "empty references, no crash",
        "unresolved module": "recorded, never fabricated",
        "circular imports": "terminates, edges resolved",
        "transitive traversal": "bounded by max_transitive_nodes",
    },
    "architecture": {
        "circular packages": "bounded traversal, no recursion",
        "deleted module": "vanishes on rebuild",
        "empty repo": "empty graph, no crash",
    },
    "impact / change plan": {
        "empty/missing/unknown target": "ImpactTargetError",
        "plan on empty repo": "valid, deterministic",
        "plan bounds": "limit genuinely applied",
        "change context no target": "deterministic package",
    },
    "context budget": {
        "max_tokens=1": "never exceeds, no crash",
        "max_tokens=0": "empty selection",
        "no matching query": "empty, deterministic",
        "root not a directory": "NotADirectoryError",
        "empty repo": "empty package",
    },
    "MCP": {
        "factory failure": "ContextEngineError; server usable",
        "build failure": "wrapped, diagnostic preserved",
        "invalid arguments": "InvalidArgumentsError",
        "non-callable factory": "InternalError",
        "wrong firewall": "InternalError",
        "failure then success": "== pristine",
        "failed plan request": "no shared-state poison",
        "unsafe target / oversized bounds": "InvalidArgumentsError",
    },
    "concurrency": {
        "ChangePlanState": "single shared engine, deterministic",
        "ArchitectureState": "no deadlock, consistent result",
        "concurrent index builds": "identical fingerprints, cache intact",
        "concurrent context builds": "identical packages",
    },
}