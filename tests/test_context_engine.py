"""Offline tests for the dependency-aware context engine (Milestone 12).

These tests never touch the network or download an embedding model. They
exercise the token estimator, configurations, dependency expansion, ranking,
budgeting, packaging, serialization, and rendering using in-memory __tmp__
repositories (and the existing synthetic fixture).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from repolens.context import (
    CandidateRole,
    ContextBudget,
    ContextCandidate,
    ContextEngine,
    ContextPackage,
    DependencyExpansionConfig,
    ExcludedCandidate,
    RetrievalConfig,
    estimate_tokens,
    render_context,
)
from repolens.context.budget import select_within_budget
from repolens.context.expansion import expand_dependencies
from repolens.context.ranking import rank_candidates
from repolens.graph import DependencyGraphBuilder
from repolens.search import CodeSearcher

SYNTHETIC_REPO = (
    Path(__file__).resolve().parent / "fixtures" / "synthetic_repository"
)


def write_file(tmp_path: Path, relative: str, source: str) -> None:
    file_path = tmp_path / relative
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(source, encoding="utf-8")


def build_lexical_engine(root, **kwargs) -> ContextEngine:
    return ContextEngine(root, searcher=CodeSearcher(root), **kwargs)


# ---------------------------------------------------------------------------
# Token estimator
# ---------------------------------------------------------------------------


def test_estimate_tokens_is_positive_and_monotonic() -> None:
    assert estimate_tokens("") == 1
    assert estimate_tokens("abcd") == 1
    assert estimate_tokens("abcdefgh") == 2
    assert estimate_tokens("a" * 100) > estimate_tokens("a" * 10)


def test_estimate_tokens_is_deterministic() -> None:
    text = "def hello(): return 42  # a comment\n"
    assert estimate_tokens(text) == estimate_tokens(text)


def test_estimate_tokens_uses_four_chars_per_token() -> None:
    assert estimate_tokens("x" * 8) == 2
    assert estimate_tokens("x" * 9) == 3


# ---------------------------------------------------------------------------
# Configurations
# ---------------------------------------------------------------------------


def test_retrieval_config_defaults() -> None:
    cfg = RetrievalConfig()
    assert cfg.strategy == "rrf"
    assert cfg.limit == 8
    assert cfg.lexical_weight == 0.5
    assert cfg.semantic_weight == 0.5
    assert cfg.rrf_k == 60


def test_retrieval_config_lexical_builds_code_searcher(tmp_path: Path) -> None:
    write_file(tmp_path, "a.py", "def f():\n    pass\n")
    searcher = RetrievalConfig(strategy="lexical").build_searcher(tmp_path)
    assert isinstance(searcher, CodeSearcher)


def test_dependency_expansion_config_defaults() -> None:
    cfg = DependencyExpansionConfig()
    assert cfg.depth == 1
    assert cfg.include_dependencies is True
    assert cfg.include_dependents is True


def test_context_budget_defaults() -> None:
    assert ContextBudget().max_tokens == 8000
    assert ContextBudget(max_tokens=None).max_tokens is None


# ---------------------------------------------------------------------------
# Dependency expansion
# ---------------------------------------------------------------------------


@pytest.fixture()
def chain_repo(tmp_path: Path) -> Path:
    write_file(tmp_path, "main.py", "from a import x\nfrom b import y\n")
    write_file(tmp_path, "a.py", "from c import z\n")
    write_file(tmp_path, "b.py", "value = 1\n")
    write_file(tmp_path, "c.py", "z = 1\n")
    return tmp_path


def graph_of(root: Path):
    return DependencyGraphBuilder(root).build()


def test_expansion_depth_zero_adds_nothing(chain_repo) -> None:
    graph = graph_of(chain_repo)
    nodes = expand_dependencies(
        graph, [Path("main.py")], DependencyExpansionConfig(depth=0)
    )
    assert nodes == []


def test_expansion_depth_one_adds_direct_dependencies(chain_repo) -> None:
    graph = graph_of(chain_repo)
    nodes = expand_dependencies(
        graph, [Path("main.py")], DependencyExpansionConfig(depth=1)
    )
    by_path = {n.path: n for n in nodes}
    assert set(by_path) == {Path("a.py"), Path("b.py")}
    assert all(n.role is CandidateRole.DEPENDENCY for n in nodes)
    assert all(n.distance == 1 for n in nodes)


def test_expansion_depth_two_adds_second_hop(chain_repo) -> None:
    graph = graph_of(chain_repo)
    nodes = expand_dependencies(
        graph, [Path("main.py")], DependencyExpansionConfig(depth=2)
    )
    by_path = {n.path: n for n in nodes}
    assert set(by_path) == {Path("a.py"), Path("b.py"), Path("c.py")}
    assert by_path[Path("c.py")].distance == 2


def test_expansion_finds_dependents(tmp_path: Path) -> None:
    write_file(tmp_path, "shared.py", "value = 1\n")
    write_file(tmp_path, "main.py", "from shared import value\n")
    graph = graph_of(tmp_path)
    nodes = expand_dependencies(
        graph, [Path("shared.py")], DependencyExpansionConfig(depth=1)
    )
    assert [(n.path, n.role, n.distance) for n in nodes] == [
        (Path("main.py"), CandidateRole.DEPENDENT, 1)
    ]


def test_expansion_can_disable_dependencies(chain_repo) -> None:
    graph = graph_of(chain_repo)
    cfg = DependencyExpansionConfig(depth=1, include_dependencies=False)
    nodes = expand_dependencies(graph, [Path("main.py")], cfg)
    assert nodes == []


def test_expansion_can_disable_dependents(tmp_path: Path) -> None:
    write_file(tmp_path, "shared.py", "value = 1\n")
    write_file(tmp_path, "main.py", "from shared import value\n")
    graph = graph_of(tmp_path)
    cfg = DependencyExpansionConfig(depth=1, include_dependents=False)
    nodes = expand_dependencies(graph, [Path("shared.py")], cfg)
    assert nodes == []


def test_expansion_prevents_cycles(tmp_path: Path) -> None:
    write_file(tmp_path, "pkg/__init__.py", "")
    write_file(tmp_path, "pkg/a.py", "from pkg.b import y\n")
    write_file(tmp_path, "pkg/b.py", "from pkg.a import x\n")
    graph = graph_of(tmp_path)
    nodes = expand_dependencies(
        graph, [Path("pkg/a.py")], DependencyExpansionConfig(depth=3)
    )
    paths = [n.path for n in nodes]
    assert len(paths) == len(set(paths)), "duplicate/revisited files"
    assert Path("pkg/b.py") in paths


def test_expansion_ignores_seeds_that_are_not_reached(tmp_path: Path) -> None:
    write_file(tmp_path, "main.py", "from a import x\n")
    write_file(tmp_path, "a.py", "value = 1\n")
    write_file(tmp_path, "unrelated.py", "value = 2\n")
    graph = graph_of(tmp_path)
    nodes = expand_dependencies(
        graph, [Path("main.py"), Path("unrelated.py")],
        DependencyExpansionConfig(depth=2),
    )
    assert [n.path for n in nodes] == [Path("a.py")]


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------


def _candidate(path, role, *, retrieval_rank=None, retrieval_score=None,
               graph_distance=None, source="") -> ContextCandidate:
    return ContextCandidate(
        path=Path(path),
        source=source,
        role=role,
        estimated_tokens=estimate_tokens(source),
        selection_reason="test",
        retrieval_rank=retrieval_rank,
        retrieval_score=retrieval_score,
        graph_distance=graph_distance,
    )


def test_primary_ranks_above_dependency() -> None:
    candidates = [
        _candidate("dep.py", CandidateRole.DEPENDENCY, graph_distance=1),
        _candidate("main.py", CandidateRole.PRIMARY, retrieval_rank=1),
    ]
    ranked = [c.path.as_posix() for c in rank_candidates(candidates)]
    assert ranked == ["main.py", "dep.py"]


def test_primary_ranked_by_retrieval_rank_then_score() -> None:
    candidates = [
        _candidate("a.py", CandidateRole.PRIMARY, retrieval_rank=2, retrieval_score=9.0),
        _candidate("c.py", CandidateRole.PRIMARY, retrieval_rank=1, retrieval_score=3.0),
        _candidate("b.py", CandidateRole.PRIMARY, retrieval_rank=3, retrieval_score=1.0),
    ]
    ranked = [c.path.as_posix() for c in rank_candidates(candidates)]
    assert ranked == ["c.py", "a.py", "b.py"]


def test_dependency_ranked_by_distance_then_role_then_path() -> None:
    candidates = [
        _candidate("util/dep.py", CandidateRole.DEPENDENCY, graph_distance=2),
        _candidate("util/caller.py", CandidateRole.DEPENDENT, graph_distance=1),
        _candidate("util/other_dep.py", CandidateRole.DEPENDENCY, graph_distance=1),
    ]
    ranked = [c.path.as_posix() for c in rank_candidates(candidates)]
    # distance 1 first; at equal distance dependents (callers) before dependencies; then path.
    assert ranked == ["util/caller.py", "util/other_dep.py", "util/dep.py"]


def test_dependency_primary_mixed_keeps_primaries_first_by_rank() -> None:
    candidates = [
        _candidate("dep.py", CandidateRole.DEPENDENCY, graph_distance=1),
        _candidate("m2.py", CandidateRole.PRIMARY, retrieval_rank=9),
        _candidate("m1.py", CandidateRole.PRIMARY, retrieval_rank=1),
    ]
    ranked = [c.path.as_posix() for c in rank_candidates(candidates)]
    assert ranked[:2] == ["m1.py", "m2.py"]
    assert ranked[2] == "dep.py"


# ---------------------------------------------------------------------------
# Budgeting
# ---------------------------------------------------------------------------


def _token_candidate(path, tokens) -> ContextCandidate:
    return ContextCandidate(
        path=Path(path),
        source="x" * (tokens * 4),
        role=CandidateRole.PRIMARY,
        estimated_tokens=tokens,
        selection_reason="test",
    )


def test_budget_never_exceeds_and_keeps_rank_order() -> None:
    ranked = [
        _token_candidate("a.py", 30),
        _token_candidate("b.py", 40),
        _token_candidate("c.py", 30),
    ]
    selected, excluded = select_within_budget(ranked, ContextBudget(max_tokens=100))
    assert [c.path.as_posix() for c in selected] == ["a.py", "b.py", "c.py"]
    assert sum(c.estimated_tokens for c in selected) <= 100
    assert excluded == []


def test_budget_skips_oversized_but_keeps_later_smaller() -> None:
    ranked = [
        _token_candidate("a.py", 60),
        _token_candidate("b.py", 80),
        _token_candidate("c.py", 30),
    ]
    selected, excluded = select_within_budget(ranked, ContextBudget(max_tokens=100))
    # a fits (60); b (80) would exceed remaining 40 -> skipped; c (30) fits.
    assert [c.path.as_posix() for c in selected] == ["a.py", "c.py"]
    assert sum(c.estimated_tokens for c in selected) == 90
    assert any(e.path == Path("b.py") and e.reason == "over_budget" for e in excluded)


def test_single_candidate_larger_than_budget_is_excluded() -> None:
    ranked = [_token_candidate("huge.py", 200)]
    selected, excluded = select_within_budget(ranked, ContextBudget(max_tokens=100))
    assert selected == []
    assert excluded == [
        ExcludedCandidate(path=Path("huge.py"), estimated_tokens=200, reason="exceeds_total_budget")
    ]


def test_zero_budget_selects_nothing() -> None:
    ranked = [_token_candidate("a.py", 5), _token_candidate("b.py", 5)]
    selected, excluded = select_within_budget(ranked, ContextBudget(max_tokens=0))
    assert selected == []
    assert len(excluded) == 2


def test_unlimited_budget_selects_everything() -> None:
    ranked = [_token_candidate("a.py", 1000), _token_candidate("b.py", 2000)]
    selected, excluded = select_within_budget(ranked, ContextBudget(max_tokens=None))
    assert [c.path.as_posix() for c in selected] == ["a.py", "b.py"]
    assert excluded == []


# ---------------------------------------------------------------------------
# Engine end-to-end
# ---------------------------------------------------------------------------


def test_engine_builds_context_package() -> None:
    engine = build_lexical_engine(
        SYNTHETIC_REPO,
        budget=ContextBudget(max_tokens=10000),
        dependency=DependencyExpansionConfig(depth=1),
    )
    pkg = engine.build_context("invoice calculation")
    assert isinstance(pkg, ContextPackage)
    assert pkg.query == "invoice calculation"
    assert pkg.selected_files
    assert all(isinstance(c, ContextCandidate) for c in pkg.selected_files)


def test_engine_primary_and_dependency_candidates() -> None:
    root = SYNTHETIC_REPO
    engine = build_lexical_engine(
        root,
        budget=ContextBudget(max_tokens=10**9),
        dependency=DependencyExpansionConfig(depth=1),
    )
    pkg = engine.build_context("application bootstrap")
    primaries = {c.path for c in pkg.primary_candidates}
    deps = {c.path for c in pkg.dependency_candidates}
    assert Path("main.py") in primaries
    # main.py imports api/routes.py, auth/login.py, database/pool.py.
    assert Path("api/routes.py") in deps
    assert Path("auth/login.py") in deps
    # A file can only appear once overall.
    combined = primaries | deps
    assert len(combined) == len(pkg.primary_candidates) + len(pkg.dependency_candidates)


def test_engine_rank_primaries_first() -> None:
    root = SYNTHETIC_REPO
    engine = build_lexical_engine(
        root,
        budget=ContextBudget(max_tokens=10**9),
        dependency=DependencyExpansionConfig(depth=1),
    )
    pkg = engine.build_context("application bootstrap")
    role_order = {
        c.path: c.role
        for c in [*pkg.primary_candidates, *pkg.dependency_candidates]
    }
    selected_roles = [c.role for c in pkg.selected_files]
    last_primary = max(
        (i for i, r in enumerate(selected_roles) if r is CandidateRole.PRIMARY),
        default=-1,
    )
    first_dep = next(
        (i for i, r in enumerate(selected_roles) if r is not CandidateRole.PRIMARY),
        None,
    )
    if first_dep is not None:
        assert last_primary < first_dep


def test_engine_budget_is_respected() -> None:
    engine = build_lexical_engine(
        SYNTHETIC_REPO,
        budget=ContextBudget(max_tokens=60),
        dependency=DependencyExpansionConfig(depth=1),
    )
    pkg = engine.build_context("application bootstrap")
    assert pkg.total_estimated_tokens <= 60
    assert len(pkg.selected_files) < len(pkg.primary_candidates)


def test_engine_empty_repository() -> None:
    from repolens.context import ContextEngine

    # empty tmp repo (no python files) with lexical searcher
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        (root / "notes.txt").write_text("hi\n")
        searcher = CodeSearcher(root)
        engine = ContextEngine(
            root, searcher=searcher, budget=ContextBudget(max_tokens=100)
        )
        pkg = engine.build_context("anything")
    assert pkg.selected_files == ()
    assert pkg.total_estimated_tokens == 0


def test_engine_invalid_root_raises(tmp_path: Path) -> None:
    from repolens.context import ContextEngine

    with pytest.raises(NotADirectoryError):
        ContextEngine(tmp_path / "missing", searcher=CodeSearcher(tmp_path))


# ---------------------------------------------------------------------------
# Serialization & rendering
# ---------------------------------------------------------------------------


def test_package_serializes_to_json() -> None:
    engine = build_lexical_engine(
        SYNTHETIC_REPO,
        budget=ContextBudget(max_tokens=10000),
        dependency=DependencyExpansionConfig(depth=1),
    )
    pkg = engine.build_context("invoice calculation")
    data = pkg.to_dict()
    assert data["query"] == "invoice calculation"
    assert data["total_estimated_tokens"] == pkg.total_estimated_tokens
    assert isinstance(data["selected_files"], list)
    assert all("path" in c and "role" in c for c in data["selected_files"])

    text = pkg.to_json()
    assert json.loads(text)["query"] == "invoice calculation"


def test_rendering_is_deterministic_and_contains_blocks() -> None:
    engine = build_lexical_engine(
        SYNTHETIC_REPO,
        budget=ContextBudget(max_tokens=10000),
        dependency=DependencyExpansionConfig(depth=1),
    )
    pkg = engine.build_context("invoice calculation")
    text = render_context(pkg)
    assert "# RepoLens Context" in text
    assert "Query:" in text
    assert "## Primary Context" in text
    assert "```python" in text
    assert text == render_context(pkg)


# ---------------------------------------------------------------------------
# Architecture-aware retrieval (Milestone 23.2)
# ---------------------------------------------------------------------------

ARCH_REPO = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "architecture_retrieval_repository"
)


def build_arch_engine(arch=None, **kwargs):
    return ContextEngine(
        ARCH_REPO,
        searcher=CodeSearcher(ARCH_REPO),
        architecture=arch,
        **kwargs,
    )


def test_engine_architecture_enabled_adds_arch_candidates() -> None:
    from repolens.architecture_retrieval import ArchitectureRetrievalConfig
    from repolens.context.candidate import INCLUSION_ARCHITECTURE

    engine = build_arch_engine(
        arch=ArchitectureRetrievalConfig(),
        budget=ContextBudget(max_tokens=10**9),
    )
    pkg = engine.build_context("how does the checkout flow work")
    arch = [c for c in pkg.selected_files if c.inclusion_reason == INCLUSION_ARCHITECTURE]
    assert arch
    assert all(c.architecture_rank is not None for c in arch)
    for c in arch:
        meta = c.architecture_metadata
        assert {"node_kind", "node_id", "package", "subsystem", "direction"} <= set(meta)
    # Architecture-only additions: lexically undiscoverable dependent + neighbor
    # modules the plain retrieval would never surface.
    billing = next(c for c in arch if c.path == Path("tests/test_billing.py"))
    assert billing.architecture_metadata["subsystem"] == "tests"
    assert billing.architecture_rank == 1
    services_init = next(
        c for c in arch if c.path == Path("store/services/__init__.py")
    )
    assert services_init.architecture_metadata["subsystem"] == "store"
    # The direct checkout module merged into the lexical primary but is present.
    assert Path("store/services/checkout.py") in {c.path for c in pkg.selected_files}


def test_engine_architecture_disabled_matches_default_behavior() -> None:
    from repolens.architecture_retrieval import ArchitectureRetrievalConfig
    from repolens.context.candidate import INCLUSION_ARCHITECTURE

    default = build_arch_engine(
        budget=ContextBudget(max_tokens=10**9)
    ).build_context("invoice checkout")
    disabled = build_arch_engine(
        arch=ArchitectureRetrievalConfig(enabled=False),
        budget=ContextBudget(max_tokens=10**9),
    ).build_context("invoice checkout")

    assert {c.path for c in default.selected_files} == {
        c.path for c in disabled.selected_files
    }
    assert all(
        c.inclusion_reason != INCLUSION_ARCHITECTURE
        for c in disabled.selected_files
    )
    assert not any(c.architecture_metadata for c in disabled.selected_files)


def test_engine_architecture_budget_is_respected() -> None:
    from repolens.architecture_retrieval import ArchitectureRetrievalConfig

    engine = build_arch_engine(
        arch=ArchitectureRetrievalConfig(),
        budget=ContextBudget(max_tokens=120),
    )
    pkg = engine.build_context("checkout flow")
    assert pkg.total_estimated_tokens <= 120


def test_engine_public_architecture_query_surface() -> None:
    from repolens.architecture_retrieval import ArchitectureRetrievalConfig

    engine = build_arch_engine(arch=ArchitectureRetrievalConfig())
    candidates = engine.architecture_candidates("checkout")
    assert candidates
    assert any(c.node.id == "store.services.checkout" for c in candidates)

    subsystems = engine.discover_subsystems()
    assert {s.id for s in subsystems} == {
        "admin", "app", "billing", "store", "tests", "web",
    }

    explanation = engine.explain_architecture_match("checkout", "store.services.checkout")
    assert explanation is not None
    assert explanation["reason"] == "architecture: direct module match"


def test_engine_render_includes_architecture_line() -> None:
    from repolens.architecture_retrieval import ArchitectureRetrievalConfig

    engine = build_arch_engine(
        arch=ArchitectureRetrievalConfig(),
        budget=ContextBudget(max_tokens=10**9),
    )
    pkg = engine.build_context("checkout flow")
    text = render_context(pkg)
    assert "Architecture:" in text


# ---------------------------------------------------------------------------
# Change-aware engine (Milestone 24.2)
# ---------------------------------------------------------------------------

CHANGE_REPO = Path(__file__).parent / "fixtures" / "change_plan_repository"


def _change_engine(root=CHANGE_REPO, **kwargs) -> ContextEngine:
    kwargs.setdefault("budget", ContextBudget(max_tokens=10**9))
    return ContextEngine(
        root,
        searcher=CodeSearcher(root),
        dependency=DependencyExpansionConfig(depth=1),
        **kwargs,
    )


def test_change_aware_build_keeps_standard_path_identical() -> None:
    engine = _change_engine()
    query = "how does checkout work"
    via_dispatcher = engine.build_context(query)
    via_standard = engine._build_context_standard(query)
    assert via_dispatcher.to_dict() == via_standard.to_dict()


def test_change_aware_build_adds_change_candidates() -> None:
    engine = _change_engine()
    pkg = engine.build_context(
        "make checkout reject empty carts",
        change_request="make checkout reject empty carts",
        change_target="app/services/checkout.py",
    )
    assert pkg.change_candidates
    change_paths = {c.path for c in pkg.change_candidates}
    assert Path("app/services/checkout.py") in change_paths
    for c in pkg.change_candidates:
        assert c.change_category is not None
        assert c.change_confidence is not None
        assert c.change_priority is not None
        assert c.inclusion_reason == "change_plan"
    # Every surviving change file is part of the selected result.
    assert change_paths <= {c.path for c in pkg.selected_files}


def test_change_candidates_team_never_outrank_direct_matches() -> None:
    engine = _change_engine()
    pkg = engine.build_context(
        "make checkout reject empty carts",
        change_request="make checkout reject empty carts",
        change_target="app/services/checkout.py",
    )
    change_paths = {c.path for c in pkg.change_candidates}
    # Retrieval primaries always come first; change-plan files are a final tier.
    role_order = [c.role for c in pkg.selected_files]
    last_primary = max(
        (i for i, r in enumerate(role_order) if r is CandidateRole.PRIMARY),
        default=-1,
    )
    first_change = next(
        (i for i, c in enumerate(pkg.selected_files) if c.path in change_paths),
        None,
    )
    # The change plan may also revisit a primary (same file), which is fine;
    # a *change-only* file must never outrank the primaries.
    change_only_first = next(
        (
            i
            for i, c in enumerate(pkg.selected_files)
            if c.path in change_paths and c.role is not CandidateRole.PRIMARY
        ),
        None,
    )
    if change_only_first is not None:
        assert last_primary < change_only_first


def test_change_aware_dedupe_keeps_primary_role() -> None:
    engine = _change_engine()
    pkg = engine.build_context(
        "make checkout reject empty carts",
        change_request="make checkout reject empty carts",
        change_target="app/services/checkout.py",
    )
    # A file that is both a retrieval primary and a change target appears once.
    by_path = {c.path: c for c in pkg.selected_files}
    checkout = by_path[Path("app/services/checkout.py")]
    assert checkout.role is CandidateRole.PRIMARY


def test_change_aware_change_candidates_survive_budget() -> None:
    engine = _change_engine(budget=ContextBudget(max_tokens=2000))
    pkg = engine.build_context(
        "make checkout reject empty carts",
        change_request="make checkout reject empty carts",
        change_target="app/services/checkout.py",
    )
    assert 0 <= pkg.total_estimated_tokens <= 2000
    assert pkg.change_candidates


def test_change_aware_is_deterministic() -> None:
    engine = _change_engine()
    query = "make checkout reject empty carts"
    a = engine.build_context(
        query,
        change_request=query,
        change_target="app/services/checkout.py",
    )
    b = engine.build_context(
        query,
        change_request=query,
        change_target="app/services/checkout.py",
    )
    assert a.to_dict() == b.to_dict()


def test_change_plan_engine_method_reuses_context_components() -> None:
    engine = _change_engine()
    shared = engine.change_plan_engine()
    assert shared._dep_graph is engine._graph
    assert shared._symbol_index is engine._symbol_index
    # A second read returns the same lazily-built engine.
    assert engine.change_plan_engine() is shared


def test_change_context_empty_repository(tmp_path: Path) -> None:
    from repolens.mcp.change_plan_tool import ChangePlanState

    empty = tmp_path / "empty"
    empty.mkdir()
    engine = _change_engine(root=empty)
    state = ChangePlanState(empty)
    plan = state.default_engine.plan("add a feature")
    pkg = engine.build_context(
        "add a feature",
        change_request="add a feature",
        change_plan=plan,
    )
    assert pkg.change_candidates == ()
