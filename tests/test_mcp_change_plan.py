"""Offline tests for the M24.2 change-plan MCP tools.

Covers ``change_plan`` and ``change_context``: argument validation, structured
responses, boundedness, determinism, error safety, lazy and shared construction
of the change-plan state, per-call bounds reusing shared components, incremental
modification, deleted files, the cache-disabled path, and protocol-level
registration. Fully offline and deterministic.
"""

from __future__ import annotations

import asyncio
import json
import shutil
from pathlib import Path

import pytest

from repolens.change_context import ChangeContextOptions, explain_change_context, plan_to_change_candidates
from repolens.change_plan import ChangePlanConfig
from repolens.context import ContextBudget, ContextEngine, ContextFirewall, DependencyExpansionConfig
from repolens.mcp import build_mcp_server
from repolens.mcp.change_plan_tool import (
    ChangePlanState,
    parse_change_context_arguments,
    parse_change_plan_arguments,
    run_change_context,
    run_change_plan,
    validate_change_target,
    validate_target_kind,
)
from repolens.mcp.errors import (
    ChangePlanError,
    InternalError,
    InvalidArgumentsError,
)
from repolens.search import CodeSearcher

ROOT = Path(__file__).parent / "fixtures" / "change_plan_repository"

#: Python file count of the fixture (cold, fresh index).
FIXTURE_FILE_COUNT = 29


def _state(root: Path) -> ChangePlanState:
    return ChangePlanState(root)


def _factory(state: ChangePlanState):
    def factory(**kwargs) -> ChangePlanState:
        return state

    return factory


@pytest.fixture()
def state() -> ChangePlanState:
    return _state(ROOT)


def _copy_fixture(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    shutil.copytree(ROOT, repo)
    return repo


def _context_engine(root: Path, max_tokens=None) -> ContextEngine:
    return ContextEngine(
        root,
        searcher=CodeSearcher(root),
        budget=(
            ContextBudget(max_tokens=max_tokens)
            if max_tokens is not None
            else ContextBudget(max_tokens=10**9)
        ),
        dependency=DependencyExpansionConfig(depth=1),
    )


def _engine_factory(root: Path):
    def factory(max_tokens=None, dependency_depth=None, **kwargs):
        engine = _context_engine(root, max_tokens)
        if dependency_depth is not None:
            engine = ContextEngine(
                root,
                searcher=CodeSearcher(root),
                budget=(
                    ContextBudget(max_tokens=max_tokens)
                    if max_tokens is not None
                    else ContextBudget(max_tokens=10**9)
                ),
                dependency=DependencyExpansionConfig(depth=dependency_depth),
            )
        return engine

    return factory


# ---------------------------------------------------------------------------
# Argument validation
# ---------------------------------------------------------------------------


def test_change_target_accepts_file_module_package_symbol() -> None:
    assert validate_change_target("app/services/checkout.py") == "app/services/checkout.py"
    assert validate_change_target("app.services.checkout") == "app.services.checkout"
    assert validate_change_target("checkout") == "checkout"
    assert validate_change_target(" Order ") == "Order"


def test_change_target_rejects_unsafe_paths() -> None:
    for bad in (
        "../escape.py",
        "a/../../b.py",
        "/absolute/path.py",
        "a\\b",
        "C:\\win",
        "a:b",
        "",
    ):
        with pytest.raises(InvalidArgumentsError):
            validate_change_target(bad)


def test_change_target_rejects_non_string() -> None:
    with pytest.raises(InvalidArgumentsError):
        validate_change_target(42)


def test_target_kind_validation() -> None:
    for kind in ("file", "module", "package", "symbol"):
        assert validate_target_kind(kind) == kind
    assert validate_target_kind(None) is None
    with pytest.raises(InvalidArgumentsError):
        validate_target_kind("class")
    with pytest.raises(InvalidArgumentsError):
        validate_target_kind(1)


def test_parse_change_plan_arguments() -> None:
    parsed = parse_change_plan_arguments(
        {"request": "fix checkout", "target": "app/services/checkout.py",
         "max_files": 10, "max_depth": 2}
    )
    assert parsed["request"] == "fix checkout"
    assert parsed["target"] == "app/services/checkout.py"
    assert parsed["max_files"] == 10
    assert parsed["max_depth"] == 2


def test_parse_change_plan_rejects_bad_input() -> None:
    with pytest.raises(InvalidArgumentsError):
        parse_change_plan_arguments(None)
    with pytest.raises(InvalidArgumentsError):
        parse_change_plan_arguments({})
    with pytest.raises(InvalidArgumentsError):
        parse_change_plan_arguments({"target": "x"})  # request missing
    with pytest.raises(InvalidArgumentsError):
        parse_change_plan_arguments({"request": "  "})
    with pytest.raises(InvalidArgumentsError):
        parse_change_plan_arguments({"request": "x", "max_files": 0})
    with pytest.raises(InvalidArgumentsError):
        parse_change_plan_arguments({"request": "x", "max_files": 10000})
    with pytest.raises(InvalidArgumentsError):
        parse_change_plan_arguments({"request": "x", "max_depth": 9})
    with pytest.raises(InvalidArgumentsError):
        parse_change_plan_arguments({"request": "x", "bogus": 1})


def test_parse_change_context_arguments() -> None:
    parsed = parse_change_context_arguments(
        {"request": "fix checkout", "include_tests": False}
    )
    assert parsed["request"] == "fix checkout"
    assert parsed["include_tests"] is False
    assert parsed["include_callers"] is True


def test_parse_change_context_rejects_bad_input() -> None:
    with pytest.raises(InvalidArgumentsError):
        parse_change_context_arguments({"request": "x", "include_tests": "no"})
    with pytest.raises(InvalidArgumentsError):
        parse_change_context_arguments({"request": "x", "max_tokens": 0})
    with pytest.raises(InvalidArgumentsError):
        parse_change_context_arguments({"request": "x", "target_kind": "class"})
    with pytest.raises(InvalidArgumentsError):
        parse_change_context_arguments({"request": "x", "nope": 1})


# ---------------------------------------------------------------------------
# change_plan tool
# ---------------------------------------------------------------------------


def test_change_plan_resolves_explicit_file_target(state) -> None:
    result = run_change_plan(
        _factory(state), "change", target="app/services/checkout.py"
    )
    assert result["status"] == "ok"
    assert result["primary_target"]["kind"] == "file"
    assert result["primary_target"]["confidence"] == "confirmed"
    assert result["primary_target"]["target"] == "app/services/checkout.py"
    assert result["affected_files"]
    assert result["statistics"]["affected_file_count"] >= 1


def test_change_plan_natural_language(state) -> None:
    result = run_change_plan(_factory(state), "make checkout reject empty carts")
    assert result["status"] == "ok"
    assert result["primary_target"] is not None
    assert result["affected_files"]
    assert result["inspection_order"]
    assert result["risk"] in ("low", "medium", "high")
    assert result["summary"]
    assert result["analysis"]["action"] in (None, "modify", "fix")


def test_change_plan_deterministic(state) -> None:
    a = run_change_plan(_factory(state), "Add refund support to checkout")
    b = run_change_plan(_factory(state), "Add refund support to checkout")
    assert a == b


def test_module_target_defining_file_is_a_change_candidate(state) -> None:
    result = run_change_plan(
        _factory(state), "change", target="app.services.checkout"
    )
    assert result["primary_target"]["kind"] == "module"
    assert result["primary_target"]["target"] == "app.services.checkout"
    assert "app/services/checkout.py" in {
        i["path"] for i in result["affected_files"]
    }
    primary_item = result["inspection_order"][0]
    assert primary_item["path"] == "app/services/checkout.py"
    assert primary_item["category"] == "primary_target"

    plan = state.default_engine.plan(
        "change", target="app.services.checkout"
    )
    candidates = plan_to_change_candidates(plan, root=ROOT)
    primary = [
        c for c in candidates if c.change_category == "primary_target"
    ]
    assert [str(c.path) for c in primary] == ["app/services/checkout.py"]


def test_change_plan_payload_separates_wall_clock_diagnostics(state) -> None:
    result = run_change_plan(_factory(state), "Add refund support to checkout")
    assert result["deterministic"] is True
    # Wall-clock time is classified as non-deterministic diagnostics, never
    # part of the deterministic statistics.
    assert "build_time" not in result["statistics"]
    assert result["statistics"]["affected_file_count"] >= 0
    assert isinstance(result["diagnostics"]["build_time"], float)
    assert "note" in result["diagnostics"]


def test_change_plan_substantive_results_deterministic_across_engines(
    tmp_path: Path,
) -> None:
    results = []
    for i in range(2):
        repo = tmp_path / f"repo{i}"
        shutil.copytree(ROOT, repo)
        state = _state(repo)
        results.append(
            run_change_plan(_factory(state), "Add refund support to checkout")
        )
    assert results[0]["statistics"] == results[1]["statistics"]
    assert results[0]["affected_files"] == results[1]["affected_files"]
    assert results[0]["inspection_order"] == results[1]["inspection_order"]
    a = {k: v for k, v in results[0].items() if k != "diagnostics"}
    b = {k: v for k, v in results[1].items() if k != "diagnostics"}
    assert a == b


def test_change_plan_serializes_to_json(state) -> None:
    result = run_change_plan(
        _factory(state), "Add refund support to checkout"
    )
    json.dumps(result)  # must be JSON-serializable


def test_change_plan_bounds(state) -> None:
    result = run_change_plan(
        _factory(state),
        "Add refund support to checkout",
        max_files=3,
        max_tests=1,
        max_targets=2,
        max_depth=1,
    )
    assert result["statistics"]["affected_file_count"] <= 3
    assert len(result["tests"]) <= 1
    assert len(result["target_candidates"]) <= 2
    assert result["target_kind"] is None


def test_change_plan_invalid_target_is_safe_error(state) -> None:
    with pytest.raises(InvalidArgumentsError):
        parse_change_plan_arguments({"request": "change", "target": "../../etc/passwd"})


def test_change_plan_deleted_file(tmp_path: Path) -> None:
    repo = _copy_fixture(tmp_path)
    (repo / "app" / "services" / "checkout.py").unlink()
    deleted = _state(repo)
    result = run_change_plan(
        _factory(deleted), "change", target="app/services/checkout.py"
    )
    assert result["status"] == "ok"
    # No crash; the plan degrades to an unresolved-mode summary.


def test_change_plan_empty_repository(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    empty = _state(repo)
    result = run_change_plan(_factory(empty), "Add refund support")
    assert result["status"] == "ok"
    assert result["primary_target"] is None
    assert result["affected_files"] == []


# ---------------------------------------------------------------------------
# change_context tool
# ---------------------------------------------------------------------------


@pytest.fixture()
def change_state() -> ChangePlanState:
    return _state(ROOT)


def test_change_context_builds_safe_package(change_state) -> None:
    firewall = ContextFirewall()
    result = run_change_context(
        _engine_factory(ROOT),
        firewall,
        _factory(change_state),
        "make checkout reject empty carts",
        target="app/services/checkout.py",
    )
    assert result["status"] == "ok"
    assert result["selected_files"]
    assert result["total_estimated_tokens"] >= 0
    assert result["change_plan"]["primary_target"]["target"] == "app/services/checkout.py"
    assert result["rendered_safe_context"]
    assert result["explanations"]
    assert all({"path", "in_plan", "selected", "survived_ranking",
                "survived_budget", "budget_status", "firewall"} <= set(e)
               for e in result["explanations"])


def test_change_context_change_files_carry_change_metadata(change_state) -> None:
    firewall = ContextFirewall()
    result = run_change_context(
        _engine_factory(ROOT),
        firewall,
        _factory(change_state),
        "make checkout reject empty carts",
        target="app/services/checkout.py",
    )
    plan = change_state.default_engine.plan(
        "make checkout reject empty carts",
        target="app/services/checkout.py",
    )
    plan_paths = {i.path for i in plan.inspection_order}
    assert result["change_files"]
    # Every change_file introduced by the plan corresponds to a plan item and
    # carries change-plan metadata in its explanation.
    for path in result["change_files"]:
        assert path in plan_paths
    assert any(
        e["change_plan_source"] and e["category"] == "primary_target"
        for e in result["explanations"]
    )


def test_change_context_budget_is_respected(change_state) -> None:
    firewall = ContextFirewall()
    capped = run_change_context(
        _engine_factory(ROOT),
        firewall,
        _factory(change_state),
        "make checkout reject empty carts",
        target="app/services/checkout.py",
        max_tokens=150,
    )
    unlimited = run_change_context(
        _engine_factory(ROOT),
        firewall,
        _factory(change_state),
        "make checkout reject empty carts",
        target="app/services/checkout.py",
    )
    assert 0 <= capped["total_estimated_tokens"] <= 150
    assert capped["total_estimated_tokens"] <= unlimited["total_estimated_tokens"]


def test_change_context_include_tests_filters_change_files(change_state) -> None:
    firewall = ContextFirewall()
    with_tests = run_change_context(
        _engine_factory(ROOT),
        firewall,
        _factory(change_state),
        "make checkout reject empty carts",
        target="app/services/checkout.py",
        include_tests=True,
    )
    without = run_change_context(
        _engine_factory(ROOT),
        firewall,
        _factory(change_state),
        "make checkout reject empty carts",
        target="app/services/checkout.py",
        include_tests=False,
    )
    # Test-only files are introduced when tests are included...
    assert "tests/test_checkout.py" in with_tests["change_files"]
    assert "tests/test_payments.py" in with_tests["change_files"]
    # ...but vanish when tests are filtered out.
    assert "tests/test_checkout.py" not in without["change_files"]
    assert "tests/test_payments.py" not in without["change_files"]
    # A test file that is also a dependency (tests/test_refunds.py imports the
    # refunds service) legitimately remains through the dependency filter.
    assert "tests/test_refunds.py" in without["change_files"]


def test_change_context_include_callers_filters_change_files(change_state) -> None:
    firewall = ContextFirewall()
    with_callers = run_change_context(
        _engine_factory(ROOT),
        firewall,
        _factory(change_state),
        "change",
        target="app/services/checkout.py",
        include_callers=True,
    )
    without = run_change_context(
        _engine_factory(ROOT),
        firewall,
        _factory(change_state),
        "change",
        target="app/services/checkout.py",
        include_callers=False,
    )
    caller_paths = {e["path"] for e in with_callers["explanations"]
                    if e["change_plan_source"]
                    and e["category"] == "direct_caller"}
    assert caller_paths  # non-trivial fixture: callers are introduced
    assert not any(
        e["change_plan_source"] and e["category"] == "direct_caller"
        for e in without["explanations"]
    )


def test_change_context_deterministic(change_state) -> None:
    firewall = ContextFirewall()
    a = run_change_context(
        _engine_factory(ROOT),
        firewall,
        _factory(change_state),
        "Add refund support to checkout",
    )
    b = run_change_context(
        _engine_factory(ROOT),
        firewall,
        _factory(change_state),
        "Add refund support to checkout",
    )
    assert a == b


def test_change_context_surfaces_module_target_defining_file(change_state) -> None:
    firewall = ContextFirewall()
    result = run_change_context(
        _engine_factory(ROOT),
        firewall,
        _factory(change_state),
        "change",
        target="app.services.checkout",
    )
    assert "app/services/checkout.py" in {
        c["path"] for c in result["selected_files"]
    }
    primary_expl = [
        e for e in result["explanations"]
        if e["category"] == "primary_target"
    ]
    assert primary_expl
    assert primary_expl[0]["path"] == "app/services/checkout.py"


def test_change_context_firewall_blocks_sensitive_file() -> None:
    import tempfile

    from repolens.context.firewall import FirewallConfig

    with tempfile.TemporaryDirectory() as d:
        root = _copy_fixture(Path(d))
        engine = _context_engine(root)
        state = _state(root)
        firewall = ContextFirewall(
            FirewallConfig(blocked_filenames=frozenset({"settings.py"}))
        )
        result = run_change_context(
            _engine_factory(root),
            firewall,
            _factory(state),
            "change",
            target="app/config/routes.py",
        )
    blocked = {str(c["path"]) for c in result["blocked_files"]}
    assert "app/config/settings.py" in blocked
    # Blocked files never appear among the selected/safe files.
    assert all(
        e["path"] != "app/config/settings.py" for e in result["explanations"]
    )


def test_change_context_invalid_engine_factory_is_safe(change_state) -> None:
    firewall = ContextFirewall()
    with pytest.raises(InternalError):
        run_change_context(
            None, firewall, _factory(change_state), "change"
        )


def test_change_context_bad_firewall_is_safe(change_state) -> None:
    with pytest.raises(InternalError):
        run_change_context(
            _engine_factory(ROOT), object(), _factory(change_state), "change"
        )


# ---------------------------------------------------------------------------
# Shared lazy state / warm reuse / no duplicate work
# ---------------------------------------------------------------------------


def test_change_state_builds_lazily_and_reuses(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("REPOLENS_CACHE_DISABLED", raising=False)
    monkeypatch.delenv("REPOLENS_CACHE_DIR", raising=False)
    monkeypatch.setattr(
        "repolens.incremental_index.home_cache_base",
        lambda: tmp_path / "cache",
    )
    repo = _copy_fixture(tmp_path)
    fresh = ChangePlanState(repo)
    assert fresh._index is None  # not built yet
    assert fresh._default_engine is None
    e1 = fresh.default_engine  # builds once
    assert fresh._index is not None
    assert fresh.default_engine is e1  # reused
    assert fresh.parsed_file_count == FIXTURE_FILE_COUNT


def test_no_duplicate_parses_across_tool_calls(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("REPOLENS_CACHE_DISABLED", raising=False)
    monkeypatch.delenv("REPOLENS_CACHE_DIR", raising=False)
    monkeypatch.setattr(
        "repolens.incremental_index.home_cache_base",
        lambda: tmp_path / "cache",
    )
    repo = _copy_fixture(tmp_path)
    state = _state(repo)
    fac = _factory(state)
    assert state.parsed_file_count is None  # untouched yet
    run_change_plan(fac, "Add refund support to checkout")
    parsed_after = state.parsed_file_count
    assert parsed_after == FIXTURE_FILE_COUNT
    run_change_plan(fac, "make checkout reject empty carts")
    run_change_context(
        _engine_factory(repo),
        ContextFirewall(),
        fac,
        "change",
        target="app/services/checkout.py",
    )
    assert state.parsed_file_count == parsed_after  # no re-parsing


def test_warm_persistent_index_no_reparse(tmp_path: Path, monkeypatch) -> None:
    from repolens.incremental_index import home_cache_base

    monkeypatch.delenv("REPOLENS_CACHE_DISABLED", raising=False)
    monkeypatch.delenv("REPOLENS_CACHE_DIR", raising=False)
    repo = _copy_fixture(tmp_path)
    cache_base = tmp_path / "cache"
    monkeypatch.setattr(
        "repolens.incremental_index.home_cache_base",
        lambda: cache_base,
    )
    first = _state(repo)
    run_change_plan(_factory(first), "Add refund support to checkout")
    assert first.parsed_file_count == FIXTURE_FILE_COUNT

    second = _state(repo)
    run_change_plan(_factory(second), "Add refund support to checkout")
    assert second.parsed_file_count == 0  # warm index reused from the cache


def test_cache_disabled_mode_still_works(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("REPOLENS_CACHE_DISABLED", "1")
    repo = _copy_fixture(tmp_path)
    state = _state(repo)
    fac = _factory(state)
    result = run_change_plan(fac, "Add refund support to checkout")
    assert result["status"] == "ok"
    assert state.parsed_file_count == FIXTURE_FILE_COUNT
    # A fresh state re-parses when built (no persistent cache).
    fresh = _state(repo)
    assert fresh.parsed_file_count is None
    assert fresh.default_engine is not None


def test_per_call_limits_share_shared_components(state) -> None:
    shared = _factory(state)
    run_change_plan(shared, "change", target="app/services/checkout.py")
    base = state.default_engine
    cfg = ChangePlanConfig(max_affected_files=3)
    per_call = state.engine(cfg)
    assert per_call is not base
    assert per_call._dep_graph is base._dep_graph
    assert per_call._symbol_index is base._symbol_index
    assert per_call._call_graph is base._call_graph
    assert per_call._arch_graph is base._arch_graph
    assert per_call._impact_analyzer is base._impact_analyzer
    # Same non-default bounds reuse the same engine instance.
    assert state.engine(cfg) is per_call


def test_incremental_modification(tmp_path: Path) -> None:
    repo = _copy_fixture(tmp_path)
    before = _state(repo)
    fac = _factory(before)
    before_result = run_change_plan(fac, "change", target="app/config/routes.py")
    assert "app/config/settings.py" in {
        i["path"] for i in before_result["inspection_order"]
    }

    (repo / "app" / "config" / "routes.py").write_text(
        "from app.config.settings import DATABASE_URL\n",
        encoding="utf-8",
    )
    after = _state(repo)
    after_result = run_change_plan(
        _factory(after), "change", target="app/config/routes.py"
    )
    assert after_result["status"] == "ok"
    assert "app/config/routes.py" in {
        i["path"] for i in after_result["inspection_order"]
    }


# ---------------------------------------------------------------------------
# Error safety
# ---------------------------------------------------------------------------


def test_factory_failure_raises_safe_change_plan_error(state) -> None:
    def boom(**kwargs):
        raise RuntimeError("boom")

    with pytest.raises(ChangePlanError):
        run_change_plan(boom, "change")


def test_non_callable_factory_raises_internal(state) -> None:
    with pytest.raises(InternalError):
        run_change_plan(None, "change")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Protocol-level registration
# ---------------------------------------------------------------------------


@pytest.fixture()
def cp_state() -> ChangePlanState:
    return _state(ROOT)


@pytest.fixture()
def change_server(cp_state):
    return build_mcp_server(
        _engine_factory(ROOT),
        ContextFirewall(),
        change_plan_factory=_factory(cp_state),
    )


def test_change_plan_tools_registered(change_server) -> None:
    tools = asyncio.run(change_server.list_tools())
    names = {t.name for t in tools}
    assert "get_context" in names  # core tool unchanged
    assert {"change_plan", "change_context"} <= names


def test_change_plan_tools_unavailable_without_factory() -> None:
    server = build_mcp_server(_engine_factory(ROOT), ContextFirewall())
    tools = asyncio.run(server.list_tools())
    names = {t.name for t in tools}
    assert names == {"get_context"}


def test_protocol_change_plan(change_server) -> None:
    async def run():
        result = await change_server.call_tool(
            "change_plan",
            {"request": "Add refund support to checkout"},
        )
        assert result.is_error is False
        data = json.loads(result.content[0].text)
        assert data["status"] == "ok"
        assert data["primary_target"] is not None
        return result

    asyncio.run(run())


def test_protocol_change_context(change_server) -> None:
    async def run():
        result = await change_server.call_tool(
            "change_context",
            {"request": "Add refund support to checkout"},
        )
        assert result.is_error is False
        data = json.loads(result.content[0].text)
        assert data["status"] == "ok"
        assert data["selected_files"]
        assert data["rendered_safe_context"]
        return result

    asyncio.run(run())


def test_protocol_invalid_argument_is_safe_error(change_server) -> None:
    async def run():
        result = await change_server.call_tool(
            "change_plan", {"request": "x", "max_files": -1}
        )
        assert result.is_error is True
        text = result.content[0].text
        assert "max_files" in text.lower()
        assert "traceback" not in text.lower()
        return result

    asyncio.run(run())


def test_protocol_unsafe_target_is_safe_error(change_server) -> None:
    async def run():
        result = await change_server.call_tool(
            "change_plan", {"request": "x", "target": "../../etc/passwd"}
        )
        assert result.is_error is True
        assert "traceback" not in result.content[0].text.lower()
        return result

    asyncio.run(run())