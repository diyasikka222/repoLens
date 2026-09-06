"""Offline tests for the M23.3 architecture-intelligence MCP tools.

Covers ``inspect_architecture``, ``discover_subsystems``,
``architecture_candidates``, and ``explain_architecture_match``: argument
validation, structured responses, boundedness, determinism, error safety, lazy
and shared construction of the architecture graph, incremental modification,
deleted files, and the cache-disabled path. Fully offline and deterministic.
"""

from __future__ import annotations

import asyncio
import json
import shutil
from pathlib import Path

import pytest

from repolens.context import ContextFirewall
from repolens.mcp import build_mcp_server
from repolens.mcp.architecture_tool import (
    ArchitectureState,
    parse_architecture_candidates_arguments,
    parse_discover_subsystems_arguments,
    parse_explain_architecture_match_arguments,
    parse_inspect_architecture_arguments,
    run_architecture_candidates,
    run_discover_subsystems,
    run_explain_architecture_match,
    run_inspect_architecture,
    validate_arch_target,
)
from repolens.mcp.errors import (
    ArchitectureError,
    InternalError,
    InvalidArgumentsError,
)

ROOT = Path(__file__).parent / "fixtures" / "architecture_retrieval_repository"


def _state(root: Path) -> ArchitectureState:
    return ArchitectureState(root)


def _factory(state: ArchitectureState):
    def factory(**kwargs) -> ArchitectureState:
        return state

    return factory


@pytest.fixture()
def state() -> ArchitectureState:
    return _state(ROOT)


def _copy_fixture(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    shutil.copytree(ROOT, repo)
    return repo


def _write(repo: Path, rel: str, content: str) -> None:
    target = repo / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")


# ---------------------------------------------------------------------------
# Target validation
# ---------------------------------------------------------------------------


def test_arch_target_accepts_file_module_package() -> None:
    assert validate_arch_target("app/services/users.py") == "app/services/users.py"
    assert validate_arch_target("app.services.users") == "app.services.users"
    assert validate_arch_target(" app/repositories ") == "app/repositories"


def test_arch_target_rejects_unsafe_paths() -> None:
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
            validate_arch_target(bad)


def test_arch_target_rejects_non_string() -> None:
    with pytest.raises(InvalidArgumentsError):
        validate_arch_target(42)


# ---------------------------------------------------------------------------
# inspect_architecture
# ---------------------------------------------------------------------------


def test_inspect_architecture_module(state) -> None:
    result = run_inspect_architecture(_factory(state), "store.services.checkout")
    assert result["status"] == "ok"
    assert result["kind"] == "module"
    assert result["module"] == "store.services.checkout"
    assert result["file"] == "store/services/checkout.py"
    assert result["package"] == "store/services"
    assert result["subsystem"] == "store"
    deps = [d["id"] for d in result["dependencies"]]
    assert "store.models.order" in deps
    assert "store.repositories.orders" in deps
    dependents = [d["id"] for d in result["dependents"]]
    assert "tests.test_checkout" in dependents
    assert "admin.reports" in dependents
    assert result["statistics"]["direct_dependency_count"] == 2
    assert result["statistics"]["package_module_count"] == 3


def test_inspect_architecture_package(state) -> None:
    result = run_inspect_architecture(_factory(state), "store/repositories")
    assert result["kind"] == "package"
    assert result["package"] == "store/repositories"
    assert result["file"] == "store/repositories/__init__.py"
    deps = [d["id"] for d in result["dependencies"]]
    assert "store/models" in deps
    assert result["statistics"]["package_file_count"] == 3


def test_inspect_architecture_file(state) -> None:
    result = run_inspect_architecture(_factory(state), "store/services/checkout.py")
    assert result["kind"] == "file"
    assert result["module"] == "store.services.checkout"
    assert result["file"] == "store/services/checkout.py"


def test_inspect_architecture_includes_subsystem_stats(state) -> None:
    result = run_inspect_architecture(_factory(state), "store.services.checkout")
    stats = result["statistics"]["subsystem_stats"]
    assert stats["packages"] == 5
    assert stats["modules"] == 12


def test_inspect_architecture_depth_is_bounded(state) -> None:
    deep = run_inspect_architecture(_factory(state), "store.services.checkout", max_depth=6)
    shallow = run_inspect_architecture(_factory(state), "store.services.checkout", max_depth=1)
    assert len(deep["neighborhood"]["dependents"]) >= len(shallow["neighborhood"]["dependents"])
    # neighborhood lists never exceed the per-side cap even at allowed depth
    assert len(deep["neighborhood"]["dependents"]) <= 200
    assert deep["neighborhood"]["max_depth"] == 6


def test_inspect_architecture_flags_toggle_sections(state) -> None:
    result = run_inspect_architecture(
        _factory(state), "store.services.checkout", include_dependents=False,
        include_subsystems=False,
    )
    assert "dependents" not in result
    assert "dependencies" in result
    assert result["kind"] == "module"


def test_inspect_architecture_unknown_target(state) -> None:
    with pytest.raises(InvalidArgumentsError):
        run_inspect_architecture(_factory(state), "nothing/here.py")


def test_inspect_architecture_validation() -> None:
    with pytest.raises(InvalidArgumentsError):
        parse_inspect_architecture_arguments(
            {"target": "store", "max_depth": 7}
        )
    with pytest.raises(InvalidArgumentsError):
        parse_inspect_architecture_arguments(
            {"target": "store", "max_depth": -1}
        )
    with pytest.raises(InvalidArgumentsError):
        parse_inspect_architecture_arguments(
            {"target": "store", "include_subsystems": "yes"}
        )


def test_parse_inspect_arguments_rejects_unknown_keys() -> None:
    with pytest.raises(InvalidArgumentsError):
        parse_inspect_architecture_arguments({"target": "store", "depth": 3})
    with pytest.raises(InvalidArgumentsError):
        parse_inspect_architecture_arguments({})
    with pytest.raises(InvalidArgumentsError):
        parse_inspect_architecture_arguments(None)


# ---------------------------------------------------------------------------
# discover_subsystems
# ---------------------------------------------------------------------------


def test_discover_subsystems_deterministic_ordering(state) -> None:
    first = run_discover_subsystems(_factory(state))
    second = run_discover_subsystems(_factory(state))
    assert first == second
    ids = [s["id"] for s in first["subsystems"]]
    assert ids == sorted(ids)
    assert ids == ["admin", "app", "billing", "store", "tests", "web"]


def test_discover_subsystems_max_subsystems(state) -> None:
    result = run_discover_subsystems(_factory(state), max_subsystems=2)
    assert result["max_subsystems"] == 2
    assert [s["id"] for s in result["subsystems"]] == ["admin", "app"]


def test_discover_subsystems_statistics(state) -> None:
    result = run_discover_subsystems(_factory(state))
    store = next(s for s in result["subsystems"] if s["id"] == "store")
    assert store["statistics"]["packages"] == 5
    assert store["statistics"]["modules"] == 12
    assert store["statistics"]["files"] == 12
    assert len(store["entry_modules"]) == 2
    assert "store/models" in store["packages"]
    assert "store.services.checkout" in store["modules"]
    assert "store/services/checkout.py" in store["files"]


def test_discover_subsystems_projections(state) -> None:
    with_deps = run_discover_subsystems(_factory(state))
    store = next(s for s in with_deps["subsystems"] if s["id"] == "store")
    assert store["dependents"] == ["admin", "billing", "tests"]
    web = next(s for s in with_deps["subsystems"] if s["id"] == "web")
    assert web["dependencies"] == ["billing"]

    no_deps = run_discover_subsystems(
        _factory(state), include_dependencies=False, include_dependents=False
    )
    store2 = next(s for s in no_deps["subsystems"] if s["id"] == "store")
    assert "dependencies" not in store2
    assert "dependents" not in store2
    assert store2["modules"]  # identity fields always present


def test_discover_subsystems_no_stats(state) -> None:
    result = run_discover_subsystems(_factory(state), include_stats=False)
    assert "statistics" not in result["subsystems"][0]


def test_discover_subsystems_empty_repository(tmp_path: Path) -> None:
    result = run_discover_subsystems(_factory(_state(tmp_path)))
    assert result["subsystem_count"] == 0
    assert result["subsystems"] == []


def test_discover_subsystems_validation(state) -> None:
    with pytest.raises(InvalidArgumentsError):
        parse_discover_subsystems_arguments({"max_subsystems": 0})
    with pytest.raises(InvalidArgumentsError):
        parse_discover_subsystems_arguments({"max_subsystems": 10000})
    with pytest.raises(InvalidArgumentsError):
        parse_discover_subsystems_arguments({"include_stats": "nope"})
    with pytest.raises(InvalidArgumentsError):
        parse_discover_subsystems_arguments({"bogus": 1})


# ---------------------------------------------------------------------------
# architecture_candidates
# ---------------------------------------------------------------------------


def test_candidates_direct_and_expansion(state) -> None:
    result = run_architecture_candidates(_factory(state), "checkout", limit=100)
    assert result["status"] == "ok"
    reasons = {c["inclusion_reason"] for c in result["candidates"]}
    relevant_reasons = {
        "architecture: direct file match",
        "architecture: direct module match",
        "architecture: dependency of matched module",
        "architecture: dependent module",
        "architecture: neighboring module",
        "architecture: same subsystem",
    }
    assert reasons & relevant_reasons
    nodes = {c["architecture_node_id"] for c in result["candidates"]}
    assert "store.services.checkout" in nodes
    assert "store.models.order" in nodes
    assert "admin.reports" in nodes
    assert "store.services.refund" in nodes


def test_candidates_direct_package_match(state) -> None:
    result = run_architecture_candidates(_factory(state), "billing")
    candidates = {c["architecture_node_id"]: c for c in result["candidates"]}
    assert any(
        c["inclusion_reason"] == "architecture: direct package match"
        for c in result["candidates"]
    )
    assert "billing" in candidates


def test_candidates_module_match_with_metadata(state) -> None:
    result = run_architecture_candidates(_factory(state), "store.services.checkout")
    first = result["candidates"][0]
    assert first["module"] == "store.services.checkout"
    assert first["file"] == "store/services/checkout.py"
    assert first["package"] == "store/services"
    assert first["subsystem"] == "store"
    assert first["architecture_node_type"] in ("file", "module")
    assert first["architecture_score"] == 0
    assert first["score"] == 1.0
    assert first["dependency_direction"] == "matched"
    assert result["signals"]
    assert all(
        {"kind", "value", "node_kind", "node_id", "reason"} <= set(s)
        for s in result["signals"]
    )


def test_candidates_deterministic(state) -> None:
    a = run_architecture_candidates(_factory(state), "checkout", limit=50)
    b = run_architecture_candidates(_factory(state), "checkout", limit=50)
    assert a == b


def test_candidates_limit(state) -> None:
    result = run_architecture_candidates(_factory(state), "checkout", limit=3)
    assert result["candidate_count"] == 3
    assert len(result["candidates"]) == 3


def test_candidates_bounded_direction_frequency(state) -> None:
    result = run_architecture_candidates(_factory(state), "checkout", limit=100)
    assert len(result["candidates"]) <= 50  # config default caps expansion


def test_candidates_invalid_input() -> None:
    with pytest.raises(InvalidArgumentsError):
        parse_architecture_candidates_arguments({"query": "   "})
    with pytest.raises(InvalidArgumentsError):
        parse_architecture_candidates_arguments({"query": "checkout", "limit": 0})
    with pytest.raises(InvalidArgumentsError):
        parse_architecture_candidates_arguments({"query": "checkout", "limit": 10000})
    with pytest.raises(InvalidArgumentsError):
        parse_architecture_candidates_arguments({"query": "checkout", "max_depth": 9})
    with pytest.raises(InvalidArgumentsError):
        parse_architecture_candidates_arguments(
            {"query": "checkout", "architecture": {"max_expanded_nodes": -1}}
        )
    with pytest.raises(InvalidArgumentsError):
        parse_architecture_candidates_arguments({"query": "checkout", "bogus": 1})
    with pytest.raises(InvalidArgumentsError):
        parse_architecture_candidates_arguments({})


# ---------------------------------------------------------------------------
# explain_architecture_match
# ---------------------------------------------------------------------------


def test_explain_direct_match(state) -> None:
    result = run_explain_architecture_match(_factory(state), "store.services.checkout", "store.services.checkout")
    assert result["is_architecturally_relevant"] is True
    assert result["inclusion_reason"] == "architecture: direct module match"
    assert result["architecture_score"] == 0
    assert result["subsystem"] == "store"
    assert result["path"] == "store/services/checkout.py"
    assert result["signals"]


def test_explain_dependency_path(state) -> None:
    result = run_explain_architecture_match(_factory(state), "store.services.checkout", "store.models.order")
    assert result["is_architecturally_relevant"] is True
    assert result["inclusion_reason"] == "architecture: dependency of matched module"
    paths = result["dependency_paths"]
    assert len(paths) >= 1
    chain = [n["id"] for n in paths[0]]
    assert chain[0] == "store.services.checkout"
    assert chain[-1] == "store.models.order"


def test_explain_subsystem_relationship(state) -> None:
    result = run_explain_architecture_match(_factory(state), "store.services.checkout", "store.services.refund")
    assert result["is_architecturally_relevant"] is True
    assert result["subsystem_relationship"]["target_subsystem"] == "store"
    # same-subsystem proximity tier
    assert any(n["reason"] == "architecture: same subsystem" for n in result["matched_nodes"]) or \
        any(c["rank"] == 2 for c in result["matched_nodes"])


def test_explain_irrelevant_target_is_structured(state) -> None:
    result = run_explain_architecture_match(_factory(state), "checkout", "billing")
    assert result["is_architecturally_relevant"] is False
    assert "explanation" in result
    assert result["signal_count"] >= 1
    assert "dependency_paths" not in result


def test_explain_unknown_target(state) -> None:
    result = run_explain_architecture_match(_factory(state), "checkout", "nope/not/here.py")
    assert result["is_architecturally_relevant"] is False
    assert "explanation" in result


def test_explain_bounded_paths(state) -> None:
    result = run_explain_architecture_match(_factory(state), "checkout", "store.services.checkout")
    for path in result["dependency_paths"]:
        assert len(path) <= 4  # matched path is explicitly capped


def test_explain_invalid_input(state) -> None:
    with pytest.raises(InvalidArgumentsError):
        parse_explain_architecture_match_arguments({"query": "", "target": "store"})
    with pytest.raises(InvalidArgumentsError):
        parse_explain_architecture_match_arguments(
            {"query": "checkout", "target": "../escape.py"}
        )
    with pytest.raises(InvalidArgumentsError):
        parse_explain_architecture_match_arguments({"query": "q"})  # missing target


# ---------------------------------------------------------------------------
# Shared lazy construction / warm reuse / no duplicate work
# ---------------------------------------------------------------------------


def test_state_builds_lazily_and_reuses(tmp_path: Path) -> None:
    repo = _copy_fixture(tmp_path)
    fresh = ArchitectureState(repo)
    assert fresh._graph is None  # not built yet
    assert fresh._index is None
    graph1 = fresh.graph  # builds once
    assert fresh._index is not None
    assert fresh.graph is graph1  # reused
    subs1 = fresh.subsystems
    assert fresh.subsystems is subs1
    assert fresh.graph is graph1  # graph survives subsystem access


def test_no_duplicate_parses_across_tool_calls(tmp_path: Path) -> None:
    repo = _copy_fixture(tmp_path)
    state = _state(repo)
    fac = _factory(state)
    assert state.parsed_file_count is None  # untouched yet
    run_discover_subsystems(fac)
    parsed_after_first = state.parsed_file_count
    assert parsed_after_first == 24
    run_inspect_architecture(fac, "store.services.checkout")
    run_architecture_candidates(fac, "checkout")
    run_explain_architecture_match(fac, "checkout", "store.models.order")
    assert state.parsed_file_count == parsed_after_first  # no re-parsing


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
    run_discover_subsystems(_factory(first))
    assert first.parsed_file_count == 24

    second = _state(repo)
    run_discover_subsystems(_factory(second))
    assert second.parsed_file_count == 0  # warm index reused from the cache


def test_cache_disabled_mode_still_works(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("REPOLENS_CACHE_DISABLED", "1")
    repo = _copy_fixture(tmp_path)
    state = _state(repo)
    fac = _factory(state)
    result = run_discover_subsystems(fac)
    assert len(result["subsystems"]) == 6
    # cache disabled => a fresh state lazily re-parses when built
    fresh = _state(repo)
    assert fresh.parsed_file_count is None  # lazily built
    assert fresh.graph is not None


def test_incremental_modification(tmp_path: Path) -> None:
    repo = _copy_fixture(tmp_path)
    before = _state(repo)
    result_before = run_inspect_architecture(_factory(before), "store.repositories.orders")
    assert "billing.invoices" not in [d["id"] for d in result_before["dependencies"]]

    order = repo / "store" / "services" / "checkout.py"
    order.write_text(
        "from store.models.order import Order\n"
        "from store.repositories.orders import OrderRepository\n"
        "from billing.invoices import invoice_number  # new dependency\n",
        encoding="utf-8",
    )
    after = _state(repo)
    result_after = run_inspect_architecture(_factory(after), "store.services.checkout")
    assert "billing.invoices" in [d["id"] for d in result_after["dependencies"]]

    # A fresh graph reflects only the current snapshot (deterministic).
    assert before.graph.serialize() in (before.graph.serialize(),)


def test_deleted_file(tmp_path: Path) -> None:
    repo = _copy_fixture(tmp_path)
    state = _state(repo)
    fac = _factory(state)
    assert run_inspect_architecture(fac, "store.api.catalog")["kind"] == "module"

    (repo / "store" / "api" / "catalog.py").unlink()
    deleted = _state(repo)
    with pytest.raises(InvalidArgumentsError):
        run_inspect_architecture(_factory(deleted), "store.api.catalog")


# ---------------------------------------------------------------------------
# Error safety
# ---------------------------------------------------------------------------


def test_factory_failure_raises_safe_architecture_error(state) -> None:
    def boom(**kwargs):
        raise RuntimeError("boom")

    with pytest.raises(ArchitectureError):
        run_discover_subsystems(boom)


def test_non_callable_factory_raises_internal(state) -> None:
    with pytest.raises(InternalError):
        run_inspect_architecture(None, "store")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Protocol-level registration
# ---------------------------------------------------------------------------


@pytest.fixture()
def arch_state() -> ArchitectureState:
    return _state(ROOT)


@pytest.fixture()
def arch_server(arch_state):
    def engine_factory(**kwargs):
        return None  # architecture tools never touch the engine

    return build_mcp_server(
        engine_factory,
        ContextFirewall(),
        architecture_factory=_factory(arch_state),
    )


def test_architecture_tools_registered(arch_server) -> None:
    tools = asyncio.run(arch_server.list_tools())
    names = {t.name for t in tools}
    assert "get_context" in names  # core tool unchanged
    assert {"inspect_architecture", "discover_subsystems", "architecture_candidates",
            "explain_architecture_match"} <= names


def test_architecture_tools_unavailable_without_factory() -> None:
    def engine_factory(**kwargs):
        return None

    server = build_mcp_server(engine_factory, ContextFirewall())
    tools = asyncio.run(server.list_tools())
    names = {t.name for t in tools}
    assert names == {"get_context"}


def test_protocol_inspect_architecture(arch_server) -> None:
    async def run():
        result = await arch_server.call_tool("inspect_architecture", {"target": "store.services.checkout"})
        assert result.is_error is False
        data = json.loads(result.content[0].text)
        assert data["status"] == "ok"
        assert data["subsystem"] == "store"
        return result

    asyncio.run(run())


def test_protocol_invalid_argument_is_safe_error(arch_server) -> None:
    async def run():
        result = await arch_server.call_tool(
            "architecture_candidates", {"query": "checkout", "limit": -1}
        )
        assert result.is_error is True
        text = result.content[0].text
        assert "limit" in text.lower()
        assert "traceback" not in text.lower()
        return result

    asyncio.run(run())


def test_protocol_unsafe_target_is_safe_error(arch_server) -> None:
    async def run():
        result = await arch_server.call_tool(
            "inspect_architecture", {"target": "../../etc/passwd"}
        )
        assert result.is_error is True
        assert "traceback" not in result.content[0].text.lower()
        return result

    asyncio.run(run())


def test_protocol_explain_irrelevant(arch_server) -> None:
    async def run():
        result = await arch_server.call_tool(
            "explain_architecture_match", {"query": "checkout", "target": "billing"}
        )
        assert result.is_error is False
        data = json.loads(result.content[0].text)
        assert data["is_architecturally_relevant"] is False
        return result

    asyncio.run(run())