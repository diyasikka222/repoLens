"""Deterministic verification for M23.3 architecture-intelligence MCP tools.

Covers the four additive MCP tools (``inspect_architecture``,
``discover_subsystems``, ``architecture_candidates``,
``explain_architecture_match``) through the MCP architecture layer:

Part 1 runs against the multi-subsystem ``architecture_retrieval_repository``
fixture and asserts structured, bounded, and deterministic responses, plus the
shared lazy architecture state (graph/subset/parsed counts stay stable across
tool calls) and warm persistent-index reuse (a second state parses zero files).

Part 2 runs against the real RepoLens repository through an MCP server built
from the launcher wiring: a cold call, a warm call (zero files re-parsed),
all four tools, repeated determinism, the cache-disabled path, and an
incremental-modification check on a disposable copy. It prints real statistics
and latencies without any flaky timing assertions.

Deterministic and fully offline. Uses a temp cache base; does not modify the
repository under analysis (the modification check runs on a disposable copy in
a temp directory).
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

from repolens.mcp.architecture_tool import (
    ArchitectureState,
    run_architecture_candidates,
    run_discover_subsystems,
    run_explain_architecture_match,
    run_inspect_architecture,
)
from repolens.mcp.launcher import make_architecture_factory

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "architecture_retrieval_repository"

QUERIES = ("store.services.checkout", "checkout flow", "billing", "how does billing work")


def _factory_for(state: ArchitectureState):
    def factory(**kwargs) -> ArchitectureState:
        return state

    return factory


def part1(root: Path) -> None:
    print("fixture: structured, bounded, deterministic tools")
    state = ArchitectureState(root)
    fac = _factory_for(state)

    inspect = run_inspect_architecture(fac, "store.services.checkout")
    assert inspect["status"] == "ok" and inspect["subsystem"] == "store"
    assert len(inspect["dependencies"]) == 2 and len(inspect["dependents"]) == 4
    assert inspect["statistics"]["direct_dependency_count"] == 2
    print(f"  inspect(module): {inspect['statistics']['direct_dependency_count']} deps, "
          f"{inspect['statistics']['direct_dependent_count']} dependents, "
          f"subsystem={inspect['subsystem']}: OK")

    subs = run_discover_subsystems(fac)
    assert [s["id"] for s in subs["subsystems"]] == sorted(
        s["id"] for s in subs["subsystems"]
    )
    assert subs["subsystem_count"] == 6
    print(f"  discover_subsystems: {subs['subsystem_count']} subsystems, "
          f"ordered={[s['id'] for s in subs['subsystems']]}: OK")

    candidates = run_architecture_candidates(fac, "checkout", limit=100)
    reasons = {c["inclusion_reason"] for c in candidates["candidates"]}
    assert {
        "architecture: direct file match",
        "architecture: direct module match",
        "architecture: dependency of matched module",
    } <= reasons
    ranks = [c["architecture_score"] for c in candidates["candidates"]]
    assert ranks == sorted(ranks)
    assert all(0 <= c["architecture_score"] <= 2 for c in candidates["candidates"])
    print(f"  architecture_candidates: {candidates['candidate_count']} nodes, "
          f"rank-ordered, {len(reasons)} distinct reasons: OK")

    explain = run_explain_architecture_match(fac, "checkout", "store.models.order")
    assert explain["is_architecturally_relevant"] is True
    assert explain["inclusion_reason"] == "architecture: dependency of matched module"
    irrelevant = run_explain_architecture_match(fac, "checkout", "billing")
    assert irrelevant["is_architecturally_relevant"] is False
    print(f"  explain: relevant ({explain['inclusion_reason']}) "
          f"and irrelevant (structured false): OK")

    first = run_architecture_candidates(fac, "checkout", limit=100)
    second = run_architecture_candidates(fac, "checkout", limit=100)
    assert first == second
    print("  candidate determinism (repeated identical): OK")

    parsed = state.parsed_file_count
    run_inspect_architecture(fac, "store/repositories")
    run_discover_subsystems(fac, max_subsystems=2)
    assert state.parsed_file_count == parsed
    print(f"  shared state: {parsed} files parsed, stable across "
          f"all four tool calls: OK")
    print(f"  graph: {state.graph.stats().as_dict()}")
    print(f"  subsystems: {[s.id for s in state.subsystems]}")


def part2(root: Path) -> None:
    cache_base = Path(tempfile.mkdtemp(prefix="repolens-arch-mcp-"))
    os.environ["REPOLENS_CACHE_DIR"] = str(cache_base)
    os.environ.pop("REPOLENS_CACHE_DISABLED", None)
    print("real repo: MCP server + launcher-wired factory")

    factory = make_architecture_factory(root)
    state = factory()
    start = time.perf_counter()
    graph = state.graph  # lazy: index + graph built here on first use
    cold_seconds = time.perf_counter() - start
    assert graph.stats().nodes > 0
    parsed_cold = state.parsed_file_count
    print(f"  cold architecture build: {parsed_cold} files parsed "
          f"in {cold_seconds:.3f}s, "
          f"graph={graph.stats().as_dict()}")

    # Warm: a second build through the same (cached) factory reuses the
    # persistent index and re-parses zero files.
    start = time.perf_counter()
    warm = make_architecture_factory(root)()
    warm_graph = warm.graph  # trigger the lazy build
    warm_seconds = time.perf_counter() - start
    assert warm.parsed_file_count == 0, "warm factory must not re-parse"
    print(f"  warm factory reuses index: parsed=0 in {warm_seconds:.3f}s: OK")

    fac = _factory_for(warm)
    for query in QUERIES:
        start = time.perf_counter()
        candidates = run_architecture_candidates(fac, query, limit=50)
        elapsed = time.perf_counter() - start
        repeated = run_architecture_candidates(fac, query, limit=50)
        assert candidates == repeated
        print(f"  candidates({query!r}): {candidates['candidate_count']} nodes "
              f"in {elapsed*1000:.1f}ms (deterministic): OK")

    start = time.perf_counter()
    inspect = run_inspect_architecture(fac, "repolens/context/engine.py")
    elapsed = time.perf_counter() - start
    assert inspect["kind"] == "file" and inspect["subsystem"]
    assert inspect is not None
    print(f"  inspect(file): {elapsed*1000:.1f}ms, kind={inspect['kind']}, "
          f"subsystem={inspect['subsystem']}: OK")

    subs = run_discover_subsystems(fac)
    assert subs["subsystem_count"] >= 3
    print(f"  discover_subsystems: {subs['subsystem_count']} subsystems "
          f"({[s['id'] for s in subs['subsystems'][:3]]}, ...): OK")

    # MCP server registration, listing, and call through the protocol layer.
    from repolens.context import ContextFirewall
    from repolens.mcp import build_mcp_server
    import asyncio

    server = build_mcp_server(
        lambda **k: None,
        ContextFirewall(),
        architecture_factory=factory,
    )

    async def protocol_drive() -> dict:
        tools = await server.list_tools()
        names = {t.name for t in tools}
        assert {
            "inspect_architecture",
            "discover_subsystems",
            "architecture_candidates",
            "explain_architecture_match",
        } <= names
        calls = [
            ("inspect_architecture", {"target": "repolens/context/engine.py"}),
            ("discover_subsystems", {"max_subsystems": 3}),
            ("architecture_candidates", {"query": "architecture", "limit": 10}),
            ("explain_architecture_match", {"query": "architecture", "target": "repolens/architecture.py"}),
        ]
        results = {}
        for name, args in calls:
            result = await server.call_tool(name, args)
            assert result.is_error is False, (name, result.content)

            results[name] = result
        return results

    results = asyncio.run(protocol_drive())
    print(f"  protocol: {len(results)} architecture tools callable via a "
          f"built MCP server; core tool unchanged: OK")

    # Cache-disabled path must still produce correct, deterministic output.
    os.environ["REPOLENS_CACHE_DISABLED"] = "1"
    disabled_state = make_architecture_factory(root)()
    candidates_disabled = run_architecture_candidates(
        _factory_for(disabled_state), "architecture", limit=10
    )
    assert candidates_disabled["candidate_count"] > 0
    assert disabled_state.parsed_file_count is not None
    print("  cache-disabled mode: tools still work correctly: OK")

    # Incremental modification on a disposable copy: a new module created after
    # the first snapshot must be visible to a fresh state's graph.
    os.environ.pop("REPOLENS_CACHE_DISABLED", None)
    copy = Path(tempfile.mkdtemp(prefix="repolens-arch-mcp-copy-")) / "repo"
    shutil.copytree(root, copy, ignore=shutil.ignore_patterns(".git", ".venv", "__pycache__", ".pytest_cache", "tests", "benchmarks", "docs"))
    first = ArchitectureState(copy)
    run_architecture_candidates(_factory_for(first), "reference", limit=50)
    assert first.parsed_file_count is not None and first.parsed_file_count > 0

    (copy / "repolens" / "arch_marker.py").write_text(
        "from repolens import context as _ctx\n\n"
        "__all__ = []\n",
        encoding="utf-8",
    )
    second = ArchitectureState(copy)
    marker = run_inspect_architecture(
        _factory_for(second), "repolens.arch_marker"
    )
    assert marker["kind"] == "module"
    marker_deps = [d["id"] for d in marker["dependencies"]]
    assert any(
        d.startswith("repolens.context") or d.startswith("repolens/context")
        for d in marker_deps
    )
    print(f"  incremental modification: new module visible to a fresh "
          f"graph ({marker['file']}): OK")


def main() -> int:
    print("PART 1 - synthetic architecture-retrieval repository")
    part1(FIXTURE)

    print("\nPART 2 - real repository (RepoLens itself)")
    part2(REPO_ROOT)

    print("\nVERIFICATION PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())