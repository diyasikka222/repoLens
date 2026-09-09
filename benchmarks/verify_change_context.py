"""Deterministic verification for M24.2 change-plan/change-context MCP tools.

Covers the two additive MCP tools (``change_plan`` and ``change_context``)
through the new MCP change-plan layer:

Part 1 runs against the ``change_plan_repository`` fixture and asserts
structured, bounded, deterministic, and JSON-safe responses for both tools:
explicit file / natural-language targets, change-aware context with surviving
change-plan files, budget respect, explanations per selected file, category
filters (``include_tests``), the shared lazy ``ChangePlanState`` (parsed counts
stable across tool calls, per-call bounds sharing the same components), and
protocol-level registration alongside the unchanged ``get_context`` tool.

Part 2 runs against the real RepoLens repository through the launcher-wired
change-plan factory: a cold change/context build (files parsed > 0), a warm
factory that reuses the persistent index (zero files re-parsed) with identical
substantive output, the cache-disabled path, repeated determinism in the real
engine context build, an incremental-modification check on a disposable copy,
and protocol-level calls through a built MCP server. It prints real statistics
and latencies without flaky timing assertions.

Deterministic and fully offline. Uses a temp cache base; does not modify the
repository under analysis (the modification check runs on a disposable copy in
a temp directory).
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

from repolens.context import ContextBudget, ContextEngine, ContextFirewall
from repolens.mcp import build_mcp_server
from repolens.mcp.change_plan_tool import (
    ChangePlanState,
    run_change_context,
    run_change_plan,
)
from repolens.mcp.launcher import make_change_plan_factory
from repolens.search import CodeSearcher

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "change_plan_repository"

CHANGE_REQUEST = "Add refund support to checkout"
CHANGE_TARGET = "app/services/checkout.py"


def _factory_for(state: ChangePlanState):
    def factory(**kwargs) -> ChangePlanState:
        return state

    return factory


def _engine_factory(root: Path):
    def factory(max_tokens=None, dependency_depth=None, **kwargs):
        return ContextEngine(
            root,
            searcher=CodeSearcher(root),
            budget=(
                ContextBudget(max_tokens=10**9)
                if max_tokens is None
                else ContextBudget(max_tokens=max_tokens)
            ),
        )

    return factory


def _substantive(response: dict) -> dict:
    """Response content excluding run-environment metadata.

    ``statistics`` carries deterministic counters, but the ``diagnostics``
    object (added in P25.6) is wall-clock run metadata — ``build_time`` and
    similar — and must be excluded from cold-vs-warm determinism comparisons
    (same contract as ``benchmarks/opencode_e2e.py``'s latency stripping).
    """
    d = dict(response)
    d.pop("statistics", None)
    d.pop("diagnostics", None)
    if "change_plan" in d:
        change_plan = dict(d["change_plan"])
        change_plan["statistics"] = "stripped"
        change_plan["diagnostics"] = "stripped"
        d["change_plan"] = change_plan
    return d


def part1(root: Path) -> None:
    print("fixture: structured, bounded, deterministic change tools")
    state = ChangePlanState(root)
    fac = _factory_for(state)
    firewall = ContextFirewall()

    # change_plan: explicit file target -> confirmed, bounded, JSON-safe.
    plan = run_change_plan(fac, "change", target=CHANGE_TARGET)
    assert plan["status"] == "ok"
    assert plan["primary_target"]["kind"] == "file"
    assert plan["primary_target"]["confidence"] == "confirmed"
    assert plan["primary_target"]["target"] == CHANGE_TARGET
    assert plan["affected_files"], "explicit file target must surface affected files"
    assert plan["risk"] in ("low", "medium", "high")
    assert plan["risk_factors"], "risk must be explainable"
    assert plan["statistics"]["affected_file_count"] == len(plan["affected_files"])
    json.dumps(plan)
    print(f"  change_plan(file): affected={len(plan['affected_files'])}, "
          f"tests={len(plan['tests'])}, risk={plan['risk']} "
          f"[{', '.join(plan['risk_factors'])}], JSON-safe: OK")

    # change_plan: natural-language request.
    nl = run_change_plan(fac, CHANGE_REQUEST)
    assert nl["primary_target"] is not None
    assert nl["inspection_order"]
    priorities = [i["priority"] for i in nl["inspection_order"]]
    assert priorities == sorted(priorities)
    print(f"  change_plan(NL): primary={nl['primary_target']['target']}, "
          f"inspection={len(nl['inspection_order'])} items: OK")

    # change_plan: bounds respected.
    bounded = run_change_plan(
        fac, "change", target=CHANGE_TARGET, max_files=3, max_tests=1, max_targets=2
    )
    assert bounded["statistics"]["affected_file_count"] <= 3
    assert len(bounded["tests"]) <= 1
    assert len(bounded["target_candidates"]) <= 2
    print(f"  change_plan bounds: affected<=3 (got "
          f"{bounded['statistics']['affected_file_count']}), tests<=1, "
          f"targets<=2: OK")

    # change_plan: deterministic.
    again = run_change_plan(fac, CHANGE_REQUEST)
    assert again == nl
    print("  change_plan determinism (repeated identical): OK")

    # change_context: safe, structured, bounded, with explanations.
    ctx = run_change_context(
        _engine_factory(root), firewall, fac, CHANGE_REQUEST, target=CHANGE_TARGET
    )
    assert ctx["status"] == "ok"
    assert ctx["selected_files"], "change-aware build must select files"
    assert ctx["change_files"], "plan files must survive into the change context"
    assert ctx["change_plan"]["primary_target"]["target"] == CHANGE_TARGET
    assert ctx["rendered_safe_context"]
    assert 0 <= ctx["total_estimated_tokens"]
    json.dumps(ctx)
    print(f"  change_context: {len(ctx['selected_files'])} selected, "
          f"{len(ctx['change_files'])} change files, "
          f"{ctx['budget']['max_tokens']} max tokens, "
          f"risk={ctx['change_plan']['risk']}, JSON-safe: OK")

    # include_tests filter: test-only files vanish from change files.
    no_tests = run_change_context(
        _engine_factory(root), firewall, fac, CHANGE_REQUEST,
        target=CHANGE_TARGET, include_tests=False,
    )
    assert "tests/test_checkout.py" not in no_tests["change_files"]
    assert "tests/test_checkout.py" in ctx["change_files"]
    print("  change_context include_tests filter (tests dropped): OK")

    # Explanations per selected file, deterministic.
    first = run_change_context(_engine_factory(root), firewall, fac, CHANGE_REQUEST)
    second = run_change_context(_engine_factory(root), firewall, fac, CHANGE_REQUEST)
    assert first == second
    for item in first["selected_files"]:
        assert item["explanation"]["path"] == item["path"]
    print(f"  change_context determinism + per-file explanations "
          f"({len(first['selected_files'])} files): OK")

    # Budget capping is respected.
    capped = run_change_context(
        _engine_factory(root), firewall, fac, CHANGE_REQUEST, max_tokens=200
    )
    assert 0 <= capped["total_estimated_tokens"] <= 200
    assert capped["total_estimated_tokens"] <= first["total_estimated_tokens"]
    print(f"  change_context budget: {capped['total_estimated_tokens']}<=200 "
          f"tokens (unlimited {first['total_estimated_tokens']}): OK")

    # Shared lazy state: parsed count stable across all tool calls.
    parsed = state.parsed_file_count
    assert parsed is not None, "index must be built after first use"
    run_change_plan(fac, CHANGE_REQUEST)
    run_change_context(_engine_factory(root), firewall, fac, CHANGE_REQUEST)
    assert state.parsed_file_count == parsed
    print(f"  shared change state: {parsed} files parsed (warm/cold), stable "
          f"across all change-tool calls: OK")

    # Per-call bounds reuse shared components through the public API.
    from repolens.change_plan import ChangePlanConfig

    base = state.default_engine
    tight = ChangePlanConfig(max_affected_files=2)
    per_call = state.engine(tight)
    assert per_call is not base
    assert per_call._dep_graph is base._dep_graph
    assert per_call._symbol_index is base._symbol_index
    assert per_call._call_graph is base._call_graph
    assert state.engine(tight) is per_call
    print(f"  per-call engines reuse shared components, default engine "
          f"cached (parsed={state.parsed_file_count}): OK")


def part2(root: Path) -> None:
    cache_base = Path(tempfile.mkdtemp(prefix="repolens-change-ctx-"))
    os.environ["REPOLENS_CACHE_DIR"] = str(cache_base)
    os.environ.pop("REPOLENS_CACHE_DISABLED", None)
    print("real repo: MCP server + launcher-wired change-plan factory")

    factory = make_change_plan_factory(root)
    state = factory()
    firewall = ContextFirewall()
    request = "Add a change-plan-aware MCP tool to the server"
    target = "repolens/mcp/change_plan_tool.py"

    start = time.perf_counter()
    engine_state = state  # lazy: index built on first use
    default = engine_state.default_engine
    cold_seconds = time.perf_counter() - start
    parsed_cold = engine_state.parsed_file_count
    cold_plan = run_change_plan(_factory_for(engine_state), request, target=target)
    assert cold_plan["status"] == "ok"
    assert cold_plan["primary_target"] is not None
    assert parsed_cold and parsed_cold > 0
    print(f"  cold change-plan build: {parsed_cold} files parsed in "
          f"{cold_seconds:.3f}s; primary={cold_plan['primary_target']['target']}, "
          f"affected={cold_plan['statistics']['affected_file_count']}, "
          f"risk={cold_plan['risk']}: OK")

    # Warm: a second factory reuses the persistent index (zero re-parses).
    start = time.perf_counter()
    warm_state = make_change_plan_factory(root)()
    warm_default = warm_state.default_engine
    warm_seconds = time.perf_counter() - start
    warm_plan = run_change_plan(_factory_for(warm_state), request, target=target)
    assert warm_state.parsed_file_count == 0, "warm factory must not re-parse"
    assert warm_plan["primary_target"] == cold_plan["primary_target"]
    assert _substantive(warm_plan) == _substantive(cold_plan)
    print(f"  warm factory reuses index: parsed=0 in {warm_seconds:.3f}s, "
          f"substantive plan identical: OK")

    # change_context on the real repo: deterministic, bounded, safe.
    fac = _factory_for(warm_state)
    start = time.perf_counter()
    ctx = run_change_context(
        _engine_factory(root), firewall, fac, request, target=target
    )
    elapsed = time.perf_counter() - start
    assert ctx["status"] == "ok"
    assert ctx["selected_files"], "real-repo change context must select files"
    assert ctx["change_plan"]["primary_target"] is not None
    repeated = run_change_context(
        _engine_factory(root), firewall, fac, request, target=target
    )
    assert _substantive(repeated) == _substantive(ctx)
    print(f"  change_context(real repo): {len(ctx['selected_files'])} selected "
          f"in {elapsed*1000:.1f}ms, plan "
          f"({ctx['change_plan']['affected_file_count']} affected files, "
          f"risk={ctx['change_plan']['risk']}), deterministic: OK")

    # Protocol-level: registration + calls alongside the core get_context tool.
    server = build_mcp_server(
        _engine_factory(root),
        firewall,
        change_plan_factory=factory,
    )

    async def protocol_drive() -> dict:
        tools = await server.list_tools()
        names = {t.name for t in tools}
        assert "get_context" in names, "core tool must remain registered"
        assert {"change_plan", "change_context"} <= names
        calls = [
            ("change_plan", {"request": request, "target": target}),
            ("change_context", {"request": request, "target": target}),
        ]
        results = {}
        for name, args in calls:
            result = await server.call_tool(name, args)
            assert result.is_error is False, (name, result.content)
            results[name] = json.loads(result.content[0].text)
        return results

    results = asyncio.run(protocol_drive())
    assert results["change_plan"]["status"] == "ok"
    assert results["change_context"]["status"] == "ok"
    print(f"  protocol: change_plan + change_context callable via a built "
          f"MCP server; core get_context unchanged: OK")

    # Cache-disabled path must still produce correct output.
    os.environ["REPOLENS_CACHE_DISABLED"] = "1"
    disabled_state = make_change_plan_factory(root)()
    disabled_plan = run_change_plan(
        _factory_for(disabled_state), request, target=target
    )
    assert disabled_plan["status"] == "ok"
    assert disabled_state.parsed_file_count is not None
    print(f"  cache-disabled mode: plan works "
          f"(parsed={disabled_state.parsed_file_count}): OK")
    os.environ.pop("REPOLENS_CACHE_DISABLED", None)

    # Incremental modification on a disposable copy: a new module created after
    # the first snapshot is visible to a fresh state without reparsing the repo.
    os.environ["REPOLENS_CACHE_DIR"] = str(cache_base)
    copy = Path(tempfile.mkdtemp(prefix="repolens-change-ctx-copy-")) / "repo"
    shutil.copytree(
        root,
        copy,
        ignore=shutil.ignore_patterns(
            ".git", ".venv", "__pycache__", ".pytest_cache",
            "tests", "benchmarks", "docs",
        ),
    )
    first = make_change_plan_factory(copy)()
    run_change_plan(_factory_for(first), "change", target=target)
    assert first.parsed_file_count and first.parsed_file_count > 0

    (copy / "repolens" / "chctx_marker.py").write_text(
        "from repolens import context as _ctx\n\n"
        "__all__ = ['CHANGE_CONTEXT_MARKER']\n",
        encoding="utf-8",
    )
    second = make_change_plan_factory(copy)()
    marker = run_change_plan(
        _factory_for(second), "change", target="repolens/chctx_marker.py"
    )
    assert marker["status"] == "ok"
    assert marker["primary_target"] is not None
    assert "chctx_marker" in marker["primary_target"]["target"]
    assert second.parsed_file_count == 1, "only the new file is parsed"
    print(f"  incremental modification: new module "
          f"{marker['primary_target']['target']} visible to a fresh state "
          f"(parsed={second.parsed_file_count}): OK")


def main() -> int:
    print("PART 1 - synthetic change-plan repository")
    part1(FIXTURE)

    print("\nPART 2 - real repository (RepoLens itself)")
    part2(REPO_ROOT)

    print("\nVERIFICATION PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())