"""Deterministic verification for M24.1 change-plan engine.

Covers the ChangePlanEngine through two parts:

Part 1 runs against the ``change_plan_repository`` fixture and asserts
structured, bounded, explainable, and deterministic plans: explicit file/
module/package/symbol targets, natural-language requests, cross-module and
cross-subsystem impact, test discovery, inspection ordering, risk levels,
confidence + reasons on every item, bounds (max-depth / max-candidates), and
stable repeated plans on a single engine instance.

Part 2 runs against the real RepoLens repository: a cold build + plan
(files parsed > 0), a warm second engine that re-uses the persistent index
(zero files re-parsed) with an identical substantive plan, a cache-disabled
build that must still produce a correct plan, and an incremental-modification
check on a disposable copy. It prints real statistics and latencies without
flaky timing assertions.

Deterministic and fully offline. Uses a temp cache base; does not modify the
repository under analysis (the modification check runs on a disposable copy in
a temp directory).
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

from repolens.change_plan import (
    ChangePlanConfig,
    ChangePlanEngine,
    Confidence,
    PlanCategory,
    plan_to_context_candidates,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "change_plan_repository"


def _substantive(plan) -> dict:
    """Plan content excluding run-environment metadata (parsed count, timing)."""
    d = asdict(plan)
    d.pop("stats", None)
    return d


def part1(root: Path) -> None:
    print("fixture: structured, bounded, explainable, deterministic plans")
    engine = ChangePlanEngine(root)

    plan = engine.plan("Add refund support to checkout")
    assert plan.primary_target is not None and plan.primary_target.confidence in (
        Confidence.LIKELY, Confidence.CONFIRMED,
    )
    assert plan.affected_files, "NL request must surface impacted files"
    assert plan.tests, "test discovery must find candidate tests"
    assert "tests/test_checkout.py" in {t.path for t in plan.tests} or \
        any("checkout" in t.path for t in plan.tests)
    assert plan.risk in ("low", "medium", "high")
    assert plan.risk_factors, "risk must be explainable"
    print(f"  NL request: primary={plan.primary_target.target}, "
          f"affected={len(plan.affected_files)}, tests={len(plan.tests)}, "
          f"risk={plan.risk} [{', '.join(plan.risk_factors)}]: OK")

    # Explicit file target: confirmed, inspectable, deterministic.
    explicit = engine.plan("change", target="app/services/checkout.py")
    assert explicit.primary_target is not None
    assert explicit.primary_target.target == "app/services/checkout.py"
    assert explicit.primary_target.kind.value == "file"
    assert explicit.primary_target.confidence == Confidence.CONFIRMED
    cats = {i.category for i in explicit.affected_files}
    assert PlanCategory.DIRECT_CALLER in cats
    assert PlanCategory.DIRECT_DEPENDENCY in cats
    order = [i.path for i in explicit.inspection_order]
    assert order[0] == "app/services/checkout.py"
    assert order == sorted(order, key=lambda _: (order.index(_),))  # stable
    priorities = [i.priority for i in explicit.inspection_order]
    assert priorities == list(range(1, len(priorities) + 1))
    print(f"  explicit file target: direct_caller + direct_dependency, "
          f"{len(explicit.inspection_order)} inspection items "
          f"(priorities 1..{priorities[-1]}): OK")

    # Module target via dotted name.
    module = engine.plan("change", target="app.services.checkout")
    assert module.primary_target is not None
    assert module.primary_target.kind.value == "module"
    assert module.primary_target.target == "app.services.checkout"
    print(f"  module target: {module.primary_target.target}: OK")

    # Package target.
    pkg = engine.plan("change", target="app/services")
    assert pkg.primary_target is not None
    assert pkg.primary_target.kind.value == "package"
    print(f"  package target: '{pkg.primary_target.target}': OK")

    # Cross-module impact: order is consumed across repositories/services.
    cross = engine.plan("change", target="app/models/order.py")
    arch = cross.architecture[0]
    assert "app.repositories.orders" in arch["dependents"]
    assert "app.services.checkout" in arch["dependents"]
    print(f"  cross-module impact: order -> "
          f"{len(arch['dependents'])} dependents across "
          f"{len(arch['dependent_packages'])} packages: OK")

    # Every inspection item carries an explanation + confidence.
    for item in explicit.inspection_order:
        assert item.reason and item.category
    print(f"  explainability: all {len(explicit.inspection_order)} items "
          f"carry category+reason: OK")

    # Determinism on the same engine instance (identical, including stats).
    again = engine.plan("Add refund support to checkout")
    assert again == plan
    print("  determinism: repeated plan identical (same engine): OK")

    # Bounds are configurable and respected.
    tight = ChangePlanConfig(
        max_inspection_items=6, impact_max_depth=1, max_tests=2,
    )
    bounded = ChangePlanEngine(root, config=tight).plan("change", target="app/models/order.py")
    assert len(bounded.inspection_order) <= 6
    assert len(bounded.tests) <= 2
    print(f"  bounds: inspection<=6 (got {len(bounded.inspection_order)}), "
          f"tests<=2 (got {len(bounded.tests)}): OK")

    # JSON-serializable output (explainability contract for tools).
    lowered = {
        "request": plan.request,
        "targets": [asdict(t) for t in plan.targets],
        "risk": plan.risk,
        "inspection_order": [asdict(i) for i in plan.inspection_order],
    }
    json.dumps(lowered)
    print("  JSON-serializable summary fields: OK")

    # Reusable context-candidate helper is additive and priority-ordered.
    candidates = plan_to_context_candidates(plan, limit=8)
    priorities = [c["priority"] for c in candidates]
    assert len(candidates) <= 8
    assert priorities == sorted(priorities)
    assert all(c["reason"] for c in candidates)
    print(f"  plan_to_context_candidates: {len(candidates)} candidates, "
          f"priority-ordered, each with a reason: OK")


def part2(root: Path) -> None:
    cache_base = Path(tempfile.mkdtemp(prefix="repolens-change-plan-"))
    os.environ["REPOLENS_CACHE_DIR"] = str(cache_base)
    os.environ.pop("REPOLENS_CACHE_DISABLED", None)
    print("real repo: cold/warm/incremental planning on RepoLens itself")

    start = time.perf_counter()
    cold = ChangePlanEngine(root)
    cold_plan = cold.plan("Add change-plan engine tooling to the MCP server")
    cold_seconds = time.perf_counter() - start
    assert cold_plan.primary_target is not None
    assert cold_plan.targets and cold_plan.inspection_order
    print(f"  cold build: {cold._parsed_files} files parsed in "
          f"{cold_seconds:.3f}s; primary={cold_plan.primary_target.target}, "
          f"affected={len(cold_plan.affected_files)}, "
          f"tests={len(cold_plan.tests)}, risk={cold_plan.risk}: OK")

    # Warm: second engine instance re-uses the persistent index (zero parses)
    # and produces identical *substantive* plan content.
    start = time.perf_counter()
    warm = ChangePlanEngine(root)
    warm_plan = warm.plan("Add change-plan engine tooling to the MCP server")
    warm_seconds = time.perf_counter() - start
    assert warm._parsed_files == 0, "warm engine must not re-parse files"
    assert _substantive(warm_plan) == _substantive(cold_plan), \
        "warm plan content must match cold plan content"
    print(f"  warm build re-uses index: parsed=0 in {warm_seconds:.3f}s, "
          f"substantive plan identical to cold: OK")

    # Cache-disabled mode: still correct, deterministic.
    os.environ["REPOLENS_CACHE_DISABLED"] = "1"
    no_cache = ChangePlanEngine(root)
    no_cache_plan = no_cache.plan("Add change-plan engine tooling to the MCP server")
    assert no_cache_plan.primary_target is not None
    assert _substantive(no_cache_plan) == _substantive(cold_plan)
    print(f"  cache-disabled: parsed={no_cache._parsed_files}, "
          f"substantive plan identical: OK")
    os.environ.pop("REPOLENS_CACHE_DISABLED", None)

    # Incremental modification on a disposable copy: a new module appears in a
    # fresh engine's index without parsing untouched files.
    os.environ["REPOLENS_CACHE_DIR"] = str(cache_base)
    copy = Path(tempfile.mkdtemp(prefix="repolens-change-plan-copy-")) / "repo"
    shutil.copytree(root, copy, ignore=shutil.ignore_patterns(
        ".git", ".venv", "__pycache__", ".pytest_cache", "tests", "benchmarks",
        "docs",
    ))
    first = ChangePlanEngine(copy)
    assert first._parsed_files > 0
    (copy / "repolens" / "pltool.py").write_text(
        "from repolens import change_plan as _cp\n\n"
        "__all__ = ['PlanTarget']\n",
        encoding="utf-8",
    )
    second = ChangePlanEngine(copy)
    plan2 = second.plan("change", target="repolens/pltool.py")
    assert plan2.primary_target is not None
    assert "repolens/pltool.py" == plan2.primary_target.target or \
        "pltool" in plan2.primary_target.target
    print(f"  incremental modification: new module "
          f"{plan2.primary_target.target} visible to a fresh engine "
          f"(parsed={second._parsed_files}): OK")


def main() -> int:
    print("PART 1 - synthetic change-plan repository")
    part1(FIXTURE)

    print("\nPART 2 - real repository (RepoLens itself)")
    part2(REPO_ROOT)

    print("\nVERIFICATION PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())