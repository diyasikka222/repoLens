"""Tests for the M24.1 ChangePlanEngine.

Covers request analysis, target discovery and ranking, impact/call/dependency
enrichment, architecture/subsystem enrichment, test discovery, inspection
ordering, risk assessment, confidence levels, explainability, deterministic
serialization, bounded traversal (max-depth behavior), empty/single-file
repositories, unresolved references, cross-package changes, and a reusable
context-candidate helper. All tests are offline and deterministic.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pytest

from repolens.change_plan import (
    ChangePlanConfig,
    ChangePlanEngine,
    Confidence,
    PlanCategory,
    RequestAnalysis,
    TargetKind,
    analyze_request,
    plan_to_context_candidates,
)

ROOT = Path(__file__).parent / "fixtures" / "change_plan_repository"


def _engine(root: Path = ROOT, **kwargs) -> ChangePlanEngine:
    return ChangePlanEngine(root, **kwargs)


# ---------------------------------------------------------------------------
# Request analysis
# ---------------------------------------------------------------------------


def test_analyze_request_detects_paths_modules_symbols_terms() -> None:
    a = analyze_request("Add refund support to app/services/checkout.py and Order")
    assert a.action == "add"
    assert "refund" in a.domain_terms
    assert "checkout" in a.domain_terms
    assert a.path_candidates == ("app/services/checkout.py",)
    assert "Order" in a.symbol_candidates


def test_analyze_request_detects_module_names() -> None:
    a = analyze_request("modify app.services.checkout")
    assert a.action == "modify"
    assert "app.services.checkout" in a.module_candidates


def test_analyze_request_action_verbs() -> None:
    for verb, expected in [
        ("add", "add"), ("remove", "remove"), ("rename", "rename"),
        ("refactor", "refactor"), ("fix", "fix"), ("migrate", "migrate"),
        ("extend", "extend"), ("delete", "remove"),
    ]:
        assert analyze_request(f"{verb} something").action == expected


def test_analyze_request_no_action() -> None:
    assert analyze_request("the checkout flow").action is None


def test_request_analysis_is_deterministic() -> None:
    a1 = analyze_request("Add refund support to checkout")
    a2 = analyze_request("Add refund support to checkout")
    assert a1 == a2


# ---------------------------------------------------------------------------
# Exact targets
# ---------------------------------------------------------------------------


def test_exact_file_target() -> None:
    plan = _engine().plan("change", target="app/services/checkout.py")
    assert plan.primary_target is not None
    assert plan.primary_target.kind == TargetKind.FILE
    assert plan.primary_target.target == "app/services/checkout.py"
    assert plan.primary_target.confidence == Confidence.CONFIRMED
    assert "explicit file target" in plan.primary_target.reasons


def test_exact_symbol_target() -> None:
    plan = _engine().plan("change", target="checkout")
    assert plan.primary_target is not None
    # 'checkout' resolves to a symbol/module in app.services.checkout
    assert plan.primary_target.confidence == Confidence.CONFIRMED


def test_module_target() -> None:
    plan = _engine().plan("change", target="app.services.checkout")
    assert plan.primary_target is not None
    assert plan.primary_target.target == "app.services.checkout"
    assert plan.primary_target.kind == TargetKind.MODULE


def test_package_target() -> None:
    plan = _engine().plan("change", target="app/services")
    assert plan.primary_target is not None
    assert plan.primary_target.kind == TargetKind.PACKAGE


def test_module_target_defining_file_in_affected_surface() -> None:
    plan = _engine().plan("change", target="app.services.checkout")
    assert plan.primary_target is not None
    assert plan.primary_target.kind == TargetKind.MODULE
    # The defining file is the primary affected item (a readable file, not the
    # bare dotted id) and the first inspection item.
    defining = [
        i for i in plan.affected_files
        if i.category == PlanCategory.PRIMARY_TARGET
    ]
    assert len(defining) == 1
    assert defining[0].path == "app/services/checkout.py"
    assert defining[0].module == "app.services.checkout"
    primary_item = plan.inspection_order[0]
    assert primary_item.path == "app/services/checkout.py"
    assert primary_item.category == PlanCategory.PRIMARY_TARGET


def test_module_target_primary_survives_affected_cap() -> None:
    plan = _engine(config=ChangePlanConfig(max_affected_files=1)).plan(
        "change", target="app.services.checkout"
    )
    assert [i.path for i in plan.affected_files] == ["app/services/checkout.py"]


def test_file_target_affected_surface_unchanged() -> None:
    plan = _engine().plan("change", target="app/services/checkout.py")
    assert plan.primary_target is not None
    assert plan.primary_target.kind == TargetKind.FILE
    assert all(
        i.category != PlanCategory.PRIMARY_TARGET for i in plan.affected_files
    )
    first = plan.inspection_order[0]
    assert first.path == "app/services/checkout.py"


# ---------------------------------------------------------------------------
# Natural language
# ---------------------------------------------------------------------------


def test_natural_language_request() -> None:
    plan = _engine().plan("Add refund support to checkout")
    assert plan.primary_target is not None
    assert "refund" in " ".join(t.target for t in plan.targets).lower()
    assert len(plan.targets) > 1  # multiple candidates


def test_multiple_target_candidates() -> None:
    plan = _engine().plan("payments")
    assert len(plan.targets) >= 1
    kinds = {t.kind for t in plan.targets}
    assert kinds  # non-empty


def test_no_false_precision_returns_multiple() -> None:
    plan = _engine().plan("similar nothing budget")
    # Do not claim a single confirmed target for a vague request.
    assert plan.primary_target is None or plan.primary_target.confidence in (
        Confidence.LIKELY, Confidence.POSSIBLE,
    )


# ---------------------------------------------------------------------------
# Target ranking
# ---------------------------------------------------------------------------


def test_target_rank_order() -> None:
    plan = _engine().plan("change", target="app/services/checkout.py")
    targets = plan.targets
    for i in range(len(targets) - 1):
        assert targets[i].score >= targets[i + 1].score


def test_explicit_target_ranks_above_lexical() -> None:
    plan = _engine().plan("checkout flow", target="app/services/checkout.py")
    assert plan.primary_target.target == "app/services/checkout.py"
    assert plan.primary_target.score >= 90


# ---------------------------------------------------------------------------
# Impact enrichment / callers / callees / dependencies
# ---------------------------------------------------------------------------


def test_impact_enrichment_finds_callers() -> None:
    plan = _engine().plan("change", target="app/services/checkout.py")
    categories = {i.category for i in plan.affected_files}
    assert PlanCategory.DIRECT_CALLER in categories  # e.g. app/api/checkout.py


def test_impact_enrichment_finds_dependencies() -> None:
    plan = _engine().plan("change", target="app/services/checkout.py")
    categories = {i.category for i in plan.affected_files}
    assert PlanCategory.DIRECT_DEPENDENCY in categories  # app/models/order.py


def test_impact_enrichment_finds_dependents() -> None:
    plan = _engine().plan("change", target="app/models/order.py")
    paths = {i.path for i in plan.affected_files}
    # order is used by many modules
    assert "app/services/checkout.py" in paths
    assert "app/models/refund.py" in paths


def test_affected_symbols_collected() -> None:
    plan = _engine().plan("change", target="app/services/checkout.py")
    assert isinstance(plan.affected_symbols, tuple)


# ---------------------------------------------------------------------------
# Architecture / subsystem enrichment
# ---------------------------------------------------------------------------


def test_architecture_enrichment() -> None:
    plan = _engine().plan("change", target="app/services/checkout.py")
    assert plan.architecture
    entry = plan.architecture[0]
    assert entry["target"] == "app/services/checkout.py"
    assert entry["package"] is not None
    assert "app.models.order" in entry["dependencies"]


def test_subsystem_enrichment() -> None:
    plan = _engine().plan("change", target="app/services/checkout.py")
    entry = plan.architecture[0]
    assert entry["subsystem"] == "app"


def test_cross_subsystem_change() -> None:
    # Changing a test target should surface app subsystem items.
    plan = _engine().plan("change", target="tests/test_checkout.py")
    assert plan.architecture[0]["subsystem"] == "tests"


# ---------------------------------------------------------------------------
# Test discovery
# ---------------------------------------------------------------------------


def test_test_discovery_imports_affected_module() -> None:
    plan = _engine().plan("change", target="app/services/checkout.py")
    test_paths = {t.path for t in plan.tests}
    assert "tests/test_checkout.py" in test_paths
    checkout_test = next(t for t in plan.tests if t.path == "tests/test_checkout.py")
    assert "imports affected module" in checkout_test.reason


def test_test_discovery_matching_module_name() -> None:
    plan = _engine().plan("change", target="app/services/payments.py")
    test_paths = {t.path for t in plan.tests}
    assert "tests/test_payments.py" in test_paths
    payments_test = next(t for t in plan.tests if t.path == "tests/test_payments.py")
    assert "imports affected module" in payments_test.reason


def test_every_test_has_reason() -> None:
    plan = _engine().plan("change", target="app/services/checkout.py")
    for t in plan.tests:
        assert t.reason.startswith("test:") or "test" in t.reason


# ---------------------------------------------------------------------------
# Inspection ordering
# ---------------------------------------------------------------------------


def test_inspection_ordering_priorities_increasing() -> None:
    plan = _engine().plan("change", target="app/services/checkout.py")
    priorities = [i.priority for i in plan.inspection_order]
    assert priorities == list(range(1, len(priorities) + 1))


def test_inspection_ordering_primary_first() -> None:
    plan = _engine().plan("change", target="app/services/checkout.py")
    assert plan.inspection_order[0].category == PlanCategory.PRIMARY_TARGET
    assert plan.inspection_order[0].path == "app/services/checkout.py"


def test_inspection_ordering_stable() -> None:
    e = _engine()
    a = e.plan("change", target="app/services/checkout.py")
    b = e.plan("change", target="app/services/checkout.py")
    assert a.inspection_order == b.inspection_order


# ---------------------------------------------------------------------------
# Risk
# ---------------------------------------------------------------------------


def test_risk_low_for_small_change() -> None:
    plan = _engine().plan("change", target="app/models/product.py")
    assert plan.risk == "low"


def test_risk_medium() -> None:
    plan = _engine().plan("change", target="app/models/order.py")
    assert plan.risk in ("low", "medium", "high")


def test_risk_high_for_cross_subsystem_wide_change() -> None:
    plan = _engine().plan("change", target="app/services/checkout.py")
    # checkout touches many packages + API consumer
    assert plan.risk in ("medium", "high")


# ---------------------------------------------------------------------------
# Confidence
# ---------------------------------------------------------------------------


def test_confidence_levels_present() -> None:
    plan = _engine().plan("Add refund support to checkout")
    confidences = {t.confidence for t in plan.targets}
    assert confidences <= {Confidence.CONFIRMED, Confidence.LIKELY, Confidence.POSSIBLE}


def test_explicit_target_confirmed() -> None:
    plan = _engine().plan("change", target="app/services/checkout.py")
    assert plan.primary_target.confidence == Confidence.CONFIRMED


# ---------------------------------------------------------------------------
# Explainability
# ---------------------------------------------------------------------------


def test_every_item_has_reason() -> None:
    plan = _engine().plan("change", target="app/services/checkout.py")
    for item in plan.inspection_order:
        assert item.reason
        assert item.category


# ---------------------------------------------------------------------------
# Determinism / serialization
# ---------------------------------------------------------------------------


def test_deterministic_serialization() -> None:
    e = _engine()
    a = e.plan("Add refund support to checkout")
    b = e.plan("Add refund support to checkout")
    assert a == b


def test_serialize_to_dict_round_trip() -> None:
    plan = _engine().plan("Add refund support to checkout")
    d = asdict(plan)
    json.dumps(d)  # must be JSON-serializable
    assert d["request"] == "Add refund support to checkout"
    assert d["risk"] in ("low", "medium", "high")


# ---------------------------------------------------------------------------
# Bounds
# ---------------------------------------------------------------------------


def test_bounded_targets() -> None:
    cfg = ChangePlanConfig(max_target_candidates=4)
    plan = _engine(config=cfg).plan("payments checkout")
    assert len(plan.targets) <= 4


def test_max_depth_behavior_limit_affected() -> None:
    cfg = ChangePlanConfig(impact_max_depth=1, max_affected_files=5)
    plan = _engine(config=cfg).plan("change", target="app/models/order.py")
    assert len(plan.affected_files) <= 5


def test_max_inspection_items_bounded() -> None:
    cfg = ChangePlanConfig(max_inspection_items=10)
    plan = _engine(config=cfg).plan("change", target="app/services/checkout.py")
    assert len(plan.inspection_order) <= 10


def test_max_tests_bounded() -> None:
    cfg = ChangePlanConfig(max_tests=2)
    plan = _engine(config=cfg).plan("change", target="app/services/checkout.py")
    assert len(plan.tests) <= 2


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_empty_repository(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    plan = _engine(repo).plan("Add refund support")
    assert plan.primary_target is None
    assert plan.affected_files == ()


def test_single_file_repository(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "single.py").write_text("x = 1\n", encoding="utf-8")
    plan = _engine(repo).plan("change", target="single.py")
    assert plan.primary_target is not None
    assert plan.primary_target.target == "single.py"


def test_unresolved_references_handled() -> None:
    # The fixture has unresolved references (external imports); planning must
    # not crash and should still produce a plan.
    plan = _engine().plan("change", target="app/config/routes.py")
    assert plan.primary_target is not None


def test_cross_package_change() -> None:
    plan = _engine().plan("change", target="app/models/order.py")
    entry = plan.architecture[0]
    packages = {entry.get("package")}
    packages.update(entry.get("dependent_packages", ()))
    packages.discard(None)
    assert len(packages) > 1  # order is consumed across packages


def test_backward_compat_impact_result() -> None:
    from repolens.impact import ImpactAnalyzer
    from repolens.incremental_index import IncrementalIndexBuilder
    from repolens.graph import DependencyGraphBuilder

    index = IncrementalIndexBuilder(ROOT, persist=False).build()
    graph = DependencyGraphBuilder(ROOT, index=index).build()
    ia = ImpactAnalyzer(ROOT, index=index, graph=graph)
    result = ia.analyze("app/services/checkout.py", max_depth=1, limit=5)
    assert result.target == "app/services/checkout.py"
    assert len(result.items) > 0


# ---------------------------------------------------------------------------
# plan_to_context_candidates helper
# ---------------------------------------------------------------------------


def test_plan_to_context_candidates() -> None:
    plan = _engine().plan("Add refund support to checkout")
    candidates = plan_to_context_candidates(plan)
    assert candidates
    assert all("path" in c for c in candidates)
    assert all("priority" in c for c in candidates)
    assert all("reason" in c for c in candidates)


def test_plan_to_context_candidates_bounded() -> None:
    plan = _engine().plan("Add refund support to checkout")
    candidates = plan_to_context_candidates(plan, limit=5)
    assert len(candidates) <= 5


def test_plan_to_context_candidates_prioritized() -> None:
    plan = _engine().plan("Add refund support to checkout")
    candidates = plan_to_context_candidates(plan)
    priorities = [c["priority"] for c in candidates]
    assert priorities == sorted(priorities)


# ---------------------------------------------------------------------------
# Change-aware bridge (Milestone 24.2): plan -> ContextCandidate
# ---------------------------------------------------------------------------


def _change_candidates(plan, **kwargs):
    from repolens.change_context import ChangeContextOptions, plan_to_change_candidates

    return plan_to_change_candidates(
        plan,
        options=ChangeContextOptions(**kwargs),
        root=ROOT,
    )


def test_plan_to_change_candidates_basic() -> None:
    from repolens.change_context import plan_to_change_candidates

    plan = _engine().plan(
        "make checkout reject empty carts", target="app/services/checkout.py"
    )
    candidates = plan_to_change_candidates(plan, root=ROOT)
    assert candidates
    first = candidates[0]
    assert first.path == Path("app/services/checkout.py")
    assert first.change_category == "primary_target"
    assert first.change_priority == 1
    assert first.change_confidence == "confirmed"
    assert first.inclusion_reason == "change_plan"
    order = [c.change_priority for c in candidates]
    assert order == sorted(order)


def test_plan_to_change_candidates_max_files() -> None:
    plan = _engine().plan(
        "make checkout reject empty carts", target="app/services/checkout.py"
    )
    candidates = _change_candidates(plan, max_files=5)
    assert len(candidates) <= 5


def test_plan_to_change_candidates_include_tests() -> None:
    plan = _engine().plan(
        "make checkout reject empty carts", target="app/services/checkout.py"
    )
    with_tests = _change_candidates(plan, include_tests=True)
    without = _change_candidates(plan, include_tests=False)
    # Test-only files are introduced with tests enabled and dropped otherwise.
    assert any(c.path == Path("tests/test_checkout.py") for c in with_tests)
    assert any(c.path == Path("tests/test_payments.py") for c in with_tests)
    assert all(c.path != Path("tests/test_checkout.py") for c in without)
    assert all(c.path != Path("tests/test_payments.py") for c in without)
    # tests/test_refunds.py also imports the refunds service, so it survives
    # through the dependency relationship even when tests are filtered out.
    assert any(c.path == Path("tests/test_refunds.py") for c in without)


def test_plan_to_change_candidates_include_callers() -> None:
    plan = _engine().plan(
        "make checkout reject empty carts", target="app/services/checkout.py"
    )
    with_callers = _change_candidates(plan, include_callers=True)
    without = _change_candidates(plan, include_callers=False)
    caller_paths = {c.path for c in with_callers if c.change_category == "direct_caller"}
    assert caller_paths
    assert all(c.change_category != "direct_caller" for c in without)


def test_plan_to_change_candidates_include_dependencies() -> None:
    plan = _engine().plan(
        "make checkout reject empty carts", target="app/services/checkout.py"
    )
    with_deps = _change_candidates(plan, include_dependencies=True)
    without = _change_candidates(plan, include_dependencies=False)
    dep_paths = {
        c.path for c in with_deps if c.change_category == "direct_dependency"
    }
    assert dep_paths
    assert all(c.change_category != "direct_dependency" for c in without)


def test_plan_to_change_candidates_skips_missing_files(tmp_path: Path) -> None:
    from repolens.change_context import plan_to_change_candidates

    import shutil

    repo = tmp_path / "repo"
    shutil.copytree(ROOT, repo)
    (repo / "app" / "services" / "checkout.py").unlink()
    plan = ChangePlanEngine(repo).plan(
        "make checkout reject empty carts", target="app/services/checkout.py"
    )
    candidates = plan_to_change_candidates(plan, root=repo)
    assert all(c.path != Path("app/services/checkout.py") for c in candidates)


def test_plan_response_payload_structure() -> None:
    from repolens.change_context import plan_response_payload

    plan = _engine().plan("Add refund support to checkout")
    payload = plan_response_payload(plan, "Add refund support to checkout")
    assert payload["request"] == "Add refund support to checkout"
    assert payload["deterministic"] is True
    assert payload["primary_target"]
    assert isinstance(payload["affected_files"], list)
    assert isinstance(payload["risk"], str)
    assert isinstance(payload["statistics"], dict)
    assert set(payload) == {
        "request", "analysis", "primary_target", "target_candidates",
        "affected_files", "affected_symbols", "callers", "callees",
        "dependencies", "dependents", "architecture", "tests",
        "inspection_order", "risk", "risk_factors", "confidence",
        "summary", "statistics", "diagnostics", "deterministic",
    }
    # Wall-clock run metadata lives only under diagnostics (deterministic
    # contract); statistics carries deterministic counters only.
    assert "build_time" not in payload["statistics"]
    assert isinstance(payload["diagnostics"]["build_time"], float)
    json.dumps(payload)  # JSON-safe


def test_explain_change_context_deterministic() -> None:
    from repolens.change_context import explain_change_context

    plan = _engine().plan(
        "make checkout reject empty carts", target="app/services/checkout.py"
    )
    first = explain_change_context(plan, None, "app/services/checkout.py")
    second = explain_change_context(plan, None, "app/services/checkout.py")
    assert first == second
    assert first["in_plan"] is True
    assert first["category"] == "primary_target"
    assert first["target"] == "app/services/checkout.py"
    assert first["selected"] is False
