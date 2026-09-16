"""Focused-context substitution selection tests (P26.2 Step 4).

Verifies that an oversized full-file candidate that would otherwise be
permanently rejected (``exceeds_total_budget``) can be substituted with a
bounded set of focused symbol slices, while the full file stays excluded and
the selection contract (budget ceiling, determinism, provenance, dedupe,
firewall) is preserved. Mirrors the guarantees asserted at every pipeline
boundary in ``test_focus.py`` and ``test_budget_diagnostics.py``.
"""

from __future__ import annotations

import json
import tempfile
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
    FirewallResult,
    INCLUSION_ARCHITECTURE,
    INCLUSION_CHANGE_PLAN,
    INCLUSION_SYMBOL_MATCH,
    estimate_tokens,
    extract_symbol_spans,
)
from repolens.context.budget import select_within_budget
from repolens.context.focus_selection import (
    DEFAULT_FALLBACK_LIMIT,
    DEFAULT_MAX_FOCUSED,
    FocusedSelection,
)
from repolens.context_trace import TraceCollector


def _source_for(*names: str) -> str:
    return "".join(f"def {name}():\n    return 1\n" for name in names)


def _big_source(n: int = 200) -> str:
    return "".join(f"def sym_{i}():\n    x = {i}\n    return x\n" for i in range(n))


def _candidate(
    path: str = "app/big.py",
    source: str | None = None,
    **overrides,
) -> ContextCandidate:
    source = _big_source() if source is None else source
    base: dict = {
        "path": Path(path),
        "source": source,
        "role": CandidateRole.PRIMARY,
        "estimated_tokens": estimate_tokens(source),
        "selection_reason": "matched signal",
        "retrieval_rank": 1,
        "retrieval_score": 0.5,
        "inclusion_reason": INCLUSION_SYMBOL_MATCH,
    }
    base.update(overrides)
    return ContextCandidate(**base)


def _select(candidates, budget, focused: FocusedSelection):
    return select_within_budget(
        candidates, ContextBudget(max_tokens=budget), focused=focused
    )


def test_oversized_required_recovered_via_focus() -> None:
    cand = _candidate()
    assert cand.estimated_tokens > 120
    events: list[dict] = []
    selected, excluded = _select(
        [cand], 120, FocusedSelection(on_event=events.append)
    )
    assert selected, "oversized candidate should be substituted, not dropped"
    assert all(c.is_focused for c in selected)
    assert all(c.path == cand.path for c in selected)
    assert not any(c.path == cand.path and not c.is_focused for c in selected)
    assert [e.path.as_posix() for e in excluded] == ["app/big.py"]
    assert excluded[0].reason == "exceeds_total_budget"
    assert events and all(e["path"] == "app/big.py" for e in events)


def test_focused_substitution_never_exceeds_budget() -> None:
    cand = _candidate()
    selected, _ = _select([cand], 120, FocusedSelection())
    assert sum(c.estimated_tokens for c in selected) <= 120
    mixed = [
        _candidate("app/a.py", _big_source()),
        _candidate("app/b.py", _big_source()),
        _candidate("app/c.py", _big_source()),
    ]
    selected, _ = _select(mixed, 4096, FocusedSelection())
    assert sum(c.estimated_tokens for c in selected) <= 4096


def test_evidence_prefers_required_symbol_over_fallback() -> None:
    cand = _candidate(source=_source_for("alpha", "beta", "gamma") + _big_source(200))
    focused = FocusedSelection(evidence={cand.path: ["gamma"]})
    selected, _ = _select([cand], 120, focused)
    assert [c.focus_name for c in selected] == ["gamma"]
    assert "sym_0" not in [c.focus_name for c in selected]


def test_fallback_is_bounded_by_default() -> None:
    cand = _candidate()
    events: list[dict] = []
    selected, _ = _select(
        [cand], 120, FocusedSelection(on_event=events.append)
    )
    assert len(events) == DEFAULT_FALLBACK_LIMIT
    assert len(selected) <= DEFAULT_FALLBACK_LIMIT


def test_evidence_is_bounded_by_max_focused() -> None:
    cand = _candidate()
    names = [f"sym_{i}" for i in range(10)]
    events: list[dict] = []
    selected, _ = _select(
        [cand], 120, FocusedSelection(
            evidence={cand.path: names},
            max_focused=4,
            on_event=events.append,
        )
    )
    assert len(events) == 4
    assert len(selected) == 4
    assert max(DEFAULT_MAX_FOCUSED, 4) >= 4


def test_candidate_symbol_acts_as_evidence() -> None:
    cand = _candidate(
        source=_source_for("ContextEngine", "helper") + _big_source(50),
        symbol="ContextEngine",
    )
    selected, _ = _select([cand], 120, FocusedSelection())
    assert [c.focus_name for c in selected] == ["ContextEngine"]


def test_fitting_candidates_unchanged_by_focused_provider() -> None:
    small = _candidate("app/tiny.py", "x = 1\n")
    base = _select([small], 120, None)
    with_focus = _select([small], 120, FocusedSelection())
    assert base == with_focus
    assert with_focus[0][0].is_focused is False


def test_nested_overlapping_spans_are_not_both_selected() -> None:
    src = (
        "def big_function():\n"
        + "    compiling = 1\n" * 120
        + "class Service:\n"
        "    def run(self):\n"
        "        return 1\n"
    )
    cand = _candidate(source=src)
    assert cand.estimated_tokens > 120
    focused = FocusedSelection(evidence={cand.path: ["Service", "run"]})
    selected, _ = _select([cand], 60, focused)
    names = [c.focus_name for c in selected]
    assert names == ["Service"], "nested method must be deduped against its class"


def test_deterministic_repeated_runs() -> None:
    cand = _candidate()
    focus = FocusedSelection(evidence={cand.path: ["sym_7", "sym_3"]})
    a = _select([cand], 120, focus)
    b = _select([cand], 120, focus)
    assert a == b
    events_a: list[dict] = []
    events_b: list[dict] = []
    _select([cand], 120, FocusedSelection(on_event=events_a.append))
    _select([cand], 120, FocusedSelection(on_event=events_b.append))
    assert events_a == events_b


def test_change_plan_provenance_preserved() -> None:
    source = _source_for("ContextEngine", "helper") + _big_source(60)
    cand = _candidate(
        source=source,
        role=CandidateRole.DEPENDENT,
        inclusion_reason=INCLUSION_CHANGE_PLAN,
        module="repolens.context.engine",
        symbol="ContextEngine",
        change_score=0.9,
        change_category="changed",
        change_confidence=0.8,
        change_relationship="direct_callee",
        change_priority=1,
    )
    selected, _ = _select([cand], 120, FocusedSelection())
    item = selected[0]
    assert item.focus_name == "ContextEngine"
    assert item.inclusion_reason == INCLUSION_CHANGE_PLAN
    assert item.module == "repolens.context.engine"
    assert item.symbol == "ContextEngine"
    assert item.change_priority == 1
    assert item.role == CandidateRole.DEPENDENT


def test_architecture_dependency_provenance_preserved() -> None:
    cand = _candidate(
        role=CandidateRole.DEPENDENCY,
        inclusion_reason=INCLUSION_ARCHITECTURE,
        graph_distance=2,
        architecture_rank=1,
        architecture_metadata={"rank": 1},
    )
    focused = FocusedSelection(evidence={cand.path: ["sym_0"]})
    selected, _ = _select([cand], 120, focused)
    item = selected[0]
    assert item.role == CandidateRole.DEPENDENCY
    assert item.graph_distance == 2
    assert item.architecture_rank == 1
    assert item.architecture_metadata == {"rank": 1}


def test_mcp_serialization_round_trip() -> None:
    cand = _candidate()
    selected, excluded = _select([cand], 120, FocusedSelection())
    package = ContextPackage(
        query="q",
        budget=ContextBudget(max_tokens=120),
        selected_files=tuple(selected),
        primary_candidates=tuple(selected),
        dependency_candidates=(),
        excluded_candidates=tuple(excluded),
        matched_symbols=("sym_0",),
    )
    data = package.to_dict()
    entry = data["selected_files"][0]
    assert entry["path"] == "app/big.py"
    assert entry["focus_name"] == "sym_0"
    assert entry["focus_start_line"] == 1
    assert entry["focus_end_line"] == 3
    assert data["excluded_candidates"][0]["reason"] == "exceeds_total_budget"
    assert json.loads(package.to_json())["selected_files"][0]["focus_name"] == "sym_0"


def test_firewall_allows_focused_entries() -> None:
    cand = _candidate()
    selected, _ = _select([cand], 120, FocusedSelection())
    package = ContextPackage(
        query="q",
        budget=ContextBudget(max_tokens=120),
        selected_files=tuple(selected),
        primary_candidates=tuple(selected),
    )
    result = FirewallResult(
        safe=True,
        allowed=("app/big.py",),
        redacted=(),
        blocked=(),
        findings=(),
        firewall_enabled=True,
        policy_version="v1",
    )
    safe = ContextFirewall().safe_package(package, result)
    focused = [
        (c.path, c.focus_name) for c in safe.safe_files
    ]
    assert ("app/big.py", "sym_0") in focused
    assert len(focused) == len(selected)
    assert not safe.blocked_files


def test_focus_events_flow_into_trace_collector() -> None:
    collector = TraceCollector()
    cand = _candidate()
    _focused_events(collector, cand)
    trace = collector.trace()
    assert trace.focus, "focus events must land on PipelineTrace.focus"
    assert all(e["path"] == "app/big.py" for e in trace.focus)
    assert trace.focus[0]["focus_reason"] == "fallback"
    assert trace.focus[0]["full_rejection_reason"] == "exceeds_total_budget"


def _focused_events(collector: TraceCollector, cand: ContextCandidate) -> None:
    from repolens.context_trace import observe_focus

    events: list[dict] = []
    select_within_budget(
        [cand], ContextBudget(max_tokens=120),
        focused=FocusedSelection(on_event=events.append),
    )
    observe_focus(collector, events)


class _Result:
    def __init__(self, path: Path, score: float = 0.5) -> None:
        self.file_path = path
        self.score = score


class _FixedSearcher:
    def __init__(self, results: list[Path]) -> None:
        self._results = results

    def search(self, query: str, limit: int):
        return [_Result(path) for path in self._results[:limit]]


@pytest.fixture
def oversized_repo(tmp_path: Path) -> Path:
    app = tmp_path / "app"
    app.mkdir()
    (app / "big.py").write_text(
        "def target():\n    return 1\n" + _big_source(200),
        encoding="utf-8",
    )
    (app / "tiny.py").write_text("def tiny():\n    return 1\n", encoding="utf-8")
    return tmp_path


def _engine(root: Path, tracer=None, budget: int = 120) -> ContextEngine:
    return ContextEngine(
        root,
        searcher=_FixedSearcher([Path("app/big.py"), Path("app/tiny.py")]),
        dependency=DependencyExpansionConfig(depth=0),
        budget=ContextBudget(max_tokens=budget),
        tracer=tracer,
    )


def test_engine_oversized_required_recovered_via_focus(oversized_repo: Path) -> None:
    collector = TraceCollector()
    package = _engine(oversized_repo, collector).build_context("target function")
    focused = [
        c for c in package.selected_files
        if c.path.as_posix() == "app/big.py"
    ]
    assert focused and all(c.is_focused for c in focused)
    assert any(
        e.path.as_posix() == "app/big.py" and e.reason == "exceeds_total_budget"
        for e in package.excluded_candidates
    )
    assert collector.trace().focus, "engine must record focus events on the trace"


def test_engine_focus_recall_and_trace_analysis(oversized_repo: Path) -> None:
    from repolens.budget_diagnostics import (
        analyze_budget_spend,
    )
    from repolens.required_file_diagnostics import classify_required_files

    collector = TraceCollector()
    package = _engine(oversized_repo, collector).build_context("target function")
    trace = collector.trace()
    report = analyze_budget_spend(
        "task",
        "context",
        trace,
        budget_max_tokens=package.budget.max_tokens,
        required_files=["app/big.py"],
        focus_events=trace.focus,
    )
    big = next(row for row in report.rows if row.path == "app/big.py")
    assert big.selected
    assert big.oversized_total
    # Accounting uses the real spend: the row's tokens sum every focused slice
    # for the path, never just the last one, and match the produced package.
    actual = sum(
        c.estimated_tokens for c in package.selected_files
        if c.path.as_posix() == "app/big.py"
    )
    assert big.estimated_tokens == actual
    assert big.estimated_tokens == big.full_tokens * 0 + actual
    assert report.summary.total_selected_tokens == sum(
        c.estimated_tokens for c in package.selected_files
    )
    assert report.summary.total_selected_tokens <= package.budget.max_tokens
    assert [row.path for row in report.required_focused()] == ["app/big.py"]
    assert report.summary.focused_generated > 0
    assert report.summary.focused_selected > 0
    assert classify_required_files(
        "task",
        "context",
        ["app/big.py"],
        trace,
        budget_max_tokens=package.budget.max_tokens,
    )[0].outcome.name == "SELECTED"


def test_engine_focus_determinism(oversized_repo: Path) -> None:
    traced = _engine(oversized_repo, TraceCollector()).build_context("target function")
    untraced = _engine(oversized_repo).build_context("target function")
    assert traced.to_dict() == untraced.to_dict()