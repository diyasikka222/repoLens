"""Budget-aware context selection diagnostics tests (P26.2).

Covers the deterministic budget-spend analysis contract: cumulative/remaining
accounting, oversized detection, exclusion categorization (exceeds_total_budget,
remaining_budget_too_small, firewall_filtered, duplicate, other),
required/supporting/test tagging, required-file loss reporting with competing
selections, report aggregation, and the guarantee that the analysis (and the
tracer it reads) never changes produced packages.
"""

from __future__ import annotations

import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path

from repolens.budget_diagnostics import (
    BudgetDiagnosticsReport,
    ExclusionCategory,
    SurfaceTag,
    analyze_budget_spend,
    categorize_reason,
    surface_tag_for_path,
)
from repolens.context import ContextBudget, ContextEngine, DependencyExpansionConfig
from repolens.context_trace import PipelineTrace, TraceCollector


def _candidate(path: str, tokens: int = 100, score: float | None = 0.5, **overrides) -> dict:
    base = {
        "path": path,
        "role": "primary",
        "estimated_tokens": tokens,
        "selection_reason": "matched signal",
        "inclusion_reason": "lexical_match",
        "retrieval_rank": 1,
        "retrieval_score": score,
        "lexical_rank": 1,
        "semantic_rank": None,
        "graph_distance": None,
        "architecture_rank": None,
        "module": None,
        "symbol": None,
        "change_score": None,
        "change_category": None,
        "change_confidence": None,
        "change_relationship": None,
        "change_priority": None,
    }
    base.update(overrides)
    return base


def _excluded(path: str, reason: str = "over_budget", tokens: int = 100) -> dict:
    return {"path": path, "estimated_tokens": tokens, "reason": reason}


def _trace(**overrides) -> PipelineTrace:
    defaults: dict = {
        "retrieval": (),
        "symbols": (),
        "dependency": (),
        "architecture": (),
        "change_plan": (),
        "ranked": (),
        "selected": (),
        "excluded": (),
        "has_change_plan": False,
    }
    defaults.update(overrides)
    return PipelineTrace(**defaults)


def _analyze(trace: PipelineTrace, *, budget: int | None = 1000, **surface):
    return analyze_budget_spend(
        "task",
        "context",
        trace,
        budget_max_tokens=budget,
        required_files=surface.get("required_files", ()),
        supporting_files=surface.get("supporting_files", ()),
        expected_tests=surface.get("expected_tests", ()),
        firewall_decisions=surface.get("firewall_decisions"),
    )


class CategorizeAndTagTests(unittest.TestCase):
    def test_categorize_reason_mappings(self) -> None:
        self.assertEqual(
            categorize_reason("exceeds_total_budget"),
            ExclusionCategory.EXCEEDS_TOTAL_BUDGET,
        )
        self.assertEqual(
            categorize_reason("over_budget"),
            ExclusionCategory.REMAINING_BUDGET_TOO_SMALL,
        )
        self.assertEqual(categorize_reason("duplicate"), ExclusionCategory.DUPLICATE)
        self.assertEqual(
            categorize_reason("some_future_reason"), ExclusionCategory.OTHER
        )
        self.assertIsNone(categorize_reason(None))

    def test_surface_tag_precedence(self) -> None:
        # required wins over test (surfaces may legally overlap, e.g. a
        # test-change task where the required file is also the expected test).
        self.assertEqual(
            surface_tag_for_path(
                "a.py", ["a.py"], ["b.py"], ["a.py"]
            ),
            SurfaceTag.REQUIRED,
        )
        self.assertEqual(
            surface_tag_for_path("b.py", ["a.py"], ["b.py"], ["a.py"]),
            SurfaceTag.SUPPORTING,
        )
        self.assertEqual(
            surface_tag_for_path("t.py", ["a.py"], [], ["t.py"]),
            SurfaceTag.TEST,
        )
        self.assertEqual(
            surface_tag_for_path("other.py", ["a.py"], ["b.py"], ["t.py"]),
            SurfaceTag.OTHER,
        )


class BudgetSpendAccountingTests(unittest.TestCase):
    def test_cumulative_and_remaining_accounting(self) -> None:
        trace = _trace(
            ranked=(
                _candidate("a.py", tokens=300, score=0.9),
                _candidate("b.py", tokens=200, score=0.8),
                _candidate("c.py", tokens=400, score=0.7),
                _candidate("d.py", tokens=200, score=0.6),
            ),
            selected=(
                _candidate("a.py", tokens=300, score=0.9),
                _candidate("b.py", tokens=200, score=0.8),
                _candidate("c.py", tokens=400, score=0.7),
            ),
            excluded=(_excluded("d.py", reason="over_budget", tokens=200),),
        )
        report = _analyze(trace, required_files=["d.py"])
        rows = {row.path: row for row in report.rows}
        self.assertEqual(rows["a.py"].cumulative_tokens_before, 0)
        self.assertEqual(rows["a.py"].cumulative_tokens_after, 300)
        self.assertEqual(rows["a.py"].remaining_after, 700)
        self.assertEqual(rows["b.py"].cumulative_tokens_before, 300)
        self.assertEqual(rows["b.py"].cumulative_tokens_after, 500)
        self.assertEqual(rows["c.py"].remaining_before, 500)
        self.assertEqual(rows["c.py"].cumulative_tokens_after, 900)
        self.assertEqual(rows["c.py"].remaining_after, 100)
        d = rows["d.py"]
        self.assertFalse(d.selected)
        self.assertEqual(d.remaining_before, 100)
        self.assertTrue(d.oversized_remaining)
        self.assertFalse(d.oversized_total)
        self.assertEqual(
            d.exclusion_category, ExclusionCategory.REMAINING_BUDGET_TOO_SMALL
        )
        self.assertEqual(d.surface_tag, SurfaceTag.REQUIRED)

        summary = report.summary
        self.assertEqual(summary.total_selected_tokens, 900)
        self.assertEqual(summary.budget_utilization, 0.9)
        self.assertEqual(summary.unused_budget, 100)
        self.assertEqual(summary.required_excluded_by_consumption, 1)
        self.assertEqual(summary.required_excluded_by_size, 0)

    def test_exceeds_total_budget_and_walk_continues(self) -> None:
        trace = _trace(
            ranked=(
                _candidate("big.py", tokens=1200),
                _candidate("small.py", tokens=500),
            ),
            selected=(_candidate("small.py", tokens=500),),
            excluded=(_excluded("big.py", reason="exceeds_total_budget", tokens=1200),),
        )
        report = _analyze(trace, budget=1000, required_files=["big.py"])
        big = next(row for row in report.rows if row.path == "big.py")
        self.assertFalse(big.selected)
        self.assertTrue(big.oversized_total)
        self.assertTrue(big.oversized_remaining)
        self.assertEqual(
            big.exclusion_category, ExclusionCategory.EXCEEDS_TOTAL_BUDGET
        )
        small = next(row for row in report.rows if row.path == "small.py")
        self.assertTrue(small.selected)
        self.assertEqual(small.cumulative_tokens_after, 500)
        self.assertEqual(small.remaining_after, 500)
        self.assertEqual(report.summary.required_excluded_by_size, 1)
        self.assertEqual(report.summary.oversized_total_count, 1)

    def test_unlimited_budget_selects_everything(self) -> None:
        trace = _trace(
            ranked=(_candidate("a.py", tokens=300), _candidate("b.py", tokens=400)),
            selected=(
                _candidate("a.py", tokens=300),
                _candidate("b.py", tokens=400),
            ),
            excluded=(),
        )
        report = _analyze(trace, budget=None)
        self.assertTrue(all(row.selected for row in report.rows))
        self.assertIsNone(report.summary.budget_utilization)
        self.assertIsNone(report.summary.unused_budget)
        self.assertIsNone(report.rows[0].remaining_before)
        self.assertEqual(report.summary.oversized_total_count, 0)

    def test_firewall_blocked_selected_file_is_a_rejection(self) -> None:
        trace = _trace(
            ranked=(_candidate("vault.py", tokens=50),),
            selected=(_candidate("vault.py", tokens=50),),
            excluded=(),
        )
        report = _analyze(
            trace,
            required_files=["vault.py"],
            firewall_decisions={"vault.py": "blocked"},
        )
        row = report.rows[0]
        self.assertTrue(row.selected)
        self.assertEqual(
            row.exclusion_category, ExclusionCategory.FIREWALL_FILTERED
        )
        self.assertEqual(row.firewall_decision, "blocked")
        self.assertEqual(report.summary.rejected_count, 1)
        self.assertEqual([r.path for r in report.required_missing()], ["vault.py"])

    def test_duplicate_category_is_representable_but_unreachable(self) -> None:
        # Dedupe runs before ranking, so no ranked candidate can be a duplicate.
        trace = _trace(
            ranked=(_candidate("a.py", tokens=50),),
            selected=(_candidate("a.py", tokens=50),),
            excluded=(),
        )
        report = _analyze(trace)
        self.assertTrue(report.rows[0].selected)
        self.assertIsNone(report.rows[0].exclusion_category)
        self.assertEqual(report.summary.rejected_count, 0)
        self.assertEqual(
            categorize_reason("duplicate"), ExclusionCategory.DUPLICATE
        )


class RequiredLossAndCompetitionTests(unittest.TestCase):
    def _loss_trace(self):
        return _trace(
            ranked=(
                _candidate("keep.py", tokens=600, score=0.9),
                _candidate("big.py", tokens=500, score=0.8),
                _candidate("late.py", tokens=200, score=0.7),
            ),
            selected=(_candidate("keep.py", tokens=600, score=0.9),),
            excluded=(
                _excluded("big.py", reason="over_budget", tokens=500),
                _excluded("late.py", reason="over_budget", tokens=200),
            ),
        )

    def test_required_lost_reports_rank_score_tokens_and_competition(self) -> None:
        trace = self._loss_trace()
        report = _analyze(trace, budget=800, required_files=["big.py", "late.py"])
        big = next(row for row in report.rows if row.path == "big.py")
        late = next(row for row in report.rows if row.path == "late.py")
        self.assertEqual(big.rank, 2)
        self.assertEqual(big.score, 0.8)
        self.assertEqual(big.estimated_tokens, 500)
        self.assertEqual(report.competing_selected(big), ("keep.py",))
        self.assertEqual(report.competing_selected(late), ("keep.py",))
        missing = report.required_missing()
        self.assertEqual([row.path for row in missing], ["big.py", "late.py"])
        self.assertEqual(
            [row.exclusion_category for row in missing],
            [
                ExclusionCategory.REMAINING_BUDGET_TOO_SMALL,
                ExclusionCategory.REMAINING_BUDGET_TOO_SMALL,
            ],
        )

    def test_top_required_lost_and_oversized_rollup(self) -> None:
        r1 = _analyze(self._loss_trace(), budget=800, required_files=["big.py"])
        over_trace = _trace(
            ranked=(_candidate("huge.py", tokens=9000),),
            excluded=(_excluded("huge.py", reason="exceeds_total_budget", tokens=9000),),
        )
        r2 = _analyze(over_trace, budget=1000, required_files=["huge.py"])
        report = BudgetDiagnosticsReport(reports=(r1, r2))
        top_lost = report.top_required_lost_to_budget()
        self.assertEqual(top_lost[0]["path"], "huge.py")
        self.assertEqual(top_lost[0]["by_size"], 1)
        self.assertEqual(top_lost[0]["by_consumption"], 0)
        self.assertEqual(top_lost[1]["path"], "big.py")
        self.assertEqual(top_lost[1]["by_consumption"], 1)
        oversized = report.top_oversized_files()
        self.assertEqual([entry["path"] for entry in oversized], ["huge.py"])
        self.assertEqual(oversized[0]["max_tokens"], 9000)


class AggregationAndDeterminismTests(unittest.TestCase):
    def _sample_reports(self):
        r1 = _analyze(
            _trace(
                ranked=(_candidate("a.py", tokens=300, score=0.9),),
                selected=(_candidate("a.py", tokens=300, score=0.9),),
            ),
            budget=1000,
            required_files=["a.py"],
        )
        r2 = _analyze(
            _trace(
                ranked=(
                    _candidate("b.py", tokens=700, score=0.8),
                    _candidate("c.py", tokens=700, score=0.6),
                ),
                selected=(_candidate("b.py", tokens=700, score=0.8),),
                excluded=(_excluded("c.py", reason="over_budget", tokens=700),),
            ),
            budget=1000,
            required_files=["c.py"],
        )
        return r1, r2

    def test_aggregate_rollup(self) -> None:
        r1, r2 = self._sample_reports()
        report = BudgetDiagnosticsReport(reports=(r1, r2))
        agg = report.aggregate()
        self.assertEqual(agg.total_selected_tokens, 1000)
        # Mean per-package utilization: 0.3 (r1) and 0.7 (r2).
        self.assertEqual(agg.budget_utilization, 0.5)
        # Total unused across both packages: 700 + 300.
        self.assertEqual(agg.unused_budget, 1000)
        self.assertEqual(agg.avg_selected_tokens, 500.0)
        self.assertEqual(agg.median_selected_tokens, 500.0)
        # a.py rank 1, c.py rank 2 -> mean 1.5, median 1.5.
        self.assertEqual(agg.avg_required_rank, 1.5)
        self.assertEqual(agg.median_required_rank, 1.5)
        self.assertEqual(agg.required_excluded_by_consumption, 1)
        categories = dict(agg.excluded_by_category)
        self.assertEqual(categories["remaining_budget_too_small"], 1)

    def test_analyze_is_deterministic(self) -> None:
        r1, r2 = self._sample_reports()
        report_a = BudgetDiagnosticsReport(reports=(r1, r2))
        report_b = BudgetDiagnosticsReport(reports=(r1, r2))
        self.assertEqual(report_a.to_dict(), report_b.to_dict())
        # Double analysis of the same trace yields identical rows.
        trace = _trace(
            ranked=(
                _candidate("b.py", tokens=700, score=0.8),
                _candidate("c.py", tokens=700, score=0.6),
            ),
            selected=(_candidate("b.py", tokens=700, score=0.8),),
            excluded=(_excluded("c.py", reason="over_budget", tokens=700),),
        )
        again_a = _analyze(trace, budget=1000, required_files=["c.py"])
        again_b = _analyze(trace, budget=1000, required_files=["c.py"])
        self.assertEqual(again_a.to_dict(), again_b.to_dict())
        self.assertEqual(r2.to_dict(), again_a.to_dict())


@dataclass(frozen=True)
class _Result:
    file_path: Path
    score: float = 0.5


class _FixedSearcher:
    def __init__(self, results: list[Path]) -> None:
        self._results = results

    def search(self, query: str, limit: int):
        return [_Result(path, score=0.5) for path in self._results[:limit]]


class EngineIntegrationTests(unittest.TestCase):
    """Whole-pipeline checks over a synthetic repo (traced engine)."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name) / "repo"
        self.root.mkdir()
        (self.root / "app").mkdir()
        (self.root / "app" / "alpha.py").write_text(
            "def alpha():\n    return 'x'\n", encoding="utf-8"
        )
        # ~1240 chars -> ~310 estimated tokens, deliberately oversized for a
        # 100-token budget.
        (self.root / "app" / "beta.py").write_text(
            "# huge\n" + "x" * 1200, encoding="utf-8"
        )

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def _engine(self, budget: int, tracer=None) -> ContextEngine:
        return ContextEngine(
            self.root,
            searcher=_FixedSearcher([Path("app/alpha.py")]),
            dependency=DependencyExpansionConfig(depth=0),
            budget=ContextBudget(max_tokens=budget),
            tracer=tracer,
        )

    def test_selected_required_file_accounting_via_engine(self) -> None:
        collector = TraceCollector()
        package = self._engine(100, collector).build_context("alpha function")
        report = analyze_budget_spend(
            "task",
            "context",
            collector.trace(),
            budget_max_tokens=package.budget.max_tokens,
            required_files=["app/alpha.py"],
        )
        alpha = next(row for row in report.rows if row.path == "app/alpha.py")
        self.assertTrue(alpha.selected)
        self.assertEqual(alpha.surface_tag, SurfaceTag.REQUIRED)
        # alpha.py selected first; beta.py is oversized for the remaining budget.
        self.assertEqual(report.rows[0].path, "app/alpha.py")

    def test_tracer_does_not_change_produced_package(self) -> None:
        traced = self._engine(100, TraceCollector()).build_context("alpha function")
        untraced = self._engine(100).build_context("alpha function")
        self.assertEqual(traced.to_dict(), untraced.to_dict())

    def test_oversized_file_excluded_via_engine(self) -> None:
        # Force beta.py into the pipeline as a retrieved candidate so the
        # oversized path is exercised end to end.
        collector = TraceCollector()
        engine = ContextEngine(
            self.root,
            searcher=_FixedSearcher(
                [Path("app/alpha.py"), Path("app/beta.py")]
            ),
            dependency=DependencyExpansionConfig(depth=0),
            budget=ContextBudget(max_tokens=100),
            tracer=collector,
        )
        package = engine.build_context("alpha or beta module internals")
        report = analyze_budget_spend(
            "task",
            "context",
            collector.trace(),
            budget_max_tokens=package.budget.max_tokens,
        )
        beta = next(row for row in report.rows if row.path == "app/beta.py")
        self.assertTrue(beta.oversized_total)
        self.assertTrue(beta.oversized_remaining)
        self.assertFalse(beta.selected)
        self.assertIsNotNone(beta.exclusion_reason)
        self.assertEqual(report.summary.oversized_remaining_count, 1)


if __name__ == "__main__":
    unittest.main()