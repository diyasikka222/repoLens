"""Required-file outcome diagnostics tests (P26.1).

Covers the classification contract (four outcomes, deterministic precedence,
tier/score extraction, firewall-decision lookup, report aggregation) plus the
guarantee that attaching a tracer never changes produced packages.
"""

from __future__ import annotations

import unittest
from pathlib import Path

from repolens.context import ContextBudget, ContextEngine
from repolens.context_trace import (
    ContextStage,
    PipelineTrace,
    TraceCollector,
)
from repolens.required_file_diagnostics import (
    RequiredFileDiagnostic,
    RequiredFileDiagnosticsReport,
    RequiredFileOutcome,
    TaskDiagnostics,
    classify_required_files,
    firewall_decisions_for_package,
)
from repolens.search import CodeSearcher

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "change_plan_repository"


def _candidate(path, **overrides) -> dict:
    base = {
        "path": path,
        "role": "primary",
        "estimated_tokens": 200,
        "selection_reason": "matched signal",
        "inclusion_reason": "lexical_match",
        "retrieval_rank": 1,
        "retrieval_score": 0.9,
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


def _excluded(path, reason="over_budget", tokens=200) -> dict:
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


class ClassifyOutcomeTests(unittest.TestCase):
    def test_selected(self) -> None:
        trace = _trace(
            retrieval=({**{"path": "a.py", "retrieval_rank": 1}, "retrieval_score": 0.9, "lexical_rank": 1, "semantic_rank": None, "signals": []},),
            ranked=(_candidate("a.py"),),
            selected=(_candidate("a.py"),),
        )
        rows = classify_required_files(
            "task", "context", ["a.py"], trace, budget_max_tokens=4096
        )
        self.assertEqual(rows[0].outcome, RequiredFileOutcome.SELECTED)
        self.assertTrue(rows[0].is_selected)
        self.assertEqual(rows[0].discovery_source, ContextStage.RETRIEVAL.value)
        self.assertEqual(rows[0].ranking_position, 1)

    def test_ranked_not_selected_reports_budget_reason(self) -> None:
        trace = _trace(
            ranked=(_candidate("b.py", retrieval_rank=2),),
            excluded=(_excluded("b.py", reason="over_budget"),),
        )
        row = classify_required_files(
            "task", "context", ["b.py"], trace, budget_max_tokens=4096
        )[0]
        self.assertEqual(
            row.outcome, RequiredFileOutcome.RANKED_NOT_SELECTED
        )
        self.assertEqual(row.excluded_reason, "over_budget")
        self.assertEqual(row.ranking_position, 1)
        self.assertEqual(row.estimated_tokens, 200)

    def test_discovered_not_ranked(self) -> None:
        trace = _trace(symbols=("c.py",))
        row = classify_required_files(
            "task", "context", ["c.py"], trace, budget_max_tokens=4096
        )[0]
        self.assertEqual(
            row.outcome, RequiredFileOutcome.DISCOVERED_NOT_RANKED
        )
        self.assertEqual(row.discovery_source, ContextStage.SYMBOL.value)
        self.assertTrue(row.from_symbol)
        self.assertFalse(row.from_retrieval)

    def test_not_discovered(self) -> None:
        trace = _trace()
        row = classify_required_files(
            "task", "context", ["nope.py"], trace, budget_max_tokens=4096
        )[0]
        self.assertEqual(row.outcome, RequiredFileOutcome.NOT_DISCOVERED)
        self.assertIsNone(row.discovery_source)
        self.assertIsNone(row.ranking_tier)

    def test_discovery_precedence_retrieval_first(self) -> None:
        trace = _trace(
            retrieval=({"path": "d.py", "retrieval_rank": 1, "retrieval_score": 0.5, "lexical_rank": 1, "semantic_rank": None, "signals": []},),
            dependency=({"path": "d.py", "role": "dependency", "distance": 1},),
            change_plan=(_candidate("d.py", inclusion_reason="change_plan"),),
            ranked=(_candidate("d.py", inclusion_reason="change_plan"),),
            selected=(_candidate("d.py", inclusion_reason="change_plan"),),
            has_change_plan=True,
        )
        row = classify_required_files(
            "task", "change_context", ["d.py"], trace, budget_max_tokens=4096
        )[0]
        self.assertEqual(row.discovery_source, ContextStage.RETRIEVAL.value)
        self.assertTrue(row.from_retrieval)
        self.assertTrue(row.from_dependency)
        self.assertTrue(row.from_change_plan)
        self.assertEqual(row.outcome, RequiredFileOutcome.SELECTED)

    def test_precedence_symbols_beat_change_plan(self) -> None:
        trace = _trace(
            symbols=("e.py",),
            change_plan=(_candidate("e.py", inclusion_reason="change_plan"),),
            ranked=(_candidate("e.py", inclusion_reason="change_plan"),),
        )
        row = classify_required_files(
            "task", "change_context", ["e.py"], trace, budget_max_tokens=4096
        )[0]
        self.assertEqual(row.discovery_source, ContextStage.SYMBOL.value)

    def test_tier_and_score_extraction(self) -> None:
        trace = _trace(
            ranked=(_candidate(
                "f.py",
                role="dependency",
                inclusion_reason="change_plan",
                change_score=7,
                retrieval_rank=None,
                retrieval_score=None,
            ),),
        )
        row = classify_required_files(
            "task", "change_context", ["f.py"], trace, budget_max_tokens=4096
        )[0]
        self.assertEqual(row.ranking_tier, 5)
        self.assertEqual(row.ranking_tier_label, "change_plan")
        self.assertEqual(row.candidate_score, 7.0)
        self.assertEqual(row.inclusion_reason, "change_plan")

    def test_architecture_tier_bucket(self) -> None:
        trace = _trace(
            ranked=(_candidate("g.py", role="dependency", architecture_rank=2),),
        )
        row = classify_required_files(
            "task", "context", ["g.py"], trace, budget_max_tokens=4096
        )[0]
        self.assertEqual(row.ranking_tier, 3)
        self.assertEqual(row.ranking_tier_label, "architecture_proximity")


class FirewallDecisionTests(unittest.TestCase):
    def test_decisions_from_safe_package(self) -> None:
        class _C:
            def __init__(self, path, decision=None):
                self.path = Path(path)
                self.decision = decision

        class _Safe:
            safe_files = (_C("app/a.py", "allowed"), _C("app/b.py", "redact"))
            blocked_files = (_C("app/secret.py"),)

        decisions = firewall_decisions_for_package(_Safe())
        self.assertEqual(decisions["app/a.py"], "allowed")
        self.assertEqual(decisions["app/b.py"], "redact")
        self.assertEqual(decisions["app/secret.py"], "blocked")

    def test_none_package_is_empty(self) -> None:
        self.assertEqual(firewall_decisions_for_package(None), {})


class ReportTests(unittest.TestCase):
    def _diag(self, task_id, strategy, path, outcome) -> RequiredFileDiagnostic:
        return RequiredFileDiagnostic(
            task_id=task_id,
            strategy=strategy,
            path=path,
            outcome=outcome,
            discovery_source=None,
            inclusion_reason=None,
            candidate_score=None,
            ranking_position=None,
            ranking_tier=None,
            ranking_tier_label=None,
            graph_distance=None,
            estimated_tokens=None,
            budget_max_tokens=4096,
            excluded_reason=None,
            firewall_decision=None,
        )

    def test_report_counts_and_aggregation(self) -> None:
        report = RequiredFileDiagnosticsReport(
            tasks=(
                TaskDiagnostics(
                    "t1",
                    "context",
                    4096,
                    (
                        self._diag("t1", "context", "a.py", RequiredFileOutcome.SELECTED),
                        self._diag("t1", "context", "b.py", RequiredFileOutcome.RANKED_NOT_SELECTED),
                    ),
                ),
                TaskDiagnostics(
                    "t2",
                    "context",
                    4096,
                    (
                        self._diag("t2", "context", "c.py", RequiredFileOutcome.NOT_DISCOVERED),
                        self._diag("t2", "context", "d.py", RequiredFileOutcome.RANKED_NOT_SELECTED),
                    ),
                ),
            )
        )
        counts = report.outcome_counts()
        self.assertEqual(counts["SELECTED"], 1)
        self.assertEqual(counts["RANKED_NOT_SELECTED"], 2)
        self.assertEqual(counts["NOT_DISCOVERED"], 1)
        self.assertEqual(counts["DISCOVERED_NOT_RANKED"], 0)
        self.assertEqual(report.for_strategy("context").__len__(), 2)
        self.assertEqual(report.strategies, ("context",))

    def test_recurrences_are_deterministic_and_ordered(self) -> None:
        report = RequiredFileDiagnosticsReport(
            tasks=(
                TaskDiagnostics(
                    "t1",
                    "context",
                    4096,
                    (
                        self._diag("t1", "context", "a.py", RequiredFileOutcome.RANKED_NOT_SELECTED),
                        self._diag("t1", "context", "b.py", RequiredFileOutcome.RANKED_NOT_SELECTED),
                    ),
                ),
                TaskDiagnostics(
                    "t2",
                    "context",
                    4096,
                    (self._diag("t2", "context", "c.py", RequiredFileOutcome.RANKED_NOT_SELECTED),),
                ),
            )
        )
        first = report.recurrences()
        second = report.recurrences()
        self.assertEqual(first, second)
        self.assertEqual(first[0]["count"], 3)
        self.assertEqual(first[0]["outcome"], "RANKED_NOT_SELECTED")
        self.assertEqual(sorted(first[0]["task_ids"]), ["t1", "t2"])


class EngineIntegrationTests(unittest.TestCase):
    def _engine(self, tracer=None) -> ContextEngine:
        return ContextEngine(
            FIXTURE,
            searcher=CodeSearcher(FIXTURE),
            budget=ContextBudget(max_tokens=4096),
            tracer=tracer,
        )

    def test_tracer_option_does_not_change_packages(self) -> None:
        plain = self._engine(None).build_context("checkout totals")
        traced = self._engine(TraceCollector()).build_context("checkout totals")
        self.assertEqual(plain.to_dict(), traced.to_dict())

    def test_classification_consistent_with_package(self) -> None:
        query = "checkout totals"
        engine = self._engine(None)
        package = engine.build_context(query)
        selected_paths = {str(c.path) for c in package.selected_files}
        self.assertTrue(selected_paths, "expected at least one selected file")

        collector = TraceCollector()
        traced_engine = self._engine(collector)
        traced_engine.build_context(query)
        trace = collector.trace()
        self.assertEqual(trace.selected_paths, frozenset(selected_paths))
        self.assertEqual(trace.ranked_paths, frozenset(trace.selected_paths) | frozenset(e["path"] for e in trace.excluded))
        self.assertTrue(trace.retrieval)
        self.assertTrue(trace.ranked)
        self.assertEqual(trace.has_change_plan, False)

        required = tuple(sorted(selected_paths))
        rows = classify_required_files(
            "checkout", "context", required, trace, budget_max_tokens=4096
        )
        self.assertTrue(all(row.outcome is RequiredFileOutcome.SELECTED for row in rows))
        self.assertEqual(rows[0].budget_max_tokens, 4096)

    def test_change_path_emits_change_plan_stage(self) -> None:
        collector = TraceCollector()
        engine = self._engine(collector)
        engine.build_context(
            "checkout totals",
            change_request="make checkout reject empty carts",
            change_target="app/api/checkout.py",
        )
        trace = collector.trace()
        self.assertTrue(trace.has_change_plan)
        self.assertEqual(trace.change_plan[0]["path"].__class__.__name__, "str")
        discovered = trace.discovered_paths()
        self.assertTrue(discovered)


if __name__ == "__main__":
    unittest.main()