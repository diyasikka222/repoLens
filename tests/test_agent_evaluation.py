"""Unit and integration tests for deterministic agent-task evaluation
(repolens/agent_evaluation.py, P25.2–P25.3).

Unit tests cover:

- the P25.2 metric math and validation of the pure (task, candidate set) →
  result pipeline: perfect/partial/zero-relevant/irrelevant-only/duplicate/
  empty-surface cases, determinism, JSON serialization, bounded results;
- the P25.3 graded context-usefulness model: required vs supporting vs test
  coverage, tokens per required file, empty/missing/irrelevant/duplicate
  surfaces, and backward compatibility of legacy task construction.

Integration tests exercise the benchmark's task corpus, producer wiring and
diagnostics against the real repository (lexical search on a subset).
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import repolens.agent_evaluation as agent_eval
from repolens.search import CodeSearcher

REPO_ROOT = Path(__file__).resolve().parent.parent


def _task(
    task_id: str = "t1",
    files: tuple[str, ...] = ("repolens/search.py",),
    **overrides,
) -> agent_eval.EvaluationTask:
    kwargs = {
        "id": task_id,
        "title": "Example task",
        "request": "change retrieval behavior",
        "expected_relevant_files": files,
    }
    kwargs.update(overrides)
    return agent_eval.EvaluationTask(**kwargs)


def _candidate(
    files: tuple[str, ...],
    *,
    strategy: str = "lexical",
    matched: tuple[str, ...] = (),
    context_size: int | None = 1200,
    selected: int | None = None,
    latency: float | None = 0.01,
) -> agent_eval.CandidateSet:
    return agent_eval.CandidateSet(
        strategy=strategy,
        retrieved_files=files,
        matched_symbols=matched,
        context_size_tokens=context_size,
        selected_file_count=len(files) if selected is None else selected,
        latency_seconds=latency,
    )


class EvaluationTaskTests(unittest.TestCase):
    def test_validates_identity_and_request(self) -> None:
        for kwargs in (
            {"id": " "},
            {"title": ""},
            {"request": None},
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    _task(**kwargs)

    def test_rejects_unknown_category(self) -> None:
        with self.assertRaises(ValueError):
            _task(category="unsupported")

    def test_normalizes_expected_files(self) -> None:
        task = _task(
            files=("repolens/search.py", "repolens/search.py", "repolens/retrieval.py"),
            target_files=(Path("repolens/context/engine.py"),),
        )
        self.assertEqual(
            task.expected_relevant_files,
            ("repolens/search.py", "repolens/retrieval.py"),
        )
        self.assertEqual(task.target_files, ("repolens/context/engine.py",))

    def test_rejects_non_relative_paths(self) -> None:
        for bad in ("/abs/path.py", "../escape.py", "a/../b.py", "a//b.py", ""):
            with self.subTest(bad=bad):
                with self.assertRaises(ValueError):
                    _task(files=(bad,))

    def test_path_error_describes_posix(self) -> None:
        with self.assertRaises(ValueError) as raised:
            _task(files=("a/../b.py",))
        self.assertIn("repository-relative POSIX paths", str(raised.exception))

    def test_accepts_all_categories(self) -> None:
        for category in agent_eval.CATEGORIES:
            task = _task(category=category)
            self.assertEqual(task.category, category)
        self.assertIn("bug_fix", agent_eval.CATEGORIES)
        self.assertIn("feature_change", agent_eval.CATEGORIES)
        self.assertIn("refactor", agent_eval.CATEGORIES)
        self.assertIn("cross_file_change", agent_eval.CATEGORIES)
        self.assertIn("test_change", agent_eval.CATEGORIES)


class CandidateSetTests(unittest.TestCase):
    def test_rejects_blank_strategy_or_paths(self) -> None:
        with self.assertRaises(ValueError):
            _candidate((), strategy=" ")
        with self.assertRaises(ValueError):
            _candidate(("",))

    def test_rejects_negative_measurements(self) -> None:
        with self.assertRaises(ValueError):
            _candidate((), context_size=-1)
        with self.assertRaises(ValueError):
            _candidate((), selected=-1)
        with self.assertRaises(ValueError):
            _candidate((), latency=-0.1)

    def test_dedupes_and_preserves_order(self) -> None:
        candidate = _candidate(("a.py", "b.py", "a.py"))
        self.assertEqual(candidate.retrieved_files, ("a.py", "b.py"))


class RunnerMetricTests(unittest.TestCase):
    def setUp(self) -> None:
        self.runner = agent_eval.AgentEvaluationRunner()

    def test_perfect_retrieval(self) -> None:
        task = _task(files=("repolens/search.py",))
        result = self.runner.evaluate(task, _candidate(("repolens/search.py",)))
        self.assertAlmostEqual(result.precision, 1.0)
        self.assertAlmostEqual(result.recall, 1.0)
        self.assertAlmostEqual(result.f1, 1.0)
        self.assertEqual(result.relevant_files_missed, ())
        self.assertEqual(result.irrelevant_files_retrieved, ())
        self.assertEqual(result.relevant_files_retrieved, ("repolens/search.py",))

    def test_partial_retrieval(self) -> None:
        task = _task(files=("a.py", "b.py"))
        result = self.runner.evaluate(task, _candidate(("a.py", "c.py")))
        self.assertAlmostEqual(result.precision, 0.5)
        self.assertAlmostEqual(result.recall, 0.5)
        self.assertAlmostEqual(result.f1, 0.5)
        self.assertEqual(result.relevant_files_missed, ("b.py",))
        self.assertEqual(result.irrelevant_files_retrieved, ("c.py",))

    def test_zero_relevant_retrieved(self) -> None:
        task = _task(files=("a.py",))
        result = self.runner.evaluate(task, _candidate(("c.py",)))
        self.assertAlmostEqual(result.precision, 0.0)
        self.assertAlmostEqual(result.recall, 0.0)
        self.assertAlmostEqual(result.f1, 0.0)

    def test_irrelevant_only_retrieval(self) -> None:
        task = _task(files=("a.py",))
        result = self.runner.evaluate(task, _candidate(("x.py", "y.py")))
        self.assertAlmostEqual(result.precision, 0.0)
        self.assertEqual(result.irrelevant_files_retrieved, ("x.py", "y.py"))

    def test_empty_candidate_set(self) -> None:
        task = _task(files=("a.py",))
        result = self.runner.evaluate(task, _candidate(()))
        self.assertAlmostEqual(result.precision, 0.0)
        self.assertAlmostEqual(result.recall, 0.0)
        self.assertAlmostEqual(result.f1, 0.0)
        self.assertEqual(result.relevant_files_missed, ("a.py",))

    def test_empty_expected_surface_is_valid(self) -> None:
        task = _task(files=())
        result = self.runner.evaluate(task, _candidate(("a.py",)))
        self.assertEqual(result.recall, 0.0)
        self.assertEqual(result.f1, 0.0)
        self.assertEqual(result.relevant_files_missed, ())

    def test_duplicate_candidates_count_once(self) -> None:
        task = _task(files=("a.py",))
        result = self.runner.evaluate(task, _candidate(("a.py", "a.py", "b.py")))
        self.assertEqual(result.retrieved_files, ("a.py", "b.py"))
        self.assertAlmostEqual(result.precision, 0.5)
        self.assertAlmostEqual(result.recall, 1.0)

    def test_symbol_and_test_recall(self) -> None:
        task = _task(
            files=("repolens/search.py",),
            expected_relevant_symbols=("CodeSearcher", "tokenize"),
            expected_tests=("tests/test_search.py",),
        )
        result = self.runner.evaluate(
            task,
            _candidate(
                ("repolens/search.py", "tests/test_search.py"),
                matched=("tokenize", "other"),
            ),
        )
        self.assertEqual(result.expected_symbols_retrieved, ("tokenize",))
        self.assertAlmostEqual(result.symbol_recall, 0.5)
        self.assertEqual(result.expected_tests_retrieved, ("tests/test_search.py",))
        self.assertAlmostEqual(result.test_recall, 1.0)

    def test_measures_are_carried_through(self) -> None:
        task = _task(files=("a.py",))
        result = self.runner.evaluate(
            task, _candidate(("a.py",), context_size=777, selected=3, latency=0.25)
        )
        self.assertEqual(result.context_size_tokens, 777)
        self.assertEqual(result.selected_file_count, 3)
        self.assertEqual(result.latency_seconds, 0.25)

    def test_bounded_result_handling(self) -> None:
        capped = agent_eval.AgentEvaluationRunner(max_candidates=3)
        task = _task(files=("a.py", "b.py", "c.py", "d.py"))
        result = capped.evaluate(
            task, _candidate(("a.py", "b.py", "c.py", "d.py"), selected=4)
        )
        self.assertEqual(result.retrieved_files, ("a.py", "b.py", "c.py"))
        self.assertAlmostEqual(result.recall, 0.75)
        with self.assertRaises(ValueError):
            agent_eval.AgentEvaluationRunner(max_candidates=0)

    def test_deterministic_repeated_evaluation(self) -> None:
        task = _task(files=("a.py", "b.py"))
        candidate = _candidate(("b.py", "c.py", "a.py"))
        first = self.runner.evaluate(task, candidate)
        for _ in range(3):
            again = self.runner.evaluate(task, candidate)
            self.assertEqual(first.to_dict(), again.to_dict())

    def test_json_serialization(self) -> None:
        task = _task(
            files=("a.py", "b.py"),
            expected_relevant_symbols=("make_bill",),
            expected_tests=("tests/test_a.py",),
        )
        result = self.runner.evaluate(task, _candidate(("a.py", "b.py")))
        payload = result.to_dict()
        decoded = json.loads(json.dumps(payload))
        self.assertEqual(decoded["precision"], 1.0)
        self.assertEqual(decoded["f1"], 1.0)
        self.assertEqual(decoded["retrieved_files"], ["a.py", "b.py"])


class RunnerCaseTests(unittest.TestCase):
    def test_case_type_validation(self) -> None:
        runner = agent_eval.AgentEvaluationRunner()
        with self.assertRaises(TypeError):
            runner.evaluate("not-a-task", _candidate(()))
        with self.assertRaises(TypeError):
            runner.evaluate(_task(), "not-a-candidate")
        with self.assertRaises(TypeError):
            agent_eval.EvaluationCase(task="x", candidate_set=_candidate(()))

    def test_report_orders_cases_and_groups_by_strategy(self) -> None:
        runner = agent_eval.AgentEvaluationRunner()
        task = _task(files=("a.py",))
        cases = [
            agent_eval.EvaluationCase(task=task, candidate_set=_candidate(("a.py",), strategy="lexical")),
            agent_eval.EvaluationCase(task=task, candidate_set=_candidate(("a.py",), strategy="hybrid")),
        ]
        report = runner.run(cases)
        self.assertEqual(report.num_cases, 2)
        self.assertEqual(report.strategies(), ("lexical", "hybrid"))
        self.assertEqual([r.strategy for r in report.results], ["lexical", "hybrid"])

    def test_strategy_summary_stats(self) -> None:
        runner = agent_eval.AgentEvaluationRunner()
        task1 = _task(task_id="a", files=("x.py",))
        task2 = _task(task_id="b", files=("x.py",))
        cases = [
            agent_eval.EvaluationCase(task=task1, candidate_set=_candidate(("x.py",), strategy="s", context_size=100, selected=1)),
            agent_eval.EvaluationCase(task=task2, candidate_set=_candidate(("nope.py",), strategy="s", context_size=300, selected=1)),
        ]
        summary = runner.run(cases).strategy_summary("s")
        self.assertEqual(summary["task_count"], 2)
        self.assertAlmostEqual(summary["recall"]["mean"], 0.5)
        self.assertAlmostEqual(summary["precision"]["mean"], 0.5)
        self.assertAlmostEqual(summary["f1"]["mean"], (2 * 0.5 * 0.5) / 1.0)
        self.assertAlmostEqual(summary["recall"]["median"], 0.5)
        self.assertAlmostEqual(summary["mean_context_size_tokens"], 200)
        self.assertEqual(summary["failures"], [])

    def test_strategy_summary_efficiency_metrics(self) -> None:
        runner = agent_eval.AgentEvaluationRunner()
        task1 = _task(task_id="a", files=("x.py",))
        task2 = _task(task_id="b", files=("x.py",))
        cases = [
            agent_eval.EvaluationCase(task=task1, candidate_set=_candidate(("x.py",), strategy="ctx", context_size=100, selected=1)),
            agent_eval.EvaluationCase(task=task2, candidate_set=_candidate(("nope.py", "y.py"), strategy="ctx", context_size=300, selected=2)),
        ]
        summary = runner.run(cases).strategy_summary("ctx")
        self.assertAlmostEqual(summary["mean_irrelevant_files"], (0.0 + 2.0) / 2.0)
        self.assertAlmostEqual(summary["mean_tokens_per_relevant_file"], (100.0 + 0.0) / 2.0)

    def test_tokens_per_relevant_is_zero_without_context_size(self) -> None:
        runner = agent_eval.AgentEvaluationRunner()
        task = _task(files=("x.py",))
        summary = runner.run(
            [agent_eval.EvaluationCase(task=task, candidate_set=_candidate(("x.py",), context_size=None, selected=1))]
        ).strategy_summary("ctx")
        self.assertEqual(summary["mean_tokens_per_relevant_file"], 0.0)

    def test_report_json_serialization(self) -> None:
        runner = agent_eval.AgentEvaluationRunner()
        task = _task(files=("a.py",))
        report = runner.run(
            [agent_eval.EvaluationCase(task=task, candidate_set=_candidate(("a.py",), strategy="lexical"))]
        )
        payload = report.to_dict()
        self.assertTrue(payload["deterministic"])
        self.assertEqual(payload["num_cases"], 1)
        self.assertIn("lexical", payload["summaries"])
        decoded = json.loads(json.dumps(payload))
        self.assertEqual(decoded["results"][0]["task_id"], "t1")


class ContextEfficiencyTests(unittest.TestCase):
    """P25.3 graded context-usefulness metrics."""

    def setUp(self) -> None:
        self.runner = agent_eval.AgentEvaluationRunner()

    def _task(
        self,
        required: tuple[str, ...] = ("a.py",),
        supporting: tuple[str, ...] = (),
        tests: tuple[str, ...] = (),
        symbols: tuple[str, ...] = (),
        **overrides,
    ) -> agent_eval.EvaluationTask:
        return agent_eval.EvaluationTask(
            id="ce",
            title="ctx task",
            request="ctx task request",
            required_files=required,
            supporting_files=supporting,
            expected_tests=tests,
            required_symbols=symbols,
            **overrides,
        )

    def test_required_vs_supporting_coverage(self) -> None:
        task = self._task(
            required=("a.py",),
            supporting=("b.py",),
            tests=("tests/test_a.py",),
        )
        eff = self.runner.context_efficiency(
            task,
            _candidate(("b.py", "tests/test_a.py", "zz.py"), strategy="context"),
        )
        self.assertAlmostEqual(eff.required_file_recall, 0.0)
        self.assertAlmostEqual(eff.supporting_file_recall, 1.0)
        self.assertAlmostEqual(eff.test_recall, 1.0)
        self.assertEqual(eff.relevant_selected_file_count, 2)
        self.assertEqual(eff.irrelevant_selected_file_count, 1)
        self.assertEqual(eff.required_files_missed, ("a.py",))
        self.assertEqual(eff.supporting_files_selected, ("b.py",))
        self.assertEqual(eff.irrelevant_files_selected, ("zz.py",))

    def test_required_file_recall(self) -> None:
        task = self._task(required=("a.py", "b.py"))
        eff = self.runner.context_efficiency(task, _candidate(("a.py",)))
        self.assertAlmostEqual(eff.required_file_recall, 0.5)

    def test_supporting_file_recall_empty_and_partial(self) -> None:
        task = self._task(supporting=("b.py", "c.py"))
        eff = self.runner.context_efficiency(task, _candidate(("b.py",)))
        self.assertAlmostEqual(eff.supporting_file_recall, 0.5)
        empty = self._task(supporting=(), required=("a.py",))
        eff0 = self.runner.context_efficiency(empty, _candidate(("a.py",)))
        self.assertEqual(eff0.supporting_file_recall, 0.0)

    def test_test_recall_empty_and_partial(self) -> None:
        task = self._task(tests=("tests/t1.py", "tests/t2.py"))
        eff = self.runner.context_efficiency(task, _candidate(("tests/t1.py",)))
        self.assertAlmostEqual(eff.test_recall, 0.5)
        empty = self._task(tests=())
        eff0 = self.runner.context_efficiency(empty, _candidate(("a.py",)))
        self.assertEqual(eff0.test_recall, 0.0)

    def test_tokens_per_required_file(self) -> None:
        task = self._task(required=("a.py", "b.py"))
        eff = self.runner.context_efficiency(
            task, _candidate(("a.py", "b.py"), context_size=400)
        )
        self.assertAlmostEqual(eff.tokens_per_required_file, 200.0)
        none_retrieved = self.runner.context_efficiency(
            task, _candidate(("c.py",), context_size=400)
        )
        self.assertEqual(none_retrieved.tokens_per_required_file, 0.0)
        none_size = self.runner.context_efficiency(
            task, _candidate(("a.py", "b.py"), context_size=None)
        )
        self.assertEqual(none_size.tokens_per_required_file, 0.0)

    def test_empty_required_surface_is_valid(self) -> None:
        task = self._task(required=(), supporting=())
        eff = self.runner.context_efficiency(task, _candidate(("x.py",)))
        self.assertEqual(eff.required_file_recall, 0.0)
        self.assertEqual(eff.required_files_missed, ())
        self.assertEqual(eff.tokens_per_required_file, 0.0)
        self.assertEqual(eff.irrelevant_selected_file_count, 1)

    def test_missing_required_files_sorted(self) -> None:
        task = self._task(required=("c.py", "a.py", "b.py"))
        eff = self.runner.context_efficiency(task, _candidate(("a.py",)))
        self.assertEqual(eff.required_files_missed, ("b.py", "c.py"))
        self.assertEqual(eff.relevant_selected_file_count, 1)

    def test_selected_relevant_and_irrelevant_counts(self) -> None:
        task = self._task(required=("a.py",), supporting=("s.py",))
        eff = self.runner.context_efficiency(
            task, _candidate(("a.py", "s.py", "zz.py", "yy.py"))
        )
        self.assertEqual(eff.relevant_selected_file_count, 2)
        self.assertEqual(eff.irrelevant_selected_file_count, 2)
        self.assertEqual(eff.irrelevant_files_selected, ("zz.py", "yy.py"))
        self.assertEqual(eff.selected_file_count, 4)

    def test_selected_file_count_passthrough(self) -> None:
        task = self._task(required=("a.py",))
        eff = self.runner.context_efficiency(
            task, _candidate(("a.py", "b.py"), selected=9)
        )
        self.assertEqual(eff.selected_file_count, 9)

    def test_duplicate_surfaces_deduped(self) -> None:
        task = self._task(required=("a.py", "a.py", "b.py"))
        self.assertEqual(task.required_files, ("a.py", "b.py"))
        eff = self.runner.context_efficiency(task, _candidate(("a.py", "a.py")))
        self.assertAlmostEqual(eff.required_file_recall, 0.5)
        self.assertEqual(eff.irrelevant_selected_file_count, 0)

    def test_overlapping_required_supporting_fails(self) -> None:
        with self.assertRaises(ValueError):
            agent_eval.EvaluationTask(
                id="o1", title="overlap", request="r",
                required_files=("a.py",), supporting_files=("a.py",),
            )
        with self.assertRaises(ValueError):
            agent_eval.EvaluationTask(
                id="o2", title="legacy overlap", request="r",
                expected_relevant_files=("a.py",), supporting_files=("a.py",),
            )

    def test_new_surface_fields_validate_paths(self) -> None:
        with self.assertRaises(ValueError):
            self._task(required=("/abs.py",))
        with self.assertRaises(ValueError):
            self._task(supporting=("a/../b.py",))
        with self.assertRaises(ValueError):
            self._task(tests=("",))

    def test_weak_case_detection(self) -> None:
        covered = self._task(required=("a.py",))
        strong = self.runner.context_efficiency(covered, _candidate(("a.py",)))
        weak = self.runner.context_efficiency(covered, _candidate(("b.py",)))
        self.assertFalse(strong.is_weak)
        self.assertTrue(weak.is_weak)

    def test_efficiency_json_serialization(self) -> None:
        task = self._task(required=("a.py",), supporting=("s.py",), tests=("tests/t.py",))
        eff = self.runner.context_efficiency(
            task, _candidate(("a.py", "s.py"), context_size=100)
        )
        decoded = json.loads(json.dumps(eff.to_dict()))
        self.assertEqual(decoded["required_file_recall"], 1.0)
        self.assertEqual(decoded["supporting_file_recall"], 1.0)
        self.assertEqual(decoded["tokens_per_required_file"], 100.0)


class BackwardCompatibilityTests(unittest.TestCase):
    """P25.2 task definitions must keep working unchanged."""

    def test_legacy_task_construction(self) -> None:
        task = _task(
            files=("a.py", "b.py"),
            expected_relevant_symbols=("S",),
            expected_tests=("tests/test_a.py",),
        )
        self.assertEqual(task.required_files_effective, ("a.py", "b.py"))
        self.assertEqual(task.required_symbols_effective, ("S",))
        self.assertEqual(task.supporting_files, ())
        self.assertEqual(task.expected_tests, ("tests/test_a.py",))

    def test_legacy_positional_construction(self) -> None:
        task = agent_eval.EvaluationTask("id1", "title", "request", ("a.py",))
        self.assertEqual(task.id, "id1")
        self.assertEqual(task.required_files_effective, ("a.py",))

    def test_merge_legacy_and_new_surfaces_deterministically(self) -> None:
        task = agent_eval.EvaluationTask(
            id="m",
            title="t",
            request="r",
            expected_relevant_files=("a.py", "shared.py"),
            required_files=("b.py", "shared.py"),
            expected_relevant_symbols=("Old",),
            required_symbols=("New", "Old"),
        )
        self.assertEqual(
            task.required_files_effective, ("b.py", "shared.py", "a.py")
        )
        self.assertEqual(task.required_symbols_effective, ("New", "Old"))

    def test_new_only_task_omits_legacy_fields(self) -> None:
        task = agent_eval.EvaluationTask(
            id="n", title="t", request="r", required_files=("a.py",)
        )
        self.assertEqual(task.expected_relevant_files, ())
        self.assertEqual(task.required_files_effective, ("a.py",))

    def test_legacy_task_evaluate_semantics_unchanged(self) -> None:
        runner = agent_eval.AgentEvaluationRunner()
        task = _task(files=("a.py", "b.py"))
        result = runner.evaluate(task, _candidate(("a.py", "c.py")))
        self.assertAlmostEqual(result.precision, 0.5)
        self.assertAlmostEqual(result.recall, 0.5)
        self.assertAlmostEqual(result.f1, 0.5)


class ReportEfficiencyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.runner = agent_eval.AgentEvaluationRunner()

    def _report(self):
        task1 = agent_eval.EvaluationTask(
            id="a", title="t", request="r",
            required_files=("x.py",),
            supporting_files=("s.py",),
            expected_tests=("tests/test_x.py",),
        )
        task2 = agent_eval.EvaluationTask(
            id="b", title="t2", request="r2",
            required_files=("x.py",),
            supporting_files=(),
        )
        cases = [
            agent_eval.EvaluationCase(
                task=task1,
                candidate_set=_candidate(
                    ("x.py", "s.py", "tests/test_x.py", "zz.py"),
                    strategy="context",
                    context_size=500,
                    selected=4,
                ),
            ),
            agent_eval.EvaluationCase(
                task=task2,
                candidate_set=_candidate(
                    ("zz.py",), strategy="context", context_size=200, selected=1
                ),
            ),
        ]
        return self.runner.run(cases)

    def test_efficiency_summary_means(self) -> None:
        summary = self._report().efficiency_summary("context")
        self.assertEqual(summary["task_count"], 2)
        self.assertAlmostEqual(summary["mean_required_file_recall"], 0.5)
        self.assertAlmostEqual(summary["mean_supporting_file_recall"], 0.5)
        self.assertAlmostEqual(summary["mean_test_recall"], 0.5)
        self.assertAlmostEqual(summary["mean_selected_file_count"], 2.5)
        self.assertAlmostEqual(summary["mean_relevant_selected_file_count"], 1.5)
        self.assertAlmostEqual(summary["mean_irrelevant_selected_file_count"], 1.0)
        self.assertAlmostEqual(summary["mean_context_size_tokens"], 350.0)
        self.assertAlmostEqual(summary["mean_tokens_per_required_file"], 250.0)
        self.assertEqual(summary["failures"], [])

    def test_deterministic_repeated_benchmark_execution(self) -> None:
        first = self._report().to_dict()
        for _ in range(2):
            self.assertEqual(first, self._report().to_dict())

    def test_weak_cases_aggregated_on_report(self) -> None:
        report = self._report()
        weak = report.weak_cases()
        self.assertEqual([result.task_id for result in weak], ["b"])


class ProducerTests(unittest.TestCase):
    def test_search_producer_for_each_result_type(self) -> None:
        task = _task(files=("repolens/search.py",))

        class FakeSearchResult:
            def __init__(self, path, symbols=()):
                self.file_path = path
                self.symbols = symbols

        class FakeSymbol:
            def __init__(self, name):
                self.name = name

        class LexicalLike:
            def search(self, query, limit=10):
                return [
                    FakeSearchResult(Path("repolens/search.py"), (FakeSymbol("CodeSearcher"),)),
                    FakeSearchResult("repolens/retrieval.py"),
                ]

        candidate = agent_eval.produce_search_candidate_set(
            task, searcher=LexicalLike(), root=None, strategy="spy", limit=5
        )
        self.assertEqual(candidate.strategy, "spy")
        self.assertEqual(
            candidate.retrieved_files,
            ("repolens/search.py", "repolens/retrieval.py"),
        )
        self.assertEqual(candidate.matched_symbols, ("CodeSearcher",))
        self.assertEqual(candidate.context_size_tokens, None)
        self.assertEqual(candidate.selected_file_count, 2)
        self.assertIsNotNone(candidate.latency_seconds)

    def test_context_producers_require_a_context_engine(self) -> None:
        task = _task(files=("repolens/search.py",))
        with self.assertRaises(AttributeError):
            agent_eval.produce_context_candidate_set(task, engine=object())

    def test_estimate_context_size_from_source(self) -> None:
        task = _task(files=("repolens/agent_evaluation.py",))
        candidate = agent_eval.produce_search_candidate_set(
            task,
            searcher=_FirstFileSearcher(Path("repolens/agent_evaluation.py")),
            root=REPO_ROOT,
            strategy="lexical",
            limit=3,
        )
        self.assertIsNotNone(candidate.context_size_tokens)
        self.assertGreater(candidate.context_size_tokens, 0)

    def test_context_producer_extracts_engine_package(self) -> None:
        from repolens.context import (
            ContextBudget,
            ContextEngine,
            DependencyExpansionConfig,
        )

        fixture = REPO_ROOT / "tests" / "fixtures" / "synthetic_repository"
        task = agent_eval.EvaluationTask(
            id="fixture",
            title="Invoice calculation",
            request="invoice calculation",
            required_files=("billing/invoice.py",),
            supporting_files=("billing/tax.py",),
        )
        engine = ContextEngine(
            fixture,
            budget=ContextBudget(max_tokens=8000),
            dependency=DependencyExpansionConfig(depth=1),
        )
        producers = (
            agent_eval.produce_context_candidate_set,
            agent_eval.produce_change_context_candidate_set,
        )
        expected_strategies = ("context", "change_context")
        for producer, strategy in zip(producers, expected_strategies):
            with self.subTest(strategy=strategy):
                candidate = producer(task, engine=engine)
                self.assertEqual(candidate.strategy, strategy)
                self.assertTrue(all((fixture / p).is_file() for p in candidate.retrieved_files))
                self.assertIsInstance(candidate.context_size_tokens, int)
                self.assertGreaterEqual(candidate.context_size_tokens, 0)
                self.assertEqual(
                    candidate.selected_file_count, len(candidate.retrieved_files)
                )
                self.assertIsInstance(candidate.matched_symbols, tuple)
                self.assertIsNotNone(candidate.latency_seconds)


class _FirstFileSearcher:
    def __init__(self, path: Path) -> None:
        self._path = path

    def search(self, query: str, limit: int = 10):
        from repolens.search import SearchResult

        return [SearchResult(file_path=self._path, score=1, matched_terms=(), symbols=())]


# ---------------------------------------------------------------------------
# Integration tests: benchmark corpus + producer wiring on the real repository
# ---------------------------------------------------------------------------


class BenchmarkDefinitionTests(unittest.TestCase):
    def setUp(self) -> None:
        sys.path.insert(0, str(REPO_ROOT / "benchmarks"))
        try:
            import agent_evaluation as benchmark  # type: ignore[import-not-found]
        finally:
            sys.path.pop(0)
        self.benchmark = benchmark
        self.tasks = benchmark.build_corpus(REPO_ROOT)

    def test_corpus_tasks_are_well_formed(self) -> None:
        self.assertGreaterEqual(len(self.tasks), 8)
        for task in self.tasks:
            with self.subTest(task_id=task.id):
                self.assertTrue(task.id)
                self.assertTrue(task.title)
                self.assertTrue(task.request)
                self.assertFalse(task.required_files_effective == ())
                self.assertIn(task.category, agent_eval.CATEGORIES)

    def test_corpus_covers_all_categories(self) -> None:
        covered = {task.category for task in self.tasks}
        self.assertEqual(covered, set(agent_eval.CATEGORIES))

    def test_corpus_surfaces_are_disjoint(self) -> None:
        for task in self.tasks:
            with self.subTest(task_id=task.id):
                self.assertEqual(
                    set(task.required_files_effective) & set(task.supporting_files),
                    set(),
                )

    def test_corpus_expected_files_exist_in_repo(self) -> None:
        for task in self.tasks:
            surface = (
                *task.required_files_effective,
                *task.supporting_files,
                *task.expected_tests,
                *task.target_files,
            )
            for path in surface:
                with self.subTest(task_id=task.id, path=path):
                    self.assertTrue((REPO_ROOT / path).is_file(), path)

    def test_corpus_is_deterministic(self) -> None:
        again = self.benchmark.build_corpus(REPO_ROOT)
        self.assertEqual(
            [task.to_dict() for task in self.tasks],
            [task.to_dict() for task in again],
        )

    def test_corpus_task_ids_unique(self) -> None:
        ids = [task.id for task in self.tasks]
        self.assertEqual(len(ids), len(set(ids)))

    def test_strategy_grouping_is_partition_of_all_strategies(self) -> None:
        self.assertEqual(
            self.benchmark.RETRIEVAL_STRATEGIES,
            ("lexical", "candidate-semantic", "hybrid"),
        )
        self.assertEqual(
            self.benchmark.CONTEXT_STRATEGIES,
            ("context", "change_context"),
        )
        self.assertFalse(
            set(self.benchmark.RETRIEVAL_STRATEGIES)
            & set(self.benchmark.CONTEXT_STRATEGIES)
        )
        self.assertEqual(
            set(self.benchmark.STRATEGIES),
            set(self.benchmark.RETRIEVAL_STRATEGIES)
            | set(self.benchmark.CONTEXT_STRATEGIES),
        )

    def test_report_output_separates_objectives(self) -> None:
        import io
        import contextlib

        runner = agent_eval.AgentEvaluationRunner()
        task = _task(files=("x.py",), expected_tests=("tests/test_x.py",))
        cases = [
            agent_eval.EvaluationCase(task=task, candidate_set=_candidate(("x.py",), strategy=s, context_size=50, selected=1))
            for s in self.benchmark.STRATEGIES
        ]
        report = runner.run(cases)
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            self.benchmark.print_report(report)
        output = buffer.getvalue()

        self.assertIn("Retrieval coverage", output)
        self.assertIn("Agent context usefulness", output)
        self.assertIn(
            "evaluated on different objectives and should not be treated as one leaderboard",
            output,
        )
        for strategy in self.benchmark.RETRIEVAL_STRATEGIES:
            line = next(line for line in output.splitlines() if line.startswith(f"- {strategy}:"))
            self.assertIn("P=", line)
            self.assertIn("F1=", line)
        for strategy in self.benchmark.CONTEXT_STRATEGIES:
            line = next(line for line in output.splitlines() if line.startswith(f"- {strategy}:"))
            self.assertIn("required-file recall=", line)
            self.assertIn("supporting-file recall=", line)
            self.assertIn("tokens_per_required=", line)
            self.assertNotIn("P=", line)


class BenchmarkDiagnosticsTests(unittest.TestCase):
    """Task-level diagnostics and weak-case detection (--diagnostics mode)."""

    def setUp(self) -> None:
        sys.path.insert(0, str(REPO_ROOT / "benchmarks"))
        try:
            import agent_evaluation as benchmark  # type: ignore[import-not-found]
        finally:
            sys.path.pop(0)
        self.benchmark = benchmark

    def _weak_report(self):
        runner = agent_eval.AgentEvaluationRunner()
        task = agent_eval.EvaluationTask(
            id="weak-task",
            title="t",
            request="r",
            required_files=("needed.py",),
            supporting_files=("extra.py",),
            expected_tests=("tests/test_x.py",),
        )
        cases = [
            agent_eval.EvaluationCase(
                task=task,
                candidate_set=_candidate(("extra.py", "zz.py"), strategy="context", context_size=500),
            ),
            agent_eval.EvaluationCase(
                task=task,
                candidate_set=_candidate(("needed.py", "extra.py"), strategy="change_context", context_size=600, selected=2),
            ),
        ]
        return runner.run(cases)

    def test_weak_context_cases_detected(self) -> None:
        report = self._weak_report()
        weak = self.benchmark.weak_context_cases(report)
        ids = [result.task_id for result in weak]
        strategies = {result.strategy for result in weak}
        self.assertEqual(ids, ["weak-task"])
        self.assertEqual(strategies, {"context"})

    def test_diagnostic_output_lists_weak_case(self) -> None:
        import contextlib
        import io

        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            self.benchmark.print_diagnostics(self._weak_report())
        output = buffer.getvalue()
        self.assertIn("weak-task[context]", output)
        self.assertIn("required_missed={needed.py}", output)
        self.assertIn("supporting_selected={extra.py}", output)
        self.assertIn("irrelevant_selected={zz.py}", output)
        self.assertIn("context_tokens=500", output)
        self.assertIn("no source content is dumped", output)

    def test_diagnostics_not_in_normal_report(self) -> None:
        import contextlib
        import io

        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            self.benchmark.print_report(self._weak_report())
        output = buffer.getvalue()
        self.assertNotIn("required_missed", output)

    def test_diagnostic_output_is_deterministic(self) -> None:
        import contextlib
        import io

        buffer1, buffer2 = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(buffer1):
            self.benchmark.print_diagnostics(self._weak_report())
        with contextlib.redirect_stdout(buffer2):
            self.benchmark.print_diagnostics(self._weak_report())
        self.assertEqual(buffer1.getvalue(), buffer2.getvalue())


class BenchmarkWiringTests(unittest.TestCase):
    """Drives the producer wiring end-to-end on the real repository (lexical only)."""

    @classmethod
    def setUpClass(cls) -> None:
        sys.path.insert(0, str(REPO_ROOT / "benchmarks"))
        try:
            import agent_evaluation as benchmark  # type: ignore[import-not-found]
        finally:
            sys.path.pop(0)
        cls.benchmark = benchmark
        cls.runner = agent_eval.AgentEvaluationRunner(max_candidates=8)

    def test_producers_return_results_for_a_task_subset(self) -> None:
        tasks = self.benchmark.build_corpus(REPO_ROOT)[:2]
        searcher = CodeSearcher(REPO_ROOT)
        cases = [
            agent_eval.EvaluationCase(
                task=task,
                candidate_set=agent_eval.produce_search_candidate_set(
                    task, searcher=searcher, root=REPO_ROOT, strategy="lexical", limit=8
                ),
            )
            for task in tasks
        ]
        report = self.runner.run(cases)
        self.assertEqual(report.num_cases, len(tasks))
        self.assertEqual(report.failures, ())
        for result in report.results:
            self.assertTrue(all(Path(REPO_ROOT / path).exists() for path in result.retrieved_files))


if __name__ == "__main__":
    unittest.main()