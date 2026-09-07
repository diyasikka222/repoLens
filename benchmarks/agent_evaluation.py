"""Deterministic agent-task evaluation benchmark (P25.2–P25.3).

Scores how much of each realistic repository-level coding task's surface
(files, symbols, tests) each RepoLens strategy makes available to an agent,
fully offline and deterministically.

Evaluated strategies:

- ``lexical`` — :class:`repolens.search.CodeSearcher` alone;
- ``candidate-semantic`` — :class:`repolens.semantic_search.SemanticSearcher`
  over lexical candidates with the deterministic fake embedding provider
  (no model, no network);
- ``hybrid`` — :class:`repolens.retrieval.HybridSearcher` fusing lexical and
  semantic rankings;
- ``context`` — ordinary :class:`repolens.context.ContextEngine` packages;
- ``change_context`` — change-aware context (plan folded into the same package
  pipeline).

The corpus targets the real RepoLens repository: every file in a task surface
is a real file, every required symbol a real definition. Criteria are split by
objective:

- **Retrieval coverage** (lexical / candidate-semantic / hybrid) is scored on
  precision, recall and F1 against the required surface;
- **Agent context usefulness** (context / change_context) is scored on graded
  required/supporting/test coverage plus package efficiency (selected files,
  irrelevant files, context tokens, tokens per required file).

These are different objectives and are not a single leaderboard.

Task-level ``--diagnostics`` output lists every context case that missed at
least one required file (task id, strategy, missed/supporting/irrelevant files,
context tokens). The normal output stays concise and deterministic.

Usage::

    python benchmarks/agent_evaluation.py
    python benchmarks/agent_evaluation.py --diagnostics
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Callable

from repolens.agent_evaluation import (
    ContextEfficiencyResult,
    EvaluationTask,
    AgentEvaluationRunner,
    CandidateSet,
    EvaluationCase,
    EvaluationReport,
    produce_change_context_candidate_set,
    produce_context_candidate_set,
    produce_search_candidate_set,
)
from repolens.context import ContextBudget, ContextEngine, DependencyExpansionConfig
from repolens.embeddings import FakeEmbeddingProvider
from repolens.retrieval import HybridSearcher
from repolens.search import CodeSearcher
from repolens.semantic_search import SemanticSearcher

REPO_ROOT = Path(__file__).resolve().parent.parent

SEARCH_LIMIT = 12
CONTEXT_MAX_TOKENS = 4096
DEPENDENCY_DEPTH = 1

STRATEGIES = (
    "lexical",
    "candidate-semantic",
    "hybrid",
    "context",
    "change_context",
)

#: Strategies scored on retrieval coverage (precision / recall / F1).
RETRIEVAL_STRATEGIES = ("lexical", "candidate-semantic", "hybrid")

#: Strategies scored on agent-context efficiency (package size vs. required
#: files), a different objective than retrieval coverage.
CONTEXT_STRATEGIES = ("context", "change_context")


def build_corpus(root: Path) -> tuple[EvaluationTask, ...]:
    """Deterministic tasks describing realistic work on the real repository.

    Each task declines its surface into *required* files/symbols (a correct
    edit touches them), *supporting* files (useful surrounding context), and
    *expected tests*. Surfaces are disjoint within a task and reference only
    real repository files and definitions.
    """
    del root
    return (
        EvaluationTask(
            id="search-ranking-bug",
            title="Lexical ranker overweights exact symbol names",
            request=(
                "The CodeSearcher ranking lets an exact symbol-name match "
                "overwhelm every other signal, so broad queries surface false "
                "positives on that symbol. Fix the scoring so import and source "
                "tokens still influence the ranking."
            ),
            required_files=("repolens/search.py",),
            supporting_files=("repolens/parser.py",),
            expected_tests=("tests/test_search.py",),
            target_files=("repolens/search.py",),
            required_symbols=("CodeSearcher", "_score"),
            category="bug_fix",
        ),
        EvaluationTask(
            id="search-result-rename",
            title="Rename the search result type across retrieval layers",
            request=(
                "Rename SearchResult to RankedFile across the lexical search, "
                "semantic search and evaluation layers, and update the consumers "
                "so the whole retrieval stack stays consistent."
            ),
            required_files=(
                "repolens/search.py",
                "repolens/semantic_search.py",
            ),
            supporting_files=("repolens/retrieval.py", "repolens/evaluation.py"),
            expected_tests=(
                "tests/test_search.py",
                "tests/test_semantic_search.py",
            ),
            target_files=("repolens/search.py",),
            required_symbols=("SearchResult", "SemanticResult"),
            category="cross_file_change",
        ),
        EvaluationTask(
            id="context-budget-option",
            title="Configurable maximum context budget",
            request=(
                "Add a configuration option that lets callers cap the maximum "
                "context budget in tokens, and thread it through the context "
                "engine so packages respect it."
            ),
            required_files=(
                "repolens/context/config.py",
                "repolens/context/engine.py",
            ),
            supporting_files=(
                "repolens/context/package.py",
                "repolens/context/budget.py",
            ),
            expected_tests=("tests/test_context_engine.py",),
            target_files=("repolens/context/config.py",),
            required_symbols=("ContextBudget", "ContextEngine"),
            category="feature_change",
        ),
        EvaluationTask(
            id="context-ranking-refactor",
            title="Extract an explicit dependency-expansion ranking stage",
            request=(
                "Refactor the context pipeline so dependency-expanded candidates "
                "are ranked in a separate, explicit stage instead of being mixed "
                "into the primary ranking."
            ),
            required_files=(
                "repolens/context/engine.py",
                "repolens/context/ranking.py",
            ),
            supporting_files=(
                "repolens/context/candidate.py",
                "repolens/context/expansion.py",
            ),
            expected_tests=("tests/test_context_engine.py",),
            target_files=("repolens/context/engine.py",),
            required_symbols=("ContextEngine", "rank_candidates"),
            category="refactor",
        ),
        EvaluationTask(
            id="change-plan-risk-signal",
            title="Expose a risk indicator on change_plan",
            request=(
                "Modify the MCP change_plan tool so every planned change exposes "
                "an explicit risk indicator, and keep the JSON response "
                "deterministic and bounded."
            ),
            required_files=(
                "repolens/mcp/change_plan_tool.py",
                "repolens/change_plan.py",
            ),
            supporting_files=("repolens/change_context.py", "repolens/mcp/server.py"),
            expected_tests=("tests/test_mcp_change_plan.py",),
            target_files=("repolens/mcp/change_plan_tool.py",),
            required_symbols=("run_change_plan", "ChangePlanEngine", "ChangePlanState"),
            category="feature_change",
        ),
        EvaluationTask(
            id="change-context-candidate-mapping",
            title="Refactor plan-item to candidate mapping",
            request=(
                "Refactor how the change-aware context layer maps planned change "
                "items into context candidates so the mapping is easier to reason "
                "about and test."
            ),
            required_files=("repolens/change_context.py",),
            supporting_files=(
                "repolens/context/candidate.py",
                "repolens/context/engine.py",
            ),
            expected_tests=(
                "tests/test_change_plan.py",
                "tests/test_mcp_change_plan.py",
            ),
            target_files=("repolens/change_context.py",),
            required_symbols=(
                "plan_to_change_candidates",
                "merge_change_candidates",
                "explain_change_context",
            ),
            category="refactor",
        ),
        EvaluationTask(
            id="hybrid-rrf-tests",
            title="Unit tests for RRF hybrid fusion",
            request=(
                "Add unit tests covering the RRF fusion strategy of the hybrid "
                "searcher in the retrieval module."
            ),
            required_files=("tests/test_retrieval.py",),
            supporting_files=("repolens/retrieval.py",),
            expected_tests=("tests/test_retrieval.py",),
            target_files=("tests/test_retrieval.py",),
            required_symbols=(),
            category="test_change",
        ),
        EvaluationTask(
            id="architecture-retrieval-ordering",
            title="Stable architecture candidate ordering",
            request=(
                "Fix architecture retrieval producing unstable candidate ordering "
                "when architecture signals tie, so repeated queries return the "
                "same ranked surface."
            ),
            required_files=("repolens/architecture_retrieval.py",),
            supporting_files=("repolens/architecture.py",),
            expected_tests=("tests/test_architecture_retrieval.py",),
            target_files=("repolens/architecture_retrieval.py",),
            required_symbols=(
                "extract_architecture_signals",
                "ArchitectureRetrievalConfig",
                "ArchitectureCandidate",
            ),
            category="bug_fix",
        ),
        EvaluationTask(
            id="incremental-index-stale-pruning",
            title="Prune stale incremental index entries",
            request=(
                "Fix the incremental index so entries for files that were removed "
                "after a rebuild are pruned and never resurface."
            ),
            required_files=("repolens/incremental_index.py",),
            supporting_files=("repolens/atomic_write.py", "repolens/index.py"),
            expected_tests=("tests/test_incremental_index.py",),
            target_files=("repolens/incremental_index.py",),
            required_symbols=("IncrementalIndexBuilder", "_prune_stale", "AnalysisCache"),
            category="bug_fix",
        ),
        EvaluationTask(
            id="dependency-expansion-cycle",
            title="Cycle-safe dependency expansion",
            request=(
                "Fix dependency expansion so a dependency cycle between two files "
                "cannot loop or duplicate expanded candidates."
            ),
            required_files=("repolens/context/expansion.py",),
            supporting_files=(
                "repolens/graph.py",
                "repolens/context/engine.py",
            ),
            expected_tests=("tests/test_context_engine.py",),
            target_files=("repolens/context/expansion.py",),
            required_symbols=("expand_dependencies", "ExpandedNode"),
            category="bug_fix",
        ),
    )


def _validate_corpus(root: Path, tasks: tuple[EvaluationTask, ...]) -> None:
    missing: list[str] = []
    for task in tasks:
        surface = (
            *task.required_files_effective,
            *task.supporting_files,
            *task.expected_tests,
            *task.target_files,
        )
        for path in surface:
            if not (root / path).is_file():
                missing.append(f"{task.id}: {path}")
    if missing:
        raise ValueError(
            "benchmark corpus references missing repository files:\n- "
            + "\n- ".join(missing)
        )


class _RunEnvironment:
    """Lazily builds and shares the retrieval/context stack over one repository."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self._lexical: CodeSearcher | None = None
        self._semantic: SemanticSearcher | None = None
        self._hybrid: HybridSearcher | None = None
        self._engine: ContextEngine | None = None

    def lexical(self) -> CodeSearcher:
        if self._lexical is None:
            self._lexical = CodeSearcher(self.root)
        return self._lexical

    def semantic(self) -> SemanticSearcher:
        if self._semantic is None:
            self._semantic = SemanticSearcher(
                self.root,
                FakeEmbeddingProvider(),
                candidate_searcher=self.lexical(),
            )
        return self._semantic

    def hybrid(self) -> HybridSearcher:
        if self._hybrid is None:
            self._hybrid = HybridSearcher(
                self.root,
                lexical_searcher=self.lexical(),
                semantic_searcher=self.semantic(),
            )
        return self._hybrid

    def engine(self) -> ContextEngine:
        if self._engine is None:
            self._engine = ContextEngine(
                self.root,
                searcher=self.lexical(),
                dependency=DependencyExpansionConfig(depth=DEPENDENCY_DEPTH),
                budget=ContextBudget(max_tokens=CONTEXT_MAX_TOKENS),
            )
        return self._engine


class _Producers:
    """One producing callable per strategy, sharing one environment."""

    def __init__(self, env: _RunEnvironment, root: Path) -> None:
        self._env = env
        self._root = root

    def make(self, strategy: str) -> Callable[[EvaluationTask], CandidateSet]:
        if strategy == "lexical":
            return lambda task: produce_search_candidate_set(
                task,
                searcher=self._env.lexical(),
                root=self._root,
                strategy=strategy,
                limit=SEARCH_LIMIT,
            )
        if strategy == "candidate-semantic":
            return lambda task: produce_search_candidate_set(
                task,
                searcher=self._env.semantic(),
                root=self._root,
                strategy=strategy,
                limit=SEARCH_LIMIT,
            )
        if strategy == "hybrid":
            return lambda task: produce_search_candidate_set(
                task,
                searcher=self._env.hybrid(),
                root=self._root,
                strategy=strategy,
                limit=SEARCH_LIMIT,
            )
        if strategy == "context":
            return lambda task: produce_context_candidate_set(
                task, engine=self._env.engine(), strategy=strategy
            )
        if strategy == "change_context":
            return lambda task: produce_change_context_candidate_set(
                task, engine=self._env.engine(), strategy=strategy
            )
        raise ValueError(f"unknown strategy {strategy!r}")


def run_benchmark(
    tasks: tuple[EvaluationTask, ...],
    root: Path,
    *,
    strategies: tuple[str, ...] = STRATEGIES,
    max_candidates: int = 20,
) -> EvaluationReport:
    """Score every task against every strategy, collecting per-case failures."""
    _validate_corpus(root, tasks)
    env = _RunEnvironment(root)
    producers = _Producers(env, root)
    runner = AgentEvaluationRunner(max_candidates=max_candidates)

    cases: list[EvaluationCase] = []
    failures: list[dict] = []
    for task in tasks:
        for strategy in strategies:
            try:
                candidate = producers.make(strategy)(task)
            except Exception as exc:  # pragma: no cover - defensive, deterministic
                failures.append(
                    {
                        "task_id": task.id,
                        "strategy": strategy,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                continue
            cases.append(EvaluationCase(task=task, candidate_set=candidate))
    report = runner.run(cases)
    return EvaluationReport(
        results=report.results,
        efficiency=report.efficiency,
        failures=tuple(failures),
    )


def print_report(report: EvaluationReport) -> None:
    """Print a concise report split into the two evaluation objectives."""
    print(f"agent-evaluation run: {report.num_cases} cases "
          f"({report.num_cases // max(len(report.strategies()), 1)} tasks x "
          f"{len(report.strategies())} strategies), deterministic offline")
    print()
    print("------------------------------------------")
    print("Retrieval coverage")
    print("(objective: maximize coverage of the required task surface;")
    print(" scored with precision / recall / F1)")
    print("------------------------------------------")
    for strategy in RETRIEVAL_STRATEGIES:
        summary = report.strategy_summary(strategy)
        print(f"- {strategy}: "
              f"P={summary['precision']['mean']:.3f} "
              f"R={summary['recall']['mean']:.3f} "
              f"F1={summary['f1']['mean']:.3f} "
              f"(median P={summary['precision']['median']:.3f} "
              f"R={summary['recall']['median']:.3f} "
              f"F1={summary['f1']['median']:.3f}) "
              f"| selected={summary['mean_selected_files']:.2f} "
              f"context={summary['mean_context_size_tokens']:.0f}t "
              f"latency={summary['mean_latency_seconds']:.3f}s "
              f"| tasks={summary['task_count']} "
              f"failures={len(summary['failures'])}")
    print()
    print("------------------------------------------")
    print("Agent context usefulness")
    print("(objective: minimal selected context that covers the required files;")
    print(" scored on graded required/supporting/test coverage and package size)")
    print("------------------------------------------")
    for strategy in CONTEXT_STRATEGIES:
        summary = report.efficiency_summary(strategy)
        print(f"- {strategy}: "
              f"required-file recall={summary['mean_required_file_recall']:.3f} "
              f"supporting-file recall={summary['mean_supporting_file_recall']:.3f} "
              f"test recall={summary['mean_test_recall']:.3f} "
              f"selected={summary['mean_selected_file_count']:.2f} "
              f"irrelevant={summary['mean_irrelevant_selected_file_count']:.2f} "
              f"context_tokens={summary['mean_context_size_tokens']:.0f} "
              f"tokens_per_required={summary['mean_tokens_per_required_file']:.0f} "
              f"| tasks={summary['task_count']} "
              f"failures={len(summary['failures'])}")
    print()
    print("NOTE: retrieval and context strategies are evaluated on different "
          "objectives and should not be treated as one leaderboard.")
    print()
    for failure in report.failures:
        print(f"- FAILURE {failure['task_id']}/{failure['strategy']}: {failure['error']}")


def weak_context_cases(
    report: EvaluationReport, strategies: tuple[str, ...] = CONTEXT_STRATEGIES
) -> tuple[ContextEfficiencyResult, ...]:
    """Every context-strategy case that missed at least one required file (weak)."""
    return tuple(
        result
        for strategy in strategies
        for result in report.efficiency_for_strategy(strategy)
        if result.is_weak
    )


def print_diagnostics(report: EvaluationReport) -> None:
    """Task-level details for weak context cases; never dumps source content."""
    cases = weak_context_cases(report)
    print(f"task-level diagnostics: {len(cases)} weak context case(s) "
          f"(required-file recall < 1.0); no source content is dumped")
    for result in cases:
        missed = ",".join(result.required_files_missed) or "-"
        supporting = ",".join(result.supporting_files_selected) or "-"
        irrelevant = ",".join(result.irrelevant_files_selected) or "-"
        print(f"- {result.task_id}[{result.strategy}]: "
              f"required_missed={{{missed}}} "
              f"supporting_selected={{{supporting}}} "
              f"irrelevant_selected={{{irrelevant}}} "
              f"context_tokens={result.context_size_tokens}")


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    show_diagnostics = "--diagnostics" in argv
    with tempfile.TemporaryDirectory(prefix="replens-agenteval-") as cache_dir:
        os.environ["REPOLENS_CACHE_DIR"] = cache_dir
        started = time.perf_counter()
        tasks = build_corpus(REPO_ROOT)
        report = run_benchmark(tasks, REPO_ROOT)
        elapsed = time.perf_counter() - started
        print_report(report)
        if show_diagnostics:
            print_diagnostics(report)
        print(f"agent-evaluation benchmark finished in {elapsed:.2f}s")
        if report.failures:
            print("WARNING: some strategy/task combinations failed")
    return 0 if not report.failures else 1


if __name__ == "__main__":
    sys.exit(main())