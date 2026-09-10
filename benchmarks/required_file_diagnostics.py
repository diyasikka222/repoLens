"""Required-file recall diagnostics benchmark (P26.1).

Runs the P25 agent-evaluation task corpus through the real context /
change-context pipeline with an attached
:class:`~repolens.context_trace.TraceCollector`, then classifies every required
file into one of:

- ``NOT_DISCOVERED`` — never a candidate at any stage;
- ``DISCOVERED_NOT_RANKED`` — discovered but absent from the ranked list;
- ``RANKED_NOT_SELECTED`` — ranked but lost at budget selection;
- ``SELECTED`` — reached the final context.

This is measurement only: it does not change retrieval, ranking, selection,
the firewall, or any OpenCode behaviour.

Per-task output lists every *missing* required file with its discovery source,
ranking tier/position, and (for budget losses) the exclusion reason, so the
report distinguishes discovery failures from ranking failures from budget
failures.

Usage::

    python benchmarks/required_file_diagnostics.py
    python benchmarks/required_file_diagnostics.py --json out.json
    python benchmarks/required_file_diagnostics.py --strategy context
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path

# Keep ``python benchmarks/required_file_diagnostics.py`` working: that form
# puts ``benchmarks/`` on sys.path but not the repo root, so the ``repolens``
# and sibling ``benchmarks`` imports below need the root prepended.
REPO_ROOT = Path(__file__).resolve().parent.parent
if __package__ in (None, ""):
    sys.path.insert(0, str(REPO_ROOT))

from repolens.agent_evaluation import EvaluationTask
from repolens.context import ContextBudget, ContextEngine, DependencyExpansionConfig
from repolens.context.firewall import ContextFirewall
from repolens.context_trace import TraceCollector
from repolens.embeddings import FakeEmbeddingProvider
from repolens.required_file_diagnostics import (
    RequiredFileDiagnosticsReport,
    RequiredFileOutcome,
    TaskDiagnostics,
    classify_required_files,
    firewall_decisions_for_package,
)
from repolens.search import CodeSearcher
from repolens.semantic_search import SemanticSearcher

from benchmarks.agent_evaluation import build_corpus

CONTEXT_MAX_TOKENS = 4096
DEPENDENCY_DEPTH = 1

DEFAULT_STRATEGIES = ("context", "change_context")


def _build_engine(root: Path, tracer: TraceCollector) -> ContextEngine:
    """Share the exact P25 agent-evaluation engine configuration (lexical
    searcher, depth 1, 4096-token budget) with an attached tracer."""
    lexical = CodeSearcher(root)
    semantic = SemanticSearcher(
        root,
        FakeEmbeddingProvider(),
        candidate_searcher=lexical,
    )
    return ContextEngine(
        root,
        searcher=semantic,
        dependency=DependencyExpansionConfig(depth=DEPENDENCY_DEPTH),
        budget=ContextBudget(max_tokens=CONTEXT_MAX_TOKENS),
        tracer=tracer,
    )


def run_diagnostics(
    tasks: tuple[EvaluationTask, ...],
    root: Path,
    *,
    strategies: tuple[str, ...] = DEFAULT_STRATEGIES,
) -> RequiredFileDiagnosticsReport:
    """Build every task through the traced pipeline and classify required files."""
    collector = TraceCollector()
    engine = _build_engine(root, collector)
    firewall = ContextFirewall()

    diagnostics: list[TaskDiagnostics] = []
    for task in tasks:
        target = task.target_files[0] if task.target_files else None
        for strategy in strategies:
            collector = TraceCollector()
            engine = _build_engine(root, collector)
            if strategy == "context":
                package = engine.build_context(task.request)
            elif strategy == "change_context":
                package = engine.build_context(
                    task.request,
                    change_request=task.request,
                    change_target=target,
                )
            else:  # pragma: no cover - guarded by callers
                raise ValueError(f"unsupported strategy {strategy!r}")

            safe = firewall.safe_package(package, firewall.inspect(package))
            trace = collector.trace()
            rows = classify_required_files(
                task.id,
                strategy,
                task.required_files_effective,
                trace,
                budget_max_tokens=(
                    package.budget.max_tokens if package.budget else None
                ),
                firewall_decisions=firewall_decisions_for_package(safe),
            )
            diagnostics.append(
                TaskDiagnostics(
                    task_id=task.id,
                    strategy=strategy,
                    budget_max_tokens=package.budget.max_tokens if package.budget else None,
                    rows=rows,
                )
            )
    return RequiredFileDiagnosticsReport(tasks=tuple(diagnostics))


def print_report(report: RequiredFileDiagnosticsReport) -> None:
    """Print a concise, actionable report."""
    print("required-file recall diagnostics (P26.1): measurement only, no "
          "retrieval/ranking/selection changes")
    print()
    for strategy in report.strategies:
        counts = report.outcome_counts(strategy)
        print(f"[{strategy}]")
        line = "  ".join(f"{key}={counts[key]}" for key in RequiredFileOutcome)
        print(f"  required-file outcomes: {line}")
        tasks = report.for_strategy(strategy)
        print(f"  tasks={len(tasks)} "
              f"selected-required={counts[RequiredFileOutcome.SELECTED.value]}")
        for task in tasks:
            missing = task.missing_rows
            if missing:
                print(f"  - {task.task_id}: recall={task.required_file_recall():.3f}"
                      f" budget={task.budget_max_tokens}")
                for row in missing:
                    src = row.discovery_source or "-"
                    extras = []
                    if row.from_change_plan:
                        extras.append("change_plan")
                    if row.from_architecture:
                        extras.append("architecture")
                    if row.graph_distance is not None:
                        extras.append(f"distance={row.graph_distance}")
                    reason = row.excluded_reason or ""
                    position = f"pos={row.ranking_position}" if row.ranking_position else ""
                    token_cost = f"tokens={row.estimated_tokens}" if row.estimated_tokens is not None else ""
                    attrs = " ".join(x for x in (position, reason, token_cost) if x)
                    print(f"      {row.path}: {row.outcome.value}"
                          f" source={src}"
                          f"{(' via ' + ','.join(extras)) if extras else ''}"
                          f"{(' [' + attrs + ']') if attrs else ''}")
        print()
    print("top recurring failure patterns:")
    occurrences = 0
    for pattern in report.recurrences():
        occurrences += pattern["count"]
        print(f"  - {pattern['count']}x {pattern['outcome']}"
              f" source={pattern['discovery_source'] or 'none'}"
              f" inclusion={pattern['inclusion_reason'] or 'none'}"
              f" tasks={','.join(pattern['task_ids'])}")
    if not occurrences:
        print("  (none — every required file reached the final context)")
    print()
    all_counts = report.outcome_counts()
    print("total:", ", ".join(f"{k}={v}" for k, v in all_counts.items()))


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    json_path: Path | None = None
    strategies = set(DEFAULT_STRATEGIES)
    args = list(argv)
    if "--json" in args:
        index = args.index("--json")
        json_path = Path(args.pop(index + 1))
        args.pop(index)
    for flag in ("--strategy",):
        while flag in args:
            index = args.index(flag)
            value = args.pop(index + 1)
            args.pop(index)
            strategies.add(value)
    strategies_tuple = tuple(sorted(s for s in strategies if s in DEFAULT_STRATEGIES))

    with tempfile.TemporaryDirectory(prefix="repolens-reqdefdiag-") as cache_dir:
        os.environ["REPOLENS_CACHE_DIR"] = cache_dir
        started = time.perf_counter()
        tasks = build_corpus(REPO_ROOT)
        report = run_diagnostics(tasks, REPO_ROOT, strategies=strategies_tuple)
        elapsed = time.perf_counter() - started
        print_report(report)
        if json_path is not None:
            json_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
            print(f"json report written to {json_path}")
        print(f"required-file diagnostics benchmark finished in {elapsed:.2f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())