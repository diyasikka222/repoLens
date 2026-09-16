"""Budget-aware context selection diagnostics benchmark (P26.2).

Runs the P25 agent-evaluation task corpus through the real context /
change-context pipeline with an attached
:class:`~repolens.context_trace.TraceCollector`, then reconstructs how the
configured token budget was spent for *every* ranked candidate: rank, score,
estimated tokens, discovery/inclusion source, required/supporting/test tag,
selection verdict, cumulative token usage, remaining budget, and precise
exclusion reason (``exceeds_total_budget`` / ``remaining_budget_too_small`` /
``firewall_filtered`` / ``duplicate`` / other).

Oversized candidates (estimated tokens exceeding the total budget, or the
remaining budget at selection time) are flagged per row and rolled up; every
required file that was RANKED_NOT_SELECTED is reported with its rank, score,
tokens, the selected files that competed for the budget, and the exact reason.

This is measurement only: it does not change retrieval, ranking, selection,
the token budget, or the firewall.

Usage::

    python benchmarks/budget_diagnostics.py
    python benchmarks/budget_diagnostics.py --json out.json
    python benchmarks/budget_diagnostics.py --strategy context
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path

# Keep ``python benchmarks/budget_diagnostics.py`` working: that form puts
# ``benchmarks/`` on sys.path but not the repo root.
REPO_ROOT = Path(__file__).resolve().parent.parent
if __package__ in (None, ""):
    sys.path.insert(0, str(REPO_ROOT))

from repolens.agent_evaluation import EvaluationTask
from repolens.budget_diagnostics import (
    BudgetDiagnosticsReport,
    BudgetSpendReport,
    ExclusionCategory,
    analyze_budget_spend,
)
from repolens.context import ContextBudget, ContextEngine, DependencyExpansionConfig
from repolens.context.firewall import ContextFirewall
from repolens.context_trace import TraceCollector
from repolens.embeddings import FakeEmbeddingProvider
from repolens.required_file_diagnostics import firewall_decisions_for_package
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


def run_budget_diagnostics(
    tasks: tuple[EvaluationTask, ...],
    root: Path,
    *,
    strategies: tuple[str, ...] = DEFAULT_STRATEGIES,
) -> BudgetDiagnosticsReport:
    """Trace every task through the real pipeline and analyze budget spend."""
    firewall = ContextFirewall()
    reports: list[BudgetSpendReport] = []
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
            reports.append(
                analyze_budget_spend(
                    task.id,
                    strategy,
                    trace,
                    budget_max_tokens=(
                        package.budget.max_tokens if package.budget else None
                    ),
                    required_files=task.required_files_effective,
                    supporting_files=task.supporting_files,
                    expected_tests=task.expected_tests,
                    firewall_decisions=firewall_decisions_for_package(safe),
                    focus_events=trace.focus,
                )
            )
    return BudgetDiagnosticsReport(reports=tuple(reports))


def print_report(report: BudgetDiagnosticsReport, *, focus_events: bool = False) -> None:
    """Print per-task spend tables plus aggregate budget diagnostics.

    Per-synthesized focus events are verbose; they print only when requested.
    """
    print(
        "budget-aware context selection diagnostics (P26.2): measurement only, "
        "no retrieval/ranking/selection/budget/firewall changes"
    )
    print()
    for strategy in report.strategies:
        print(f"[{strategy}]")
        packages = report.for_strategy(strategy)
        for package in packages:
            print(
                f"  - {package.task_id}: budget={package.budget_max_tokens}"
                f" selected={package.summary.selected_count}"
                f" rejected={package.summary.rejected_count}"
                f" utilization={_fmt(package.summary.budget_utilization)}"
                f" used={package.summary.total_selected_tokens}"
                f" unused={package.summary.unused_budget}"
            )
            for row in package.rows:
                status = _row_status(row)
                source = row.discovery_source or "-"
                score = _fmt(row.score)
                tag = row.surface_tag.value
                flags = []
                if row.oversized_total:
                    flags.append("OVERSIZED-TOTAL")
                if row.oversized_remaining:
                    flags.append("OVERSIZED-REM")
                flag_text = f" {' '.join(flags)}" if flags else ""
                print(
                    f"      #{row.rank:>2} {row.path}"
                    f" [{tag}] tok={row.estimated_tokens}"
                    f" source={source} inclusion={row.inclusion_reason or '-'}"
                    f" score={score}"
                    f" {status}{flag_text}"
                )
            for row in package.required_missing():
                competing = package.competing_selected(row)
                competed = ",".join(competing) if competing else "(none)"
                print(
                    f"      REQUIRED LOST #{row.rank:>2} {row.path}"
                    f" score={_fmt(row.score)} tok={row.estimated_tokens}"
                    f" reason={row.exclusion_reason or row.exclusion_category or '-'}"
                    f" competing=[{competed}]"
                )
            for event in package.focus_events if focus_events else ():
                print(
                    f"      FOCUS {event['path']}::{event['focus_name']}"
                    f" lines={event['focus_start_line']}-{event['focus_end_line']}"
                    f" focused_tok={event['focused_estimated_tokens']}"
                    f" full_tok={event['full_estimated_tokens']}"
                    f" focus_reason={event['focus_reason']}"
                    f" full_rejected={event['full_rejection_reason']}"
                )
            if package.required_focused():
                print("      REQUIRED RECOVERED VIA FOCUS:")
                for row in package.required_focused():
                    print(
                        f"        {row.path}: rank=#{row.rank}"
                        f" full_tok={row.full_tokens}"
                        f" reason={row.exclusion_reason or '-'}"
                    )
        print()
        agg = report.aggregate(strategy)
        print(
            f"  aggregate: budget={agg.budget_max_tokens}"
            f" utilization={_fmt(agg.budget_utilization)}"
            f" used={agg.total_selected_tokens}"
            f" unused={agg.unused_budget}"
            f" selected={agg.selected_count} rejected={agg.rejected_count}"
        )
        print(
            f"  focused: generated={agg.focused_generated}"
            f" selected={agg.focused_selected}"
        )
        print(
            f"  oversized: total={agg.oversized_total_count}"
            f" remaining={agg.oversized_remaining_count}"
        )
        category_line = " ".join(
            f"{category}={count}" for category, count in agg.excluded_by_category
        )
        print(f"  excluded-by-category: {category_line}")
        print(
            f"  required lost: by_size={agg.required_excluded_by_size}"
            f" by_consumption={agg.required_excluded_by_consumption}"
        )
        print(
            f"  selected size: avg={_fmt(agg.avg_selected_tokens)}"
            f" median={_fmt(agg.median_selected_tokens)}"
        )
        print(
            f"  required rank: avg={_fmt(agg.avg_required_rank)}"
            f" median={_fmt(agg.median_required_rank)}"
        )
        print("  top oversized files:")
        for entry in report.top_oversized_files(strategy):
            print(
                f"    - {entry['path']}: x{entry['count']}"
                f" max_tokens={entry['max_tokens']}"
                f" tasks={','.join(entry['tasks'])}"
            )
        print("  top required files lost to budget:")
        for entry in report.top_required_lost_to_budget(strategy):
            print(
                f"    - {entry['path']}: lost x{entry['lost_count']}"
                f" by_size={entry['by_size']} by_consumption={entry['by_consumption']}"
                f" max_tokens={entry['max_tokens']}"
                f" tasks={','.join(entry['tasks'])}"
            )
        print()

    _print_verdict(report)
    print()

    total = report.aggregate()
    print(
        "total:",
        " ".join(
            f"{category}={count}"
            for category, count in total.excluded_by_category
        ),
    )


def _row_status(row) -> str:
    if not row.selected:
        return f"EXCLUDED {row.exclusion_reason or 'unknown'}"
    if row.exclusion_category is ExclusionCategory.FIREWALL_FILTERED:
        return "SELECTED->FIREWALL_FILTERED"
    remaining = row.remaining_after
    remaining_text = f" rem={remaining}" if remaining is not None else ""
    return f"SELECTED cum={row.cumulative_tokens_after}{remaining_text}"


def _print_verdict(report: BudgetDiagnosticsReport) -> None:
    print("failure attribution (required files RANKED_NOT_SELECTED):")
    for strategy in report.strategies:
        agg = report.aggregate(strategy)
        by_size = agg.required_excluded_by_size
        by_consumption = agg.required_excluded_by_consumption
        total_lost = by_size + by_consumption
        if total_lost == 0:
            verdict = "none — no required file lost to budget"
        elif by_size and by_consumption:
            verdict = "D (combination: both oversized files and earlier consumption)"
        elif by_size:
            verdict = "A (files individually exceeding the total budget)"
        elif by_consumption:
            verdict = "B (earlier selections consuming the budget)"
        else:  # pragma: no cover - defensive
            verdict = "C (poor ranking)"
        focused = agg.focused_selected
        recovered = sum(
            len(package.required_focused())
            for package in report.for_strategy(strategy)
        )
        focus_note = (
            f" focused_recovered_required={recovered}"
            if recovered
            else ""
        )
        print(f"  [{strategy}] lost={total_lost} by_size={by_size} "
              f"by_consumption={by_consumption} focused_selected={focused}"
              f"{focus_note} -> {verdict}")


def _fmt(value) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    json_path: Path | None = None
    strategies = set(DEFAULT_STRATEGIES)
    args = list(argv)
    focus_events = False
    if "--focus-events" in args:
        args.remove("--focus-events")
        focus_events = True
    if "--json" in args:
        index = args.index("--json")
        json_path = Path(args.pop(index + 1))
        args.pop(index)
    strategies_tuple = tuple(sorted(s for s in strategies if s in DEFAULT_STRATEGIES))

    with tempfile.TemporaryDirectory(prefix="repolens-budgetdiag-") as cache_dir:
        os.environ["REPOLENS_CACHE_DIR"] = cache_dir
        started = time.perf_counter()
        tasks = build_corpus(REPO_ROOT)
        report = run_budget_diagnostics(tasks, REPO_ROOT, strategies=strategies_tuple)
        elapsed = time.perf_counter() - started
        print_report(report, focus_events=focus_events)
        if json_path is not None:
            json_path.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
            print(f"json report written to {json_path}")
        print(f"budget diagnostics benchmark finished in {elapsed:.2f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())