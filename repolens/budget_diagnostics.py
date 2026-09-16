"""Budget-aware context selection diagnostics (P26.2).

Given a P25 agent-evaluation task and the :class:`~repolens.context_trace.PipelineTrace`
a context build produced, this module explains exactly how the configured
context token budget was spent: for every ranked candidate it reports the
rank, best-available score, estimated tokens, discovery/inclusion source,
task-surface tag (required/supporting/test), the selection verdict, and for
selected files the cumulative token usage and remaining budget *at that point
in the walk*. Rejected files are classified into
:class:`ExclusionCategory` buckets; oversized candidates are flagged
individually and in aggregate.

This is measurement only. It is a pure function of already-produced artifacts
(the frozen trace, the task surface, the budget, and the firewall verdicts):

- deterministic: identical trace + task + budget -> identical rows;
- no LLM calls, no retrieval/ranking/selection/firewall changes;
- reuses the P26.1 tracing infrastructure (:mod:`repolens.context_trace`)
  without adding a second tracing system;
- preserves the existing public surface of the P26.1 diagnostic modules.

The budget walk is *reconstructed*, not re-run: selection is positionally
deterministic, so walking ``trace.ranked`` in order and attributing every
candidate to its traced outcome (``trace.selected`` / ``trace.excluded``)
reproduces the engine's exact cumulative budget accounting. Selected-file
token counts are taken from the ``selected`` snapshots (which reflect any
truncation), so the accounting always matches the final package.

The companion CLI (``benchmarks/budget_diagnostics.py``) runs this against the
P25 evaluation corpus and prints the per-task spend table plus aggregates.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from repolens.context_trace import PipelineTrace

# ---------------------------------------------------------------------------
# Classification enums
# ---------------------------------------------------------------------------


class ExclusionCategory(str, Enum):
    """Why a ranked candidate did not reach the final context.

    Mirrors the existing selection reasons produced by
    :mod:`repolens.context.budget` (``exceeds_total_budget``,
    ``over_budget``) plus the post-selection firewall and a taxonomy slot for
    duplicates (structurally impossible today: dedupe happens before ranking).
    """

    EXCEEDS_TOTAL_BUDGET = "exceeds_total_budget"
    REMAINING_BUDGET_TOO_SMALL = "remaining_budget_too_small"
    FIREWALL_FILTERED = "firewall_filtered"
    DUPLICATE = "duplicate"
    OTHER = "other"


class SurfaceTag(str, Enum):
    """A candidate's role in the task surface (a file may match several)."""

    REQUIRED = "required"
    SUPPORTING = "supporting"
    TEST = "test"
    OTHER = "other"


def categorize_reason(reason: str | None) -> ExclusionCategory | None:
    """Map an engine exclusion reason onto the :class:`ExclusionCategory` taxonomy."""
    if reason is None:
        return None
    if reason == "exceeds_total_budget":
        return ExclusionCategory.EXCEEDS_TOTAL_BUDGET
    if reason == "over_budget":
        return ExclusionCategory.REMAINING_BUDGET_TOO_SMALL
    if reason == "duplicate":
        return ExclusionCategory.DUPLICATE
    return ExclusionCategory.OTHER


def surface_tag_for_path(
    path: str,
    required: Iterable[str],
    supporting: Iterable[str],
    expected_tests: Iterable[str],
) -> SurfaceTag:
    """Tag ``path`` by its task-surface membership (required precedence)."""
    required_set = set(_posix_str(p) for p in required)
    if path in required_set:
        return SurfaceTag.REQUIRED
    supporting_set = set(_posix_str(p) for p in supporting)
    if path in supporting_set:
        return SurfaceTag.SUPPORTING
    tests_set = set(_posix_str(p) for p in expected_tests)
    if path in tests_set:
        return SurfaceTag.TEST
    return SurfaceTag.OTHER


# ---------------------------------------------------------------------------
# Row + summary model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BudgetSpendRow:
    """One ranked candidate with its full budget accounting at selection time."""

    task_id: str
    strategy: str
    #: 1-based position in the deduped, ranked candidate order.
    rank: int
    path: str
    #: Best-available numeric score (retrieval_score, else change_score).
    score: float | None
    estimated_tokens: int
    #: Complete full-file estimate for the path (equals ``estimated_tokens``
    #: unless the path was recovered via focused substitution, which carries the
    #: smaller focused slice in ``estimated_tokens``). Drives size attribution.
    full_tokens: int | None
    discovery_source: str | None
    inclusion_reason: str | None
    surface_tag: SurfaceTag
    selected: bool
    #: Estimated tokens used before/after this candidate in the selection walk.
    cumulative_tokens_before: int
    cumulative_tokens_after: int
    #: Remaining budget at selection time (None when the budget is unlimited).
    remaining_before: int | None
    remaining_after: int | None
    exclusion_reason: str | None
    exclusion_category: ExclusionCategory | None
    firewall_decision: str | None
    #: Estimated tokens exceed the *total* context budget.
    oversized_total: bool
    #: Estimated tokens exceed the remaining budget at selection time.
    oversized_remaining: bool

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "strategy": self.strategy,
            "rank": self.rank,
            "path": self.path,
            "score": _rounded(self.score),
            "estimated_tokens": self.estimated_tokens,
            "full_tokens": self.full_tokens,
            "discovery_source": self.discovery_source,
            "inclusion_reason": self.inclusion_reason,
            "surface_tag": self.surface_tag.value,
            "selected": self.selected,
            "cumulative_tokens_before": self.cumulative_tokens_before,
            "cumulative_tokens_after": self.cumulative_tokens_after,
            "remaining_before": self.remaining_before,
            "remaining_after": self.remaining_after,
            "exclusion_reason": self.exclusion_reason,
            "exclusion_category": (
                self.exclusion_category.value if self.exclusion_category else None
            ),
            "firewall_decision": self.firewall_decision,
            "oversized_total": self.oversized_total,
            "oversized_remaining": self.oversized_remaining,
        }


@dataclass(frozen=True)
class BudgetSummary:
    """Per-package (or rolled-up) budget-spend aggregates."""

    budget_max_tokens: int | None
    total_selected_tokens: int
    budget_utilization: float | None
    unused_budget: int | None
    selected_count: int
    rejected_count: int
    oversized_total_count: int
    oversized_remaining_count: int
    #: Required files dropped because they alone exceed the whole budget.
    required_excluded_by_size: int
    #: Required files dropped because earlier selections consumed the budget.
    required_excluded_by_consumption: int
    #: Focused candidates synthesized during selection (P26.2 Step 4).
    focused_generated: int = 0
    #: Focused candidates that reached the final selection (P26.2 Step 4).
    focused_selected: int = 0
    excluded_by_category: tuple[tuple[str, int], ...] = ()
    avg_selected_tokens: float | None = None
    median_selected_tokens: float | None = None
    avg_required_rank: float | None = None
    median_required_rank: float | None = None

    def to_dict(self) -> dict:
        return {
            "budget_max_tokens": self.budget_max_tokens,
            "total_selected_tokens": self.total_selected_tokens,
            "budget_utilization": _rounded(self.budget_utilization),
            "unused_budget": self.unused_budget,
            "selected_count": self.selected_count,
            "rejected_count": self.rejected_count,
            "oversized_total_count": self.oversized_total_count,
            "oversized_remaining_count": self.oversized_remaining_count,
            "required_excluded_by_size": self.required_excluded_by_size,
            "required_excluded_by_consumption": self.required_excluded_by_consumption,
            "focused_generated": self.focused_generated,
            "focused_selected": self.focused_selected,
            "excluded_by_category": {
                category: count for category, count in self.excluded_by_category
            },
            "avg_selected_tokens": _rounded(self.avg_selected_tokens),
            "median_selected_tokens": _rounded(self.median_selected_tokens),
            "avg_required_rank": _rounded(self.avg_required_rank),
            "median_required_rank": _rounded(self.median_required_rank),
        }


@dataclass(frozen=True)
class BudgetSpendReport:
    """Full budget-spend analysis for one (task, strategy) package."""

    task_id: str
    strategy: str
    budget_max_tokens: int | None
    rows: tuple[BudgetSpendRow, ...]
    summary: BudgetSummary
    #: Focused-context substitution events (P26.2 Step 4), in capture order.
    focus_events: tuple[dict, ...] = ()

    def required_missing(self) -> tuple[BudgetSpendRow, ...]:
        """Required-surface files that did not reach the final context."""
        return tuple(
            row
            for row in self.rows
            if row.surface_tag is SurfaceTag.REQUIRED and not self._reached(row)
        )

    @staticmethod
    def _reached(row: BudgetSpendRow) -> bool:
        return row.selected and row.exclusion_category is not ExclusionCategory.FIREWALL_FILTERED

    def required_focused(self) -> tuple[BudgetSpendRow, ...]:
        """Required-surface files recovered specifically via focused substitution."""
        recovered = {
            row.path
            for row in self.rows
            if row.surface_tag is SurfaceTag.REQUIRED
            and (row.oversized_total or row.oversized_remaining)
        }
        selected = {
            row.path
            for row in self.rows
            if row.surface_tag is SurfaceTag.REQUIRED and row.selected
        }
        if not recovered:
            return ()
        return tuple(
            row
            for row in self.rows
            if row.surface_tag is SurfaceTag.REQUIRED
            and row.path in recovered
            and row.path in selected
        )

    def competing_selected(self, row: BudgetSpendRow) -> tuple[str, ...]:
        """Selected files that consumed budget before ``row`` was considered."""
        return tuple(
            other.path for other in self.rows if other.selected and other.rank < row.rank
        )

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "strategy": self.strategy,
            "budget_max_tokens": self.budget_max_tokens,
            "summary": self.summary.to_dict(),
            "rows": [row.to_dict() for row in self.rows],
            "focus_events": [dict(event) for event in self.focus_events],
        }


# ---------------------------------------------------------------------------
# Pure analysis (trace + task surface -> rows)
# ---------------------------------------------------------------------------


def analyze_budget_spend(
    task_id: str,
    strategy: str,
    trace: PipelineTrace,
    *,
    budget_max_tokens: int | None,
    required_files: Iterable[str] = (),
    supporting_files: Iterable[str] = (),
    expected_tests: Iterable[str] = (),
    firewall_decisions: Mapping[str, str] | None = None,
    focus_events: Iterable[Mapping] = (),
) -> BudgetSpendReport:
    """Reconstruct the per-candidate budget accounting from a captured trace.

    ``budget_max_tokens`` is the context budget the build used; the surface
    iterables are the task's required/supporting/test paths; ``firewall_decisions``
    (optional) is a path -> verdict mapping from
    :func:`repolens.required_file_diagnostics.firewall_decisions_for_package`;
    ``focus_events`` (optional, P26.2 Step 4) are the focused-context
    substitution events captured on the trace, used for focused accounting.

    The walk never re-runs selection: it follows ``trace.ranked`` in order,
    attributes each candidate to its traced outcome, and carries the traced
    token counts forward. Deterministic for identical inputs.
    """
    selected_by_path: dict[str, int] = {}
    for entry in trace.selected:
        path = entry["path"]
        selected_by_path[path] = (
            selected_by_path.get(path, 0) + int(entry.get("estimated_tokens") or 0)
        )
    excluded_by_path = {entry["path"]: entry for entry in trace.excluded}
    firewall = dict(firewall_decisions or {})

    rows: list[BudgetSpendRow] = []
    cumulative = 0
    for rank, entry in enumerate(trace.ranked, start=1):
        path = entry["path"]
        selected = path in selected_by_path
        full_estimate = int(entry.get("estimated_tokens") or 0)
        tokens = (
            selected_by_path[path] if selected else full_estimate
        )
        remaining_before = (
            None if budget_max_tokens is None else budget_max_tokens - cumulative
        )
        cumulative_after = cumulative + (tokens if selected else 0)
        remaining_after = (
            None
            if budget_max_tokens is None
            else budget_max_tokens - cumulative_after
        )

        exclusion_reason = None
        if not selected:
            excluded_entry = excluded_by_path.get(path)
            if excluded_entry is not None:
                exclusion_reason = excluded_entry.get("reason")

        firewall_decision = firewall.get(path)
        category = categorize_reason(exclusion_reason)
        # A selected file that the firewall blocks still spent budget at
        # selection, but from the agent's perspective it is a rejection.
        if selected and firewall_decision == "blocked":
            category = ExclusionCategory.FIREWALL_FILTERED

        oversized_total = (
            budget_max_tokens is not None and full_estimate > budget_max_tokens
        )
        oversized_remaining = (
            budget_max_tokens is not None
            and remaining_before is not None
            and full_estimate > remaining_before
        )

        rows.append(
            BudgetSpendRow(
                task_id=task_id,
                strategy=strategy,
                rank=rank,
                path=path,
                score=_best_score(entry),
                estimated_tokens=tokens,
                full_tokens=full_estimate,
                discovery_source=(
                    trace.discovery_source(path).value
                    if trace.discovery_source(path) is not None
                    else None
                ),
                inclusion_reason=entry.get("inclusion_reason"),
                surface_tag=surface_tag_for_path(
                    path, required_files, supporting_files, expected_tests
                ),
                selected=selected,
                cumulative_tokens_before=cumulative,
                cumulative_tokens_after=cumulative_after,
                remaining_before=remaining_before,
                remaining_after=remaining_after,
                exclusion_reason=exclusion_reason,
                exclusion_category=category,
                firewall_decision=firewall_decision,
                oversized_total=oversized_total,
                oversized_remaining=oversized_remaining,
            )
        )
        cumulative = cumulative_after

    summary = _summarize_rows(rows, budget_max_tokens)
    focused = tuple(dict(event) for event in focus_events)
    if focused:
        summary = replace(
            summary,
            focused_generated=len(focused),
            focused_selected=sum(
                1
                for entry in trace.selected
                if entry.get("focus_start_line") is not None
            ),
        )
    return BudgetSpendReport(
        task_id=task_id,
        strategy=strategy,
        budget_max_tokens=budget_max_tokens,
        rows=tuple(rows),
        summary=summary,
        focus_events=focused,
    )


def _best_score(entry: dict) -> float | None:
    retrieval_score = entry.get("retrieval_score")
    if retrieval_score is not None:
        return float(retrieval_score)
    change_score = entry.get("change_score")
    if change_score is not None:
        return float(change_score)
    return None


def _summarize_rows(
    rows: Sequence[BudgetSpendRow],
    budget_max_tokens: int | None,
) -> BudgetSummary:
    selected = [row for row in rows if row.selected]
    total_selected = sum(row.estimated_tokens for row in selected)
    rejected = [row for row in rows if not row.selected]
    oversized_total = [row for row in rows if row.oversized_total]
    oversized_remaining = [row for row in rows if row.oversized_remaining]

    category_counts: dict[ExclusionCategory, int] = {}
    for row in rejected + [row for row in rows if row.selected and row.exclusion_category]:
        if row.exclusion_category is not None:
            category_counts[row.exclusion_category] = (
                category_counts.get(row.exclusion_category, 0) + 1
            )

    required_rows = [
        row for row in rows if row.surface_tag is SurfaceTag.REQUIRED
    ]
    required_by_size = sum(
        1
        for row in required_rows
        if row.exclusion_category is ExclusionCategory.EXCEEDS_TOTAL_BUDGET
    )
    required_by_consumption = sum(
        1
        for row in required_rows
        if row.exclusion_category is ExclusionCategory.REMAINING_BUDGET_TOO_SMALL
    )

    return BudgetSummary(
        budget_max_tokens=budget_max_tokens,
        total_selected_tokens=total_selected,
        budget_utilization=(
            (total_selected / budget_max_tokens)
            if budget_max_tokens
            else None
        ),
        unused_budget=(
            budget_max_tokens - total_selected if budget_max_tokens is not None else None
        ),
        selected_count=len(selected),
        rejected_count=len(rejected)
        + sum(1 for row in selected if row.exclusion_category is ExclusionCategory.FIREWALL_FILTERED),
        oversized_total_count=len(oversized_total),
        oversized_remaining_count=len(oversized_remaining),
        required_excluded_by_size=required_by_size,
        required_excluded_by_consumption=required_by_consumption,
        excluded_by_category=tuple(
            sorted((category.value, count) for category, count in category_counts.items())
        ),
        avg_selected_tokens=_mean([row.estimated_tokens for row in selected]),
        median_selected_tokens=_median([row.estimated_tokens for row in selected]),
        avg_required_rank=_mean([row.rank for row in required_rows]),
        median_required_rank=_median([row.rank for row in required_rows]),
    )


@dataclass(frozen=True)
class BudgetDiagnosticsReport:
    """Deterministic aggregate budget-spend report over many packages."""

    reports: tuple[BudgetSpendReport, ...]

    @property
    def strategies(self) -> tuple[str, ...]:
        seen: list[str] = []
        for report in self.reports:
            if report.strategy not in seen:
                seen.append(report.strategy)
        return tuple(seen)

    def for_strategy(self, strategy: str | None) -> tuple[BudgetSpendReport, ...]:
        if strategy is None:
            return self.reports
        return tuple(r for r in self.reports if r.strategy == strategy)

    def aggregate(self, strategy: str | None = None) -> BudgetSummary:
        """Roll up every package under ``strategy`` (or all) into one summary.

        Row-level metrics (counts, sizes, ranks, oversized flags) are summed /
        averaged over every row across the packages. ``budget_utilization`` is
        the *mean* per-package utilization and ``unused_budget`` the total
        unused tokens across packages — both meaningful only when every package
        in the roll-up shares one budget.
        """
        reports = self.for_strategy(strategy)
        rows: list[BudgetSpendRow] = []
        for report in reports:
            rows.extend(report.rows)
        budgets = {report.budget_max_tokens for report in reports}
        budget = budgets.pop() if len(budgets) == 1 else None
        summary = _summarize_rows(rows, budget)
        focused_generated = sum(
            report.summary.focused_generated for report in reports
        )
        focused_selected = sum(
            report.summary.focused_selected for report in reports
        )
        if focused_generated or focused_selected:
            summary = replace(
                summary,
                focused_generated=focused_generated,
                focused_selected=focused_selected,
            )
        if budget is not None:
            utilizations = [
                report.summary.budget_utilization
                for report in reports
                if report.summary.budget_utilization is not None
            ]
            mean_utilization = (
                float(statistics.mean(utilizations)) if utilizations else None
            )
            unused_total = sum(
                unused
                for unused in (
                    report.summary.unused_budget for report in reports
                )
                if unused is not None
            )
            summary = replace(
                summary,
                budget_utilization=(
                    round(mean_utilization, 6) if mean_utilization is not None else None
                ),
                unused_budget=unused_total,
            )
        return summary

    def top_oversized_files(
        self, strategy: str | None = None, *, top: int = 10
    ) -> tuple[dict, ...]:
        """Files whose estimated tokens exceed the total budget, most frequent first.

        ``count`` is how many packages surfaced the file oversized;
        ``max_tokens`` the largest estimate seen. Deterministic ordering.
        """
        buckets: dict[str, dict] = {}
        for package in self.for_strategy(strategy):
            for row in package.rows:
                if not row.oversized_total:
                    continue
                bucket = buckets.setdefault(
                    row.path,
                    {"path": row.path, "count": 0, "max_tokens": 0, "tasks": []},
                )
                bucket["count"] += 1
                bucket["max_tokens"] = max(
                    bucket["max_tokens"], row.full_tokens or row.estimated_tokens
                )
                if row.task_id not in bucket["tasks"]:
                    bucket["tasks"].append(row.task_id)
        ordered = sorted(
            buckets.values(),
            key=lambda b: (-b["count"], -b["max_tokens"], b["path"]),
        )
        return tuple(ordered[:top])

    def top_required_lost_to_budget(
        self, strategy: str | None = None, *, top: int = 10
    ) -> tuple[dict, ...]:
        """Required files lost to budget selection, most frequent first.

        ``by_size`` counts losses where the file alone exceeds the whole
        budget; ``by_consumption`` where earlier selections consumed the
        budget; ``max_tokens`` the largest estimate seen.
        """
        buckets: dict[str, dict] = {}
        for package in self.for_strategy(strategy):
            for row in package.rows:
                if (
                    row.surface_tag is not SurfaceTag.REQUIRED
                    or row.exclusion_category is None
                    or row.exclusion_category
                    not in (
                        ExclusionCategory.EXCEEDS_TOTAL_BUDGET,
                        ExclusionCategory.REMAINING_BUDGET_TOO_SMALL,
                    )
                ):
                    continue
                bucket = buckets.setdefault(
                    row.path,
                    {
                        "path": row.path,
                        "lost_count": 0,
                        "by_size": 0,
                        "by_consumption": 0,
                        "max_tokens": 0,
                        "tasks": [],
                    },
                )
                bucket["lost_count"] += 1
                if row.exclusion_category is ExclusionCategory.EXCEEDS_TOTAL_BUDGET:
                    bucket["by_size"] += 1
                else:
                    bucket["by_consumption"] += 1
                bucket["max_tokens"] = max(bucket["max_tokens"], row.estimated_tokens)
                if row.task_id not in bucket["tasks"]:
                    bucket["tasks"].append(row.task_id)
        ordered = sorted(
            buckets.values(),
            key=lambda b: (-b["lost_count"], -b["max_tokens"], b["path"]),
        )
        return tuple(ordered[:top])

    def to_dict(self) -> dict:
        return {
            "strategies": list(self.strategies),
            "aggregate": {
                strategy: self.aggregate(strategy).to_dict()
                for strategy in self.strategies
            },
            "top_oversized_files": {
                strategy: [dict(entry) for entry in self.top_oversized_files(strategy)]
                for strategy in self.strategies
            },
            "top_required_lost_to_budget": {
                strategy: [dict(entry) for entry in self.top_required_lost_to_budget(strategy)]
                for strategy in self.strategies
            },
            "reports": [report.to_dict() for report in self.reports],
        }


# ---------------------------------------------------------------------------
# Small numeric helpers
# ---------------------------------------------------------------------------


def _mean(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return round(float(statistics.mean(values)), 6)


def _median(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return round(float(statistics.median(values)), 6)


def _rounded(value: float | None) -> float | None:
    if value is None:
        return None
    return round(float(value), 6)


def _posix_str(value: object) -> str:
    if isinstance(value, Path):
        return value.as_posix()
    return str(value).replace("\\", "/")