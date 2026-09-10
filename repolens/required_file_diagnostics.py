"""Required-file outcome diagnostics (P26.1).

Given a P25 agent-evaluation task and the :class:`~repolens.context_trace.PipelineTrace`
a context build produced, this module explains, for every *required file*,
which pipeline stage cost its presence in the final context:

- :attr:`RequiredFileOutcome.NOT_DISCOVERED` — the file never became any kind
  of candidate (never entered retrieval, symbol, dependency, architecture, or
  change-plan discovery);
- :attr:`RequiredFileOutcome.DISCOVERED_NOT_RANKED` — the file was discovered
  by some stage but did not survive into the deduped, ranked candidate list;
- :attr:`RequiredFileOutcome.RANKED_NOT_SELECTED` — the file was ranked but
  was lost during budget selection (typically ``over_budget`` /
  ``exceeds_total_budget``);
- :attr:`RequiredFileOutcome.SELECTED` — the file reached the final context.

Classification is a pure function of the frozen trace plus the task surface —
deterministic, offline, and it never re-runs retrieval or ranking (the trace
is produced by the engine itself through the existing candidate models).

Design constraints honoured here:

- deterministic: identical trace + task → identical rows;
- no LLM calls;
- no retrieval/ranking/context-selection changes: this module only reads
  already-produced artifacts;
- reuses the existing candidate model: snapshots come from
  :mod:`repolens.context_trace`, firewall decisions from the existing
  :class:`~repolens.context.firewall.firewall.ContextFirewall` output.

The companion CLI (``benchmarks/required_file_diagnostics.py``) runs this
against the P25 evaluation corpus and prints an actionable report.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Sequence

from repolens.context_trace import ContextStage, PipelineTrace

# ---------------------------------------------------------------------------
# Outcome model
# ---------------------------------------------------------------------------


class RequiredFileOutcome(str, Enum):
    """Why a required file did (or did not) reach the final context."""

    NOT_DISCOVERED = "NOT_DISCOVERED"
    DISCOVERED_NOT_RANKED = "DISCOVERED_NOT_RANKED"
    RANKED_NOT_SELECTED = "RANKED_NOT_SELECTED"
    SELECTED = "SELECTED"


#: Ranking-tier labels mirroring the engine's deterministic ranking policy
#: (see :mod:`repolens.context.ranking`). They are *labels* derived from a
#: candidate's metadata, never a re-implementation of ranking.
_TIER_LABELS: dict[int, str] = {
    0: "retrieval_primary",
    1: "architecture_direct",
    2: "architecture_neighbour",
    3: "architecture_proximity",
    4: "dependency",
    5: "change_plan",
}

#: Firewall decisions a required file can carry, in determinized form.
_SAFE_DECISIONS = frozenset({"allowed", "redact"})


def _tier_for(entry: dict) -> tuple[int, str]:
    """Nominal ranking tier for a ranked-candidate snapshot.

    Mirrors the bucket structure of ``repolens.context.ranking._candidate_key``
    without re-running ranking: primary → 0; change-plan-only → 5;
    dependency/architecture → the architecture bucket (0..2) or 4 for generic
    dependencies (no architecture signal).
    """
    if entry.get("role") == "primary":
        return 0, _TIER_LABELS[0]
    if entry.get("inclusion_reason") == "change_plan":
        return 5, _TIER_LABELS[5]
    arch_rank = entry.get("architecture_rank")
    if arch_rank is not None:
        bucket = min(int(arch_rank), 2)
        return 1 + bucket, _TIER_LABELS[1 + bucket]
    return 4, _TIER_LABELS[4]


def _candidate_score(entry: dict | None) -> float | None:
    """Best-available numeric score for a candidate snapshot (or None)."""
    if entry is None:
        return None
    retrieval_score = entry.get("retrieval_score")
    if retrieval_score is not None:
        return float(retrieval_score)
    change_score = entry.get("change_score")
    if change_score is not None:
        return float(change_score)
    return None


def _est_tokens(entry: dict | None, excluded_entry: dict | None) -> int | None:
    if excluded_entry is not None:
        return int(excluded_entry.get("estimated_tokens") or 0)
    if entry is not None:
        return int(entry.get("estimated_tokens")) if entry.get("estimated_tokens") is not None else None
    return None


@dataclass(frozen=True)
class RequiredFileDiagnostic:
    """One required file's full pipeline outcome for one (task, strategy)."""

    task_id: str
    strategy: str
    path: str
    outcome: RequiredFileOutcome
    #: Highest-priority discovery stage (or None when never discovered).
    discovery_source: str | None
    inclusion_reason: str | None
    candidate_score: float | None
    ranking_position: int | None
    ranking_tier: int | None
    ranking_tier_label: str | None
    graph_distance: int | None
    estimated_tokens: int | None
    budget_max_tokens: int | None
    excluded_reason: str | None
    firewall_decision: str | None
    #: Discovery-source membership flags (a file may have several sources).
    from_retrieval: bool = False
    from_symbol: bool = False
    from_dependency: bool = False
    from_architecture: bool = False
    from_change_plan: bool = False

    @property
    def is_selected(self) -> bool:
        return self.outcome is RequiredFileOutcome.SELECTED

    @property
    def is_missing(self) -> bool:
        return not self.is_selected

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "strategy": self.strategy,
            "path": self.path,
            "outcome": self.outcome.value,
            "discovery_source": self.discovery_source,
            "inclusion_reason": self.inclusion_reason,
            "candidate_score": self.candidate_score,
            "ranking_position": self.ranking_position,
            "ranking_tier": self.ranking_tier,
            "ranking_tier_label": self.ranking_tier_label,
            "graph_distance": self.graph_distance,
            "estimated_tokens": self.estimated_tokens,
            "budget_max_tokens": self.budget_max_tokens,
            "excluded_reason": self.excluded_reason,
            "firewall_decision": self.firewall_decision,
            "from_retrieval": self.from_retrieval,
            "from_symbol": self.from_symbol,
            "from_dependency": self.from_dependency,
            "from_architecture": self.from_architecture,
            "from_change_plan": self.from_change_plan,
        }


# ---------------------------------------------------------------------------
# Classification (pure function of trace + task surface)
# ---------------------------------------------------------------------------


def classify_required_files(
    task_id: str,
    strategy: str,
    required_files: Sequence[str],
    trace: PipelineTrace,
    *,
    budget_max_tokens: int | None,
    firewall_decisions: Mapping[str, str] | None = None,
) -> tuple[RequiredFileDiagnostic, ...]:
    """Classify every required file against a captured pipeline trace.

    ``budget_max_tokens`` is the context budget used for the build.
    ``firewall_decisions`` (optional) maps a repository-relative path to a
    deterministic firewall verdict string for files that appeared in the
    produced package (``allowed`` / ``redact`` / ``blocked`` / ``not_present``).
    """
    selected_paths = trace.selected_paths
    ranked_paths = trace.ranked_paths
    ranked_order = [entry["path"] for entry in trace.ranked]
    discovered_by_stage = trace.discovered_by_stage()
    discovered_paths = trace.discovered_paths()
    excluded_by_path = {e["path"]: e for e in trace.excluded}

    rows: list[RequiredFileDiagnostic] = []
    for path in required_files:
        path = _posix_str(path)
        candidate_entry = trace.candidate_entry(path)
        excluded_entry = excluded_by_path.get(path)

        if path in selected_paths:
            outcome = RequiredFileOutcome.SELECTED
        elif path in ranked_paths:
            outcome = RequiredFileOutcome.RANKED_NOT_SELECTED
        elif path in discovered_paths:
            outcome = RequiredFileOutcome.DISCOVERED_NOT_RANKED
        else:
            outcome = RequiredFileOutcome.NOT_DISCOVERED

        if candidate_entry is not None:
            position = (ranked_order.index(path) + 1) if path in ranked_order else None
            tier, tier_label = _tier_for(candidate_entry)
            inclusion = candidate_entry.get("inclusion_reason")
            tokens = _est_tokens(candidate_entry, excluded_entry)
            graph_distance = candidate_entry.get("graph_distance")
        else:
            position = None
            tier, tier_label = None, None
            inclusion = None
            tokens = _est_tokens(None, excluded_entry)
            graph_distance = excluded_entry.get("distance") if excluded_entry else None

        source = trace.discovery_source(path)
        source_value = source.value if source is not None else None
        membership = {
            ContextStage.RETRIEVAL: path in discovered_by_stage[ContextStage.RETRIEVAL],
            ContextStage.SYMBOL: path in discovered_by_stage[ContextStage.SYMBOL],
            ContextStage.DEPENDENCY: path in discovered_by_stage[ContextStage.DEPENDENCY],
            ContextStage.ARCHITECTURE: path in discovered_by_stage[ContextStage.ARCHITECTURE],
            ContextStage.CHANGE_PLAN: path in discovered_by_stage[ContextStage.CHANGE_PLAN],
        }

        firewall = None
        if firewall_decisions is not None:
            firewall = firewall_decisions.get(path)

        rows.append(
            RequiredFileDiagnostic(
                task_id=task_id,
                strategy=strategy,
                path=path,
                outcome=outcome,
                discovery_source=source_value,
                inclusion_reason=inclusion,
                candidate_score=_candidate_score(candidate_entry),
                ranking_position=position,
                ranking_tier=tier,
                ranking_tier_label=tier_label,
                graph_distance=graph_distance,
                estimated_tokens=tokens,
                budget_max_tokens=budget_max_tokens,
                excluded_reason=excluded_entry.get("reason") if excluded_entry else None,
                firewall_decision=firewall,
                from_retrieval=membership[ContextStage.RETRIEVAL],
                from_symbol=membership[ContextStage.SYMBOL],
                from_dependency=membership[ContextStage.DEPENDENCY],
                from_architecture=membership[ContextStage.ARCHITECTURE],
                from_change_plan=membership[ContextStage.CHANGE_PLAN],
            )
        )
    return tuple(rows)


def _posix_str(value: object) -> str:
    text = str(value)
    return text.replace("\\", "/")


# ---------------------------------------------------------------------------
# Aggregation and report
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TaskDiagnostics:
    """Outcome rows for one (task, strategy) package."""

    task_id: str
    strategy: str
    budget_max_tokens: int | None
    rows: tuple[RequiredFileDiagnostic, ...]

    def outcome_counts(self) -> dict[str, int]:
        counts = {outcome.value: 0 for outcome in RequiredFileOutcome}
        for row in self.rows:
            counts[row.outcome.value] += 1
        return counts

    def required_file_recall(self) -> float:
        selected = sum(1 for row in self.rows if row.is_selected)
        return selected / len(self.rows) if self.rows else 0.0

    @property
    def missing_rows(self) -> tuple[RequiredFileDiagnostic, ...]:
        return tuple(row for row in self.rows if row.is_missing)

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "strategy": self.strategy,
            "budget_max_tokens": self.budget_max_tokens,
            "required_file_recall": round(self.required_file_recall(), 6),
            "outcome_counts": self.outcome_counts(),
            "rows": [row.to_dict() for row in self.rows],
        }


@dataclass(frozen=True)
class RequiredFileDiagnosticsReport:
    """Deterministic aggregate report over all (task, strategy) diagnostics."""

    tasks: tuple[TaskDiagnostics, ...]

    @property
    def strategies(self) -> tuple[str, ...]:
        seen: list[str] = []
        for task in self.tasks:
            if task.strategy not in seen:
                seen.append(task.strategy)
        return tuple(seen)

    def for_strategy(self, strategy: str) -> tuple[TaskDiagnostics, ...]:
        return tuple(task for task in self.tasks if task.strategy == strategy)

    def outcome_counts(self, strategy: str | None = None) -> dict[str, int]:
        tasks = self.for_strategy(strategy) if strategy is not None else self.tasks
        counts = {outcome.value: 0 for outcome in RequiredFileOutcome}
        for task in tasks:
            for outcome, count in task.outcome_counts().items():
                counts[outcome] += count
        return counts

    def recurrences(
        self, strategy: str | None = None, *, top: int = 10
    ) -> tuple[dict, ...]:
        """Most frequent (outcome, discovery source) failure patterns.

        Deterministically ordered: frequency descending, then pattern key
        lexicographically. Only *missing* required files count.
        """
        buckets: dict[tuple, dict] = {}
        tasks = self.for_strategy(strategy) if strategy is not None else self.tasks
        for task in tasks:
            for row in task.missing_rows:
                key = (
                    row.outcome.value,
                    row.discovery_source,
                    row.inclusion_reason or "no_inclusion",
                )
                bucket = buckets.setdefault(
                    key,
                    {
                        "outcome": row.outcome.value,
                        "discovery_source": row.discovery_source,
                        "inclusion_reason": row.inclusion_reason,
                        "count": 0,
                        "task_ids": [],
                    },
                )
                bucket["count"] += 1
                if row.task_id not in bucket["task_ids"]:
                    bucket["task_ids"].append(row.task_id)
        ordered = sorted(
            buckets.values(),
            key=lambda b: (-b["count"], b["outcome"], b["discovery_source"] or "", b["inclusion_reason"] or ""),
        )
        return tuple(ordered[:top])

    def to_dict(self) -> dict:
        return {
            "tasks": [task.to_dict() for task in self.tasks],
            "outcome_counts": self.outcome_counts(),
            "recurrences": self.recurrences(),
        }


# ---------------------------------------------------------------------------
# Firewall-decision adapter (reuses the existing ContextFirewall output)
# ---------------------------------------------------------------------------


def firewall_decisions_for_package(safe_package) -> dict[str, str]:
    """Map repo-relative paths to firewall verdicts from a ``SafeContextPackage``.

    ``allowed`` / ``redact`` come from ``safe_files``; ``blocked`` from
    ``blocked_files``; ``not_present`` for paths absent from both. Returns an
    empty mapping when ``safe_package`` is None.
    """
    if safe_package is None:
        return {}
    decisions: dict[str, str] = {}
    for candidate in getattr(safe_package, "safe_files", ()):
        decision = getattr(candidate, "decision", "allowed")
        decisions[_posix_str(candidate.path)] = (
            decision if decision in _SAFE_DECISIONS else "allowed"
        )
    for candidate in getattr(safe_package, "blocked_files", ()):
        decisions[_posix_str(candidate.path)] = "blocked"
    return decisions