"""Opt-in, observational tracing of the context pipeline (P26.1).

This module gives the :class:`~repolens.context.ContextEngine` an optional,
purely observational *tracer* so tooling can answer "why did a file not reach
the final context?" without re-implementing retrieval or touching selection
behaviour.

Design:

- :class:`ContextPipelineTracer` is a no-op-by-default observer. The engine
  emits stage-boundary observations only when a tracer is attached
  (``ContextEngine(..., tracer=tracer)``); with ``tracer=None`` the pipeline is
  byte-for-byte unchanged.
- Observations are *snapshots* (plain JSON-safe dicts / path strings), never
  live objects, so a tracer cannot mutate or influence the pipeline.
- :class:`TraceCollector` accumulates the snapshots into a :class:`PipelineTrace`
  consumed by :mod:`repolens.required_file_diagnostics`.
- Stage ordering follows the engine: retrieval, symbol matching, dependency
  expansion, architecture candidates, and (for change-aware builds)
  change-plan candidates are *discovery* stages; ranking and selection are the
  final two stages.

The snapshot helpers below are shared by the engine and the collector so the
two never drift (no duplicated serialization logic).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Sequence


class ContextStage(str, Enum):
    """Stage boundaries the tracer can observe.

    ``RETRIEVAL``, ``SYMBOL``, ``DEPENDENCY``, ``ARCHITECTURE`` and
    ``CHANGE_PLAN`` are discovery stages; ``RANKING`` and ``SELECTION`` close
    the pipeline.
    """

    RETRIEVAL = "retrieval"
    SYMBOL = "symbol"
    DEPENDENCY = "dependency"
    ARCHITECTURE = "architecture"
    CHANGE_PLAN = "change_plan"
    RANKING = "ranking"
    SELECTION = "selection"


def _posix(path: object) -> str:
    if isinstance(path, Path):
        return path.as_posix()
    return str(path)


# ---------------------------------------------------------------------------
# Snapshot helpers (single source of trace serialization)
# ---------------------------------------------------------------------------


def snapshot_retrieval(meta: dict) -> dict:
    """Snapshot one primary-retrieval metadata dict (see engine._retrieval_metadata)."""
    return {
        "path": _posix(meta["file_path"]),
        "retrieval_rank": meta.get("retrieval_rank"),
        "retrieval_score": meta.get("retrieval_score"),
        "lexical_rank": meta.get("lexical_rank"),
        "semantic_rank": meta.get("semantic_rank"),
        "signals": list(meta.get("signals") or ()),
    }


def snapshot_dependency(node) -> dict:
    """Snapshot one dependency-expansion node (``ExpandedNode``)."""
    return {
        "path": _posix(node.path),
        "role": node.role.value if getattr(node.role, "value", None) else str(node.role),
        "distance": node.distance,
    }


def snapshot_architecture(match) -> dict:
    """Snapshot one architecture match (``ArchitectureCandidate``)."""
    return {
        "path": _posix(match.path),
        "rank": match.rank,
        "reason": match.reason,
    }


def snapshot_candidate(candidate) -> dict:
    """Snapshot one :class:`~repolens.context.ContextCandidate`.

    Carries every field the diagnostic needs (retrieval signals, graph
    distance, architecture and change-plan metadata, tokens, reasons).
    """
    return {
        "path": _posix(candidate.path),
        "role": (
            candidate.role.value
            if getattr(candidate.role, "value", None)
            else str(candidate.role)
        ),
        "estimated_tokens": candidate.estimated_tokens,
        "selection_reason": candidate.selection_reason,
        "inclusion_reason": candidate.inclusion_reason,
        "retrieval_rank": candidate.retrieval_rank,
        "retrieval_score": candidate.retrieval_score,
        "lexical_rank": candidate.lexical_rank,
        "semantic_rank": candidate.semantic_rank,
        "graph_distance": candidate.graph_distance,
        "architecture_rank": candidate.architecture_rank,
        "module": candidate.module,
        "symbol": candidate.symbol,
        "change_score": candidate.change_score,
        "change_category": candidate.change_category,
        "change_confidence": candidate.change_confidence,
        "change_relationship": candidate.change_relationship,
        "change_priority": candidate.change_priority,
    }


def snapshot_excluded(candidate) -> dict:
    """Snapshot one :class:`~repolens.context.ExcludedCandidate`."""
    return {
        "path": _posix(candidate.path),
        "estimated_tokens": candidate.estimated_tokens,
        "reason": candidate.reason,
    }


# ---------------------------------------------------------------------------
# Tracer protocol + emit helpers
# ---------------------------------------------------------------------------


class ContextPipelineTracer:
    """Optional observer of context-pipeline stage boundaries.

    Every hook defaults to a no-op; the engine emits nothing when no tracer is
    attached. Hooks receive only immutable snapshot dicts/strings.
    """

    def on_retrieval(self, entries: Sequence[dict]) -> None:
        """Primary retrieval results, in retrieval order."""

    def on_symbols(self, paths: Sequence[str]) -> None:
        """Symbol-match file paths (sorted)."""

    def on_dependency(self, entries: Sequence[dict]) -> None:
        """Dependency-expanded nodes (role, distance)."""

    def on_architecture(self, entries: Sequence[dict]) -> None:
        """Architecture matches (rank, reason)."""

    def on_change_plan(self, entries: Sequence[dict]) -> None:
        """Change-plan candidates (full planned set, pre-dedupe)."""

    def on_ranked(self, entries: Sequence[dict]) -> None:
        """All candidates after dedupe and ranking, in rank order."""

    def on_selection(self, selected: Sequence[dict], excluded: Sequence[dict]) -> None:
        """Surviving selection and budget-excluded candidates."""


def observe_standard_stages(
    tracer: ContextPipelineTracer | None,
    *,
    retrieval_meta: Sequence[dict],
    symbol_paths,
    dependency_nodes,
    arch_matches,
) -> None:
    """Emit the discovery-stage observations shared by both engine paths.

    No-op when ``tracer`` is None. ``retrieval_meta`` are the engine's
    ``_retrieval_metadata`` dicts; ``symbol_paths`` an iterable of ``Path``;
    ``dependency_nodes`` the ``ExpandedNode`` list; ``arch_matches`` the
    architecture-match list.
    """
    if tracer is None:
        return
    tracer.on_symbols(sorted(_posix(p) for p in symbol_paths))
    tracer.on_retrieval([snapshot_retrieval(meta) for meta in retrieval_meta])
    if dependency_nodes:
        tracer.on_dependency([snapshot_dependency(node) for node in dependency_nodes])
    if arch_matches:
        tracer.on_architecture([snapshot_architecture(match) for match in arch_matches])


def observe_change_candidates(
    tracer: ContextPipelineTracer | None,
    change_candidates: Sequence,
) -> None:
    """Emit the full planned change-candidate set (pre-dedupe)."""
    if tracer is None:
        return
    tracer.on_change_plan([snapshot_candidate(c) for c in change_candidates])


def observe_rank_selection(
    tracer: ContextPipelineTracer | None,
    *,
    ranked: Sequence,
    selected: Sequence,
    excluded: Sequence,
) -> None:
    """Emit the ranked list and the selection/exclusion result."""
    if tracer is None:
        return
    tracer.on_ranked([snapshot_candidate(c) for c in ranked])
    tracer.on_selection(
        [snapshot_candidate(c) for c in selected],
        [snapshot_excluded(e) for e in excluded],
    )


# ---------------------------------------------------------------------------
# Collector
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PipelineTrace:
    """Captured, stage-grouped trace of one context build.

    All paths are repository-relative POSIX strings. Discovery stages keep the
    per-stage entry snapshots; ``ranked`` is the deduped, ranked ordered list;
    ``selected`` / ``excluded`` the final selection outcome.
    """

    retrieval: tuple[dict, ...] = ()
    symbols: tuple[str, ...] = ()
    dependency: tuple[dict, ...] = ()
    architecture: tuple[dict, ...] = ()
    change_plan: tuple[dict, ...] = ()
    ranked: tuple[dict, ...] = ()
    selected: tuple[dict, ...] = ()
    excluded: tuple[dict, ...] = ()
    has_change_plan: bool = field(default=False)

    @property
    def selected_paths(self) -> frozenset[str]:
        return frozenset(entry["path"] for entry in self.selected)

    @property
    def ranked_paths(self) -> frozenset[str]:
        return frozenset(entry["path"] for entry in self.ranked)

    #: Discovery precedence (highest priority discovered stage first). A file
    #: seen by several stages is attributed to the most specific source.
    DISCOVERY_PRECEDENCE: tuple[ContextStage, ...] = (
        ContextStage.RETRIEVAL,
        ContextStage.SYMBOL,
        ContextStage.DEPENDENCY,
        ContextStage.ARCHITECTURE,
        ContextStage.CHANGE_PLAN,
    )

    def entries_for(self, stage: ContextStage) -> tuple[dict, ...]:
        """Entries recorded under ``stage`` (discovery stages only)."""
        if stage is ContextStage.RETRIEVAL:
            return self.retrieval
        if stage is ContextStage.SYMBOL:
            return tuple({"path": path} for path in self.symbols)
        if stage is ContextStage.DEPENDENCY:
            return self.dependency
        if stage is ContextStage.ARCHITECTURE:
            return self.architecture
        if stage is ContextStage.CHANGE_PLAN:
            return self.change_plan
        raise ValueError(f"{stage} is not a discovery stage")

    def discovered_by_stage(self) -> dict[ContextStage, frozenset[str]]:
        """Map each discovery stage to the paths it produced."""
        return {
            stage: frozenset(entry["path"] for entry in self.entries_for(stage))
            for stage in self.DISCOVERY_PRECEDENCE
        }

    def discovered_paths(self) -> frozenset[str]:
        """Every path that entered the pipeline at any discovery stage."""
        paths: set[str] = set()
        for stage in self.DISCOVERY_PRECEDENCE:
            paths.update(
                entry["path"] for entry in self.entries_for(stage)
            )
        return frozenset(paths)

    def discovery_source(self, path: str) -> ContextStage | None:
        """Highest-priority discovery stage for ``path`` (or None)."""
        seen = self.discovered_by_stage()
        for stage in self.DISCOVERY_PRECEDENCE:
            if path in seen[stage]:
                return stage
        return None

    def candidate_entry(self, path: str) -> dict | None:
        """First ranked-candidate snapshot whose path is ``path`` (or None)."""
        for entry in self.ranked:
            if entry["path"] == path:
                return entry
        return None


class TraceCollector(ContextPipelineTracer):
    """Accumulates tracer observations into a :class:`PipelineTrace`.

    Hooks are pure accumulators in stage order; the assembled trace is frozen
    and JSON-safe.
    """

    def __init__(self) -> None:
        self._retrieval: list[dict] = []
        self._symbols: list[str] = []
        self._dependency: list[dict] = []
        self._architecture: list[dict] = []
        self._change_plan: list[dict] = []
        self._ranked: list[dict] = []
        self._selected: list[dict] = []
        self._excluded: list[dict] = []
        self._has_change_plan = False

    def on_retrieval(self, entries: Sequence[dict]) -> None:
        self._retrieval.extend(dict(e) for e in entries)

    def on_symbols(self, paths: Sequence[str]) -> None:
        self._symbols.extend(str(p) for p in paths)

    def on_dependency(self, entries: Sequence[dict]) -> None:
        self._dependency.extend(dict(e) for e in entries)

    def on_architecture(self, entries: Sequence[dict]) -> None:
        self._architecture.extend(dict(e) for e in entries)

    def on_change_plan(self, entries: Sequence[dict]) -> None:
        self._change_plan.extend(dict(e) for e in entries)
        self._has_change_plan = True

    def on_ranked(self, entries: Sequence[dict]) -> None:
        self._ranked.extend(dict(e) for e in entries)

    def on_selection(self, selected: Sequence[dict], excluded: Sequence[dict]) -> None:
        self._selected.extend(dict(e) for e in selected)
        self._excluded.extend(dict(e) for e in excluded)

    def trace(self) -> PipelineTrace:
        """Return the frozen, stage-ordered trace."""
        return PipelineTrace(
            retrieval=tuple(self._retrieval),
            symbols=tuple(sorted(set(self._symbols))),
            dependency=tuple(self._dependency),
            architecture=tuple(self._architecture),
            change_plan=tuple(self._change_plan),
            ranked=tuple(self._ranked),
            selected=tuple(self._selected),
            excluded=tuple(self._excluded),
            has_change_plan=self._has_change_plan,
        )