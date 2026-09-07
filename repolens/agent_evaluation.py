"""Deterministic agent-task evaluation foundation (P25.2).

A coding agent is only useful if the context it receives actually covers the
files a realistic repository-level task touches. This module evaluates that
claim offline and deterministically: each :class:`EvaluationTask` describes a
plausible developer request (with the surface of files a correct edit would
touch), each :class:`CandidateSet` is one strategy's answer (retrieved files,
matched symbols, selected-file count, context size), and
:class:`AgentEvaluationRunner` compares the two side by side.

The produced :class:`EvaluationResult` reports, per case:

- precision / recall / F1 of the retrieved file set against the expected
  surface;
- the relevant files that were retrieved, missed, and the irrelevant files
  that were retrieved;
- which expected symbols and expected test files surfaced;
- context size (tokens), number of selected files, and latency where
  measurable.

Nothing here calls a model or the network: strategies are built on top of the
existing RepoLens retrieval and context engines, and every ordering is
deterministic by construction (retrieved candidates are kept in the order the
strategy returned them; misses are sorted; expected surfaces are normalized
up front).

The module deliberately does not re-implement retrieval: :func:`produce_*`
helpers simply translate existing searcher and context-engine outputs into an
:class:`CandidateSet`. The benchmark driving the whole thing lives in
``benchmarks/agent_evaluation.py``.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from repolens.context import estimate_tokens

# ---------------------------------------------------------------------------
# Task model
# ---------------------------------------------------------------------------

#: The supported task categories used by the benchmark corpus.
CATEGORIES = (
    "bug_fix",
    "feature_change",
    "refactor",
    "cross_file_change",
    "test_change",
)

#: Evaluation is fully offline and order-stable; exposed to serializers.
DETERMINISTIC = True

#: Default cap applied to any candidate set inside the runner.
DEFAULT_MAX_CANDIDATES = 50


def _require_scalar(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value.strip()


def _normalize_paths(values: Sequence[object], label: str) -> tuple[str, ...]:
    """Normalize to a deduplicated, order-preserving tuple of repo-relative paths.

    Paths must stay inside the repository: absolute paths, backslashes and
    ``.``/``..`` segments are rejected because ``CandidateSet`` and evaluation
    always reason in repository-relative ``a/b.py`` coordinates.
    """
    results: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value).strip()
        if (
            not text
            or text.startswith("/")
            or "\\" in text
            or ":" in text
        ):
            raise ValueError(
                f"{label} entries must be repository-relative paths: {value!r}"
            )
        parts = text.split("/")
        if any(part in ("", ".", "..") for part in parts):
            raise ValueError(
                f"{label} entries must be repository-relative POSIX paths with "
                f"no absolute, backslash, or '.'/'..' segments: {value!r}"
            )
        if text not in seen:
            seen.add(text)
            results.append(text)
    return tuple(results)


def _normalize_names(values: Sequence[object], label: str) -> tuple[str, ...]:
    results: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value).strip()
        if not text:
            raise ValueError(f"{label} entries must be non-empty strings")
        if text not in seen:
            seen.add(text)
            results.append(text)
    return tuple(results)


@dataclass(frozen=True)
class EvaluationTask:
    """One realistic repository-level coding chore with its expected surface.

    The *expected surface* is the ground truth an idealized agent would need:
    the files (and, when useful, the symbols and tests) that a correct edit
    would touch. ``target_files``/``target_symbols`` are the explicit
    anchoring focus the task points at (e.g. the primary file to change),
    distinct from the broader expected surface used for scoring.

    The request field is what is fed to retrieval / context strategies.
    """

    id: str
    title: str
    request: str
    expected_relevant_files: tuple[str, ...]
    target_files: tuple[str, ...] = ()
    expected_relevant_symbols: tuple[str, ...] = ()
    expected_tests: tuple[str, ...] = ()
    category: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _require_scalar(self.id, "id"))
        object.__setattr__(self, "title", _require_scalar(self.title, "title"))
        object.__setattr__(self, "request", _require_scalar(self.request, "request"))
        if self.category is not None and self.category not in CATEGORIES:
            raise ValueError(
                f"category {self.category!r} not in {CATEGORIES}"
            )
        object.__setattr__(
            self,
            "expected_relevant_files",
            _normalize_paths(self.expected_relevant_files, "expected_relevant_files"),
        )
        object.__setattr__(
            self, "target_files", _normalize_paths(self.target_files, "target_files")
        )
        object.__setattr__(
            self,
            "expected_relevant_symbols",
            _normalize_names(
                self.expected_relevant_symbols, "expected_relevant_symbols"
            ),
        )
        object.__setattr__(
            self, "expected_tests", _normalize_paths(self.expected_tests, "expected_tests")
        )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "request": self.request,
            "expected_relevant_files": list(self.expected_relevant_files),
            "target_files": list(self.target_files),
            "expected_relevant_symbols": list(self.expected_relevant_symbols),
            "expected_tests": list(self.expected_tests),
            "category": self.category,
        }


# ---------------------------------------------------------------------------
# Candidate / case model
# ---------------------------------------------------------------------------


def _as_posix(path: object) -> str:
    if hasattr(path, "as_posix"):
        return path.as_posix()  # type: ignore[attr-defined]
    return str(path)


@dataclass(frozen=True)
class CandidateSet:
    """One strategy's deterministic answer for one task.

    ``retrieved_files`` are repository-relative in the strategy's ranking
    order; ``matched_symbols`` are the symbol names that surfaced alongside
    the retrieved files. ``context_size_tokens``/``selected_file_count`` are
    ``None`` when the producing strategy cannot measure them.
    """

    strategy: str
    retrieved_files: tuple[str, ...]
    matched_symbols: tuple[str, ...] = ()
    context_size_tokens: int | None = None
    selected_file_count: int | None = None
    latency_seconds: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "strategy", _require_scalar(self.strategy, "strategy")
        )
        object.__setattr__(
            self,
            "retrieved_files",
            _normalize_paths(self.retrieved_files, "retrieved_files"),
        )
        object.__setattr__(
            self,
            "matched_symbols",
            _normalize_names(self.matched_symbols, "matched_symbols"),
        )
        if self.context_size_tokens is not None and self.context_size_tokens < 0:
            raise ValueError("context_size_tokens must be non-negative")
        if self.selected_file_count is not None and self.selected_file_count < 0:
            raise ValueError("selected_file_count must be non-negative")
        if self.latency_seconds is not None and self.latency_seconds < 0:
            raise ValueError("latency_seconds must be non-negative")


@dataclass(frozen=True)
class EvaluationCase:
    """One task pinned to one candidate set for scoring."""

    task: EvaluationTask
    candidate_set: CandidateSet

    def __post_init__(self) -> None:
        if not isinstance(self.task, EvaluationTask):
            raise TypeError("task must be an EvaluationTask")
        if not isinstance(self.candidate_set, CandidateSet):
            raise TypeError("candidate_set must be a CandidateSet")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _unique_in_order(values: Iterable[str]) -> tuple[str, ...]:
    results: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value not in seen:
            seen.add(value)
            results.append(value)
    return tuple(results)


def _metrics(relevant: set[str], retrieved: tuple[str, ...]) -> tuple[float, float, float]:
    """Return (precision, recall, f1) for retrieved against expected relevant files."""
    n_retrieved = len(retrieved)
    n_relevant = len(relevant)
    hits = sum(1 for path in retrieved if path in relevant)
    precision = hits / n_retrieved if n_retrieved else 0.0
    recall = hits / n_relevant if n_relevant else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


# ---------------------------------------------------------------------------
# Results and report
# ---------------------------------------------------------------------------

_ROUND = 6


def _rounded(value: float) -> float:
    return round(value, _ROUND)


@dataclass(frozen=True)
class EvaluationResult:
    """Per-case metrics for one (task, candidate set) pair."""

    case: EvaluationCase
    precision: float
    recall: float
    f1: float
    symbol_recall: float
    test_recall: float
    retrieved_files: tuple[str, ...]
    relevant_files_retrieved: tuple[str, ...]
    relevant_files_missed: tuple[str, ...]
    irrelevant_files_retrieved: tuple[str, ...]
    expected_symbols_retrieved: tuple[str, ...]
    expected_tests_retrieved: tuple[str, ...]
    context_size_tokens: int | None
    selected_file_count: int | None
    latency_seconds: float | None

    @property
    def task_id(self) -> str:
        return self.case.task.id

    @property
    def strategy(self) -> str:
        return self.case.candidate_set.strategy

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "strategy": self.strategy,
            "precision": _rounded(self.precision),
            "recall": _rounded(self.recall),
            "f1": _rounded(self.f1),
            "symbol_recall": _rounded(self.symbol_recall),
            "test_recall": _rounded(self.test_recall),
            "retrieved_files": list(self.retrieved_files),
            "relevant_files_retrieved": list(self.relevant_files_retrieved),
            "relevant_files_missed": list(self.relevant_files_missed),
            "irrelevant_files_retrieved": list(self.irrelevant_files_retrieved),
            "expected_symbols_retrieved": list(self.expected_symbols_retrieved),
            "expected_tests_retrieved": list(self.expected_tests_retrieved),
            "context_size_tokens": self.context_size_tokens,
            "selected_file_count": self.selected_file_count,
            "latency_seconds": self.latency_seconds,
        }


@dataclass(frozen=True)
class EvaluationReport:
    """Complete, ordered evaluation output for one run."""

    results: tuple[EvaluationResult, ...]
    failures: tuple[dict, ...] = ()

    @property
    def num_cases(self) -> int:
        return len(self.results)

    def strategies(self) -> tuple[str, ...]:
        return _unique_in_order(result.strategy for result in self.results)

    def for_strategy(self, strategy: str) -> tuple[EvaluationResult, ...]:
        return tuple(result for result in self.results if result.strategy == strategy)

    def strategy_summary(self, strategy: str) -> dict:
        """Compact per-strategy summary used by the benchmark report."""
        results = self.for_strategy(strategy)
        values = {
            key: _mean_and_median(
                [float(getattr(result, key)) for result in results]
            )
            for key in ("precision", "recall", "f1", "symbol_recall", "test_recall")
        }
        mean_selected = _mean_number(
            [result.selected_file_count for result in results]
        )
        mean_context = _mean_number(
            [result.context_size_tokens for result in results]
        )
        mean_latency = _mean_number(
            [result.latency_seconds for result in results]
        )
        mean_irrelevant = _mean_number(
            [float(len(result.irrelevant_files_retrieved)) for result in results]
        )
        mean_tokens_per_relevant = _mean_number(
            [_tokens_per_relevant_file(result) for result in results]
        )
        failures = [item for item in self.failures if item["strategy"] == strategy]
        return {
            "strategy": strategy,
            "task_count": len(results),
            "precision": {"mean": _rounded(values["precision"][0]), "median": _rounded(values["precision"][1])},
            "recall": {"mean": _rounded(values["recall"][0]), "median": _rounded(values["recall"][1])},
            "f1": {"mean": _rounded(values["f1"][0]), "median": _rounded(values["f1"][1])},
            "mean_symbol_recall": _rounded(values["symbol_recall"][0]),
            "mean_test_recall": _rounded(values["test_recall"][0]),
            "mean_selected_files": _rounded(mean_selected),
            "mean_context_size_tokens": _rounded(mean_context),
            "mean_irrelevant_files": _rounded(mean_irrelevant),
            "mean_tokens_per_relevant_file": _rounded(mean_tokens_per_relevant),
            "mean_latency_seconds": _rounded(mean_latency),
            "failures": failures,
        }

    def to_dict(self) -> dict:
        return {
            "deterministic": DETERMINISTIC,
            "num_cases": self.num_cases,
            "results": [result.to_dict() for result in self.results],
            "summaries": {
                strategy: self.strategy_summary(strategy)
                for strategy in self.strategies()
            },
            "failures": list(self.failures),
        }


def _mean_number(values: Sequence[float | int | None]) -> float:
    present = [float(value) for value in values if value is not None]
    return sum(present) / len(present) if present else 0.0


def _mean_and_median(values: Sequence[float]) -> tuple[float, float]:
    return (sum(values) / len(values) if values else 0.0), (statistics.median(values) if values else 0.0)


def _tokens_per_relevant_file(result: EvaluationResult) -> float:
    """Context tokens spent per required file actually retrieved (0 when none)."""
    if result.context_size_tokens is None:
        return 0.0
    relevant_retrieved = len(result.relevant_files_retrieved)
    if not relevant_retrieved:
        return 0.0
    return result.context_size_tokens / relevant_retrieved


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


class AgentEvaluationRunner:
    """Scores candidate sets against the expected task surface.

    The runner is deliberately small: it owns metric computation, candidate
    capping (bounded result handling), and ordering. Producing the candidate
    sets is left to the strategy adapters below (or to callers), so nothing in
    this class touches the repository, a model, or the network.
    """

    def __init__(self, *, max_candidates: int = DEFAULT_MAX_CANDIDATES) -> None:
        if max_candidates <= 0:
            raise ValueError("max_candidates must be positive")
        self._max_candidates = max_candidates

    @property
    def max_candidates(self) -> int:
        return self._max_candidates

    def evaluate(self, task: EvaluationTask, candidate_set: CandidateSet) -> EvaluationResult:
        """Score one (task, candidate set) pair deterministically."""
        if not isinstance(task, EvaluationTask):
            raise TypeError("task must be an EvaluationTask")
        if not isinstance(candidate_set, CandidateSet):
            raise TypeError("candidate_set must be a CandidateSet")

        retrieved = candidate_set.retrieved_files[: self._max_candidates]
        relevant = set(task.expected_relevant_files)
        precision, recall, f1 = _metrics(relevant, retrieved)

        retrieved_set = set(retrieved)
        relevant_retrieved = tuple(path for path in retrieved if path in relevant)
        relevant_missed = tuple(sorted(relevant - retrieved_set))
        irrelevant_retrieved = tuple(path for path in retrieved if path not in relevant)

        expected_symbols = tuple(task.expected_relevant_symbols)
        retrieved_symbols = set(candidate_set.matched_symbols)
        symbols_retrieved = tuple(
            symbol for symbol in expected_symbols if symbol in retrieved_symbols
        )
        symbol_recall = (
            (len(symbols_retrieved) / len(expected_symbols))
            if expected_symbols
            else 0.0
        )

        expected_tests = tuple(task.expected_tests)
        tests_retrieved = tuple(path for path in expected_tests if path in retrieved_set)
        test_recall = len(tests_retrieved) / len(expected_tests) if expected_tests else 0.0

        return EvaluationResult(
            case=EvaluationCase(task=task, candidate_set=candidate_set),
            precision=precision,
            recall=recall,
            f1=f1,
            symbol_recall=symbol_recall,
            test_recall=test_recall,
            retrieved_files=retrieved,
            relevant_files_retrieved=relevant_retrieved,
            relevant_files_missed=relevant_missed,
            irrelevant_files_retrieved=irrelevant_retrieved,
            expected_symbols_retrieved=symbols_retrieved,
            expected_tests_retrieved=tests_retrieved,
            context_size_tokens=candidate_set.context_size_tokens,
            selected_file_count=candidate_set.selected_file_count,
            latency_seconds=candidate_set.latency_seconds,
        )

    def run(self, cases: Iterable[EvaluationCase]) -> EvaluationReport:
        """Score every case, preserving iteration order."""
        results = tuple(self.evaluate(case.task, case.candidate_set) for case in cases)
        return EvaluationReport(results=results)


# ---------------------------------------------------------------------------
# Strategy adapters (reuse existing RepoLens retrieval/context APIs)
# ---------------------------------------------------------------------------


def _symbol_names(symbols: Sequence[object]) -> tuple[str, ...]:
    names: list[str] = []
    for symbol in symbols:
        name = getattr(symbol, "name", None)
        names.append(name if isinstance(name, str) else str(symbol))
    return tuple(names)


def _estimate_context_size(root: Path | str | None, paths: Sequence[str]) -> int | None:
    """Deterministic token footprint of the retrieved files when a root is set."""
    if root is None:
        return None
    root_path = Path(root)
    total = 0
    for path in paths:
        try:
            total += estimate_tokens(root_path.joinpath(path).read_text(encoding="utf-8"))
        except OSError:
            continue
    return total


def produce_search_candidate_set(
    task: EvaluationTask,
    *,
    searcher: object,
    root: Path | str | None = None,
    strategy: str = "lexical",
    limit: int = 10,
) -> CandidateSet:
    """Translate a ``searcher.search(request, limit=...)`` result into a CandidateSet.

    Works for :class:`~repolens.search.CodeSearcher`,
    :class:`~repolens.semantic_search.SemanticSearcher`, and
    :class:`~repolens.retrieval.HybridSearcher` outputs; symbol coverage is
    taken when the searcher reports it (lexical) and is empty otherwise.
    """
    if limit <= 0:
        raise ValueError("limit must be positive")
    started = time.perf_counter()
    results = searcher.search(task.request, limit=limit)  # type: ignore[attr-defined]
    latency = time.perf_counter() - started

    retrieved: list[str] = []
    symbols: list[str] = []
    for result in results:
        retrieved.append(_as_posix(result.file_path))
        symbols.extend(_symbol_names(getattr(result, "symbols", ()) or ()))
    retrieved_tuple = tuple(retrieved)
    return CandidateSet(
        strategy=strategy,
        retrieved_files=retrieved_tuple,
        matched_symbols=_unique_in_order(symbols),
        context_size_tokens=_estimate_context_size(root, retrieved_tuple),
        selected_file_count=len(retrieved_tuple),
        latency_seconds=latency,
    )


def _candidate_set_from_package(
    strategy: str,
    package: object,
    latency: float,
) -> CandidateSet:
    """Translate a :class:`repolens.context.package.ContextPackage` into a CandidateSet."""
    selected = tuple(
        _as_posix(item.path)
        for item in getattr(package, "selected_files", ())
    )
    matched = tuple(getattr(package, "matched_symbols", ()) or ())
    return CandidateSet(
        strategy=strategy,
        retrieved_files=selected,
        matched_symbols=matched,
        context_size_tokens=_as_int(getattr(package, "total_estimated_tokens", None)),
        selected_file_count=len(selected),
        latency_seconds=latency,
    )


def _as_int(value: object) -> int | None:
    return None if value is None else int(value)


def produce_context_candidate_set(
    task: EvaluationTask,
    *,
    engine: object,
    strategy: str = "context",
) -> CandidateSet:
    """Ordinary context: ``engine.build_context(task.request)`` unchanged."""
    started = time.perf_counter()
    package = engine.build_context(task.request)  # type: ignore[attr-defined]
    latency = time.perf_counter() - started
    return _candidate_set_from_package(strategy, package, latency)


def produce_change_context_candidate_set(
    task: EvaluationTask,
    *,
    engine: object,
    strategy: str = "change_context",
) -> CandidateSet:
    """Change-aware context: folds the change plan for the task into the package.

    ``change_request`` is the task request and the change target defaults to
    the task's first explicit target file (request-only planning when the task
    declares none). The plan engine is shared/lazy inside the context engine,
    so no repository re-parse happens.
    """
    change_target = task.target_files[0] if task.target_files else None
    started = time.perf_counter()
    package = engine.build_context(  # type: ignore[attr-defined]
        task.request,
        change_request=task.request,
        change_target=change_target,
    )
    latency = time.perf_counter() - started
    return _candidate_set_from_package(strategy, package, latency)