"""Deterministic agent-task evaluation foundation (P25.2–P25.3).

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

P25.3 adds an explicit *agent context usefulness* model that does **not** treat
every expected file as equally mandatory. A task distinguishes:

- **required files** — genuinely needed to understand or implement the task;
- **supporting files** — useful surrounding context, not strictly required;
- **expected tests** — tests relevant to validating the task;
- **required symbols** — definitions an agent needs to inspect.

The :class:`ContextEfficiencyResult` then scores an actual context package on
required/supporting/test coverage plus package size (selected files, context
tokens, tokens per required file). This answers "was the context RepoLens built
*both* small and useful" instead of only "did retrieval find the right files."

Backward compatibility: the P25.2 ``expected_relevant_files`` /
``expected_relevant_symbols`` fields remain supported and are merged into the
new ``required_files`` / ``required_symbols`` surfaces deterministically (see
:attr:`EvaluationTask.required_files_effective`), so every existing task
definition stays valid unchanged.

Nothing here calls a model or the network: strategies are built on top of the
existing RepoLens retrieval and context engines, and every ordering is
deterministic by construction (retrieved candidates are kept in the order the
strategy returned them; misses are sorted; expected surfaces are normalized
up front). The module deliberately does not re-implement retrieval: the
:func:`produce_*` helpers simply translate existing searcher and context-engine
outputs into a :class:`CandidateSet`. The benchmark driving the whole thing
lives in ``benchmarks/agent_evaluation.py``.
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

    P25.3 surface model: ``required_files`` are genuinely mandatory,
    ``supporting_files`` merely helpful, ``expected_tests`` validate the task,
    and ``required_symbols`` are definitions to inspect. The P25.2
    ``expected_relevant_files`` / ``expected_relevant_symbols`` fields still
    work and are merged into the required surfaces via
    :attr:`required_files_effective` / :attr:`required_symbols_effective`, so
    existing task definitions remain valid unchanged. Required and supporting
    surfaces must not overlap: that would make scoring ambiguous.
    """

    id: str
    title: str
    request: str
    expected_relevant_files: tuple[str, ...] = ()
    target_files: tuple[str, ...] = ()
    expected_relevant_symbols: tuple[str, ...] = ()
    expected_tests: tuple[str, ...] = ()
    category: str | None = None
    required_files: tuple[str, ...] = ()
    supporting_files: tuple[str, ...] = ()
    required_symbols: tuple[str, ...] = ()

    @property
    def required_files_effective(self) -> tuple[str, ...]:
        """Deduped required surface: ``required_files`` plus the legacy
        ``expected_relevant_files`` (both are normalized, order preserved)."""
        return _unique_in_order((*self.required_files, *self.expected_relevant_files))

    @property
    def required_symbols_effective(self) -> tuple[str, ...]:
        """Deduped required symbols: ``required_symbols`` plus the legacy
        ``expected_relevant_symbols``."""
        return _unique_in_order(
            (*self.required_symbols, *self.expected_relevant_symbols)
        )

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
        object.__setattr__(
            self, "required_files", _normalize_paths(self.required_files, "required_files")
        )
        object.__setattr__(
            self,
            "supporting_files",
            _normalize_paths(self.supporting_files, "supporting_files"),
        )
        object.__setattr__(
            self, "required_symbols", _normalize_names(self.required_symbols, "required_symbols")
        )
        overlap = set(self.required_files_effective) & set(self.supporting_files)
        if overlap:
            raise ValueError(
                "required and supporting surfaces must not overlap; "
                f"files in both: {sorted(overlap)}"
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
            "required_files": list(self.required_files),
            "supporting_files": list(self.supporting_files),
            "required_symbols": list(self.required_symbols),
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
class ContextEfficiencyResult:
    """Agent-facing context usefulness metrics for one produced package (P25.3).

    Unlike :class:`EvaluationResult` (retrieval coverage), this result treats
    the task surface as graded: it distinguishes required files (mandatory),
    supporting files (helpful), and expected tests, and it measures the actual
    package produced by a context strategy (selected files, context tokens,
    tokens per required file). Denominators of zero (empty required/supporting/
    test surfaces) are handled deterministically by returning 0.0 recall and
    ``tokens_per_required_file``.
    """

    case: EvaluationCase
    required_file_recall: float
    supporting_file_recall: float
    test_recall: float
    selected_file_count: int
    relevant_selected_file_count: int
    irrelevant_selected_file_count: int
    context_size_tokens: int | None
    tokens_per_required_file: float
    required_files_missed: tuple[str, ...]
    supporting_files_selected: tuple[str, ...]
    irrelevant_files_selected: tuple[str, ...]

    @property
    def task_id(self) -> str:
        return self.case.task.id

    @property
    def strategy(self) -> str:
        return self.case.candidate_set.strategy

    @property
    def is_weak(self) -> bool:
        """A task case that missed at least one required file."""
        return self.required_file_recall < 1.0

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "strategy": self.strategy,
            "required_file_recall": _rounded(self.required_file_recall),
            "supporting_file_recall": _rounded(self.supporting_file_recall),
            "test_recall": _rounded(self.test_recall),
            "selected_file_count": self.selected_file_count,
            "relevant_selected_file_count": self.relevant_selected_file_count,
            "irrelevant_selected_file_count": self.irrelevant_selected_file_count,
            "context_size_tokens": self.context_size_tokens,
            "tokens_per_required_file": _rounded(self.tokens_per_required_file),
            "required_files_missed": list(self.required_files_missed),
            "supporting_files_selected": list(self.supporting_files_selected),
            "irrelevant_files_selected": list(self.irrelevant_files_selected),
        }


@dataclass(frozen=True)
class EvaluationReport:
    """Complete, ordered evaluation output for one run.

    ``results`` carries retrieval-coverage scores per case and ``efficiency``
    the P25.3 agent-context-usefulness scores; they are 1:1 per evaluated
    case. ``failures`` records per-case production errors.
    """

    results: tuple[EvaluationResult, ...]
    efficiency: tuple[ContextEfficiencyResult, ...] = ()
    failures: tuple[dict, ...] = ()

    @property
    def num_cases(self) -> int:
        return len(self.results)

    def strategies(self) -> tuple[str, ...]:
        return _unique_in_order(result.strategy for result in self.results)

    def for_strategy(self, strategy: str) -> tuple[EvaluationResult, ...]:
        return tuple(result for result in self.results if result.strategy == strategy)

    def efficiency_for_strategy(
        self, strategy: str
    ) -> tuple[ContextEfficiencyResult, ...]:
        return tuple(result for result in self.efficiency if result.strategy == strategy)

    def weak_cases(self) -> tuple[ContextEfficiencyResult, ...]:
        """All context cases that missed at least one required file."""
        return tuple(result for result in self.efficiency if result.is_weak)

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

    def efficiency_summary(self, strategy: str) -> dict:
        """Per-strategy means of the P25.3 agent-context-usefulness metrics."""
        efficiency = self.efficiency_for_strategy(strategy)

        def _mean(key: str) -> float:
            return _rounded(
                _mean_number([float(getattr(result, key)) for result in efficiency])
            )

        failures = [item for item in self.failures if item["strategy"] == strategy]
        return {
            "strategy": strategy,
            "task_count": len(efficiency),
            "mean_required_file_recall": _mean("required_file_recall"),
            "mean_supporting_file_recall": _mean("supporting_file_recall"),
            "mean_test_recall": _mean("test_recall"),
            "mean_selected_file_count": _rounded(
                _mean_number([result.selected_file_count for result in efficiency])
            ),
            "mean_relevant_selected_file_count": _mean("relevant_selected_file_count"),
            "mean_irrelevant_selected_file_count": _mean("irrelevant_selected_file_count"),
            "mean_context_size_tokens": _rounded(
                _mean_number([result.context_size_tokens for result in efficiency])
            ),
            "mean_tokens_per_required_file": _mean("tokens_per_required_file"),
            "failures": failures,
        }

    def to_dict(self) -> dict:
        return {
            "deterministic": DETERMINISTIC,
            "num_cases": self.num_cases,
            "results": [result.to_dict() for result in self.results],
            "efficiency": [result.to_dict() for result in self.efficiency],
            "summaries": {
                strategy: self.strategy_summary(strategy)
                for strategy in self.strategies()
            },
            "efficiency_summaries": {
                strategy: self.efficiency_summary(strategy)
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
        """Score one (task, candidate set) pair deterministically (retrieval coverage).

        The scoring surface is the task's required files: for P25.2-style tasks
        this is exactly ``expected_relevant_files``; the P25.3 surface model is
        consumed by :meth:`context_efficiency`.
        """
        if not isinstance(task, EvaluationTask):
            raise TypeError("task must be an EvaluationTask")
        if not isinstance(candidate_set, CandidateSet):
            raise TypeError("candidate_set must be a CandidateSet")

        retrieved = candidate_set.retrieved_files[: self._max_candidates]
        relevant = set(task.required_files_effective)
        precision, recall, f1 = _metrics(relevant, retrieved)

        retrieved_set = set(retrieved)
        relevant_retrieved = tuple(path for path in retrieved if path in relevant)
        relevant_missed = tuple(sorted(relevant - retrieved_set))
        irrelevant_retrieved = tuple(path for path in retrieved if path not in relevant)

        expected_symbols = tuple(task.required_symbols_effective)
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

    def context_efficiency(
        self, task: EvaluationTask, candidate_set: CandidateSet
    ) -> ContextEfficiencyResult:
        """Score an actual context package on graded, agent-facing usefulness.

        Uses the full candidate set as produced (context packages are already
        bounded by their budget), distinguishing required, supporting and test
        coverage, and measuring the selected-package size. Zero denominators
        are deterministic: empty required/supporting/test surfaces yield 0.0
        recall and ``tokens_per_required_file``.
        """
        if not isinstance(task, EvaluationTask):
            raise TypeError("task must be an EvaluationTask")
        if not isinstance(candidate_set, CandidateSet):
            raise TypeError("candidate_set must be a CandidateSet")

        selected = candidate_set.retrieved_files
        required = set(task.required_files_effective)
        supporting = set(task.supporting_files)
        tests = set(task.expected_tests)

        required_selected = tuple(path for path in selected if path in required)
        supporting_selected = tuple(path for path in selected if path in supporting)
        relevant_surface = required | supporting | tests
        irrelevant_selected = tuple(path for path in selected if path not in relevant_surface)

        required_file_recall = (
            len(required_selected) / len(required) if required else 0.0
        )
        supporting_file_recall = (
            len(supporting_selected) / len(supporting) if supporting else 0.0
        )
        test_recall = (
            sum(1 for path in selected if path in tests) / len(tests) if tests else 0.0
        )

        context_size = candidate_set.context_size_tokens
        tokens_per_required_file = (
            (context_size / len(required_selected))
            if context_size is not None and required_selected
            else 0.0
        )

        return ContextEfficiencyResult(
            case=EvaluationCase(task=task, candidate_set=candidate_set),
            required_file_recall=required_file_recall,
            supporting_file_recall=supporting_file_recall,
            test_recall=test_recall,
            selected_file_count=(
                candidate_set.selected_file_count
                if candidate_set.selected_file_count is not None
                else len(selected)
            ),
            relevant_selected_file_count=len(selected) - len(irrelevant_selected),
            irrelevant_selected_file_count=len(irrelevant_selected),
            context_size_tokens=context_size,
            tokens_per_required_file=tokens_per_required_file,
            required_files_missed=tuple(sorted(required - set(selected))),
            supporting_files_selected=supporting_selected,
            irrelevant_files_selected=irrelevant_selected,
        )

    def run(self, cases: Iterable[EvaluationCase]) -> EvaluationReport:
        """Score every case (retrieval coverage + agent context usefulness),
        preserving iteration order."""
        results: list[EvaluationResult] = []
        efficiency: list[ContextEfficiencyResult] = []
        for case in cases:
            results.append(self.evaluate(case.task, case.candidate_set))
            efficiency.append(self.context_efficiency(case.task, case.candidate_set))
        return EvaluationReport(
            results=tuple(results), efficiency=tuple(efficiency)
        )


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
    """Translate a :class:`repolens.context.package.ContextPackage` into a CandidateSet.

    Extracts the selected files in package order, preserves the package token
    count and latency, and collects selected symbols where available: the
    package-level ``matched_symbols`` plus each selected candidate's ``symbol``
    (the definition a change-aware or symbol-retrieval candidate anchors on).
    """
    selected: list[str] = []
    per_file_symbols: list[str] = []
    for item in getattr(package, "selected_files", ()):
        selected.append(_as_posix(item.path))
        symbol = getattr(item, "symbol", None)
        if isinstance(symbol, str) and symbol:
            per_file_symbols.append(symbol)
    matched = tuple(getattr(package, "matched_symbols", ()) or ())
    matched_symbols = _unique_in_order((*matched, *per_file_symbols))
    return CandidateSet(
        strategy=strategy,
        retrieved_files=tuple(selected),
        matched_symbols=matched_symbols,
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