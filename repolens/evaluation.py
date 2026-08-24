"""Deterministic evaluation of RepoLens retrieval quality.

This module measures how well :class:`repolens.search.CodeSearcher` ranks
files for a set of query/ground-truth pairs. It is pure evaluation
infrastructure: no LLMs, embeddings, randomness, or network access. Running
the same cases against the same repository always yields identical numbers.

Data model
----------
- :class:`EvaluationCase` — one query plus the repository-relative files
  considered relevant to it (the ground truth).
- :class:`CaseEvaluation` — per-case outcome: the ranked retrieved files,
  the ground truth, precision@k, recall@k, the reciprocal rank, and the
  1-based rank of the first relevant file (``None`` if none was retrieved).
- :class:`EvaluationReport` — all case evaluations plus aggregate means,
  including mean reciprocal rank (MRR).

Metrics
-------
With ``retrieved`` the top-k files returned by search and ``relevant`` the
ground-truth set:

- ``precision@k = |retrieved ∩ relevant| / |retrieved|``
  (defined as ``0.0`` when nothing was retrieved).
- ``recall@k = |retrieved ∩ relevant| / |relevant|``
  (defined as ``0.0`` when the ground truth is empty).
- ``reciprocal rank = 1 / rank``, where ``rank`` is the 1-based position of
  the first relevant file in ``retrieved``; ``0.0`` when no relevant file
  was retrieved.

All metrics lie in ``[0.0, 1.0]``. The runner drives any object satisfying
the :class:`Searcher` protocol (``search(query, limit=...)`` returning
ranked results with a ``file_path`` attribute); it defaults to the lexical
:class:`~repolens.search.CodeSearcher`, whose baseline behavior is
unchanged, and can equally evaluate a :class:`~repolens.semantic_search.SemanticSearcher`
without duplicating any evaluation logic.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from repolens.search import CodeSearcher

DEFAULT_K = 5


@runtime_checkable
class Searcher(Protocol):
    """Anything the evaluator can query for ranked repository files."""

    def search(self, query: str, limit: int) -> Sequence[Any]:
        """Return ranked results (objects exposing ``file_path``)."""
        ...


@dataclass(frozen=True)
class EvaluationCase:
    """One evaluation query and its ground-truth relevant files.

    ``relevant_files`` accepts any iterable of paths (or strings) and is
    normalized to a frozenset of repository-relative :class:`~pathlib.Path`.
    """

    query: str
    relevant_files: Iterable[Path | str] = frozenset()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "relevant_files",
            frozenset(Path(item) for item in self.relevant_files),
        )


def _as_path_set(paths: Iterable[Path | str]) -> frozenset[Path]:
    return frozenset(Path(item) for item in paths)


def precision_at_k(
    retrieved: Sequence[Path], relevant: Iterable[Path | str]
) -> float:
    """Fraction of retrieved files that are relevant; ``0.0`` if none retrieved."""
    if not retrieved:
        return 0.0
    relevant_set = _as_path_set(relevant)
    hits = sum(1 for item in retrieved if Path(item) in relevant_set)
    return hits / len(retrieved)


def recall_at_k(
    retrieved: Sequence[Path], relevant: Iterable[Path | str]
) -> float:
    """Fraction of relevant files that were retrieved; ``0.0`` if no ground truth."""
    relevant_set = _as_path_set(relevant)
    if not relevant_set:
        return 0.0
    hits = sum(1 for item in retrieved if Path(item) in relevant_set)
    return hits / len(relevant_set)


def first_relevant_rank(
    retrieved: Sequence[Path], relevant: Iterable[Path | str]
) -> int | None:
    """1-based rank of the first relevant retrieved file, or ``None``."""
    relevant_set = _as_path_set(relevant)
    for rank, item in enumerate(retrieved, start=1):
        if Path(item) in relevant_set:
            return rank
    return None


def reciprocal_rank(
    retrieved: Sequence[Path], relevant: Iterable[Path | str]
) -> float:
    """``1 / rank`` of the first relevant file; ``0.0`` when there is none."""
    rank = first_relevant_rank(retrieved, relevant)
    return 1.0 / rank if rank is not None else 0.0


@dataclass(frozen=True)
class CaseEvaluation:
    """The outcome of evaluating one case at a given k."""

    query: str
    relevant_files: frozenset[Path]
    retrieved_files: tuple[Path, ...]
    precision_at_k: float
    recall_at_k: float
    reciprocal_rank: float
    first_relevant_rank: int | None


@dataclass(frozen=True)
class EvaluationReport:
    """Aggregated evaluations for a run of cases at one value of k."""

    k: int
    case_evaluations: tuple[CaseEvaluation, ...]

    @property
    def num_cases(self) -> int:
        return len(self.case_evaluations)

    @property
    def mean_precision_at_k(self) -> float:
        if not self.case_evaluations:
            return 0.0
        return (
            sum(item.precision_at_k for item in self.case_evaluations)
            / self.num_cases
        )

    @property
    def mean_recall_at_k(self) -> float:
        if not self.case_evaluations:
            return 0.0
        return (
            sum(item.recall_at_k for item in self.case_evaluations)
            / self.num_cases
        )

    @property
    def mean_reciprocal_rank(self) -> float:
        if not self.case_evaluations:
            return 0.0
        return (
            sum(item.reciprocal_rank for item in self.case_evaluations)
            / self.num_cases
        )


class EvaluationRunner:
    """Runs evaluation cases against a repository.

    The default searcher is the lexical :class:`CodeSearcher`; pass any
    other :class:`Searcher` (for example a ``SemanticSearcher`` built on the
    same root) via ``searcher`` to measure it with identical code::

        runner = EvaluationRunner(repo_root)
        runner = EvaluationRunner(
            repo_root, searcher=SemanticSearcher(repo_root, provider)
        )

        report = runner.evaluate(cases, k=5)
        report.mean_reciprocal_rank
    """

    def __init__(self, root: Path | str, searcher: Searcher | None = None) -> None:
        self.root = Path(root)
        self._searcher: Searcher = (
            searcher if searcher is not None else CodeSearcher(self.root)
        )

    def evaluate(
        self, cases: Iterable[EvaluationCase], k: int = DEFAULT_K
    ) -> EvaluationReport:
        """Evaluate every case at cutoff ``k``, preserving input order."""
        if k < 1:
            raise ValueError(f"k must be at least 1, got {k}")
        evaluations = tuple(self.evaluate_case(case, k=k) for case in cases)
        return EvaluationReport(k=k, case_evaluations=evaluations)

    def evaluate_case(
        self, case: EvaluationCase, k: int = DEFAULT_K
    ) -> CaseEvaluation:
        """Search for one case's query and compare against its ground truth."""
        if k < 1:
            raise ValueError(f"k must be at least 1, got {k}")
        results = self._searcher.search(case.query, limit=k)
        retrieved = tuple(result.file_path for result in results)
        relevant = case.relevant_files
        return CaseEvaluation(
            query=case.query,
            relevant_files=relevant,
            retrieved_files=retrieved,
            precision_at_k=precision_at_k(retrieved, relevant),
            recall_at_k=recall_at_k(retrieved, relevant),
            reciprocal_rank=reciprocal_rank(retrieved, relevant),
            first_relevant_rank=first_relevant_rank(retrieved, relevant),
        )
