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

import time
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


# ---------------------------------------------------------------------------
# Strategy benchmarking
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CaseMetrics:
    """Per-case metrics with latency and resource usage for one query.

    ``candidate_count`` and ``embedded_count`` are ``None`` for strategies
    where the metric does not apply (e.g. lexical search has no embeddings;
    full-semantic search has no candidate count).

    ``relevant_in_candidates`` is ``None`` for strategies with no candidate
    stage; otherwise it records whether every ground-truth relevant file
    survived candidate generation (True/False). A ``False`` here explains a
    candidate-semantic recall failure that full semantic avoids.
    """

    query: str
    retrieved_files: tuple[Path, ...]
    relevant_files: frozenset[Path]
    precision_at_k: float
    recall_at_k: float
    reciprocal_rank: float
    first_relevant_rank: int | None
    search_latency_seconds: float
    candidate_count: int | None = None
    embedded_count: int | None = None
    relevant_in_candidates: bool | None = None


@dataclass(frozen=True)
class BenchmarkResult:
    """Aggregate result of benchmarking one retrieval strategy."""

    strategy: str
    k: int
    case_metrics: tuple[CaseMetrics, ...]
    build_time_seconds: float
    mean_precision_at_k: float
    mean_recall_at_k: float
    mean_reciprocal_rank: float
    mean_search_latency_seconds: float
    mean_embedded_count: float | None
    total_embedded: int | None
    total_candidates: int | None


def _sub_searcher(searcher: Searcher) -> Any:
    """For a hybrid, resolve to the underlying semantic searcher."""
    for attr in ("_semantic",):
        inner = getattr(searcher, attr, None)
        if inner is not None:
            return inner
    return searcher


def _get_candidate_searcher(searcher: Searcher) -> Any | None:
    """Return the candidate generator of a candidate-based searcher, or None."""
    return getattr(_sub_searcher(searcher), "_candidate_searcher", None)


def _get_candidate_count(searcher: Searcher) -> int | None:
    """Return the candidate limit for a candidate-based searcher, or None."""
    candidate_searcher = _get_candidate_searcher(searcher)
    if candidate_searcher is None:
        return None
    return getattr(_sub_searcher(searcher), "_candidate_limit", None)


def _get_embedded_count(searcher: Searcher) -> int:
    """Return the number of cached embeddings for a semantic searcher."""
    vectors = getattr(_sub_searcher(searcher), "_vectors_by_path", None)
    if vectors is None:
        return 0
    return len(vectors)


def _relevant_in_candidates(
    searcher: Searcher, query: str, relevant: frozenset[Path]
) -> bool | None:
    """Whether the ground-truth files are in the candidate set for a query.

    Returns ``None`` when the searcher has no candidate stage (lexical,
    full-semantic), else True/False. The candidate set reflects the
    searcher's configured ``candidate_limit``, so a tight limit that drops a
    ground-truth file yields ``False``. Computing this requires one extra
    candidate-generator call; it is cheap when candidates are cached.
    """
    candidate_searcher = _get_candidate_searcher(searcher)
    if candidate_searcher is None:
        return None
    limit = getattr(_sub_searcher(searcher), "_candidate_limit", None)
    if limit is None:
        return None
    candidate_paths = {
        result.file_path for result in candidate_searcher.search(query, limit=limit)
    }
    if not relevant:
        return None
    return relevant <= candidate_paths


def benchmark_strategy(
    strategy: str,
    root: Path | str,
    cases: Iterable[EvaluationCase],
    *,
    k: int = DEFAULT_K,
    provider: Any | None = None,
    candidate_limit: int = 40,
    lexical_weight: float = 0.5,
    semantic_weight: float = 0.5,
    rrf_k: int = 60,
) -> BenchmarkResult:
    """Benchmark a retrieval strategy and return per-case + aggregate metrics.

    Supports four strategy names:

    - ``"lexical"`` — :class:`CodeSearcher` (baseline, no embeddings).
    - ``"semantic"`` — Full-semantic :class:`SemanticSearcher` that embeds
      *all* files (no candidate limit; for comparison against candidate-based).
    - ``"candidate-semantic"`` — Candidate-based :class:`SemanticSearcher`
      (embeds only lexical top-N; production configuration).
    - ``"hybrid"`` — :class:`HybridSearcher` (default: weighted 0.5/0.5).

    ``provider`` must be provided for every strategy except ``"lexical"``.
    ``candidate_limit`` controls how many files the candidate-semantic
    strategy embeds (default 40).
    """
    root = Path(root)
    cases_list = list(cases)
    if not cases_list:
        raise ValueError("No evaluation cases provided")

    from repolens.local_embeddings import LocalEmbeddingProvider

    if provider is None and strategy != "lexical":
        provider = LocalEmbeddingProvider()

    # -- build searcher -------------------------------------------------------

    build_start = time.perf_counter()

    if strategy == "lexical":
        searcher: Searcher = CodeSearcher(root)

    elif strategy == "semantic":
        # Full-semantic: embed every file, no candidate filtering.
        from repolens.parser import PythonParser
        from repolens.index import SymbolIndexBuilder

        _parser = PythonParser()
        _symbols_by_path: dict[Path, tuple] = {}
        index = SymbolIndexBuilder(root).build()
        from collections import defaultdict
        by_path: dict[Path, list] = defaultdict(list)
        for sym in index.get_all_symbols():
            by_path[sym.file_path].append(sym)
        _symbols_by_path = {p: tuple(s) for p, s in by_path.items()}

        def _read_file(path: Path) -> tuple[str, Any | None]:
            try:
                source = (root / path).read_text(encoding="utf-8")
            except (OSError, ValueError):
                return "", None
            try:
                return source, _parser.parse_source(source, file_path=path)
            except SyntaxError:
                return source, None

        def _compose_document(path: Path) -> str:
            source, analysis = _read_file(path)
            lines = [f"path: {path.as_posix()}"]
            if analysis is not None:
                imports = [item.module for item in analysis.imports]
                imports += [
                    f"{item.module}.{item.name}" if item.module else item.name
                    for item in analysis.from_imports
                ]
                if imports:
                    lines.append("imports: " + " ".join(imports))
            symbols = _symbols_by_path.get(path, ())
            functions = [s.name for s in symbols if s.kind.value == "function"]
            classes = [s.name for s in symbols if s.kind.value == "class"]
            methods = [
                f"{s.parent_class}.{s.name}" if s.parent_class else s.name
                for s in symbols
                if s.kind.value == "method"
            ]
            if functions:
                lines.append("functions: " + ", ".join(functions))
            if classes:
                lines.append("classes: " + ", ".join(classes))
            if methods:
                lines.append("methods: " + ", ".join(methods))
            lines.append("source:")
            lines.append(source)
            return "\n".join(lines)

        # Discover all files and embed everything up front.
        from repolens.scanner import RepositoryScanner
        all_paths = RepositoryScanner(root).discover_python_files()
        documents = [_compose_document(path) for path in all_paths]
        vectors = list(provider.embed_texts(documents))
        _vectors_by_path = dict(zip(all_paths, vectors))

        # Build a minimal searcher that uses pre-embedded vectors.
        class _FullSemanticSearcher:
            def __init__(self, vectors_by_path: dict[Path, Any], provider_: Any):
                self._vectors_by_path = vectors_by_path
                self._provider = provider_

            def search(self, query: str, limit: int = 10):
                from repolens.semantic_search import SemanticResult, cosine_similarity
                qvec = self._provider.embed_text(query)
                scored = []
                for path, vec in self._vectors_by_path.items():
                    sim = cosine_similarity(qvec, vec)
                    if sim > 0.0:
                        scored.append((path, sim))
                scored.sort(key=lambda x: (-x[1], x[0].as_posix()))
                return [SemanticResult(file_path=p, similarity=s) for p, s in scored[:limit]]

        searcher = _FullSemanticSearcher(_vectors_by_path, provider)

    elif strategy == "candidate-semantic":
        from repolens.semantic_search import SemanticSearcher
        searcher = SemanticSearcher(root, provider, candidate_limit=candidate_limit)

    elif strategy == "hybrid":
        from repolens.semantic_search import SemanticSearcher
        from repolens.retrieval import FusionStrategy, HybridSearcher

        lex = CodeSearcher(root)
        sem = SemanticSearcher(
            root, provider, candidate_searcher=lex, candidate_limit=candidate_limit
        )
        searcher = HybridSearcher(
            root,
            lexical_searcher=lex,
            semantic_searcher=sem,
            lexical_weight=lexical_weight,
            semantic_weight=semantic_weight,
            strategy=FusionStrategy.WEIGHTED,
            rrf_k=rrf_k,
        )

    else:
        raise ValueError(f"Unknown strategy: {strategy!r}")

    build_time = time.perf_counter() - build_start

    # -- warm-up query --------------------------------------------------------

    warmup = searcher.search(cases_list[0].query, limit=k)

    # -- per-case metrics -----------------------------------------------------

    case_metrics: list[CaseMetrics] = []
    for case in cases_list:
        t0 = time.perf_counter()
        results = searcher.search(case.query, limit=k)
        elapsed = time.perf_counter() - t0
        retrieved = tuple(r.file_path for r in results)
        candidate_count = _get_candidate_count(searcher)
        embedded_count = _get_embedded_count(searcher) if strategy != "lexical" else None
        relevant_in_cands = _relevant_in_candidates(
            searcher, case.query, case.relevant_files
        )

        case_metrics.append(
            CaseMetrics(
                query=case.query,
                retrieved_files=retrieved,
                relevant_files=case.relevant_files,
                precision_at_k=precision_at_k(retrieved, case.relevant_files),
                recall_at_k=recall_at_k(retrieved, case.relevant_files),
                reciprocal_rank=reciprocal_rank(retrieved, case.relevant_files),
                first_relevant_rank=first_relevant_rank(retrieved, case.relevant_files),
                search_latency_seconds=elapsed,
                candidate_count=candidate_count,
                embedded_count=embedded_count,
                relevant_in_candidates=relevant_in_cands,
            )
        )

    # -- aggregate ------------------------------------------------------------

    n = len(case_metrics)
    mean_p = sum(m.precision_at_k for m in case_metrics) / n
    mean_r = sum(m.recall_at_k for m in case_metrics) / n
    mean_rr = sum(m.reciprocal_rank for m in case_metrics) / n
    mean_lat = sum(m.search_latency_seconds for m in case_metrics) / n

    embedded_counts = [m.embedded_count for m in case_metrics if m.embedded_count is not None]
    mean_emb = (sum(embedded_counts) / len(embedded_counts)) if embedded_counts else None
    total_emb = max(embedded_counts) if embedded_counts else None

    candidate_counts = [m.candidate_count for m in case_metrics if m.candidate_count is not None]
    total_cand = max(candidate_counts) if candidate_counts else None

    return BenchmarkResult(
        strategy=strategy,
        k=k,
        case_metrics=tuple(case_metrics),
        build_time_seconds=build_time,
        mean_precision_at_k=mean_p,
        mean_recall_at_k=mean_r,
        mean_reciprocal_rank=mean_rr,
        mean_search_latency_seconds=mean_lat,
        mean_embedded_count=mean_emb,
        total_embedded=total_emb,
        total_candidates=total_cand,
    )
