"""Production benchmarking harness for RepoLens (Milestone 20).

Measures the real-world operational characteristics of the existing RepoLens
pipeline against a repository — scanning, cold and warm indexing, incremental
updates, embedding-cache behavior, per-strategy retrieval latency, and
context-generation latency/size — without redesigning any underlying system.

Design notes
------------
* **Structural, not wall-clock**: every stage reports counts (parsed files,
  cache hits/misses, candidates, selected files, context size) in addition to
  timings. No absolute real-time threshold is asserted anywhere, so the
  harness is portable across machines and CI runners.
* **Monotonic timers**: all durations come from :func:`time.perf_counter`.
* **Deterministic embeddings**: automated runs use the offline
  :class:`~repolens.embeddings.FakeEmbeddingProvider` by default; an external
  provider may be injected. No network API is ever called.
* **Non-destructive**: the cold/warm index phases write only to a fresh
  temporary cache directory; the mutation phases (modify/add/delete) run on a
  fresh temporary *copy* of the repository, so the user's repository is never
  modified by this harness.
* **Peak memory**: :mod:`tracemalloc` is used only when ``measure_memory`` is
  requested, because it perturbs allocation behaviour and slows every
  operation.

The :class:`ProductionReport` returned by :func:`run_production_benchmark`
aggregates every stage; a printable form is available via
:meth:`ProductionReport.to_text`. The bundled CLI
``benchmarks/production_benchmark.py`` renders it for an arbitrary repository.
"""

from __future__ import annotations

import shutil
import tempfile
import time
import tracemalloc
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median
from typing import Any

from repolens.embeddings import EmbeddingProvider, FakeEmbeddingProvider
from repolens.incremental_index import IncrementalIndexBuilder
from repolens.scanner import RepositoryScanner
from repolens.search import CodeSearcher

DEFAULT_QUERIES: tuple[str, ...] = (
    "how is authentication handled",
    "database connection and pooling",
    "error handling and logging",
    "utility helpers and application entry point",
    "payment processing and billing",
)

DEFAULT_CANDIDATE_LIMIT = 40
DEFAULT_REPEATS = 3

#: Top-level names skipped when copying a repository for mutation workflows.
_SKIP_NAMES = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    "node_modules",
    ".benchmark_data",
    ".pytest_cache",
    ".repolens-cache",
    ".repolens-index",
    ".repolens_embeddings",
    "models",
    ".cache",
    ".eggs",
    "*.egg-info",
}


@dataclass(frozen=True)
class StageReport:
    """One measured stage: a name, a wall-clock time, and scalar metrics."""

    name: str
    elapsed_ms: float
    metrics: Mapping[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        return self.metrics.get(key, default)

    def metric_text(self) -> str:
        """Render ``k=v k=v ...`` (``None`` values omitted)."""
        parts = []
        for key, value in self.metrics.items():
            if value is None:
                continue
            parts.append(f"{key}={value}")
        return " ".join(parts)


@dataclass(frozen=True)
class ProductionReport:
    """Aggregate of every stage measured for one repository."""

    repository: Path
    files_discovered: int
    discovery: StageReport
    cold_index: StageReport
    warm_index: StageReport
    incremental: tuple[StageReport, ...]
    embedding_cache: tuple[StageReport, ...]
    retrieval: tuple[StageReport, ...]
    context: StageReport

    def to_text(self) -> str:
        """Render a concise, human-readable report (see documentation)."""
        lines = [f"Repository: {self.repository}"]
        lines.append(f"Files discovered: {self.files_discovered}")
        lines.append(f"Python files: {self.files_discovered}")

        def add(report: StageReport | None, indent: str = "  ") -> None:
            if report is None:
                return
            text = f"{indent}{report.name}: {report.elapsed_ms} ms"
            metric_text = report.metric_text()
            lines.append(f"{text}  {metric_text}".rstrip())

        add(self.discovery)
        lines.append("Cold index:")
        add(self.cold_index)
        lines.append("Warm index:")
        add(self.warm_index)
        if self.incremental:
            lines.append("Incremental update (fresh temp copy of the repo):")
            for report in self.incremental:
                add(report)
        if self.embedding_cache:
            lines.append("Embedding / cache statistics (fresh temp copy):")
            for report in self.embedding_cache:
                add(report)
        if self.retrieval:
            lines.append("Retrieval latency (median over repeated runs):")
            for report in self.retrieval:
                add(report)
        lines.append("Context generation:")
        add(self.context)
        return "\n".join(lines)


class CountingEmbeddingProvider(FakeEmbeddingProvider):
    """A :class:`FakeEmbeddingProvider` that counts embedded texts.

    ``embedding_identity`` (used by the persistent cache) is identical to
    ``FakeEmbeddingProvider`` of the same dimension, so cold/warm searchers
    with different instances share cache entries.
    """

    def __init__(self, dimensions: int = 1024) -> None:
        super().__init__(dimensions=dimensions)
        self.embedded_documents = 0
        self.embedded_queries = 0

    def embed_texts(self, texts: Sequence[str]):
        self.embedded_documents += len(texts)
        return super().embed_texts(texts)

    def embed_text(self, text: str):
        self.embedded_queries += 1
        return super().embed_text(text)


def _measure_stage(
    name: str,
    fn,
    *,
    measure_memory: bool,
) -> tuple[StageReport, Any]:
    """Time ``fn()`` → ``(metrics, payload)`` and wrap it in a StageReport.

    Uses :func:`time.perf_counter`. When ``measure_memory`` is set,
    :mod:`tracemalloc` is started before ``fn`` and the peak traced memory is
    recorded as ``peak_mem_bytes``.
    """
    if measure_memory:
        tracemalloc.start()
    started = time.perf_counter()
    metrics, payload = fn()
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    peak = None
    if measure_memory:
        peak = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
    if peak is not None:
        metrics = {**metrics, "peak_mem_bytes": peak}
    return (
        StageReport(name=name, elapsed_ms=round(elapsed_ms, 3), metrics=metrics),
        payload,
    )


def _copy_repo_for_benchmark(source: Path, dest: Path) -> None:
    """Copy a repository into ``dest``, skipping caches and virtualenvs."""
    shutil.copytree(
        source, dest, ignore=shutil.ignore_patterns(*_SKIP_NAMES, "*.pyc")
    )


def _python_file_count(root: Path) -> int:
    return len(RepositoryScanner(root).discover_python_files())


# ---------------------------------------------------------------------------
# Stage 1 — repository discovery
# ---------------------------------------------------------------------------


def benchmark_discovery(root: Path, *, measure_memory: bool = False) -> StageReport:
    """Measure repository scanning: time to discover all Python files."""
    root = Path(root)

    def work():
        files = RepositoryScanner(root).discover_python_files()
        return {"files_discovered": len(files), "python_files": len(files)}, None

    report, _ = _measure_stage("discovery", work, measure_memory=measure_memory)
    return report


# ---------------------------------------------------------------------------
# Stage 2 — cold vs warm indexing
# ---------------------------------------------------------------------------


def benchmark_index_builds(
    root: Path,
    *,
    cache_dir: Path | str | None = None,
    measure_memory: bool = False,
) -> tuple[StageReport, StageReport]:
    """Measure a clean (cold) index build and an unchanged (warm) rebuild.

    ``cache_dir`` is a directory to reuse between the two builds; when ``None``
    a fresh temporary directory is created (and removed) so the user's caches
    are never touched. Both builds use the *same* cache directory, so the warm
    build is expected to parse zero files.
    """
    root = Path(root)
    owned = cache_dir is None
    base = Path(tempfile.mkdtemp(prefix="repolens-bench-")) if owned else Path(cache_dir)
    cold_cache = base / "index"
    try:
        def build():
            index = IncrementalIndexBuilder(root, cache_dir=cold_cache).build()
            return index.stats.as_dict(), index

        cold, cold_payload = _measure_stage(
            "cold_index", build, measure_memory=measure_memory
        )
        warm, warm_payload = _measure_stage(
            "warm_index", build, measure_memory=measure_memory
        )
        return cold, warm
    finally:
        if owned:
            shutil.rmtree(base, ignore_errors=True)


def benchmark_incremental_workflow(
    root: Path,
    *,
    measure_memory: bool = False,
) -> tuple[StageReport, ...]:
    """Run the A–E incremental-index verification on a temp copy of ``root``.

    Returns one :class:`StageReport` per transition:

    A. cold build — parses every Python file;
    B. warm rebuild — parses zero files (all cache hits);
    C. one Python file modified by appending a comment — reparses exactly one;
    D. one Python file added — parses exactly one;
    E. one Python file deleted — removes its stale cache entry.

    The repository passed to the builder is a freshly copied scratch repo, so
    the caller's repository is never mutated. Returns an empty tuple when the
    repository has no Python files to exercise.
    """
    root = Path(root)
    with tempfile.TemporaryDirectory(prefix="repolens-workflow-") as tmp:
        work = Path(tmp) / "repo"
        _copy_repo_for_benchmark(root, work)
        scanner = RepositoryScanner(work)
        files = scanner.discover_python_files()
        if not files:
            return ()
        cache = Path(tmp) / "index_cache"
        reports: list[StageReport] = []

        def build():
            index = IncrementalIndexBuilder(work, cache_dir=cache).build()
            return index.stats.as_dict(), index

        cold, _ = _measure_stage("cold", build, measure_memory=measure_memory)
        reports.append(cold)

        warm, _ = _measure_stage("warm", build, measure_memory=measure_memory)
        reports.append(warm)

        target = files[0]
        modified = work / target
        _append_line(modified, "# repolens m20 benchmark comment")
        modified_report, _ = _measure_stage("modified", build, measure_memory=measure_memory)
        reports.append(modified_report)

        added = work / "m20_added_module.py"
        added.write_text("def m20_added_function():\n    return 42\n", encoding="utf-8")
        added_report, _ = _measure_stage("added", build, measure_memory=measure_memory)
        reports.append(added_report)

        if len(files) > 1:
            delete_target = files[1]
        else:
            delete_target = Path("m20_delete_target.py")
        (work / delete_target).unlink()
        deleted_report, _ = _measure_stage("deleted", build, measure_memory=measure_memory)
        reports.append(deleted_report)

        return tuple(reports)


def _append_line(path: Path, line: str) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"\n{line}\n")


# ---------------------------------------------------------------------------
# Stage 3 — embedding cache validation
# ---------------------------------------------------------------------------


def benchmark_embedding_cache(
    root: Path,
    *,
    queries: Sequence[str] = DEFAULT_QUERIES,
    candidate_limit: int = DEFAULT_CANDIDATE_LIMIT,
    measure_memory: bool = False,
) -> tuple[StageReport, ...]:
    """Measure embedding-cache cold/warm/one-file-changed behaviour.

    Runs on a fresh temporary copy of ``root`` with a fresh on-disk cache:

    1. **cold** — a new searcher embeds every candidate document;
    2. **warm** — a "fresh process" searcher sharing the cache embeds nothing;
    3. **changed** — after one candidate source file changes, exactly that
       document is embedded again (the rest are cache hits).

    Returns an empty tuple when there is no Python file to embed. Only
    document vectors are ever stored in the persistent cache.
    """
    root = Path(root)
    with tempfile.TemporaryDirectory(prefix="repolens-emb-") as tmp:
        work = Path(tmp) / "repo"
        _copy_repo_for_benchmark(root, work)
        scanner = RepositoryScanner(work)
        files = scanner.discover_python_files()
        if not files:
            return ()
        query = _query_with_candidates(work, queries, candidate_limit)
        if query is None:
            return ()
        cache_dir = Path(tmp) / "emb_cache"
        cold_provider = CountingEmbeddingProvider()

        def cold():
            provider = cold_provider
            searcher = _build_candidate_searcher(work, provider, cache_dir)
            results = searcher.search(query, limit=10)
            metrics = {
                "embedded_documents": provider.embedded_documents,
                "embedded_queries": provider.embedded_queries,
                "cache_hits": searcher.cache_stats["hits"],
                "cache_misses": searcher.cache_stats["misses"],
                "results": len(results),
            }
            return metrics, searcher

        cold_report, cold_searcher = _measure_stage(
            "cold_embedding", cold, measure_memory=measure_memory
        )

        warm_provider = CountingEmbeddingProvider()

        def warm():
            provider = warm_provider
            searcher = _build_candidate_searcher(work, provider, cache_dir)
            results = searcher.search(query, limit=10)
            metrics = {
                "embedded_documents": provider.embedded_documents,
                "embedded_queries": provider.embedded_queries,
                "cache_hits": searcher.cache_stats["hits"],
                "cache_misses": searcher.cache_stats["misses"],
                "results": len(results),
            }
            return metrics, searcher

        warm_report, _ = _measure_stage("warm_embedding", warm, measure_memory=measure_memory)

        # Change one candidate file (chosen via the lexical candidate set).
        candidate_paths = [
            r.file_path for r in CodeSearcher(work).search(query, limit=candidate_limit)
        ]
        if not candidate_paths:
            return (cold_report, warm_report)
        changed_file = work / candidate_paths[0]
        _append_line(changed_file, "# m20 changed candidate content")
        changed_provider = CountingEmbeddingProvider()

        def changed():
            provider = changed_provider
            searcher = _build_candidate_searcher(work, provider, cache_dir)
            results = searcher.search(query, limit=10)
            metrics = {
                "embedded_documents": provider.embedded_documents,
                "embedded_queries": provider.embedded_queries,
                "cache_hits": searcher.cache_stats["hits"],
                "cache_misses": searcher.cache_stats["misses"],
                "results": len(results),
            }
            return metrics, searcher

        changed_report, _ = _measure_stage(
            "changed_embedding", changed, measure_memory=measure_memory
        )

        return (cold_report, warm_report, changed_report)


def _query_with_candidates(
    root: Path, queries: Sequence[str], candidate_limit: int
) -> str | None:
    """Return the first query whose lexical candidate set is non-empty."""
    searcher = CodeSearcher(root)
    for query in queries:
        if searcher.search(query, limit=candidate_limit):
            return query
    return None


def _build_candidate_searcher(work: Path, provider, cache_dir):
    """Build a candidate-based SemanticSearcher sharing ``cache_dir``."""
    from repolens.embedding_cache import make_repo_cache
    from repolens.semantic_search import SemanticSearcher

    return SemanticSearcher(
        work,
        provider,
        candidate_searcher=CodeSearcher(work),
        cache=make_repo_cache(work, directory=cache_dir),
    )


# ---------------------------------------------------------------------------
# Stage 4 — retrieval latency
# ---------------------------------------------------------------------------


def benchmark_retrieval(
    root: Path,
    queries: Sequence[str],
    *,
    provider: EmbeddingProvider | None = None,
    candidate_limit: int = DEFAULT_CANDIDATE_LIMIT,
    repeats: int = DEFAULT_REPEATS,
    measure_memory: bool = False,
) -> tuple[StageReport, ...]:
    """Measure per-strategy retrieval latency for representative queries.

    Strategies measured (each reusing the same scripted index snapshot):

    - ``lexical`` — :class:`~repolens.search.CodeSearcher` (no embeddings);
    - ``semantic`` — :class:`~repolens.semantic_search.SemanticSearcher` with
      an unbounded candidate limit (embeds the full candidate surface);
    - ``candidate-semantic`` — candidate-based semantic (default limit);
    - ``rrf`` — hybrid with reciprocal-rank fusion;
    - ``weighted`` — hybrid with weighted-sum fusion.

    Each strategy is run ``repeats`` times per query; the report carries the
    median/min/max latency in milliseconds, the number of runs, and, where
    relevant, the number of documents embedded.

    The default provider is :class:`~repolens.embeddings.FakeEmbeddingProvider`
    (deterministic, offline).
    """
    root = Path(root)
    provider = provider or FakeEmbeddingProvider()
    queries = list(queries)
    index = IncrementalIndexBuilder(root, persist=False).build()
    num_files = max(1, len(index.files))
    with tempfile.TemporaryDirectory(prefix="repolens-latency-") as tmp:
        from repolens.embedding_cache import make_repo_cache
        from repolens.retrieval import FusionStrategy, HybridSearcher
        from repolens.semantic_search import SemanticSearcher

        cache = make_repo_cache(root, directory=Path(tmp) / "emb")
        lexical = CodeSearcher(root, index=index)
        candidate_semantic = SemanticSearcher(
            root,
            provider,
            candidate_searcher=lexical,
            candidate_limit=candidate_limit,
            cache=cache,
            index=index,
        )
        full_semantic = SemanticSearcher(
            root,
            provider,
            candidate_searcher=lexical,
            candidate_limit=num_files,
            cache=cache,
            index=index,
        )
        rrf = HybridSearcher(
            root,
            lexical_searcher=lexical,
            semantic_searcher=candidate_semantic,
            strategy=FusionStrategy.RRF,
        )
        weighted = HybridSearcher(
            root,
            lexical_searcher=lexical,
            semantic_searcher=candidate_semantic,
            strategy=FusionStrategy.WEIGHTED,
        )
        strategies = {
            "lexical": lexical,
            "semantic": full_semantic,
            "candidate-semantic": candidate_semantic,
            "rrf": rrf,
            "weighted": weighted,
        }

        reports = []
        for strategy_name, searcher in strategies.items():
            latencies_ms: list[float] = []
            total_results = 0
            started = time.perf_counter()
            for _ in range(max(1, repeats)):
                for query in queries:
                    elapsed_started = time.perf_counter()
                    results = searcher.search(query, limit=10)
                    elapsed_ms = (time.perf_counter() - elapsed_started) * 1000.0
                    latencies_ms.append(elapsed_ms)
                    total_results += len(results)
            loop_elapsed_ms = (time.perf_counter() - started) * 1000.0
            number_of_runs = max(1, repeats) * len(queries) if queries else 0
            metrics: dict[str, Any] = {
                "runs": number_of_runs,
                "median_ms": round(median(latencies_ms), 3) if latencies_ms else None,
                "min_ms": round(min(latencies_ms), 3) if latencies_ms else None,
                "max_ms": round(max(latencies_ms), 3) if latencies_ms else None,
                "results": total_results,
            }
            rate_metrics = _rate_metrics(searcher)
            metrics.update(rate_metrics)
            reports.append(
                StageReport(
                    name=strategy_name,
                    elapsed_ms=round(loop_elapsed_ms, 3),
                    metrics=metrics,
                )
            )
        return tuple(reports)


def _rate_metrics(searcher: Any) -> dict[str, Any]:
    """Extract embedding/cache counters from any searcher (hybrid-aware)."""
    inner = getattr(searcher, "_semantic", searcher)
    metrics: dict[str, Any] = {}
    if hasattr(inner, "cache_stats"):
        stats = inner.cache_stats
        metrics["cache_hits"] = stats.get("hits", 0)
        metrics["cache_misses"] = stats.get("misses", 0)
        metrics["embedded_documents"] = stats.get("embedded_documents", 0)
    return metrics


# ---------------------------------------------------------------------------
# Stage 5 — context generation
# ---------------------------------------------------------------------------


def benchmark_context_engine(
    root: Path,
    queries: Sequence[str],
    *,
    provider: EmbeddingProvider | None = None,
    cache_dir: Path | str | None = None,
    measure_memory: bool = False,
) -> StageReport:
    """Measure :class:`~repolens.context.ContextEngine.build_context` latency.

    Reports per-query median latency plus structural counters: number of
    candidate files before budgeting, selected files, and the resulting
    estimated context size (always at or under the budget).

    Uses the default RRF hybrid searcher over a scripted incremental snapshot,
    with the persistent embedding cache enabled when ``cache_dir`` is given
    (otherwise an ephemeral in-memory embedding cache).
    """
    from repolens.context import ContextBudget, ContextEngine, RetrievalConfig

    root = Path(root)
    provider = provider or FakeEmbeddingProvider()
    queries = list(queries)
    index = IncrementalIndexBuilder(root, persist=False).build()
    budget = ContextBudget()
    owned = cache_dir is None

    def build_engine():
        if owned:
            embedding_cache = None
        else:
            from repolens.embedding_cache import make_repo_cache

            embedding_cache = make_repo_cache(root, directory=Path(cache_dir))
        searcher = RetrievalConfig().build_searcher(
            root,
            embedding_provider=provider,
            embedding_cache=embedding_cache,
            index=index,
        )
        return ContextEngine(
            root, searcher=searcher, budget=budget, index=index
        )

    engine = build_engine()

    latencies_ms: list[float] = []
    candidates_total = 0
    selected_total = 0
    sizes: list[int] = []
    budget_value = budget.max_tokens

    def work():
        nonlocal candidates_total, selected_total
        for query in queries:
            started = time.perf_counter()
            package = engine.build_context(query)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            latencies_ms.append(elapsed_ms)
            candidates = (
                len(package.primary_candidates) + len(package.dependency_candidates)
            )
            candidates_total += candidates
            selected_total += len(package.selected_files)
            sizes.append(package.total_estimated_tokens)
        metrics: dict[str, Any] = {
            "queries": len(queries) if queries else 0,
            "median_ms": round(median(latencies_ms), 3) if latencies_ms else None,
            "min_ms": round(min(latencies_ms), 3) if latencies_ms else None,
            "max_ms": round(max(latencies_ms), 3) if latencies_ms else None,
            "budget": budget_value,
        }
        if sizes:
            metrics["context_size_median"] = int(median(sizes))
            metrics["context_size_max"] = max(sizes)
        if queries:
            metrics["candidates"] = candidates_total // len(queries)
            metrics["selected"] = selected_total // len(queries)
        return metrics, None

    report, _ = _measure_stage("context", work, measure_memory=measure_memory)
    return report


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_production_benchmark(
    root: Path | str,
    *,
    queries: Sequence[str] | None = None,
    cache_dir: Path | str | None = None,
    candidate_limit: int = DEFAULT_CANDIDATE_LIMIT,
    repeats: int = DEFAULT_REPEATS,
    measure_memory: bool = False,
    provider: EmbeddingProvider | None = None,
) -> ProductionReport:
    """Run the full production benchmark against ``root``.

    This is the single entry point used by ``benchmarks/production_benchmark.py``
    and by the final-verification workflow. All mutation-based stages run on a
    fresh temporary copy of the repository; only read-only stages (discovery,
    cold/warm index builds, retrieval, context) ever touch the real repository,
    and the index/embedding caches they use live in temporary directories.
    """
    root = Path(root)
    if not root.is_dir():
        raise NotADirectoryError(f"repository root is not a directory: {root}")
    effective_queries = list(queries) if queries else list(DEFAULT_QUERIES)
    provider = provider or FakeEmbeddingProvider()

    discovery = benchmark_discovery(root, measure_memory=measure_memory)
    files_discovered = discovery.get("files_discovered", 0)
    cold_index, warm_index = benchmark_index_builds(
        root, cache_dir=cache_dir, measure_memory=measure_memory
    )
    incremental = benchmark_incremental_workflow(root, measure_memory=measure_memory)
    embedding_cache = benchmark_embedding_cache(
        root,
        queries=effective_queries,
        candidate_limit=candidate_limit,
        measure_memory=measure_memory,
    )
    retrieval = benchmark_retrieval(
        root,
        effective_queries,
        provider=provider,
        candidate_limit=candidate_limit,
        repeats=repeats,
        measure_memory=measure_memory,
    )
    context = benchmark_context_engine(
        root,
        effective_queries,
        provider=provider,
        cache_dir=cache_dir,
        measure_memory=measure_memory,
    )

    return ProductionReport(
        repository=root,
        files_discovered=files_discovered,
        discovery=discovery,
        cold_index=cold_index,
        warm_index=warm_index,
        incremental=incremental,
        embedding_cache=embedding_cache,
        retrieval=retrieval,
        context=context,
    )