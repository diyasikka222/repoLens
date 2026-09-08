#!/usr/bin/env python3
"""Performance and scalability benchmark for RepoLens (Phase 25.4).

Measures every major production path — discovery, indexing (cold / warm /
cache-disabled), incremental updates, embedding-cache behaviour, retrieval
strategies, context generation, architecture, impact analysis, change
planning, and change-aware context — across deterministic synthetic
repositories of growing size, and against a real repository when run with a
path argument::

    python benchmarks/performance_scalability.py                 # synthetic only
    python benchmarks/performance_scalability.py .               # real repository
    python benchmarks/performance_scalability.py --scale medium  # one synthetic scale
    python benchmarks/performance_scalability.py --queries q.txt /path/to/repo

Design notes
------------
* **Deterministic and offline.** Synthetic corpora are generated from a fixed
  seed; embeddings use the offline
  :class:`~repolens.embeddings.FakeEmbeddingProvider` by default. Nothing is
  downloaded and no credentials are required.
* **Structural, not wall-clock.** Every operation reports counts (files
  parsed, cache hits/misses, candidates, selected files, context size) plus a
  deterministic fingerprint, alongside timings from ``time.perf_counter()``.
  No absolute real-time threshold is asserted anywhere, so results are
  portable across machines; the numbers are meant to be diffed on the same
  machine over time.
* **Non-destructive.** Mutation stages run on a fresh temporary copy of the
  repository; all caches live under a temporary base unless ``--cache-dir``
  is given. The measured repository is never modified.
* **Reuses the production harness.** Discovery, cold/warm indexing, the
  incremental workflow, and embedding-cache behaviour come from
  :mod:`repolens.production_benchmark` verbatim. The new operations build on
  the same public RepoLens classes the MCP server uses (incremental index,
  dependency graph, symbol/reference/call graphs, architecture graph, impact
  analyzer, change-plan engine, context engine, firewall); no ranking,
  budgeting, or planning logic is re-implemented here.

Synthetic corpora (exact sizes documented in ``docs/performance.md``):

* ``small``  — 9  Python files (2 packages x 2 modules + 2 ``__init__`` + ``shared.py`` + 2 tests)
* ``medium`` — 25 Python files (4 packages x 4 modules + 4 ``__init__`` + ``shared.py`` + 4 tests)
* ``large``  — 51 Python files (6 packages x 6 modules + 6 ``__init__`` + ``shared.py`` + 8 tests)

Each corpus contains packages, modules, functions, classes, cross-file
imports and calls, package-level re-exports, and tests, so every measured
pipeline sees realistic dependency and call structure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median
from typing import Any, Callable

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from repolens.architecture import ArchitectureGraphBuilder  # noqa: E402
from repolens.call_graph import CallGraphBuilder  # noqa: E402
from repolens.change_plan import ChangePlanConfig, ChangePlanEngine  # noqa: E402
from repolens.change_context import ChangeContextOptions  # noqa: E402
from repolens.context import ContextBudget, ContextEngine, ContextFirewall, RetrievalConfig  # noqa: E402
from repolens.embeddings import EmbeddingProvider, FakeEmbeddingProvider  # noqa: E402
from repolens.graph import DependencyGraphBuilder  # noqa: E402
from repolens.impact import ImpactAnalyzer  # noqa: E402
from repolens.index import SymbolIndexBuilder  # noqa: E402
from repolens.incremental_index import IncrementalIndexBuilder, RepositoryIndex  # noqa: E402
from repolens.production_benchmark import (  # noqa: E402
    DEFAULT_CANDIDATE_LIMIT,
    DEFAULT_QUERIES,
    DEFAULT_REPEATS,
    StageReport,
    benchmark_discovery,
    benchmark_embedding_cache,
    benchmark_incremental_workflow,
    benchmark_index_builds,
)
from repolens.references import ReferenceIndexBuilder  # noqa: E402
from repolens.retrieval import FusionStrategy, HybridSearcher  # noqa: E402
from repolens.search import CodeSearcher  # noqa: E402
from repolens.semantic_search import SemanticSearcher  # noqa: E402
from repolens.subsystems import discover_subsystems  # noqa: E402

# ---------------------------------------------------------------------------
# Synthetic corpus generator
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScalePreset:
    """Deterministic structural recipe for one synthetic corpus scale."""

    packages: int
    modules: int
    functions: int
    classes: int
    methods: int
    tests: int

    @property
    def total_python_files(self) -> int:
        """Exact number of generated ``.py`` files for this preset."""
        return self.packages * self.modules + self.packages + 1 + self.tests


SCALE_PRESETS: dict[str, ScalePreset] = {
    "small": ScalePreset(packages=2, modules=2, functions=3, classes=1, methods=2, tests=2),
    "medium": ScalePreset(packages=4, modules=4, functions=4, classes=2, methods=3, tests=4),
    "large": ScalePreset(packages=6, modules=6, functions=5, classes=3, methods=3, tests=8),
}


@dataclass(frozen=True)
class CorpusSpec:
    """What :func:`generate_synthetic_repository` produced or describes."""

    root: Path
    scale: str
    seed: int
    files: tuple[str, ...]
    queries: tuple[str, ...]
    counts: dict[str, int]


def _module_prev(packages: int, modules: int, i: int, j: int) -> tuple[int, int]:
    """The sibling module a module reuses (same package, or previous package)."""
    if j > 0:
        return i, j - 1
    if i > 0:
        return i - 1, modules - 1
    return 0, 0


def _shared_source(seed: int) -> str:
    salt = random.Random(seed).randint(1, 999)
    return (
        f'"""{seed} synthetic shared utilities."""\n'
        "from __future__ import annotations\n\n"
        "def transform_0(value):\n"
        "    return value + 1\n\n"
        "def transform_1(value):\n"
        f"    return value * {salt % 7 + 1}\n\n"
        "def bootstrap(value):\n"
        "    return transform_0(value) + transform_1(value)\n"
    )


def _module_source(
    seed: int, i: int, j: int, *, functions: int, classes: int, methods: int,
    modules: int, packages: int, salt: int,
) -> str:
    """Deterministic source for ``pkg_<i>.module_<j>`` with cross-file deps."""
    prev_i, prev_j = _module_prev(packages, modules, i, j)
    uses_shared = prev_i == 0 and prev_j == 0
    if uses_shared:
        peer_import = "import shared as peer"
        from_import = "from shared import transform_1 as imported_apply"
        peer_call = "peer.transform_1(value + {})"
    else:
        peer_import = f"import pkg_{prev_i}.module_{prev_j} as peer"
        from_import = f"from pkg_{prev_i}.module_{prev_j} import apply_1 as imported_apply"
        peer_call = "peer.apply_1(value + {})"

    lines = [
        f'"""{seed} synthetic module pkg_{i}.module_{j}."""',
        "from __future__ import annotations",
        "",
        peer_import,
        from_import,
        "",
        "ROOTS = []",
        "",
    ]
    for c in range(classes):
        base = "" if c == 0 else f"(Handler_{c - 1})"
        lines.append(f"class Handler_{c}{base}:")
        lines.append('    """Handler chained to its sibling module."""')
        for m in range(methods):
            call = peer_call.format(m + salt)
            lines.append(f"    def handle_{m}(self, value):")
            lines.append(f"        return {call}")
        lines.append("")
    for k in range(functions):
        lines.append(f"def apply_{k}(value, scale={salt}):")
        lines.append('    """Deterministic transform helper."""')
        lines.append(f"    return value * scale + {k}")
        lines.append("")
        lines.append(f"def run_{k}(value):")
        call = peer_call.format(k)
        lines.append(f"    result = {call}")
        lines.append("    imported = imported_apply(value)")
        lines.append("    ROOTS.append(result + imported)")
        lines.append("    return result")
        lines.append("")
    if i == 0 and j == 0:
        lines.append(f"def bootstrap_{i}(value):")
        lines.append("    return apply_0(value) + apply_1(value)")
        lines.append("")
    return "\n".join(lines)


def _package_init_source(i: int, modules: int) -> str:
    lines = [f'"""{i} package of the synthetic corpus."""', ""]
    for j in range(modules):
        lines.append(f"from pkg_{i}.module_{j} import run_0")
    lines.append(f"from pkg_{i}.module_0 import bootstrap_{i}")
    lines.append("")
    return "\n".join(lines)


def _test_source(i: int) -> str:
    return (
        f'"""{i} tests for the synthetic corpus."""\n'
        f"from pkg_{i} import bootstrap_{i}\n"
        f"from pkg_{i}.module_0 import apply_0\n\n"
        f"def test_bootstrap_{i}():\n"
        f"    assert bootstrap_{i}(1) >= 1\n\n"
        f"def test_apply_{i}():\n"
        f"    assert isinstance(apply_0(1), int)\n"
    )


def _generate_repository(root: Path, preset: ScalePreset, seed: int, scale: str) -> CorpusSpec:
    """Write a deterministic synthetic repository under ``root``."""
    salt = random.Random(seed).randint(1, 997)
    files: list[str] = []
    counts: dict[str, int] = {}

    root.mkdir(parents=True, exist_ok=True)

    (root / "shared.py").write_text(_shared_source(seed), encoding="utf-8")
    files.append("shared.py")

    module_files = 0
    function_count = 0
    method_count = 0
    class_count = 0
    for i in range(preset.packages):
        pkg_dir = root / f"pkg_{i}"
        pkg_dir.mkdir(parents=True, exist_ok=True)
        (pkg_dir / "__init__.py").write_text(
            _package_init_source(i, preset.modules), encoding="utf-8"
        )
        files.append(f"pkg_{i}/__init__.py")
        for j in range(preset.modules):
            source = _module_source(
                seed, i, j,
                functions=preset.functions,
                classes=preset.classes,
                methods=preset.methods,
                modules=preset.modules,
                packages=preset.packages,
                salt=salt,
            )
            (pkg_dir / f"module_{j}.py").write_text(source, encoding="utf-8")
            files.append(f"pkg_{i}/module_{j}.py")
            module_files += 1
            function_count += preset.functions
            class_count += preset.classes
            method_count += preset.classes * preset.methods

    tests_dir = root / "tests"
    tests_dir.mkdir(parents=True, exist_ok=True)
    for i in range(preset.tests):
        (tests_dir / f"test_pkg_{i}.py").write_text(
            _test_source(i), encoding="utf-8"
        )
        files.append(f"tests/test_pkg_{i}.py")

    files.sort()
    counts.update(
        {
            "python_files": len(files),
            "module_files": module_files,
            "init_files": preset.packages,
            "shared_files": 1,
            "test_files": preset.tests,
            "functions": function_count,
            "classes": class_count,
            "methods": method_count,
            "packages": preset.packages,
        }
    )
    queries = _corpus_queries(files, seed)
    return CorpusSpec(
        root=root,
        scale=scale,
        seed=seed,
        files=tuple(files),
        queries=queries,
        counts=counts,
    )


def scale_preset(scale: str) -> ScalePreset:
    """Return the structural preset for ``scale`` (raises on unknown names)."""
    if scale not in SCALE_PRESETS:
        choices = ", ".join(sorted(SCALE_PRESETS))
        raise ValueError(f"unknown scale {scale!r}; expected one of: {choices}")
    return SCALE_PRESETS[scale]


def generate_synthetic_repository(root: Path, scale: str, seed: int = 7) -> CorpusSpec:
    """Generate a deterministic synthetic repository at ``root``.

    Identical ``seed`` and ``scale`` always produce byte-identical trees;
    different seeds produce different content (every source embeds
    seed-derived constants). Returns a :class:`CorpusSpec` describing what was
    written, including a deterministic query corpus matched to the generated
    names.
    """
    return _generate_repository(Path(root), scale_preset(scale), seed, scale)


def _corpus_queries(files: list[str], seed: int) -> tuple[str, ...]:
    """Deterministic queries that lexically hit the generated tree."""
    modules = sorted(
        f[:-3].replace("/", ".")
        for f in files
        if f.startswith("pkg_") and f.endswith(".py") and not f.endswith("__init__.py")
    )
    queries = [
        f"{seed}_shared_transform_bootstrap_module_function",
        "shared transform bootstrap",
    ]
    for mod in modules[: min(6, len(modules))]:
        queries.append(f"{mod} run pipeline")
    queries.append("pkg_1 module function apply")
    return tuple(queries)


# ---------------------------------------------------------------------------
# Timing + fingerprint helpers
# ---------------------------------------------------------------------------


def _percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    k = (len(ordered) - 1) * (p / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(ordered) - 1)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


def _latency_stats(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"median_ms": None, "p95_ms": None, "min_ms": None, "max_ms": None}
    return {
        "median_ms": round(median(values), 3),
        "p95_ms": round(_percentile(values, 95.0), 3),
        "min_ms": round(min(values), 3),
        "max_ms": round(max(values), 3),
    }


def _fingerprint(*parts: Any) -> str:
    """Deterministic short fingerprint of JSON-safe ``parts``."""
    payload = json.dumps(parts, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _index_fingerprint(index: RepositoryIndex) -> str:
    """Stable fingerprint of the per-file content hashes in an index."""
    entries = sorted(
        (path.as_posix(), index.by_path[path].content_hash)
        for path in index.files
    )
    return _fingerprint(entries)


def _measure(name: str, fn: Callable[[], tuple[dict[str, Any], Any]]) -> tuple[StageReport, Any]:
    """Time ``fn`` → ``(metrics, payload)`` into a :class:`StageReport`."""
    started = time.perf_counter()
    metrics, payload = fn()
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return (
        StageReport(name=name, elapsed_ms=round(elapsed_ms, 3), metrics=metrics),
        payload,
    )


# ---------------------------------------------------------------------------
# Index + shared components
# ---------------------------------------------------------------------------


def benchmark_cache_disabled_index(root: Path, *, repeats: int = 2) -> tuple[StageReport, ...]:
    """Index with persistence disabled; every build must parse every file.

    Reports one :class:`StageReport` per build with ``files_parsed`` equal to
    the discovered count and ``cache_hits == 0`` — the structural evidence
    that the persistent cache was genuinely bypassed.
    """
    root = Path(root)
    reports: list[StageReport] = []
    for run in range(max(1, repeats)):
        name = "cache_disabled" if run == 0 else f"cache_disabled_repeat_{run + 1}"

        def build():
            index = IncrementalIndexBuilder(root, persist=False).build()
            return index.stats.as_dict(), index

        report, index = _measure(name, build)
        report.metrics["fingerprint"] = _index_fingerprint(index)
        reports.append(report)
    return tuple(reports)


def _build_shared_components(root: Path, index: RepositoryIndex) -> dict[str, Any]:
    """Derive the dependency/symbol/reference/call components from one index."""
    dependency_graph = DependencyGraphBuilder(root, index=index).build()
    symbol_index = SymbolIndexBuilder(root, index=index).build()
    reference_index = ReferenceIndexBuilder(root, index=index).build()
    call_graph = CallGraphBuilder(
        root, index=index, reference_index=reference_index,
        symbol_index=symbol_index,
    ).build()
    return {
        "dependency_graph": dependency_graph,
        "symbol_index": symbol_index,
        "reference_index": reference_index,
        "call_graph": call_graph,
    }


# ---------------------------------------------------------------------------
# Retrieval (H–L) and context (M)
# ---------------------------------------------------------------------------


def _build_searchers(root: Path, index: RepositoryIndex, provider, embedding_cache, candidate_limit: int) -> dict[str, Any]:
    """The five production searchers over one snapshot (mirrors the harness)."""
    lexical = CodeSearcher(root, index=index)
    candidate_semantic = SemanticSearcher(
        root, provider, candidate_searcher=lexical,
        candidate_limit=candidate_limit, cache=embedding_cache, index=index,
    )
    full_semantic = SemanticSearcher(
        root, provider, candidate_searcher=lexical,
        candidate_limit=max(1, len(index.files)), cache=embedding_cache,
        index=index,
    )
    rrf = HybridSearcher(
        root, lexical_searcher=lexical, semantic_searcher=candidate_semantic,
        strategy=FusionStrategy.RRF,
    )
    weighted = HybridSearcher(
        root, lexical_searcher=lexical, semantic_searcher=candidate_semantic,
        strategy=FusionStrategy.WEIGHTED,
    )
    return {
        "lexical": lexical,
        "candidate-semantic": candidate_semantic,
        "semantic": full_semantic,
        "rrf": rrf,
        "weighted": weighted,
    }


def _result_rows(searcher, query: str) -> list:
    results = searcher.search(query, limit=10)
    return sorted(
        (r.file_path.as_posix(), round(float(getattr(r, "score", 0.0)), 4))
        for r in results
    )


def benchmark_retrieval_scaled(
    root: Path,
    queries: list[str],
    *,
    index: RepositoryIndex,
    provider: EmbeddingProvider,
    candidate_limit: int,
    repeats: int,
    embedding_cache,
) -> tuple[StageReport, ...]:
    """Per-strategy retrieval latency (median / p95 / min / max) + fingerprints."""
    root = Path(root)
    searchers = _build_searchers(root, index, provider, embedding_cache, candidate_limit)
    reports: list[StageReport] = []
    for strategy, searcher in searchers.items():
        latencies: list[float] = []
        result_rows: list[list[Any]] = []
        total_results = 0
        started = time.perf_counter()
        for _ in range(max(1, repeats)):
            for query in queries:
                elapsed_started = time.perf_counter()
                rows = _result_rows(searcher, query)
                elapsed_ms = (time.perf_counter() - elapsed_started) * 1000.0
                latencies.append(elapsed_ms)
                result_rows.append([query, rows])
                total_results += len(rows)
        loop_elapsed_ms = (time.perf_counter() - started) * 1000.0
        metrics = _latency_stats(latencies)
        metrics.update(
            {
                "runs": max(1, repeats) * len(queries) if queries else 0,
                "results": total_results,
                "fingerprint": _fingerprint(sorted(result_rows)),
            }
        )
        inner = getattr(searcher, "_semantic", searcher)
        if hasattr(inner, "cache_stats"):
            stats = inner.cache_stats
            metrics["cache_hits"] = stats.get("hits", 0)
            metrics["cache_misses"] = stats.get("misses", 0)
            metrics["embedded_documents"] = stats.get("embedded_documents", 0)
        reports.append(
            StageReport(name=strategy, elapsed_ms=round(loop_elapsed_ms, 3), metrics=metrics)
        )
    return tuple(reports)


def _avg(values: list[int]) -> int:
    return round(sum(values) / len(values)) if values else 0


def benchmark_context_scaled(
    root: Path,
    queries: list[str],
    *,
    index: RepositoryIndex,
    provider: EmbeddingProvider,
    embedding_cache,
    budget: ContextBudget,
) -> tuple[StageReport, ContextEngine]:
    """Context-generation latency, size, budget compliance, and fingerprint."""
    root = Path(root)
    searcher = RetrievalConfig().build_searcher(
        root, embedding_provider=provider, embedding_cache=embedding_cache,
        index=index,
    )
    engine = ContextEngine(root, searcher=searcher, budget=budget, index=index)
    latencies: list[float] = []
    candidates: list[int] = []
    selected: list[int] = []
    sizes: list[int] = []
    fingerprints: list[str] = []

    for query in queries:
        elapsed_started = time.perf_counter()
        package = engine.build_context(query)
        elapsed_ms = (time.perf_counter() - elapsed_started) * 1000.0
        latencies.append(elapsed_ms)
        candidates.append(
            len(package.primary_candidates) + len(package.dependency_candidates)
        )
        selected.append(len(package.selected_files))
        sizes.append(package.total_estimated_tokens)
        fingerprints.append(
            _fingerprint(
                sorted(
                    (c.path.as_posix(), c.role, c.estimated_tokens)
                    for c in package.selected_files
                )
            )
        )

    metrics = _latency_stats(latencies)
    metrics.update(
        {
            "queries": len(queries) if queries else 0,
            "budget": budget.max_tokens,
            "candidates_avg": _avg(candidates),
            "selected_avg": _avg(selected),
            "context_size_median": int(median(sizes)) if sizes else None,
            "context_size_max": max(sizes) if sizes else None,
            "fingerprint": _fingerprint(sorted(fingerprints)) if fingerprints else None,
        }
    )
    report = StageReport(
        name="context", elapsed_ms=round(sum(latencies), 3), metrics=metrics
    )
    return report, engine


# ---------------------------------------------------------------------------
# Architecture (N)
# ---------------------------------------------------------------------------


def benchmark_architecture(
    root: Path, *, index: RepositoryIndex, dependency_graph,
) -> tuple[StageReport, StageReport]:
    """Architecture-graph construction and deterministic query cost."""
    root = Path(root)

    def build():
        graph = ArchitectureGraphBuilder(root, index=index, graph=dependency_graph).build()
        stats = graph.stats()
        metrics = dict(stats.as_dict())
        metrics["fingerprint"] = _fingerprint(graph.serialize())
        return metrics, graph

    build_report, graph = _measure("architecture_build", build)

    module_nodes = graph.modules()
    sample = module_nodes[: min(40, len(module_nodes))]
    latencies: list[float] = []
    rows: list[list[Any]] = []
    for node in sample:
        elapsed_started = time.perf_counter()
        _ = graph.get_module_node(node.id)
        deps = graph.dependencies_of(node)
        dependents = graph.dependents_of(node)
        elapsed_ms = (time.perf_counter() - elapsed_started) * 1000.0
        latencies.append(elapsed_ms)
        rows.append(
            [node.id, [d.id for d in deps], [d.id for d in dependents]]
        )
    query_metrics = _latency_stats(latencies)
    query_metrics.update(
        {
            "nodes_queried": len(sample),
            "fingerprint": _fingerprint(sorted(rows)),
        }
    )
    query_report = StageReport(
        name="architecture_query", elapsed_ms=round(sum(latencies), 3),
        metrics=query_metrics,
    )
    return build_report, query_report


# ---------------------------------------------------------------------------
# Impact (O)
# ---------------------------------------------------------------------------


def impact_targets(index: RepositoryIndex, *, limit: int = 8) -> list[str]:
    """Deterministic module targets that resolve unambiguously."""
    files = sorted(index.files, key=lambda p: p.as_posix())
    targets: list[str] = []
    for path in files:
        if path.name == "__init__.py" or "tests/" in path.as_posix():
            continue
        module = ".".join((*path.parts[:-1], path.stem))
        targets.append(module)
        if len(targets) >= limit:
            break
    if not targets and files:
        targets.append(files[0].as_posix())
    return targets


def benchmark_impact(
    root: Path,
    *,
    index: RepositoryIndex,
    dependency_graph,
    symbol_index,
    call_graph,
    targets: list[str],
) -> StageReport:
    """Impact-analysis latency, affected counts, risk, and fingerprint."""
    root = Path(root)
    analyzer = ImpactAnalyzer(
        root, index=index, graph=dependency_graph, symbol_index=symbol_index,
        reference_graph=call_graph,
    )
    latencies: list[float] = []
    affected: list[int] = []
    risks: list[str] = []
    fingerprints: list[str] = []
    for target in targets:
        elapsed_started = time.perf_counter()
        result = analyzer.analyze(target, max_depth=3, limit=200)
        elapsed_ms = (time.perf_counter() - elapsed_started) * 1000.0
        latencies.append(elapsed_ms)
        affected.append(len(result.items))
        risks.append(result.risk.value)
        fingerprints.append(_fingerprint(result.to_dict()))
    metrics = _latency_stats(latencies)
    metrics.update(
        {
            "targets": len(targets) if targets else 0,
            "affected_avg": _avg(affected),
            "affected_max": max(affected) if affected else None,
            "fingerprint": _fingerprint(sorted(fingerprints)) if fingerprints else None,
        }
    )
    return StageReport(
        name="impact", elapsed_ms=round(sum(latencies), 3), metrics=metrics
    )


# ---------------------------------------------------------------------------
# Change planning (P) and change-aware context (Q)
# ---------------------------------------------------------------------------


def plan_requests(queries: list[str], *, limit: int = 6) -> list[str]:
    """Deterministic change requests derived from the query corpus."""
    requests = [
        "refactor the module handling function apply and run internals",
        "add a new package module reusing the shared transform pipeline",
        "modify the bootstrap entry point in the first package",
        "fix the cross-package peer call used at module boundaries",
    ]
    requests.extend(queries[: max(0, limit - len(requests))])
    return requests[:limit]


def benchmark_change_plan(
    root: Path,
    *,
    index: RepositoryIndex,
    components: dict[str, Any],
    requests: list[str],
) -> tuple[StageReport, StageReport, ChangePlanEngine]:
    """Change-plan engine construction and per-request plan latency."""
    root = Path(root)
    architecture = ArchitectureGraphBuilder(
        root, index=index, graph=components["dependency_graph"],
    ).build()
    subsystems = discover_subsystems(architecture)

    def build_engine():
        engine = ChangePlanEngine(
            root,
            index=index,
            dependency_graph=components["dependency_graph"],
            symbol_index=components["symbol_index"],
            call_graph=components["call_graph"],
            arch_graph=architecture,
            subsystems=subsystems,
            impact_analyzer=ImpactAnalyzer(
                root, index=index, graph=components["dependency_graph"],
                symbol_index=components["symbol_index"],
                reference_graph=components["call_graph"],
            ),
            searcher=CodeSearcher(root, index=index),
            config=ChangePlanConfig(),
        )
        return {"parsed_files": index.stats.files_parsed}, engine

    build_report, engine = _measure("plan_engine_build", build_engine)
    build_report.metrics["fingerprint"] = _fingerprint(architecture.serialize())

    latencies: list[float] = []
    target_counts: list[int] = []
    affected_counts: list[int] = []
    risk_values: set[str] = set()
    fingerprints: list[str] = []
    for request in requests:
        elapsed_started = time.perf_counter()
        plan = engine.plan(request)
        elapsed_ms = (time.perf_counter() - elapsed_started) * 1000.0
        latencies.append(elapsed_ms)
        target_counts.append(len(plan.targets))
        affected_counts.append(len(plan.affected_files))
        risk_values.add(plan.risk)
        fingerprints.append(
            _fingerprint(
                {
                    "primary": plan.primary_target.target if plan.primary_target else None,
                    "affected": [i.path for i in plan.affected_files],
                    "tests": [t.path for t in plan.tests],
                    "risk": plan.risk,
                    "stats": plan.stats,
                }
            )
        )
    metrics = _latency_stats(latencies)
    metrics.update(
        {
            "requests": len(requests) if requests else 0,
            "targets_avg": _avg(target_counts),
            "affected_avg": _avg(affected_counts),
            "risks": ",".join(sorted(risk_values)),
            "fingerprint": _fingerprint(sorted(fingerprints)) if fingerprints else None,
        }
    )
    request_report = StageReport(
        name="change_plan_request", elapsed_ms=round(sum(latencies), 3),
        metrics=metrics,
    )
    return build_report, request_report, engine


def benchmark_change_context(
    root: Path,
    *,
    context_engine: ContextEngine,
    plan_engine: ChangePlanEngine,
    requests: list[str],
    firewall: ContextFirewall,
) -> StageReport:
    """End-to-end change-aware context: plan + fold + firewall inspection."""
    options = ChangeContextOptions()
    latencies: list[float] = []
    selected_counts: list[int] = []
    change_counts: list[int] = []
    tokens: list[int] = []
    fingerprints: list[str] = []
    for request in requests:
        elapsed_started = time.perf_counter()
        plan = plan_engine.plan(request)
        package = context_engine.build_context(
            request,
            change_request=request,
            change_target=plan.primary_target.target if plan.primary_target else None,
            change_plan=plan,
            change_options=options,
        )
        result = firewall.inspect(package)
        safe = firewall.safe_package(package, result)
        elapsed_ms = (time.perf_counter() - elapsed_started) * 1000.0
        latencies.append(elapsed_ms)
        selected_counts.append(len(safe.safe_files))
        change_counts.append(len(package.change_candidates))
        tokens.append(safe.total_estimated_tokens)
        fingerprints.append(
            _fingerprint(
                sorted(c.path for c in safe.safe_files)
                + [safe.budget.max_tokens]
            )
        )
    metrics = _latency_stats(latencies)
    metrics.update(
        {
            "requests": len(requests) if requests else 0,
            "selected_avg": _avg(selected_counts),
            "change_candidates_avg": _avg(change_counts),
            "tokens_median": int(median(tokens)) if tokens else None,
            "tokens_max": max(tokens) if tokens else None,
            "budget": ContextBudget().max_tokens,
            "fingerprint": _fingerprint(sorted(fingerprints)) if fingerprints else None,
        }
    )
    return StageReport(
        name="change_context", elapsed_ms=round(sum(latencies), 3), metrics=metrics
    )


# ---------------------------------------------------------------------------
# Orchestration + report
# ---------------------------------------------------------------------------


@dataclass
class ScalabilityReport:
    """Aggregate report for one repository/scene run."""

    repository: Path
    files_discovered: int
    python_files: int
    groups: dict[str, tuple[StageReport, ...]] = field(default_factory=dict)
    checks: list[tuple[str, bool, str]] = field(default_factory=list)
    scenes: list[str] = field(default_factory=list)

    def to_text(self) -> str:
        lines = [
            f"Repository: {self.repository}",
            f"Files discovered: {self.files_discovered}",
            f"Python files: {self.python_files}",
        ]
        for group, reports in self.groups.items():
            lines.append(f"{group}:")
            for report in reports:
                text = f"  {report.name}: {report.elapsed_ms} ms"
                metric_text = report.metric_text()
                lines.append(f"{text}  {metric_text}".rstrip())
        if self.checks:
            lines.append("Sanity checks:")
            for name, passed, detail in self.checks:
                marker = "PASS" if passed else "FAIL"
                lines.append(f"  [{marker}] {name}{(' — ' + detail) if detail else ''}")
        return "\n".join(lines)


def run_scalability_benchmark(
    root: Path | str,
    *,
    scale: str | None = None,
    seed: int = 7,
    queries: list[str] | None = None,
    repeats: int = DEFAULT_REPEATS,
    candidate_limit: int = DEFAULT_CANDIDATE_LIMIT,
    cache_dir: Path | str | None = None,
    measure_memory: bool = False,
    provider: EmbeddingProvider | None = None,
) -> ScalabilityReport:
    """Run the full performance/scalability benchmark against ``root``."""
    root = Path(root)
    if not root.is_dir():
        raise NotADirectoryError(f"repository root is not a directory: {root}")
    provider = provider or FakeEmbeddingProvider()
    effective_queries = list(queries) if queries else list(DEFAULT_QUERIES)

    owned_cache = cache_dir is None
    cache_base = Path(cache_dir) if cache_dir is not None else Path(
        tempfile.mkdtemp(prefix="repolens-scale-")
    )
    cache_base.mkdir(parents=True, exist_ok=True)

    groups: dict[str, tuple[StageReport, ...]] = {}
    checks: list[tuple[str, bool, str]] = []
    try:
        discovery = benchmark_discovery(root, measure_memory=measure_memory)
        files_discovered = discovery.get("files_discovered", 0)
        cold, warm = benchmark_index_builds(
            root, cache_dir=cache_base / "index", measure_memory=measure_memory
        )
        cold.metrics["fingerprint"] = _index_fingerprint(
            IncrementalIndexBuilder(root, cache_dir=cache_base / "index").build()
        )
        warm.metrics["fingerprint"] = _index_fingerprint(
            IncrementalIndexBuilder(root, cache_dir=cache_base / "index").build()
        )
        cache_disabled = benchmark_cache_disabled_index(root, repeats=2)
        groups["Discovery and indexing (A–D)"] = (
            discovery, cold, warm, *cache_disabled,
        )

        incremental = benchmark_incremental_workflow(root, measure_memory=measure_memory)
        embedding = benchmark_embedding_cache(
            root, queries=effective_queries, candidate_limit=candidate_limit,
            measure_memory=measure_memory,
        )
        if incremental:
            groups["Incremental updates (E–G)"] = incremental
        if embedding:
            groups["Embedding cache statistics"] = embedding

        index = IncrementalIndexBuilder(
            root, cache_dir=cache_base / "index"
        ).build()
        components = _build_shared_components(root, index)

        from repolens.embedding_cache import make_repo_cache

        emb_cache_dir = cache_base / "emb"
        emb_cache_dir.mkdir(parents=True, exist_ok=True)
        retrieval = benchmark_retrieval_scaled(
            root, effective_queries, index=index, provider=provider,
            candidate_limit=candidate_limit, repeats=repeats,
            embedding_cache=make_repo_cache(root, directory=emb_cache_dir),
        )
        groups["Retrieval latency (H–L)"] = retrieval

        context_report, context_engine = benchmark_context_scaled(
            root, effective_queries, index=index, provider=provider,
            embedding_cache=make_repo_cache(root, directory=emb_cache_dir),
            budget=ContextBudget(),
        )
        groups["Context generation (M)"] = (context_report,)

        arch_build, arch_query = benchmark_architecture(
            root, index=index, dependency_graph=components["dependency_graph"],
        )
        groups["Architecture (N)"] = (arch_build, arch_query)

        impact = benchmark_impact(
            root, index=index, dependency_graph=components["dependency_graph"],
            symbol_index=components["symbol_index"],
            call_graph=components["call_graph"],
            targets=impact_targets(index, limit=8),
        )
        groups["Impact analysis (O)"] = (impact,)

        requests = plan_requests(effective_queries)
        plan_build, plan_requests_report, plan_engine = benchmark_change_plan(
            root, index=index, components=components, requests=requests,
        )
        groups["Change planning (P)"] = (plan_build, plan_requests_report)

        change_context = benchmark_change_context(
            root, context_engine=context_engine, plan_engine=plan_engine,
            requests=requests, firewall=ContextFirewall(),
        )
        groups["Change-aware context (Q)"] = (change_context,)

        checks = _build_checks(
            cold=cold, warm=warm, cache_disabled=cache_disabled,
            incremental=incremental, embedding=embedding,
            retrieval=retrieval, context=context_report,
            architecture=(arch_build, arch_query), impact=impact,
            change_plan=(plan_build, plan_requests_report),
            change_context=change_context,
        )
        return ScalabilityReport(
            repository=root,
            files_discovered=files_discovered,
            python_files=files_discovered,
            groups=groups,
            checks=checks,
            scenes=[scale] if scale is not None else [],
        )
    finally:
        if owned_cache:
            shutil.rmtree(cache_base, ignore_errors=True)


def _build_checks(
    *,
    cold: StageReport,
    warm: StageReport,
    cache_disabled: tuple[StageReport, ...],
    incremental: tuple[StageReport, ...],
    embedding: tuple[StageReport, ...],
    retrieval: tuple[StageReport, ...],
    context: StageReport,
    architecture: tuple[StageReport, StageReport],
    impact: StageReport,
    change_plan: tuple[StageReport, StageReport],
    change_context: StageReport,
) -> list[tuple[str, bool, str]]:
    """Structural (machine-independent) regression checks."""
    checks: list[tuple[str, bool, str]] = []
    discovered = cold.get("files_discovered", 0)

    checks.append((
        "warm rebuild parses zero files",
        warm.get("files_parsed", -1) == 0,
        f"parsed={warm.get('files_parsed')}, hits={warm.get('cache_hits')}",
    ))
    if cache_disabled:
        parsed = [r.get("files_parsed", -1) for r in cache_disabled]
        hits = [r.get("cache_hits", -1) for r in cache_disabled]
        checks.append((
            "cache-disabled indexing parses everything with zero hits",
            all(p == discovered for p in parsed) and all(h == 0 for h in hits),
            f"parsed={parsed}, hits={hits}",
        ))

    renamed = {r.name: r for r in incremental}
    checks.append((
        "incremental modification reparses exactly one file",
        "modified" in renamed and renamed["modified"].get("files_parsed", -1) == 1,
        f"parsed={renamed.get('modified', StageReport('modified', -1)).get('files_parsed')}",
    ))
    checks.append((
        "incremental addition parses exactly one file",
        "added" in renamed and renamed["added"].get("files_parsed", -1) == 1,
        f"parsed={renamed.get('added', StageReport('added', -1)).get('files_parsed')}",
    ))
    checks.append((
        "incremental deletion removes exactly one entry",
        "deleted" in renamed and renamed["deleted"].get("files_removed", -1) == 1,
        f"removed={renamed.get('deleted', StageReport('deleted', -1)).get('files_removed')}",
    ))

    if len(embedding) >= 2:
        warm_emb = embedding[1]
        checks.append((
            "warm embedding reuses the cache (no re-embed)",
            warm_emb.get("embedded_documents", -1) == 0 and warm_emb.get("cache_hits", 0) > 0,
            f"embedded={warm_emb.get('embedded_documents')}, hits={warm_emb.get('cache_hits')}",
        ))

    lexical = next((r for r in retrieval if r.name == "lexical"), None)
    checks.append((
        "lexical retrieval returns results",
        lexical is not None and lexical.get("results", 0) > 0,
        f"results={lexical.get('results') if lexical else None}",
    ))
    for strategy in ("rrf", "weighted"):
        found = next((r for r in retrieval if r.name == strategy), None)
        checks.append((
            f"{strategy} retrieval measured",
            found is not None and found.get("median_ms") is not None,
            "hybrid strategy produced latencies",
        ))

    checks.append((
        "context stays within the token budget",
        context.get("context_size_max", context.get("budget", 0) + 1) <= context.get("budget", 0),
        f"size_max={context.get('context_size_max')}, budget={context.get('budget')}",
    ))
    checks.append((
        "architecture graph is deterministic",
        bool(architecture[0].metrics.get("fingerprint")),
        f"nodes={architecture[0].get('nodes')}, edges={architecture[0].get('edges')}",
    ))
    checks.append((
        "impact analysis resolved targets",
        impact.get("targets", 0) > 0,
        f"targets={impact.get('targets')}, affected_avg={impact.get('affected_avg')}",
    ))
    checks.append((
        "change plan produced a primary target",
        change_plan[1].get("requests", 0) > 0,
        f"requests={change_plan[1].get('requests')}",
    ))
    checks.append((
        "change-aware context respected the budget",
        change_context.get("tokens_max", change_context.get("budget", 0) + 1) <= change_context.get("budget", 0),
        f"tokens_max={change_context.get('tokens_max')}, budget={change_context.get('budget')}",
    ))
    return checks


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="performance_scalability.py",
        description=(
            "Measure RepoLens performance/scalability on synthetic corpora "
            "or a real repository (default: the current directory)."
        ),
    )
    parser.add_argument(
        "root", type=str, nargs="?", default=None,
        help="Local repository to measure. When omitted, deterministic "
        "synthetic corpora are generated instead.",
    )
    parser.add_argument(
        "--scale", type=str, default="all",
        help="Synthetic scale(s): all, small, medium, large (default: all).",
    )
    parser.add_argument(
        "--seed", type=int, default=7,
        help="Random seed for synthetic corpora (default: 7).",
    )
    parser.add_argument(
        "--queries", type=str, default=None,
        help="Newline-delimited text file of representative queries.",
    )
    parser.add_argument(
        "--cache-dir", type=str, default=None,
        help="Reuse this cache directory (default: a fresh temporary dir).",
    )
    parser.add_argument(
        "--candidate-limit", type=int, default=DEFAULT_CANDIDATE_LIMIT,
        help="Candidate limit for candidate-based semantic search (default: 40).",
    )
    parser.add_argument(
        "--repeats", type=int, default=DEFAULT_REPEATS,
        help="Repeated runs per query for latency statistics (default: 3).",
    )
    parser.add_argument(
        "--measure-memory", action="store_true",
        help="Instrument peak memory with tracemalloc (slower, perturbing).",
    )
    parser.add_argument(
        "--corpora-dir", type=str, default=None,
        help="Directory for synthetic corpora (default: a fresh temp dir).",
    )
    parser.add_argument(
        "--keep", action="store_true",
        help="Keep generated synthetic corpora on disk after the run.",
    )
    return parser.parse_args(argv)


def _load_queries(path: str | None) -> list[str] | None:
    if path is None:
        return None
    queries = [
        line.strip()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not queries:
        raise SystemExit(f"No queries found in {path}")
    return queries


def _fmt(value) -> str:
    return "-" if value is None else str(value)


def _run_real_repo(args: argparse.Namespace) -> int:
    root = Path(args.root).expanduser().resolve()
    if not root.is_dir():
        print(f"error: not a directory: {root}", file=sys.stderr)
        return 2
    queries = _load_queries(args.queries)
    print(f"Running performance/scalability benchmark against {root}")
    print(f"  default budget: {ContextBudget().max_tokens} estimated tokens")
    print(f"  repeats: {args.repeats}  candidate-limit: {args.candidate_limit}")
    print()
    report = run_scalability_benchmark(
        root,
        queries=queries,
        candidate_limit=args.candidate_limit,
        repeats=args.repeats,
        cache_dir=args.cache_dir,
        measure_memory=args.measure_memory,
    )
    print(report.to_text())
    print()
    print("NOTE: these numbers are measurements, not pass/fail assertions.")
    return 0


def _run_synthetic(args: argparse.Namespace) -> int:
    wanted = args.scale.split(",")
    if "all" in wanted:
        wanted = list(SCALE_PRESETS)
    for scale in wanted:
        if scale not in SCALE_PRESETS:
            choices = ", ".join(sorted(SCALE_PRESETS))
            raise SystemExit(
                f"unknown --scale {scale!r}; expected one of: {choices}, all"
            )
    owned_corpora = args.corpora_dir is None
    corpora_base = Path(
        args.corpora_dir if args.corpora_dir else tempfile.mkdtemp(prefix="repolens-corpora-")
    )
    corpora_base.mkdir(parents=True, exist_ok=True)
    summary: list[dict[str, Any]] = []
    try:
        for scale in wanted:
            corpus_dir = corpora_base / f"{scale}_{args.seed}"
            spec = generate_synthetic_repository(corpus_dir, scale, args.seed)
            print(f"Scale: {scale} ({spec.counts['python_files']} python files)")
            report = run_scalability_benchmark(
                spec.root,
                scale=scale,
                queries=list(spec.queries),
                seed=args.seed,
                candidate_limit=args.candidate_limit,
                repeats=args.repeats,
                cache_dir=args.cache_dir,
                measure_memory=args.measure_memory,
            )
            print()
            print(report.to_text())
            print()
            indexing = report.groups.get("Discovery and indexing (A–D)", ())
            cold = next((r for r in indexing if r.name == "cold_index"), None)
            retrieval = report.groups.get("Retrieval latency (H–L)", ())
            context = next(iter(report.groups.get("Context generation (M)", ())), None)
            plan_requests_report = next(
                (
                    r for r in report.groups.get("Change planning (P)", ())
                    if r.name == "change_plan_request"
                ),
                None,
            )
            summary.append(
                {
                    "scale": scale,
                    "files": spec.counts["python_files"],
                    "cold_index_ms": cold.elapsed_ms if cold else None,
                    "retrieval_p95_ms": next(
                        (r.get("p95_ms") for r in retrieval if r.name == "lexical"), None
                    ),
                    "context_p95_ms": context.get("p95_ms") if context else None,
                    "change_plan_p95_ms": (
                        plan_requests_report.get("p95_ms") if plan_requests_report else None
                    ),
                }
            )
    finally:
        if not owned_corpora and not args.keep:
            shutil.rmtree(corpora_base, ignore_errors=True)
        if owned_corpora and args.keep:
            print(f"Corpora kept under {corpora_base}")
    if len(summary) > 1:
        print("Scalability summary (timings are machine-specific):")
        header = (
            f"{'scale':<8} {'files':>7} {'cold_index_ms':>14} "
            f"{'retrieval_p95':>14} {'context_p95':>13} {'plan_p95':>10}"
        )
        print(header)
        for row in summary:
            print(
                f"{row['scale']:<8} {row['files']:>7} "
                f"{_fmt(row['cold_index_ms']):>14} "
                f"{_fmt(row['retrieval_p95_ms']):>14} "
                f"{_fmt(row['context_p95_ms']):>13} "
                f"{_fmt(row['change_plan_p95_ms']):>10}"
            )
    print()
    print("NOTE: these numbers are measurements, not pass/fail assertions.")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.root is not None:
        return _run_real_repo(args)
    return _run_synthetic(args)


if __name__ == "__main__":
    sys.exit(main())