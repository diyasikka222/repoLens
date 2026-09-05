"""Tests for the production benchmark harness (Milestone 20).

The harness measures *structural* invariants (counts) plus relative timings;
it must not assert machine-dependent absolute-time thresholds. These tests
verify the shape of every stage, the deterministic counts, and the memory
instrumentation path, all on a small synthetic repository.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from repolens.incremental_index import IncrementalIndexBuilder
from repolens.production_benchmark import (
    DEFAULT_QUERIES,
    CountingEmbeddingProvider,
    ProductionReport,
    StageReport,
    benchmark_context_engine,
    benchmark_discovery,
    benchmark_embedding_cache,
    benchmark_incremental_workflow,
    benchmark_index_builds,
    benchmark_retrieval,
    run_production_benchmark,
)

FILES = {
    "auth/login.py": "def do_login(user):\n    return user\n",
    "auth/session.py": "class Session:\n    def start(self):\n        pass\n",
    "auth/__init__.py": "",
    "database/pool.py": "def acquire(pool_id):\n    return pool_id\n",
    "database/connection.py": "def connect(dsn):\n    return dsn\n",
    "billing/invoice.py": (
        "def create_invoice(amount):\n"
        "    tax = compute_tax(amount)\n"
        "    return invoice(tax)\n"
    ),
    "billing/tax.py": "def compute_tax(amount):\n    return amount * 0.2\n",
}


def _write_repo(root: Path) -> None:
    for relative, source in FILES.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8")


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    _write_repo(root)
    return root


# ---------------------------------------------------------------------------
# Individual stages
# ---------------------------------------------------------------------------


def test_discovery_counts_python_files(repo: Path) -> None:
    report = benchmark_discovery(repo)
    assert report.name == "discovery"
    assert report.get("files_discovered") == len(FILES)
    assert report.get("python_files") == len(FILES)
    assert report.elapsed_ms >= 0


def test_cold_warm_index_counts(repo: Path) -> None:
    cold, warm = benchmark_index_builds(repo)
    assert cold.get("files_discovered") == len(FILES)
    assert cold.get("files_parsed") == len(FILES)
    assert cold.get("cache_hits") == 0
    assert warm.get("files_parsed") == 0
    assert warm.get("cache_hits") == len(FILES)


def test_cold_warm_index_use_same_cache_dir(repo: Path) -> None:
    cold, warm = benchmark_index_builds(repo, cache_dir=str(repo.parent / "idx"))
    assert cold.get("files_parsed") == len(FILES)
    assert warm.get("files_parsed") == 0


def test_incremental_workflow_transitions(repo: Path) -> None:
    reports = benchmark_incremental_workflow(repo)
    names = [r.name for r in reports]
    assert names == ["cold", "warm", "modified", "added", "deleted"]
    cold, warm, modified, added, deleted = reports
    assert cold.get("files_parsed") == len(FILES)
    assert warm.get("files_parsed") == 0
    assert warm.get("cache_hits") == len(FILES)
    assert modified.get("files_parsed") == 1
    assert added.get("files_parsed") == 1
    assert deleted.get("files_removed") == 1


def test_incremental_workflow_empty_repo(tmp_path: Path) -> None:
    empty = tmp_path / "empty_repo"
    empty.mkdir()
    assert benchmark_incremental_workflow(empty) == ()


def test_embedding_cache_cold_warm_changed(repo: Path) -> None:
    reports = benchmark_embedding_cache(repo, candidate_limit=40)
    assert [r.name for r in reports] == [
        "cold_embedding", "warm_embedding", "changed_embedding",
    ]
    cold, warm, changed = reports
    assert cold.get("embedded_documents", 0) > 0
    assert cold.get("cache_misses", 0) > 0
    assert warm.get("embedded_documents", 0) == 0
    assert warm.get("cache_hits", 0) > 0
    assert changed.get("embedded_documents", 0) >= 1


def test_embedding_cache_empty_repo(tmp_path: Path) -> None:
    empty = tmp_path / "empty_repo"
    empty.mkdir()
    assert benchmark_embedding_cache(empty) == ()


def test_retrieval_reports_all_strategies_with_medians(repo: Path) -> None:
    reports = benchmark_retrieval(repo, DEFAULT_QUERIES, repeats=2)
    names = {r.name for r in reports}
    assert names == {"lexical", "semantic", "candidate-semantic", "rrf", "weighted"}
    for report in reports:
        assert report.get("median_ms") is not None
        assert report.get("runs") == 2 * len(DEFAULT_QUERIES)
        assert report.get("median_ms") <= report.get("max_ms")


def test_context_engine_budget_respected(repo: Path) -> None:
    report = benchmark_context_engine(repo, ["auth login session"], measure_memory=False)
    assert report.name == "context"
    assert report.get("queries") == 1
    assert report.get("budget") == 8000
    assert report.get("context_size_median", 0) <= 8000
    assert report.get("selected", 0) >= 0


def test_memory_instrumentation_reports_peak(repo: Path) -> None:
    report = benchmark_discovery(repo, measure_memory=True)
    assert "peak_mem_bytes" in report.metrics
    assert report.get("peak_mem_bytes") >= 0


# ---------------------------------------------------------------------------
# Orchestrated report
# ---------------------------------------------------------------------------


def test_run_production_benchmark_report_shape(repo: Path) -> None:
    report = run_production_benchmark(repo, queries=["auth login"], repeats=1)
    assert isinstance(report, ProductionReport)
    assert report.repository == repo
    assert report.files_discovered == len(FILES)
    assert report.discovery.name == "discovery"
    assert report.cold_index.get("files_parsed") == len(FILES)
    assert report.warm_index.get("files_parsed") == 0
    assert len(report.incremental) == 5
    assert len(report.embedding_cache) == 3
    assert {r.name for r in report.retrieval} == {
        "lexical", "semantic", "candidate-semantic", "rrf", "weighted",
    }
    assert report.context.name == "context"


def test_run_production_benchmark_non_directory(tmp_path: Path) -> None:
    with pytest.raises(NotADirectoryError):
        run_production_benchmark(tmp_path / "nope")


def test_report_to_text_covers_all_sections(repo: Path) -> None:
    report = run_production_benchmark(repo, queries=["auth"], repeats=1)
    text = report.to_text()
    for expected in (
        "Repository:",
        "Files discovered:",
        "Python files:",
        "Cold index:",
        "Warm index:",
        "Incremental update",
        "Embedding / cache statistics",
        "Retrieval latency",
        "Context generation:",
        "cold_embedding",
    ):
        assert expected in text


# ---------------------------------------------------------------------------
# Counting provider keeps embedding cost observable
# ---------------------------------------------------------------------------


def test_counting_provider_counts_documents_and_queries() -> None:
    provider = CountingEmbeddingProvider()
    provider.embed_texts(["a", "b"])
    provider.embed_text("query")
    assert provider.embedded_documents == 2
    assert provider.embedded_queries == 1


def test_stage_report_omits_none_metrics_in_text() -> None:
    report = StageReport(name="x", elapsed_ms=1.0, metrics={"a": None, "b": 2})
    assert "a=" not in report.metric_text()
    assert "b=2" in report.metric_text()