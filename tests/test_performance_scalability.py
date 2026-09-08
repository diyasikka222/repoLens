"""Tests for the performance & scalability benchmark (Phase 25.4).

Like the production-benchmark tests, these verify *structural* invariants of
the benchmark itself (deterministic corpora, exact file counts, cache-regime
behavior, per-op metric shape, budget compliance).  They never assert
machine-dependent wall-clock thresholds.  All tests run offline on the
deterministic synthetic corpora.

The benchmark lives under ``benchmarks/`` (not part of the installed
package), so this module inserts the repository root into ``sys.path`` before
importing it — the same trick the benchmark scripts themselves use.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.performance_scalability import (  # noqa: E402
    SCALE_PRESETS,
    generate_synthetic_repository,
    run_scalability_benchmark,
    scale_preset,
)


@pytest.mark.parametrize(
    "scale, expected",
    [
        ("small", 9),
        ("medium", 25),
        ("large", 51),
    ],
)
def test_preset_file_counts(scale: str, expected: int) -> None:
    assert scale_preset(scale).total_python_files == expected
    assert SCALE_PRESETS[scale].total_python_files == expected


def _generate(root: Path, scale: str, seed: int = 7):
    return generate_synthetic_repository(root, scale, seed=seed)


@pytest.fixture
def small_spec(tmp_path: Path):
    return _generate(tmp_path / "small_7", "small", seed=7)


@pytest.fixture
def small_repo(small_spec) -> Path:
    return small_spec.root


@pytest.fixture
def run_small(small_spec):
    """Run the full benchmark against the small corpus with matching queries."""

    def _run(**kwargs):
        kwargs.setdefault("scale", "small")
        kwargs.setdefault("seed", 7)
        kwargs.setdefault("repeats", 1)
        kwargs.setdefault("queries", list(small_spec.queries))
        return run_scalability_benchmark(small_spec.root, **kwargs)

    return _run


def test_generator_is_deterministic(tmp_path: Path) -> None:
    first = _generate(tmp_path / "a", "small", seed=7)
    second = _generate(tmp_path / "b", "small", seed=7)
    assert first.files == second.files
    assert first.queries == second.queries
    assert first.counts == second.counts
    for relative in first.files:
        assert (first.root / relative).read_bytes() == (second.root / relative).read_bytes()


def test_generator_seed_changes_bytes(tmp_path: Path) -> None:
    default = _generate(tmp_path / "d", "small", seed=7)
    altered = _generate(tmp_path / "a", "small", seed=99)
    assert default.files == altered.files  # same structure…
    assert any(
        (default.root / rel).read_bytes() != (altered.root / rel).read_bytes()
        for rel in default.files  # …but different content
    )


def test_generated_corpus_structure_small(small_spec) -> None:
    root = small_spec.root
    assert small_spec.counts["python_files"] == 9
    assert small_spec.counts["packages"] == 2
    assert small_spec.counts["init_files"] == 2
    assert small_spec.counts["module_files"] == 4
    assert small_spec.counts["shared_files"] == 1
    assert small_spec.counts["test_files"] == 2
    assert (root / "shared.py").is_file()
    for package in ("pkg_0", "pkg_1"):
        assert (root / package / "__init__.py").is_file()
        for module in ("module_0", "module_1"):
            assert (root / package / f"{module}.py").is_file()
        assert (root / "tests" / f"test_{package}.py").is_file()
    for relative in small_spec.files:
        assert (root / relative).is_file()


def test_generated_modules_cross_import(small_repo: Path) -> None:
    module = (small_repo / "pkg_0" / "module_0.py").read_text(encoding="utf-8")
    assert "import shared as peer" in module
    assert "peer.transform_1(" in module
    package = (small_repo / "pkg_0" / "__init__.py").read_text(encoding="utf-8")
    assert "from pkg_0.module_0 import run_0" in package
    tests = (small_repo / "tests" / "test_pkg_0.py").read_text(encoding="utf-8")
    assert "pkg_0" in tests


def _stage(report, group: str, name: str):
    for stage in report.groups[group]:
        if stage.name == name:
            return stage
    raise AssertionError(f"stage {name!r} missing from group {group!r}")


def test_full_pipeline_all_checks_pass(run_small) -> None:
    report = run_small()
    assert report.files_discovered == 9
    assert report.python_files == 9
    assert set(report.groups) == {
        "Discovery and indexing (A–D)",
        "Incremental updates (E–G)",
        "Embedding cache statistics",
        "Retrieval latency (H–L)",
        "Context generation (M)",
        "Architecture (N)",
        "Impact analysis (O)",
        "Change planning (P)",
        "Change-aware context (Q)",
    }
    assert report.checks, "benchmark must produce sanity checks"
    failing = [name for name, passed, _ in report.checks if not passed]
    assert not failing, f"structural checks failed: {failing}"
    assert "Sanity checks" in report.to_text()


def test_cache_regimes(run_small) -> None:
    report = run_small()
    cold = _stage(report, "Discovery and indexing (A–D)", "cold_index")
    warm = _stage(report, "Discovery and indexing (A–D)", "warm_index")
    assert cold.get("files_parsed") == 9 and cold.get("cache_hits") == 0
    assert warm.get("files_parsed") == 0 and warm.get("cache_hits") == 9
    assert cold.get("fingerprint") == warm.get("fingerprint")
    cache_disabled = [s for s in report.groups["Discovery and indexing (A–D)"] if s.name.startswith("cache_disabled")]
    assert len(cache_disabled) == 2
    for stage in cache_disabled:
        assert stage.get("files_parsed") == 9
        assert stage.get("cache_hits") == 0


def test_incremental_band(run_small) -> None:
    report = run_small()
    modified = _stage(report, "Incremental updates (E–G)", "modified")
    added = _stage(report, "Incremental updates (E–G)", "added")
    deleted = _stage(report, "Incremental updates (E–G)", "deleted")
    assert modified.get("files_parsed") == 1
    assert added.get("files_parsed") == 1
    assert added.get("files_discovered") == 10
    assert deleted.get("files_removed") == 1


def test_embedding_cache_band(run_small) -> None:
    report = run_small()
    cold = _stage(report, "Embedding cache statistics", "cold_embedding")
    warm = _stage(report, "Embedding cache statistics", "warm_embedding")
    assert cold.get("embedded_documents") == 9 and cold.get("cache_hits") == 0
    assert warm.get("embedded_documents") == 0 and warm.get("cache_hits") == 9


def test_retrieval_metrics(run_small) -> None:
    report = run_small()
    retrieval = report.groups["Retrieval latency (H–L)"]
    assert {s.name for s in retrieval} == {
        "lexical", "candidate-semantic", "semantic", "rrf", "weighted",
    }
    for stage in retrieval:
        assert stage.get("runs", 0) > 0
        assert stage.get("results", 0) > 0
        assert stage.get("fingerprint")
        for key in ("median_ms", "p95_ms", "min_ms", "max_ms"):
            assert stage.get(key) is not None


def test_context_stays_within_budget(run_small) -> None:
    report = run_small()
    context = _stage(report, "Context generation (M)", "context")
    assert context.get("selected_avg", 0) > 0
    assert context.get("context_size_max", 0) <= context.get("budget")


def test_architecture_impact_planning_and_change_context(run_small) -> None:
    report = run_small()
    arch = _stage(report, "Architecture (N)", "architecture_build")
    assert arch.get("nodes", 0) > 0
    assert arch.get("fingerprint")
    impact = _stage(report, "Impact analysis (O)", "impact")
    assert impact.get("targets", 0) > 0
    assert impact.get("affected_avg", 0) > 0
    plan_requests = _stage(report, "Change planning (P)", "change_plan_request")
    assert plan_requests.get("requests", 0) > 0
    change_context = _stage(report, "Change-aware context (Q)", "change_context")
    assert change_context.get("tokens_max", 0) <= change_context.get("budget")


def test_run_accepts_explicit_queries_and_cache(run_small, tmp_path: Path) -> None:
    shared = tmp_path / "cache"
    queries = ["bootstrap dispatch", "pkg_0.module_0 pipeline"]
    report = run_small(queries=queries, cache_dir=shared)
    assert report.checks
    assert shared.is_dir()
    # A second run against the same cache directory also passes.
    second = run_small(queries=queries, cache_dir=shared)
    assert all(passed for _, passed, _ in second.checks)


def test_real_repo_cli_mode(small_repo: Path) -> None:
    from benchmarks.performance_scalability import main

    assert main(["--scale", "small", "--seed", "7", "--repeats", "1", str(small_repo)]) == 0


def test_synthetic_cli_mode(tmp_path: Path) -> None:
    from benchmarks.performance_scalability import main

    corpora = tmp_path / "corpora"
    assert main(["--scale", "small", "--repeats", "1", "--corpora-dir", str(corpora), "--keep"]) == 0
    assert (corpora / "small_7").is_dir()


def test_large_preset_generates(tmp_path: Path) -> None:
    spec = _generate(tmp_path / "large", "large", seed=7)
    expected = SCALE_PRESETS["large"].total_python_files
    assert expected == 51
    assert spec.counts["python_files"] == expected
    assert len(spec.files) == expected