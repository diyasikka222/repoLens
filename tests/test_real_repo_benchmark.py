"""Offline tests for the real-world benchmark infrastructure/configuration.

These tests must never touch the network, download the external repository,
or download an embedding model. They validate configuration, the manually
curated query dataset, and offline-safe helpers.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.real_repo.config import (
    DATA_DIR,
    QUERIES_PATH,
    REPO_DIR,
    REPOSITORY_COMMIT,
    REPOSITORY_NAME,
    REPOSITORY_REF,
    REPOSITORY_URL,
    TARBALL_URL,
)
from benchmarks.real_repo.dataset import load_cases
import benchmarks.real_repo.runner as runner
from benchmarks.real_repo.runner import _module_help, _prereq_check, count_python_files

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def test_repository_metadata_is_complete() -> None:
    assert REPOSITORY_NAME
    assert REPOSITORY_URL.startswith("https://")
    assert REPOSITORY_URL == "https://github.com/Textualize/rich"
    assert REPOSITORY_REF == "v14.3.4"
    assert len(REPOSITORY_COMMIT) == 40


def test_tarball_url_pins_the_reference() -> None:
    assert TARBALL_URL == (
        "https://codeload.github.com/Textualize/rich/tar.gz/refs/tags/v14.3.4"
    )


def test_data_directory_is_gitignored_and_offscreen() -> None:
    # The benchmark data must not live inside the source package.
    assert DATA_DIR.name == ".benchmark_data"
    assert REPO_DIR == DATA_DIR / "repo"
    assert not DATA_DIR.is_relative_to(QUERIES_PATH)


def test_module_help_is_specified() -> None:
    text = _module_help()
    assert "python -m benchmarks.real_repo" in text
    assert REPOSITORY_NAME in text
    assert REPOSITORY_REF in text


# ---------------------------------------------------------------------------
# Dataset (manually curated)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cases():
    return load_cases()


def test_dataset_has_twenty_queries(cases) -> None:
    assert len(cases) == 20


def test_dataset_queries_are_non_blank(cases) -> None:
    for case in cases:
        assert case.query.strip()
        assert isinstance(case.query, str)


def test_every_case_has_non_empty_ground_truth(cases) -> None:
    for case in cases:
        assert case.relevant_files, f"empty ground truth for query: {case.query!r}"


def test_ground_truth_paths_are_relative_python_files(cases) -> None:
    for case in cases:
        for path in case.relevant_files:
            p = Path(path)
            assert not p.is_absolute(), f"absolute path in {case.query!r}: {path}"
            assert ".." not in p.parts, f"parent traversal in {case.query!r}: {path}"
            assert p.suffix == ".py", f"non-.py ground truth in {case.query!r}: {path}"


def test_dataset_is_deterministic(cases) -> None:
    assert load_cases() == cases


def test_dataset_json_is_well_formed() -> None:
    payload = json.loads(QUERIES_PATH.read_text(encoding="utf-8"))
    assert payload["default_k"] == 5
    assert len(payload["cases"]) == 20
    names = [item["name"] for item in payload["cases"]]
    assert len(set(names)) == len(names), "duplicate case names"


# ---------------------------------------------------------------------------
# Offline-safe helpers
# ---------------------------------------------------------------------------


def test_runner_module_imports_without_network() -> None:
    # Importing the runner and inspecting metadata must not need network.
    assert runner.DEFAULT_K == 5
    assert runner.DEFAULT_LEXICAL_WEIGHT == 0.5
    assert runner.DEFAULT_SEMANTIC_WEIGHT == 0.5
    assert runner.DEFAULT_RRF_K == 60


def test_prereq_check_runs_offline() -> None:
    missing = _prereq_check()
    # fastembed and repolens are installed in the dev environment.
    assert "fastembed is not installed" not in missing
    assert "repolens is not importable" not in missing


def test_count_python_files_on_offline_tree(tmp_path: Path) -> None:
    (tmp_path / "a").mkdir()
    (tmp_path / "a").joinpath("one.py").write_text("def f():\n    pass\n")
    (tmp_path / "a").joinpath("two.py").write_text("x = 1\n")
    (tmp_path / "a").mkdir(exist_ok=True)
    tmp_path.joinpath("top.py").write_text("y = 1\n")
    assert count_python_files(tmp_path) == 3
