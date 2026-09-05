"""Tests for structured observability diagnostics (Milestone 20).

Diagnostics are opt-in, never alter returned results, and never carry source
contents or secrets. These tests cover enable/disable semantics, the JSON
record shape, the ``timed`` helper, and the integration points (incremental
index build and context generation).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from repolens import diagnostics
from repolens.context import ContextBudget, ContextEngine
from repolens.incremental_index import IncrementalIndexBuilder
from repolens.search import CodeSearcher

VALID = "def alpha(x: int) -> int:\n    return x + 1\n"


def _write(repo: Path, name: str, source: str) -> None:
    path = repo / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")
    return path


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    _write(root, "a.py", VALID)
    return root


@pytest.fixture(autouse=True)
def _reset_diagnostics():
    diagnostics.reset()
    yield
    diagnostics.reset()


def _captured(caplog: pytest.LogCaptureFixture) -> list[dict]:
    return [
        json.loads(record.message)
        for record in caplog.records
        if record.name == "repolens.diagnostics"
    ]


# ---------------------------------------------------------------------------
# Enable / disable semantics
# ---------------------------------------------------------------------------


def test_disabled_emits_nothing(caplog: pytest.LogCaptureFixture) -> None:
    diagnostics.disable()
    with caplog.at_level(logging.DEBUG, logger="repolens.diagnostics"):
        diagnostics.record("index_build", repository="/x", files_parsed=3)
    assert _captured(caplog) == []


def test_enable_emits_json_record(caplog: pytest.LogCaptureFixture) -> None:
    diagnostics.enable()
    with caplog.at_level(logging.DEBUG, logger="repolens.diagnostics"):
        diagnostics.record("index_build", repository="/repo", files_parsed=3)
    records = _captured(caplog)
    assert len(records) == 1
    record = records[0]
    assert record["operation"] == "index_build"
    assert record["repository"] == "/repo"
    assert record["files_parsed"] == 3


def test_disable_after_enable_suppresses(caplog: pytest.LogCaptureFixture) -> None:
    diagnostics.enable()
    diagnostics.disable()
    with caplog.at_level(logging.DEBUG, logger="repolens.diagnostics"):
        diagnostics.record("context_build", selected=1)
    assert _captured(caplog) == []


def test_env_var_enables_and_non_string_values_fall_back_to_str(
    caplog: pytest.LogCaptureFixture, monkeypatch
) -> None:
    monkeypatch.setenv("REPOLENS_DIAGNOSTICS", "1")
    diagnostics.reset()
    with caplog.at_level(logging.DEBUG, logger="repolens.diagnostics"):
        diagnostics.record("index_build", repository=Path("/some/path"), files_parsed=1)
    records = _captured(caplog)
    assert records[0]["repository"] == "/some/path"


def test_false_env_val_is_disabled(
    caplog: pytest.LogCaptureFixture, monkeypatch
) -> None:
    monkeypatch.setenv("REPOLENS_DIAGNOSTICS", "0")
    diagnostics.reset()
    with caplog.at_level(logging.DEBUG, logger="repolens.diagnostics"):
        diagnostics.record("index_build")
    assert _captured(caplog) == []


# ---------------------------------------------------------------------------
# timed helper
# ---------------------------------------------------------------------------


def test_timed_records_elapsed_ms(caplog: pytest.LogCaptureFixture) -> None:
    diagnostics.enable()
    with caplog.at_level(logging.DEBUG, logger="repolens.diagnostics"):
        with diagnostics.timed("stage", repository="/r", files=1):
            pass
    records = _captured(caplog)
    assert len(records) == 1
    assert records[0]["operation"] == "stage"
    assert "elapsed_ms" in records[0]
    assert records[0]["files"] == 1


# ---------------------------------------------------------------------------
# Integration points
# ---------------------------------------------------------------------------


def test_index_build_emits_diagnostics(
    repo: Path, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    diagnostics.enable()
    with caplog.at_level(logging.DEBUG, logger="repolens.diagnostics"):
        IncrementalIndexBuilder(repo, cache_dir=tmp_path / "cache").build()
    records = _captured(caplog)
    index_records = [r for r in records if r["operation"] == "index_build"]
    assert len(index_records) == 1
    index_record = index_records[0]
    assert index_record["files_discovered"] == 1
    assert index_record["files_parsed"] == 1
    assert index_record["cache_hits"] == 0
    assert "elapsed_ms" in index_record
    # Never any source contents or secret-like fields.
    assert "source" not in index_record
    assert "content" not in index_record


def test_context_build_emits_diagnostics(
    repo: Path, caplog: pytest.LogCaptureFixture
) -> None:
    diagnostics.enable()
    engine = ContextEngine(repo, searcher=CodeSearcher(repo), budget=ContextBudget())
    with caplog.at_level(logging.DEBUG, logger="repolens.diagnostics"):
        engine.build_context("alpha function")
    records = _captured(caplog)
    context_records = [r for r in records if r["operation"] == "context_build"]
    assert len(context_records) == 1
    record = context_records[0]
    assert record["selected"] == len(engine.build_context("alpha function").selected_files)
    assert record["budget"] == 8000
    assert "context_size" in record
    assert "intent" in record
    # Diagnostics never carry package/source contents.
    assert "source" not in record


def test_diagnostics_do_not_change_returned_results(repo: Path) -> None:
    engine = ContextEngine(repo, searcher=CodeSearcher(repo), budget=ContextBudget())
    plain = engine.build_context("alpha function")
    diagnostics.enable()
    enabled = engine.build_context("alpha function")
    diagnostics.disable()
    assert plain.to_dict() == enabled.to_dict()