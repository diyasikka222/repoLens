"""Tests for incremental repository indexing (Milestone 18).

The incremental index persists per-file :class:`ModuleAnalysis` keyed by a
sha256 content hash, so a rebuild of an unchanged repository re-parses nothing.
These tests are fully offline and deterministic (no OpenAI/network).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from repolens.embeddings import FakeEmbeddingProvider
from repolens.graph import DependencyGraphBuilder
from repolens.incremental_index import (
    CACHE_SCHEMA_VERSION,
    IncrementalIndexBuilder,
    repository_identity,
)
from repolens.index import SymbolIndexBuilder
from repolens.parser import PythonParser
from repolens.search import CodeSearcher
from repolens.semantic_search import SemanticSearcher

A = "import os\n\ndef alpha(x: int) -> int:\n    return x + 1\n\nclass Beta:\n    def meth(self):\n        pass\n"
B = "from .a import alpha, Beta\n\ndef gamma():\n    return alpha(2)\n"


def _write(repo: Path, name: str, content: str) -> Path:
    p = repo / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return p


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    _write(root, "a.py", A)
    _write(root, "b.py", B)
    return root


@pytest.fixture
def cache(tmp_path: Path) -> Path:
    return tmp_path / "index_cache"


def _builder(repo: Path, cache: Path) -> IncrementalIndexBuilder:
    return IncrementalIndexBuilder(repo, cache_dir=cache)


def _symbol_names(repo: Path, cache: Path) -> list[str]:
    index = _builder(repo, cache).build()
    return sorted(s.name for s in index.symbols)


def test_clean_build_parses_every_file(repo: Path, cache: Path) -> None:
    index = _builder(repo, cache).build()
    stats = index.stats.as_dict()
    assert stats["files_discovered"] == 2
    assert stats["files_parsed"] == 2
    assert stats["cache_misses"] == 2
    assert stats["cache_hits"] == 0
    assert stats["files_removed"] == 0


def test_unchanged_rebuild_hits_cache(repo: Path, cache: Path) -> None:
    _builder(repo, cache).build()
    index = _builder(repo, cache).build()
    stats = index.stats.as_dict()
    assert stats["files_parsed"] == 0
    assert stats["cache_hits"] == 2
    assert stats["cache_misses"] == 0
    assert _symbol_names(repo, cache) == ["Beta", "alpha", "gamma", "meth"]


def test_single_file_modification_reparses_only_that_file(
    repo: Path, cache: Path
) -> None:
    _builder(repo, cache).build()
    _write(repo, "a.py", A.replace("x + 1", "x + 2"))
    index = _builder(repo, cache).build()
    stats = index.stats.as_dict()
    assert stats["files_parsed"] == 1
    assert stats["cache_hits"] == 1
    assert stats["cache_misses"] == 1


def test_new_file_is_parsed(repo: Path, cache: Path) -> None:
    _builder(repo, cache).build()
    _write(repo, "c.py", "def new_func():\n    return 1\n")
    index = _builder(repo, cache).build()
    stats = index.stats.as_dict()
    assert stats["files_discovered"] == 3
    assert stats["files_parsed"] == 1
    assert stats["cache_hits"] == 2
    assert "new_func" in {s.name for s in index.symbols}


def test_deleted_file_is_pruned(repo: Path, cache: Path) -> None:
    _builder(repo, cache).build()
    (repo / "b.py").unlink()
    index = _builder(repo, cache).build()
    stats = index.stats.as_dict()
    assert stats["files_discovered"] == 1
    assert stats["files_parsed"] == 0
    assert stats["files_removed"] == 1
    assert index.files == (Path("a.py"),)


def test_multiple_repos_are_isolated(tmp_path: Path, cache: Path) -> None:
    repo = tmp_path / "repo"
    other = tmp_path / "other"
    repo.mkdir()
    other.mkdir()
    _write(repo, "a.py", A)
    _write(other, "a.py", "import json\n\ndef delta():\n    pass\n")
    r1 = _builder(repo, cache).build()
    r2 = _builder(other, cache).build()
    # Same relative path but different repos resolve to distinct cache entries.
    assert r1.analysis_for(Path("a.py")) != r2.analysis_for(Path("a.py"))
    assert r1.stats.files_parsed == 1
    assert r2.stats.files_parsed == 1


def test_repository_identity_is_stable_for_same_root(repo: Path) -> None:
    assert repository_identity(repo) == repository_identity(repo.resolve())


def test_corrupt_cache_entry_is_treated_as_miss(
    repo: Path, cache: Path
) -> None:
    index = _builder(repo, cache).build()
    # Corrupt every cached JSON file.
    for entry in cache.glob("*.json"):
        entry.write_text("{not json", encoding="utf-8")
    index2 = _builder(repo, cache).build()
    assert index2.stats.files_parsed == index.stats.files_parsed
    assert index2.stats.cache_hits == 0


def test_schema_version_change_invalidates_cache(repo: Path, cache: Path) -> None:
    _builder(repo, cache).build()
    # Rewrite entries to claim a newer schema (simulating a format bump).
    for entry in cache.glob("*.json"):
        import json as _json

        payload = _json.loads(entry.read_text())
        payload["schema"] = CACHE_SCHEMA_VERSION + 1
        entry.write_text(_json.dumps(payload))
    index = _builder(repo, cache).build()
    assert index.stats.files_parsed == 2
    assert index.stats.cache_hits == 0


def test_parser_failure_raises_and_leaves_no_stale_entry(
    repo: Path, cache: Path
) -> None:
    _builder(repo, cache).build()
    _write(repo, "a.py", "def broken(:\n")
    with pytest.raises(SyntaxError):
        _builder(repo, cache).build()
    # The stale valid entry for the old a.py (different content hash) must not
    # be reused; a subsequent valid version must reparse. Nothing is persisted
    # for the broken version.
    for entry in cache.glob("*.json"):
        import json as _json

        payload = _json.loads(entry.read_text())
        if payload.get("path") == "a.py" and payload["schema"] == CACHE_SCHEMA_VERSION:
            assert "broken" not in _json.dumps(payload["analysis"])


def test_clear_restores_clean_build(repo: Path, cache: Path) -> None:
    _builder(repo, cache).build()
    for entry in cache.glob("*.json"):
        entry.unlink()
    index = _builder(repo, cache).build()
    assert index.stats.files_parsed == 2
    assert index.stats.cache_hits == 0


def test_index_backed_symbol_index_matches_standalone(
    repo: Path, cache: Path
) -> None:
    snapshot = _builder(repo, cache).build()
    standalone = SymbolIndexBuilder(repo).build()
    backed = SymbolIndexBuilder(repo, index=snapshot).build()
    assert sorted(s.name for s in standalone.get_all_symbols()) == sorted(
        s.name for s in backed.get_all_symbols()
    )


def test_index_backed_code_searcher_matches_standalone(
    repo: Path, cache: Path
) -> None:
    snapshot = _builder(repo, cache).build()
    standalone = CodeSearcher(repo)
    backed = CodeSearcher(repo, index=snapshot)
    for q in ("alpha", "Beta", "gamma", "meth", "os"):
        assert [r.file_path.name for r in standalone.search(q)] == [
            r.file_path.name for r in backed.search(q)
        ]
        assert [r.score for r in standalone.search(q)] == [
            r.score for r in backed.search(q)
        ]


def test_index_backed_graph_matches_standalone(repo: Path, cache: Path) -> None:
    snapshot = _builder(repo, cache).build()
    standalone = DependencyGraphBuilder(repo).build()
    backed = DependencyGraphBuilder(repo, index=snapshot).build()
    assert standalone.get_all_nodes() == backed.get_all_nodes()
    assert standalone.get_all_edges() == backed.get_all_edges()


def test_index_backed_semantic_search_finds_results(
    repo: Path, cache: Path
) -> None:
    snapshot = _builder(repo, cache).build()
    searcher = SemanticSearcher(
        repo, FakeEmbeddingProvider(), index=snapshot
    )
    results = searcher.search("beta method", limit=5)
    assert results
    assert all(r.file_path in {Path("a.py"), Path("b.py")} for r in results)


def test_persist_false_keeps_cache_in_memory(repo: Path, cache: Path) -> None:
    builder = IncrementalIndexBuilder(repo, cache_dir=cache, persist=False)
    first = builder.build()
    assert first.stats.files_parsed == 2
    # Same builder reuses the in-memory cache; nothing written to disk.
    second = builder.build()
    assert second.stats.files_parsed == 0
    assert not cache.exists()


def test_parser_round_trip_preserves_analysis(repo: Path, cache: Path) -> None:
    snapshot = _builder(repo, cache).build()
    a = snapshot.analysis_for(Path("a.py"))
    parser = PythonParser()
    fresh = parser.parse_source(A, file_path=Path("a.py"))
    assert a.imports == fresh.imports
    assert a.from_imports == fresh.from_imports
    assert a.classes == fresh.classes
    assert a.functions == fresh.functions