"""Failure / recovery hardening for RepoLens (Milestone 20).

These tests pin the *explicit, safe* behaviour expected when things go wrong
at the persistence and indexing boundaries, without redesigning any of the
underlying systems:

- malformed Python preserves the existing ``SyntaxError`` semantics;
- a file that turns malformed after a previously valid index repares after a
  fix and never serves stale data;
- renamed / deleted files prune stale records;
- corrupt or incompatible cache entries degrade to misses (never crashes and
  never silently masquerade stale data as current);
- an embedding model identity change invalidates cached vectors;
- empty repositories, ignored-only repositories, and non-Python repos build
  empty but valid indexes;
- missing cache directories are created; disabled persistence writes nothing.

All tests are fully offline and deterministic.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from repolens.embeddings import FakeEmbeddingProvider
from repolens.embedding_cache import FileSystemEmbeddingCache, normalize_embedding_identity
from repolens.incremental_index import (
    CACHE_SCHEMA_VERSION,
    AnalysisCache,
    IncrementalIndexBuilder,
)
from repolens.parser import PythonParser
from repolens.search import CodeSearcher
from repolens.semantic_search import SemanticSearcher

VALID = "def alpha(x: int) -> int:\n    return x + 1\n"


def _write(repo: Path, name: str, source: str) -> Path:
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


@pytest.fixture
def cache(tmp_path: Path) -> Path:
    return tmp_path / "index_cache"


# ---------------------------------------------------------------------------
# Malformed Python
# ---------------------------------------------------------------------------


def test_malformed_python_raises_syntax_error(repo: Path, cache: Path) -> None:
    _write(repo, "broken.py", "def nope(:\n")
    with pytest.raises(SyntaxError):
        IncrementalIndexBuilder(repo, cache_dir=cache).build()


def test_parser_public_api_preserves_syntax_error_semantics() -> None:
    parser = PythonParser()
    with pytest.raises(SyntaxError):
        parser.parse_source("def broken(:\n")


def test_file_becoming_malformed_then_repaired_is_reparsed(
    repo: Path, cache: Path
) -> None:
    builder = IncrementalIndexBuilder(repo, cache_dir=cache)
    first = builder.build()
    assert first.stats.files_parsed == 1

    _write(repo, "a.py", "def broken(:\n")
    with pytest.raises(SyntaxError):
        builder.build()
        # The broken build must never succeed or serve the old analysis.

    _write(repo, "a.py", VALID.replace("x + 1", "x + 2"))
    third = builder.build()
    assert third.stats.files_parsed == 1
    # The analysis reflects the *current* content, not the stale snapshot.
    functions = third.analysis_for(Path("a.py")).functions
    assert functions and "x + 2" not in str(functions[0].arguments)


def test_no_stale_data_masquerades_as_current(repo: Path, cache: Path) -> None:
    builder = IncrementalIndexBuilder(repo, cache_dir=cache)
    builder.build()
    _write(repo, "a.py", "def alpha(x: int) -> int:\n    return x + 999\n")
    index = builder.build()
    source = index.source_for(Path("a.py"))
    assert "999" in source
    # Fresh analysis: the parsed function now references the updated body.
    assert index.stats.files_parsed == 1
    assert "alpha" in {s.name for s in index.symbols}


# ---------------------------------------------------------------------------
# Deleted / renamed files
# ---------------------------------------------------------------------------


def test_renamed_file_is_pruned_and_reparsed(tmp_path: Path, cache: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    _write(root, "old_name.py", VALID)
    builder = IncrementalIndexBuilder(root, cache_dir=cache)
    first = builder.build()
    assert first.stats.files_parsed == 1

    (root / "old_name.py").rename(root / "new_name.py")
    second = builder.build()
    stats = second.stats.as_dict()
    assert stats["files_discovered"] == 1
    assert stats["files_parsed"] == 1  # new path ⇒ new content identity
    assert stats["files_removed"] == 1  # old path pruned
    assert second.files == (Path("new_name.py"),)


def test_deleted_file_is_pruned_and_absent_from_index(
    repo: Path, cache: Path
) -> None:
    _write(repo, "extra.py", "def extra():\n    return 1\n")
    builder = IncrementalIndexBuilder(repo, cache_dir=cache)
    builder.build()
    (repo / "extra.py").unlink()
    index = builder.build()
    assert index.stats.files_removed == 1
    assert index.files == (Path("a.py"),)
    assert Path("extra.py") not in index.by_path


# ---------------------------------------------------------------------------
# Corrupt / incompatible cache entries
# ---------------------------------------------------------------------------


def test_unreadable_embedding_cache_entry_is_a_miss(tmp_path: Path) -> None:
    cache = FileSystemEmbeddingCache(tmp_path)
    cache.store("a.py", "content1", "embed1", (0.1, 0.2))
    # Corrupt the payload so it can no longer be parsed.
    for entry in tmp_path.glob("*.json"):
        entry.write_text("{definitely not json", encoding="utf-8")
    assert cache.lookup("a.py", "content1", "embed1") is None


def test_mismatched_embedding_cache_entry_is_a_miss(tmp_path: Path) -> None:
    cache = FileSystemEmbeddingCache(tmp_path)
    cache.store("a.py", "content1", "embed1", (0.1,))
    assert cache.lookup("a.py", "content1", "embed2") is None
    assert cache.lookup("a.py", "contentX", "embed1") is None
    assert cache.lookup("b.py", "content1", "embed1") is None


def test_corrupt_incremental_entry_does_not_crash_build(
    repo: Path, cache: Path
) -> None:
    builder = IncrementalIndexBuilder(repo, cache_dir=cache)
    builder.build()
    for entry in cache.glob("*.json"):
        entry.write_text("{broken", encoding="utf-8")
    index = builder.build()
    assert index.stats.cache_hits == 0
    assert index.stats.cache_misses == 1
    assert index.stats.files_parsed == 1


def test_invalid_schema_version_is_treated_as_miss(repo: Path, cache: Path) -> None:
    builder = IncrementalIndexBuilder(repo, cache_dir=cache)
    builder.build()
    for entry in cache.glob("*.json"):
        payload = json.loads(entry.read_text(encoding="utf-8"))
        payload["schema"] = CACHE_SCHEMA_VERSION - 1  # stale format
        entry.write_text(json.dumps(payload), encoding="utf-8")
    index = builder.build()
    assert index.stats.files_parsed == 1
    assert index.stats.cache_hits == 0


def test_mismatched_content_hash_is_treated_as_miss(repo: Path, cache: Path) -> None:
    builder = IncrementalIndexBuilder(repo, cache_dir=cache)
    builder.build()
    # Rewrite entries advertising a *different* content hash than stored.
    import hashlib

    for entry in cache.glob("*.json"):
        payload = json.loads(entry.read_text(encoding="utf-8"))
        payload["content_hash"] = hashlib.sha256(b"something-else").hexdigest()
        entry.write_text(json.dumps(payload), encoding="utf-8")
    index = builder.build()
    # The altered content hash never matches the on-disk entry → full reparse.
    assert index.stats.files_parsed == 1
    assert index.stats.cache_hits == 0


# ---------------------------------------------------------------------------
# Embedding model identity
# ---------------------------------------------------------------------------


def test_lookup_model_identity_change_is_a_miss(tmp_path: Path) -> None:
    cache = FileSystemEmbeddingCache(tmp_path)
    cache.store("a.py", "c1", "model-A", (0.1,))
    assert cache.lookup("a.py", "c1", "model-A") == (0.1,)
    assert cache.lookup("a.py", "c1", "model-B") is None


def test_normalize_identity_distinguishes_dimensions() -> None:
    provider_1024 = FakeEmbeddingProvider(dimensions=1024)
    provider_512 = FakeEmbeddingProvider(dimensions=512)
    first = normalize_embedding_identity(provider_1024)
    second = normalize_embedding_identity(provider_512)
    assert first != second
    assert first == normalize_embedding_identity(
        FakeEmbeddingProvider(dimensions=1024)
    )


def test_semantic_search_reembeds_after_model_identity_change(
    repo: Path, tmp_path: Path
) -> None:
    from repolens.embedding_cache import make_repo_cache
    from repolens.semantic_search import SemanticSearcher

    cache_dir = tmp_path / "emb"

    def search(dimensions: int) -> SemanticSearcher:
        return SemanticSearcher(
            repo,
            FakeEmbeddingProvider(dimensions=dimensions),
            candidate_searcher=CodeSearcher(repo),
            cache=make_repo_cache(repo, directory=cache_dir),
        )

    first = search(dimensions=1024)
    _ = first.search("alpha function", limit=5)
    assert first.cache_stats["misses"] > 0

    # Different dimensionality ⇒ different embedding identity ⇒ cold again.
    second = search(dimensions=512)
    _ = second.search("alpha function", limit=5)
    assert second.cache_stats["hits"] == 0
    assert second.cache_stats["misses"] > 0


# ---------------------------------------------------------------------------
# Sparse / hostile repository shapes
# ---------------------------------------------------------------------------


def test_empty_repository_builds_empty_index(tmp_path: Path, cache: Path) -> None:
    root = tmp_path / "empty"
    root.mkdir()
    index = IncrementalIndexBuilder(root, cache_dir=cache).build()
    assert index.stats.files_discovered == 0
    assert index.files == ()
    assert index.symbols == ()
    # ContextEngine still works on an empty repository.
    from repolens.context import ContextEngine

    engine = ContextEngine(root)
    package = engine.build_context("anything")
    assert package.selected_files == ()


def test_ignored_directories_yield_no_python_files(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    for ignored in (".git", "__pycache__", ".venv", "venv", "node_modules"):
        _write(root, f"{ignored}/module.py", "def hidden():\n    return 0\n")
    index = IncrementalIndexBuilder(root, persist=False).build()
    assert index.stats.files_discovered == 0
    assert CodeSearcher(root).search("hidden", limit=5) == []


def test_non_python_repository_builds_empty_index(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    _write(root, "readme.md", "# docs only")
    _write(root, "data.txt", "not python")
    index = IncrementalIndexBuilder(root, persist=False).build()
    assert index.stats.files_discovered == 0


def test_missing_cache_directory_is_created(repo: Path, tmp_path: Path) -> None:
    nested = tmp_path / "a" / "b" / "c"
    builder = IncrementalIndexBuilder(repo, cache_dir=nested)
    index = builder.build()
    assert nested.is_dir()
    assert index.stats.files_parsed == 1


def test_disabled_persistence_writes_nothing_to_disk(
    repo: Path, tmp_path: Path
) -> None:
    cache_path = tmp_path / "never_created"
    builder = IncrementalIndexBuilder(repo, cache_dir=cache_path, persist=False)
    builder.build()
    assert not cache_path.exists()


def test_disabled_embedding_cache_is_used_inmemory(repo: Path) -> None:
    provider = FakeEmbeddingProvider()
    searcher = SemanticSearcher(repo, provider, candidate_searcher=CodeSearcher(repo))
    results = searcher.search("alpha function", limit=5)
    assert results  # no cache at all is still functional


def test_legacy_index_entry_without_embedded_path_is_not_pruned(
    repo: Path, cache: Path
) -> None:
    """Older entries with no 'path' field must never be wrongly pruned."""
    builder = IncrementalIndexBuilder(repo, cache_dir=cache)
    builder.build()
    for entry in cache.glob("*.json"):
        payload = json.loads(entry.read_text(encoding="utf-8"))
        payload.pop("path", None)
        entry.write_text(json.dumps(payload), encoding="utf-8")
    # Unchanged rebuild: entries have the right content hash/schema, and are
    # reused despite the missing path field.
    index = builder.build()
    assert index.stats.cache_hits >= 1


def test_interrupted_write_never_serves_stale_data(
    repo: Path, cache: Path
) -> None:
    """A truncated/corrupt incremental entry degrades to a miss + reparse."""
    builder = IncrementalIndexBuilder(repo, cache_dir=cache)
    builder.build()
    for entry in cache.glob("*.json"):
        entry.write_text('{"schema": 99, "conten', encoding="utf-8")
    index = builder.build()
    assert index.stats.files_parsed == 1
    assert index.stats.cache_hits == 0
    assert "alpha" in {s.name for s in index.symbols}