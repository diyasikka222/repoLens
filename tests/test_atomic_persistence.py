"""Tests for atomic, interruption-safe persistence (Milestone 20).

The persistent caches (:class:`~repolens.embedding_cache.FileSystemEmbeddingCache`
and :class:`~repolens.incremental_index.AnalysisCache`) write through
:func:`repolens.atomic_write.atomic_write_text` — a temporary sibling file is
written, flushed, fsynced, and moved into place with :func:`os.replace`. These
tests verify that:

- stored records round-trip exactly;
- no canonical file is ever left half-written (the ``*.part-*.tmp`` partials
  are ignored by readers and swept by ``clear``);
- a simulated interrupted write (a stale partial where the canonical file
  should be) degrades to a cache *miss*, never a crash;
- partial sweeps never delete real entries.
"""

from __future__ import annotations

import json
from pathlib import Path

from repolens.atomic_write import atomic_write_text, sweep_stale_partials
from repolens.embedding_cache import FileSystemEmbeddingCache
from repolens.incremental_index import AnalysisCache
from repolens.parser import Function, ModuleAnalysis


# ---------------------------------------------------------------------------
# atomic_write_text
# ---------------------------------------------------------------------------


def test_atomic_write_creates_file(tmp_path: Path) -> None:
    target = tmp_path / "a.json"
    atomic_write_text(target, '{"x": 1}')
    assert target.read_text(encoding="utf-8") == '{"x": 1}'
    assert list(tmp_path.glob("*.part-*.tmp")) == []


def test_atomic_write_replaces_existing_content(tmp_path: Path) -> None:
    target = tmp_path / "a.json"
    atomic_write_text(target, "old")
    atomic_write_text(target, "new")
    assert target.read_text(encoding="utf-8") == "new"


def test_atomic_write_preserves_old_file_on_failure(tmp_path: Path) -> None:
    target = tmp_path / "a.json"
    atomic_write_text(target, "old")
    # Force the replace to fail by pointing at a non-empty directory.
    broken = tmp_path / "sub"
    broken.mkdir()
    import pytest

    with pytest.raises(OSError):
        atomic_write_text(broken, "boom")
    assert (tmp_path / "a.json").read_text(encoding="utf-8") == "old"


def test_stale_partial_is_swept_without_touching_real_entries(tmp_path: Path) -> None:
    (tmp_path / "keep.json").write_text("{}", encoding="utf-8")
    (tmp_path / "keep.json.part-1.tmp").write_text("half", encoding="utf-8")
    (tmp_path / "other.json.part-2.tmp").write_text("half", encoding="utf-8")
    removed = sweep_stale_partials(tmp_path)
    assert removed == 2
    assert (tmp_path / "keep.json").exists()
    assert list(tmp_path.glob("*.part-*.tmp")) == []


# ---------------------------------------------------------------------------
# Simulated interrupted embedding-cache write
# ---------------------------------------------------------------------------


def test_interrupted_write_canonical_missing_is_a_miss(tmp_path: Path) -> None:
    """A stale partial with no canonical entry must yield a miss, not a crash."""
    cache = FileSystemEmbeddingCache(tmp_path)
    cache.store("a.py", "c1", "e1", (0.1, 0.2, 0.3))
    entry = next(tmp_path.glob("*.json"))
    resting_place = entry.read_text(encoding="utf-8")
    entry.unlink()
    # Simulate a crashed writer that only got as far as the temp file.
    (tmp_path / f"{entry.name}.part-crash.tmp").write_text(
        resting_place + "TRUNCATED", encoding="utf-8"
    )
    assert cache.lookup("a.py", "c1", "e1") is None


def test_interrupted_write_truncated_canonical_is_a_miss(tmp_path: Path) -> None:
    """Even a (legacy/naive) truncated canonical file must be a safe miss."""
    cache = FileSystemEmbeddingCache(tmp_path)
    cache.store("a.py", "c1", "e1", (0.1, 0.2, 0.3))
    (next(tmp_path.glob("*.json"))).write_text("{broken", encoding="utf-8")
    assert cache.lookup("a.py", "c1", "e1") is None


def test_store_after_interruption_recovers_cleanly(tmp_path: Path) -> None:
    cache = FileSystemEmbeddingCache(tmp_path)
    cache.store("a.py", "c1", "e1", (0.1, 0.2, 0.3))
    (tmp_path / "a.py.json.part-stale.tmp").write_text("junk", encoding="utf-8")
    cache.store("a.py", "c1", "e1", (0.4, 0.5, 0.6))
    assert cache.lookup("a.py", "c1", "e1") == (0.4, 0.5, 0.6)
    # The stale partial is gone after clear() sweeps.
    cache.clear()
    assert list(tmp_path.glob("*.part-*.tmp")) == []


# ---------------------------------------------------------------------------
# FileSystemEmbeddingCache atomicity
# ---------------------------------------------------------------------------


def test_embedding_store_round_trips(tmp_path: Path) -> None:
    cache = FileSystemEmbeddingCache(tmp_path)
    cache.store("a.py", "content-hash-a", "embed-id", (0.1, 0.2, 0.3))
    assert cache.lookup("a.py", "content-hash-a", "embed-id") == (0.1, 0.2, 0.3)


def test_embedding_store_leaves_no_partials(tmp_path: Path) -> None:
    cache = FileSystemEmbeddingCache(tmp_path)
    for index in range(5):
        cache.store(f"f{index}.py", f"c{index}", "e", (float(index),))
    assert list(tmp_path.glob("*.part-*.tmp")) == []
    assert len(list(tmp_path.glob("*.json"))) == 5
    for entry in tmp_path.glob("*.json"):
        payload = json.loads(entry.read_text(encoding="utf-8"))
        assert payload["path"] in {f"f{i}.py" for i in range(5)}


def test_embedding_clear_sweeps_partials(tmp_path: Path) -> None:
    cache = FileSystemEmbeddingCache(tmp_path)
    cache.store("a.py", "c1", "e1", (0.1,))
    (tmp_path / "x.json.part-9.tmp").write_text("junk", encoding="utf-8")
    cache.clear()
    assert list(tmp_path.glob("*.json")) == []
    assert list(tmp_path.glob("*.part-*.tmp")) == []


def test_embedding_cache_missing_directory_created(tmp_path: Path) -> None:
    nested = tmp_path / "does" / "not" / "exist"
    cache = FileSystemEmbeddingCache(nested)
    assert cache.directory.is_dir()
    cache.store("a.py", "c1", "e1", (0.1,))
    assert cache.lookup("a.py", "c1", "e1") == (0.1,)


# ---------------------------------------------------------------------------
# AnalysisCache atomicity
# ---------------------------------------------------------------------------


def _analysis() -> ModuleAnalysis:
    return ModuleAnalysis(
        file_path=None,
        functions=[Function(name="alpha", arguments=[], line=1)],
    )


def test_index_cache_store_round_trips(tmp_path: Path) -> None:
    cache = AnalysisCache(tmp_path)
    cache.store("a.py", "hash1", _analysis())
    restored = cache.lookup("a.py", "hash1")
    assert restored is not None
    assert restored.functions == [Function(name="alpha", arguments=[], line=1)]


def test_index_cache_store_leaves_no_partials(tmp_path: Path) -> None:
    cache = AnalysisCache(tmp_path)
    for index in range(5):
        cache.store(f"f{index}.py", f"hash{index}", _analysis())
    assert list(tmp_path.glob("*.part-*.tmp")) == []
    assert len(list(tmp_path.glob("*.json"))) == 5


def test_index_clear_sweeps_partials(tmp_path: Path) -> None:
    cache = AnalysisCache(tmp_path)
    cache.store("a.py", "hash1", _analysis())
    (tmp_path / "x.json.part-3.tmp").write_text("junk", encoding="utf-8")
    cache.clear()
    assert list(tmp_path.glob("*.json")) == []
    assert list(tmp_path.glob("*.part-*.tmp")) == []


def test_index_cache_missing_directory_created(tmp_path: Path) -> None:
    nested = tmp_path / "nested" / "cache"
    cache = AnalysisCache(nested)
    cache.store("a.py", "hash1", _analysis())
    assert cache.lookup("a.py", "hash1") is not None