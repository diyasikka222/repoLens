"""Tests for the persistent embedding cache and its SemanticSearcher integration."""

import json
from pathlib import Path

import pytest

from repolens.embedding_cache import (
    FileSystemEmbeddingCache,
    content_identity,
    default_cache_dir,
    make_repo_cache,
    normalize_embedding_identity,
    repository_identity,
)
from repolens.embeddings import FakeEmbeddingProvider
from repolens.semantic_search import SemanticSearcher


def write_file(tmp_path: Path, relative: str, source: str) -> None:
    file_path = tmp_path / relative
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(source, encoding="utf-8")


class RecordingProvider(FakeEmbeddingProvider):
    """Fake provider that records how many documents it embedded."""

    def __init__(self) -> None:
        super().__init__()
        self.total_documents_embedded = 0

    def embed_texts(self, texts):
        self.total_documents_embedded += len(texts)
        return super().embed_texts(texts)

    def embed_text(self, text):
        return super().embed_text(text)


def build_corpus(tmp_path: Path) -> None:
    write_file(tmp_path, "payments/refund.py", "def refund_transaction():\n    pass\n")
    write_file(tmp_path, "payments/charge.py", "def charge_card():\n    pass\n")
    write_file(tmp_path, "auth/login.py", "def do_login():\n    pass\n")


def temp_cache_dir(tmp_path: Path) -> str:
    return str(tmp_path / "seqcache")


def search(searcher: SemanticSearcher, query: str, limit: int = 10):
    return [r.file_path.as_posix() for r in searcher.search(query, limit=limit)]


# ---------------------------------------------------------------------------
# Cache abstraction unit tests
# ---------------------------------------------------------------------------


def temp_dir(tmp_path: Path) -> Path:
    return tmp_path / "cache"


def test_lookup_miss_when_empty(tmp_path: Path) -> None:
    cache = FileSystemEmbeddingCache(temp_dir(tmp_path))
    assert cache.lookup("a.py", content_identity(b"x"), "emb") is None


def test_store_then_lookup_hit(tmp_path: Path) -> None:
    cache = FileSystemEmbeddingCache(temp_dir(tmp_path))
    vector = (0.1, 0.2, 0.3)
    cache.store("a.py", content_identity(b"x"), "emb", vector)
    assert cache.lookup("a.py", content_identity(b"x"), "emb") == pytest.approx(vector)


def test_content_change_is_a_miss(tmp_path: Path) -> None:
    cache = FileSystemEmbeddingCache(temp_dir(tmp_path))
    cache.store("a.py", content_identity(b"old"), "emb", (1.0,))
    assert cache.lookup("a.py", content_identity(b"new"), "emb") is None


def test_embedding_change_is_a_miss(tmp_path: Path) -> None:
    cache = FileSystemEmbeddingCache(temp_dir(tmp_path))
    cache.store("a.py", content_identity(b"x"), "emb-old", (1.0,))
    assert cache.lookup("a.py", content_identity(b"x"), "emb-new") is None


def test_path_change_is_a_miss(tmp_path: Path) -> None:
    cache = FileSystemEmbeddingCache(temp_dir(tmp_path))
    cache.store("a.py", content_identity(b"x"), "emb", (1.0,))
    assert cache.lookup("b.py", content_identity(b"x"), "emb") is None


def test_corrupt_json_is_a_miss(tmp_path: Path) -> None:
    cache = FileSystemEmbeddingCache(temp_dir(tmp_path))
    cache.store("a.py", content_identity(b"x"), "emb", (1.0, 2.0))
    entry = next(temp_dir(tmp_path).glob("*.json"))
    entry.write_text("not json{{{", encoding="utf-8")
    assert cache.lookup("a.py", content_identity(b"x"), "emb") is None


def test_identity_mismatch_in_file_is_a_miss(tmp_path: Path) -> None:
    cache = FileSystemEmbeddingCache(temp_dir(tmp_path))
    cache.store("a.py", content_identity(b"x"), "emb", (1.0, 2.0))
    entry = next(temp_dir(tmp_path).glob("*.json"))
    payload = json.loads(entry.read_text(encoding="utf-8"))
    payload["content"] = "tampered"
    entry.write_text(json.dumps(payload), encoding="utf-8")
    assert cache.lookup("a.py", content_identity(b"x"), "emb") is None


def test_malformed_vector_is_a_miss(tmp_path: Path) -> None:
    cache = FileSystemEmbeddingCache(temp_dir(tmp_path))
    cache.store("a.py", content_identity(b"x"), "emb", (1.0, 2.0))
    entry = next(temp_dir(tmp_path).glob("*.json"))
    payload = json.loads(entry.read_text(encoding="utf-8"))
    payload["vector"] = ["not", "numbers"]
    entry.write_text(json.dumps(payload), encoding="utf-8")
    assert cache.lookup("a.py", content_identity(b"x"), "emb") is None


def test_clear_removes_all_entries(tmp_path: Path) -> None:
    cache = FileSystemEmbeddingCache(temp_dir(tmp_path))
    cache.store("a.py", content_identity(b"x"), "emb", (1.0,))
    cache.store("b.py", content_identity(b"y"), "emb", (2.0,))
    cache.clear()
    assert list(temp_dir(tmp_path).glob("*.json")) == []
    assert cache.lookup("a.py", content_identity(b"x"), "emb") is None


def test_default_cache_dir_is_outside_repo_but_scoped(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("REPOLENS_CACHE_DIR", str(tmp_path / "global-cache"))
    base = default_cache_dir(None)
    repo = tmp_path / "repo"
    repo.mkdir()
    cache_dir = default_cache_dir(repo)
    assert str(cache_dir).startswith(str(base))
    assert cache_dir.parent == base  # hash id appended directly under base


def test_repository_identity_is_deterministic_and_distinct() -> None:
    a = repository_identity("/tmp/some/repo")
    b = repository_identity("/tmp/some/repo")
    c = repository_identity("/tmp/some/other")
    assert a == b
    assert a != c


def test_normalize_embedding_identity_distinguishes_models() -> None:
    p1 = FakeEmbeddingProvider(dimensions=128)
    p2 = FakeEmbeddingProvider(dimensions=256)
    assert normalize_embedding_identity(p1) != normalize_embedding_identity(p2)


def test_make_repo_cache_respects_explicit_directory(tmp_path: Path) -> None:
    cache = make_repo_cache(tmp_path, directory=tmp_path / "custom-cache")
    cache.store("a.py", content_identity(b"x"), "emb", (1.0,))
    assert list((tmp_path / "custom-cache").glob("*.json")) != []


def test_make_repo_cache_default_location(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("REPOLENS_CACHE_DIR", str(tmp_path / "scoped-cache"))
    cache = make_repo_cache(tmp_path)
    repoid = repository_identity(tmp_path)
    assert str(cache.directory) == str((tmp_path / "scoped-cache") / repoid)


# --- A. Cache miss ----------------------------------------------------------


def test_a_cache_miss_persists_vector(tmp_path: Path) -> None:
    build_corpus(tmp_path)
    cache = make_repo_cache(tmp_path, directory=temp_cache_dir(tmp_path))
    provider = RecordingProvider()
    searcher = SemanticSearcher(tmp_path, provider, cache=cache)

    assert search(searcher, "refund payment") != []
    assert provider.total_documents_embedded > 0
    assert searcher.cache_stats["misses"] == provider.total_documents_embedded
    # vectors were persisted for exactly the embedded candidates
    assert (
        len(list((tmp_path / "seqcache").glob("*.json")))
        == provider.total_documents_embedded
    )


# --- B. Cache hit -----------------------------------------------------------


def test_b_cache_hit_does_not_call_provider(tmp_path: Path) -> None:
    build_corpus(tmp_path)
    cache_path = temp_cache_dir(tmp_path)
    cache = make_repo_cache(tmp_path, directory=cache_path)
    first = RecordingProvider()
    first_searcher = SemanticSearcher(tmp_path, first, cache=cache)
    search(first_searcher, "refund payment")
    first_embedded = first.total_documents_embedded
    assert first_embedded > 0

    # A fresh searcher with a fresh provider, sharing the same on-disk cache.
    second = RecordingProvider()
    second_searcher = SemanticSearcher(
        tmp_path, second, cache=make_repo_cache(tmp_path, directory=cache_path)
    )
    assert search(second_searcher, "refund payment") != []
    assert second.total_documents_embedded == 0
    assert second_searcher.cache_stats["hits"] == first_embedded
    assert second_searcher.cache_stats["misses"] == 0


# --- C. Content invalidation ------------------------------------------------


def test_c_content_invalidation_reembeds_changed_file(tmp_path: Path) -> None:
    build_corpus(tmp_path)
    cache_path = temp_cache_dir(tmp_path)
    cache = make_repo_cache(tmp_path, directory=cache_path)
    first = RecordingProvider()
    SemanticSearcher(tmp_path, first, cache=cache).search("refund payment")
    first_embedded = first.total_documents_embedded

    # Modify one candidate file's content.
    write_file(
        tmp_path,
        "payments/refund.py",
        "def refund_transaction():\n    return 'refund user money now'\n",
    )
    changed = RecordingProvider()
    changed_searcher = SemanticSearcher(
        tmp_path, changed, cache=make_repo_cache(tmp_path, directory=cache_path)
    )
    search(changed_searcher, "refund payment")

    # The changed candidate is re-embedded, unchanged candidates hit the cache.
    assert 0 < changed.total_documents_embedded <= first_embedded
    assert (
        changed_searcher.cache_stats["hits"]
        == first_embedded - changed.total_documents_embedded
    )
    assert changed_searcher.cache_stats["embedded_documents"] == changed.total_documents_embedded


# --- D. Model/provider invalidation -----------------------------------------


def test_d_model_invalidation_does_not_reuse_old_vector(tmp_path: Path) -> None:
    build_corpus(tmp_path)
    cache_path = temp_cache_dir(tmp_path)
    cache = make_repo_cache(tmp_path, directory=cache_path)
    provider_a = RecordingProvider()
    SemanticSearcher(tmp_path, provider_a, cache=cache).search("refund payment")
    assert provider_a.total_documents_embedded > 0

    # A different embedding identity (different dimensions -> different model).
    provider_b = RecordingProvider()
    provider_b.dimensions = 2048
    second = SemanticSearcher(tmp_path, provider_b, cache=cache)
    search(second, "refund payment")
    assert provider_b.total_documents_embedded > 0
    assert second.cache_stats["hits"] == 0


# --- E. Restart persistence -------------------------------------------------


def test_e_restart_persistence_loads_from_cache(tmp_path: Path) -> None:
    build_corpus(tmp_path)
    cache_path = temp_cache_dir(tmp_path)
    provider = RecordingProvider()

    searcher1 = SemanticSearcher(
        tmp_path, provider, cache=make_repo_cache(tmp_path, directory=cache_path)
    )
    search(searcher1, "refund payment")
    first_embedded = provider.total_documents_embedded
    assert first_embedded > 0

    # Simulate a process restart: brand-new searcher + brand-new provider,
    # same on-disk cache, same query.
    provider2 = RecordingProvider()
    searcher2 = SemanticSearcher(
        tmp_path, provider2, cache=make_repo_cache(tmp_path, directory=cache_path)
    )
    assert search(searcher2, "refund payment") == search(searcher1, "refund payment")
    assert provider2.total_documents_embedded == 0  # nothing re-embedded
    assert searcher2.cache_stats["hits"] == first_embedded


# --- F. Corrupt cache -------------------------------------------------------


def test_f_corrupt_cache_is_a_miss_and_search_succeeds(tmp_path: Path) -> None:
    build_corpus(tmp_path)
    cache_path = temp_cache_dir(tmp_path)
    cache = make_repo_cache(tmp_path, directory=cache_path)
    first = RecordingProvider()
    SemanticSearcher(tmp_path, first, cache=cache).search("refund payment")

    # Corrupt every cache entry on disk.
    for entry in (tmp_path / "seqcache").glob("*.json"):
        entry.write_text("### not valid json ###", encoding="utf-8")

    provider = RecordingProvider()
    searcher = SemanticSearcher(
        tmp_path, provider, cache=make_repo_cache(tmp_path, directory=cache_path)
    )
    result = search(searcher, "refund payment")
    assert result != []  # search still succeeds
    assert provider.total_documents_embedded == first.total_documents_embedded
    assert searcher.cache_stats["hits"] == 0
    assert searcher.cache_stats["misses"] == first.total_documents_embedded


# --- G. Multiple repositories -----------------------------------------------


def test_g_repositories_do_not_collide(tmp_path: Path) -> None:
    repo_a = tmp_path / "repoA"
    repo_b = tmp_path / "repoB"
    write_file(repo_a, "payments/refund.py", "def refund_transaction():\n    pass\n")
    write_file(repo_b, "payments/refund.py", "def refund_transaction():\n    pass\n")

    cache_a = make_repo_cache(repo_a, directory=temp_cache_dir(tmp_path))
    provider_a = RecordingProvider()
    SemanticSearcher(repo_a, provider_a, cache=cache_a).search("refund payment")
    first_a = provider_a.total_documents_embedded

    # repo B uses a separate cache directory, so its embedding identity is
    # distinct and it must embed from scratch.
    cache_b = make_repo_cache(repo_b, directory=temp_cache_dir(tmp_path) + "B")
    provider_b = RecordingProvider()
    searcher_b = SemanticSearcher(repo_b, provider_b, cache=cache_b)
    search(searcher_b, "refund payment")
    assert provider_b.total_documents_embedded == first_a
    assert repository_identity(repo_a) != repository_identity(repo_b)


# --- H. Candidate behavior --------------------------------------------------


def test_h_only_candidates_are_embedded(tmp_path: Path) -> None:
    build_corpus(tmp_path)
    # A file that never shows up as a candidate for this query.
    write_file(tmp_path, "docs/notes.py", "def unrelated_topic():\n    pass\n")
    cache_path = temp_cache_dir(tmp_path)
    provider = RecordingProvider()
    searcher = SemanticSearcher(tmp_path, provider, cache=make_repo_cache(tmp_path, directory=cache_path))

    search(searcher, "refund payment")
    embedded = provider.total_documents_embedded
    cached_paths = {Path(p.stem) for p in (tmp_path / "seqcache").glob("*.json")}
    assert embedded < 4  # strictly fewer than the full 4-file corpus
    # The unrelated non-candidate was not persisted.
    assert len(list((tmp_path / "seqcache").glob("*.json"))) == embedded
    assert searcher.cache_stats["embedded_documents"] == embedded


# --- I. In-memory fast layer preserved --------------------------------------


def test_i_in_memory_cache_keeps_repeated_searches_fast(tmp_path: Path) -> None:
    build_corpus(tmp_path)
    cache_path = temp_cache_dir(tmp_path)
    provider = RecordingProvider()
    searcher = SemanticSearcher(
        tmp_path, provider, cache=make_repo_cache(tmp_path, directory=cache_path)
    )

    search(searcher, "refund payment")
    first_embedded = provider.total_documents_embedded
    # Same query again in the same process: no provider call, no disk lookups.
    search(searcher, "refund payment")
    assert provider.total_documents_embedded == first_embedded
    assert searcher.cache_stats["hits"] == 0  # in-memory layer served it


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def temp_cache_dir(tmp_path: Path) -> str:
    return str(tmp_path / "seqcache")