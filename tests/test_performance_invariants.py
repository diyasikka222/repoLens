"""Performance *invariants* (Milestone 20).

Instead of asserting brittle real-time thresholds ("must finish in < 0.5s"),
these tests pin deterministic structural guarantees that keep performance
bounded:

- an unchanged rebuild parses ZERO files;
- a single-file modification reparses exactly ONE file;
- semantic candidate retrieval embeds only the candidate documents (never the
  whole repository);
- the persistent cache prevents re-embedding unchanged documents;
- the context budget is never exceeded;
- duplicate files are never selected twice;
- the number of unique symbols/files stays consistent across cache regimes.

All tests are offline, deterministic, and machine-independent.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from repolens.context import ContextBudget, ContextEngine
from repolens.embeddings import FakeEmbeddingProvider
from repolens.embedding_cache import make_repo_cache
from repolens.incremental_index import IncrementalIndexBuilder
from repolens.search import CodeSearcher
from repolens.semantic_search import SemanticSearcher

A = "import os\n\ndef alpha(x: int) -> int:\n    return x + 1\n"
B = "from a import alpha\n\ndef beta():\n    return alpha(2)\n"


def _write(repo: Path, name: str, source: str) -> Path:
    path = repo / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")
    return path


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


# ---------------------------------------------------------------------------
# Indexing invariants
# ---------------------------------------------------------------------------


def test_unchanged_build_parses_zero_files(repo: Path, cache: Path) -> None:
    builder = IncrementalIndexBuilder(repo, cache_dir=cache)
    first = builder.build()
    assert first.stats.files_parsed == 2
    second = builder.build()
    assert second.stats.files_parsed == 0
    assert second.stats.cache_hits == 2


def test_single_file_modification_reparses_one_file(
    repo: Path, cache: Path
) -> None:
    builder = IncrementalIndexBuilder(repo, cache_dir=cache)
    builder.build()
    _write(repo, "a.py", A.replace("x + 1", "x + 2"))
    index = builder.build()
    assert index.stats.files_parsed == 1
    assert index.stats.cache_hits == 1


def test_duplicate_files_never_selected(repo: Path) -> None:
    engine = ContextEngine(repo, budget=ContextBudget(max_tokens=100_000))
    package = engine.build_context("alpha beta function")
    paths = [c.path for c in package.selected_files]
    assert len(paths) == len(set(paths))  # no duplicates


# ---------------------------------------------------------------------------
# Embedding invariants
# ---------------------------------------------------------------------------


class CountingEmbeddingProvider(FakeEmbeddingProvider):
    """Fake provider that counts how many *documents* it embedded."""

    def __init__(self) -> None:
        super().__init__()
        self.documents_embedded = 0

    def embed_texts(self, texts):
        self.documents_embedded += len(texts)
        return super().embed_texts(texts)


def test_candidate_semantic_embeds_only_candidates(tmp_path: Path) -> None:
    repo = tmp_path / "bigger"
    repo.mkdir()
    _write(repo, "a.py", A)
    _write(repo, "b.py", B)
    _write(repo, "c.py", "def gamma():\n    return 3\n")
    _write(repo, "d.py", "class Delta:\n    def run(self):\n        return 4\n")
    _write(repo, "e.py", "def epsilon():\n    return 5\n")

    provider = CountingEmbeddingProvider()
    lexical = CodeSearcher(repo)
    candidate_paths = {
        result.file_path for result in lexical.search("alpha", limit=40)
    }
    searcher = SemanticSearcher(
        repo, provider, candidate_searcher=lexical, candidate_limit=40
    )
    _ = searcher.search("alpha function", limit=5)
    # Exactly the candidate documents were embedded — never the whole repo.
    assert provider.documents_embedded == len(candidate_paths)
    assert provider.documents_embedded < len(
        IncrementalIndexBuilder(repo, persist=False).build().files
    )


def test_persistent_cache_avoids_reembedding_unchanged_documents(
    repo: Path, tmp_path: Path
) -> None:
    cache_dir = tmp_path / "emb"

    def search(provider: CountingEmbeddingProvider) -> SemanticSearcher:
        return SemanticSearcher(
            repo,
            provider,
            candidate_searcher=CodeSearcher(repo),
            cache=make_repo_cache(repo, directory=cache_dir),
        )

    first = CountingEmbeddingProvider()
    s1 = search(first)
    _ = s1.search("alpha function", limit=5)
    assert first.documents_embedded > 0  # cold: everything embedded

    second = CountingEmbeddingProvider()
    s2 = search(second)
    _ = s2.search("alpha function", limit=5)
    assert second.documents_embedded == 0  # warm: persistent cache hit path
    assert s2.cache_stats["hits"] > 0


def test_query_vectors_never_enter_the_persistent_cache(
    repo: Path, tmp_path: Path
) -> None:
    """Only repository *document* vectors are persisted (never query vectors)."""
    import json

    cache_dir = tmp_path / "emb"
    searcher = SemanticSearcher(
        repo,
        FakeEmbeddingProvider(),
        candidate_searcher=CodeSearcher(repo),
        cache=make_repo_cache(repo, directory=cache_dir),
    )
    _ = searcher.search("alpha function", limit=5)
    _ = searcher.search("beta from alpha", limit=5)

    entries = list(cache_dir.glob("*.json"))
    assert entries, "the persistent cache should have been written"
    repository_files = {Path("a.py"), Path("b.py")}
    for entry in entries:
        payload = json.loads(entry.read_text(encoding="utf-8"))
        # A stored entry is keyed to a repository file's path and content hash.
        assert Path(payload["path"]) in repository_files
        assert payload["content"]  # the document content identity
        assert "vector" in payload


def test_persistent_cache_stores_no_query_query_marker(
    repo: Path, tmp_path: Path
) -> None:
    """The persisted payloads confirm documents, not queries, were stored."""
    import json

    cache_dir = tmp_path / "emb"
    searcher = SemanticSearcher(
        repo,
        FakeEmbeddingProvider(),
        candidate_searcher=CodeSearcher(repo),
        cache=make_repo_cache(repo, directory=cache_dir),
    )
    _ = searcher.search("alpha function", limit=5)
    texts = [
        json.loads(e.read_text(encoding="utf-8"))["path"]
        for e in cache_dir.glob("*.json")
    ]
    assert all(text.strip() for text in texts)
    assert "alpha function" not in texts  # query text is never a cache key


def test_embedding_model_change_reembeds(repo: Path, tmp_path: Path) -> None:
    cache_dir = tmp_path / "emb"

    def search(provider: CountingEmbeddingProvider) -> int:
        searcher = SemanticSearcher(
            repo,
            provider,
            candidate_searcher=CodeSearcher(repo),
            cache=make_repo_cache(repo, directory=cache_dir),
        )
        _ = searcher.search("alpha function", limit=5)
        return provider.documents_embedded

    first = CountingEmbeddingProvider()
    cold = search(first)
    second = CountingEmbeddingProvider()
    warm = search(second)
    # A *different* model identity (different dimensions) re-embeds everything.
    other = CountingEmbeddingProvider()
    other.dimensions = 512
    model_changed = search(other)
    assert cold > 0
    assert warm == 0
    assert model_changed > 0


# ---------------------------------------------------------------------------
# Context budget invariants
# ---------------------------------------------------------------------------


def test_context_budget_never_exceeded(repo: Path) -> None:
    for budget in (60, 120, 500, 2000):
        engine = ContextEngine(repo, budget=ContextBudget(max_tokens=budget))
        package = engine.build_context("how does alpha and beta work")
        assert package.total_estimated_tokens <= budget


def test_context_budget_unlimited_when_none(repo: Path) -> None:
    engine = ContextEngine(repo, budget=ContextBudget(max_tokens=None))
    package = engine.build_context("alpha beta")
    assert package.total_estimated_tokens > 0
    assert package.selected_files  # nothing dropped for exceeding a budget


def test_retrieval_results_are_unique_across_strategies(repo: Path) -> None:
    from repolens.retrieval import FusionStrategy, HybridSearcher

    provider = FakeEmbeddingProvider()
    lexical = CodeSearcher(repo)
    semantic = SemanticSearcher(repo, provider, candidate_searcher=lexical)
    hybrid = HybridSearcher(
        repo,
        lexical_searcher=lexical,
        semantic_searcher=semantic,
        strategy=FusionStrategy.RRF,
    )
    paths = [r.file_path for r in hybrid.search("alpha", limit=10)]
    assert len(paths) == len(set(paths))