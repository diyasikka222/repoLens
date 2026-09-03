"""Real persistence verification for the M17 embedding cache.

Performs the milestone's end-to-end persistence check with a fresh temporary
repository:

1. candidate semantic search (records how many documents were embedded);
2. a "fresh process" SemanticSearcher sharing the same on-disk cache runs the
   same query and must embed ZERO documents (all cache hits);
3. modifying one candidate source file and re-running re-embeds only that
   changed file.

Deterministic and fully offline (FakeEmbeddingProvider). Prints exact counts.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from repolens.embeddings import FakeEmbeddingProvider
from repolens.embedding_cache import make_repo_cache
from repolens.semantic_search import SemanticSearcher

QUERY = "refund a card payment"
CANDIDATE_FILE = "payments/refund.py"


class CountingProvider(FakeEmbeddingProvider):
    """Fake provider that counts embedded documents."""

    def __init__(self) -> None:
        super().__init__()
        self.total_documents_embedded = 0

    def embed_texts(self, texts):
        self.total_documents_embedded += len(texts)
        return super().embed_texts(texts)

    def embed_text(self, text):
        return super().embed_text(text)


def write_file(root: Path, relative: str, source: str) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")


def build_repo(root: Path) -> None:
    write_file(root, CANDIDATE_FILE, "def refund_transaction():\n    return 'refund the payment'\n")
    write_file(root, "payments/charge.py", "def charge_card():\n    pass\n")
    write_file(root, "auth/login.py", "def do_login():\n    pass\n")
    write_file(root, "storage/ledger.py", "def record_entry():\n    pass\n")


def run_searcher(root: Path, provider: CountingProvider) -> int:
    searcher = SemanticSearcher(root, provider, cache=make_repo_cache(root, directory=root / ".seqcache"))
    hits = list(searcher.search(QUERY, limit=5))
    return provider.total_documents_embedded, len(hits), searcher.cache_stats


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        build_repo(root)

        # 1. First run — cold cache.
        provider1 = CountingProvider()
        embedded1, results1, stats1 = run_searcher(root, provider1)
        print(f"first-run: documents embedded = {embedded1}, results = {results1}")

        # 2. Fresh process sharing the same on-disk cache.
        provider2 = CountingProvider()
        embedded2, results2, stats2 = run_searcher(root, provider2)
        print(f"fresh-process: documents embedded = {embedded2}, results = {results2}")
        print(f"  cache hits = {stats2['hits']}  misses = {stats2['misses']}")

        # 3. Modify one candidate file, re-run.
        write_file(root, CANDIDATE_FILE, "def refund_transaction():\n    return 'refund user money immediately now'\n")
        provider3 = CountingProvider()
        embedded3, results3, stats3 = run_searcher(root, provider3)
        print(f"changed-file: documents embedded = {embedded3}, results = {results3}")
        print(f"  cache hits = {stats3['hits']}  misses = {stats3['misses']}")

        # 4. Deleting the cache restores clean-cache behavior.
        from repolens.embedding_cache import FileSystemEmbeddingCache
        FileSystemEmbeddingCache(root / ".seqcache").clear()
        provider4 = CountingProvider()
        embedded4, results4, _ = run_searcher(root, provider4)
        print(f"after-cache-clear: documents embedded = {embedded4}, results = {results4}")

        ok = (
            embedded1 > 0
            and embedded2 == 0 and results2 == results1
            and 0 < embedded3 <= embedded1 and results3 == results1
            and embedded4 == embedded1 and results4 == results1
        )
        print()
        print("VERIFICATION " + ("PASSED" if ok else "FAILED"))
        return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())