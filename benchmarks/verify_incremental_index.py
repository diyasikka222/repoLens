"""Real persistence verification for the M18 incremental index.

Builds the incremental :class:`RepositoryIndex` for a temporary repository and
demonstrates:

1. clean build — parses every Python file;
2. unchanged rebuild (fresh builder, same cache) — parses ZERO files (all cache
   hits) and derived structures (symbol index, code search, dependency graph)
   still match the standalone path exactly;
3. single-file modification — re-parses only that file;
4. new file added — parses only the new file;
5. file deleted — prunes its cache entry;
6. clearing the index cache restores clean-build behavior.

Deterministic and fully offline. Prints exact counts.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from repolens.graph import DependencyGraphBuilder
from repolens.incremental_index import IncrementalIndexBuilder
from repolens.index import SymbolIndexBuilder
from repolens.search import CodeSearcher


def write_file(root: Path, relative: str, source: str) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")


def build_repo(root: Path) -> None:
    write_file(root, "payments/refund.py", "import os\n\ndef refund_transaction():\n    return 'refund the payment'\n")
    write_file(root, "payments/charge.py", "from .refund import refund_transaction\n\ndef charge_card():\n    return refund_transaction()\n")
    write_file(root, "auth/login.py", "def do_login():\n    pass\n")


def build(cache: Path, root: Path):
    return IncrementalIndexBuilder(root, cache_dir=cache).build()


def derived_match(root: Path, snapshot) -> bool:
    s1 = SymbolIndexBuilder(root).build()
    s2 = SymbolIndexBuilder(root, index=snapshot).build()
    c1 = CodeSearcher(root)
    c2 = CodeSearcher(root, index=snapshot)
    g1 = DependencyGraphBuilder(root).build()
    g2 = DependencyGraphBuilder(root, index=snapshot).build()
    if [s.name for s in s1.get_all_symbols()] != [s.name for s in s2.get_all_symbols()]:
        return False
    if [r.file_path.as_posix() for r in c1.search("refund")] != [r.file_path.as_posix() for r in c2.search("refund")]:
        return False
    if g1.get_all_edges() != g2.get_all_edges():
        return False
    return True


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "repo"
        root.mkdir()
        cache = Path(tmp) / "index_cache"
        build_repo(root)

        # 1. Clean build.
        r1 = build(cache, root)
        print("clean-build:  " + str(r1.stats.as_dict()))
        assert r1.stats.files_parsed == 3

        # 2. Unchanged rebuild (fresh builder object, same cache).
        r2 = build(cache, root)
        derived_match_ok = derived_match(root, r2)
        print("unchanged:    " + str(r2.stats.as_dict()))
        print(f"  derived structures match standalone = {derived_match_ok}")
        assert r2.stats.files_parsed == 0 and r2.stats.cache_hits == 3

        # 3. Modify one file.
        write_file(root, "payments/refund.py", "import os, re\n\ndef refund_transaction(amount):\n    return 'refund now absolutely'\n")
        r3 = build(cache, root)
        print("modified-file:" + str(r3.stats.as_dict()))
        assert r3.stats.files_parsed == 1 and r3.stats.cache_hits == 2

        # 4. New file added.
        write_file(root, "audit/report.py", "def build_report():\n    pass\n")
        r4 = build(cache, root)
        print("new-file:     " + str(r4.stats.as_dict()))
        assert r4.stats.files_parsed == 1 and r4.stats.files_discovered == 4

        # 5. Delete a file.
        (root / "auth/login.py").unlink()
        r5 = build(cache, root)
        print("deleted-file: " + str(r5.stats.as_dict()))
        assert r5.stats.files_removed == 1 and r5.stats.files_discovered == 3

        # 6. Clear the index cache -> clean-build behavior.
        for entry in cache.glob("*.json"):
            entry.unlink()
        r6 = build(cache, root)
        print("after-clear:  " + str(r6.stats.as_dict()))
        assert r6.stats.files_parsed == 3

        ok = (
            r1.stats.files_parsed == 3
            and r2.stats.files_parsed == 0 and r2.stats.cache_hits == 3 and derived_match_ok
            and r3.stats.files_parsed == 1
            and r4.stats.files_parsed == 1
            and r5.stats.files_removed == 1
            and r6.stats.files_parsed == 3
        )
        print()
        print("VERIFICATION " + ("PASSED" if ok else "FAILED"))
        return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())