"""Atomic, interruption-safe file writes for RepoLens caches.

The persistent caches (:class:`~repolens.embedding_cache.FileSystemEmbeddingCache`
and :class:`~repolens.incremental_index.AnalysisCache`) write small JSON
records. A naive ``path.write_text(...)`` truncates the target and then
writes; a crash between truncation and completion leaves a half-written JSON
file that a future process must treat as a miss (or worse, mis-parse).

:func:`atomic_write_text` avoids that class of corruption by writing to a
temporary sibling file, flushing and fsyncing it, and then moving it into
place with :func:`os.replace`, which is atomic on POSIX and Windows. At any
point a reader sees either the old complete file or the new complete file —
never a truncated one.

Crash behavior
--------------
A crash before :func:`os.replace` leaves only a stale ``<name>.part-<id>.tmp``
sibling. Readers only ever open the canonical file name, so stale partials
are ignored; eager consumers (``clear()``) sweep them opportunistically. No
distributed locking is implied: RepoLens caches are single-process by design
and this module only guarantees no *half-written target file* survives an
interruption.

Failure behavior
----------------
If writing or replacing raises, the temporary file is removed and the
original target is left untouched. The caller keeps its own error handling.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


def atomic_write_text(path: str | Path, content: str) -> None:
    """Write ``content`` to ``path`` atomically.

    The file is written to a unique temporary sibling, flushed, fsynced, and
    moved over the destination with :func:`os.replace`. On any failure the
    temporary file is removed and the destination is left unchanged.

    Raises the underlying :class:`OSError` (for example when the directory is
    not writable); callers decide how to surface that.
    """
    target = Path(path)
    directory = target.parent
    directory.mkdir(parents=True, exist_ok=True)

    fd, tmp_name = tempfile.mkstemp(
        dir=directory, prefix=f"{target.name}.part-", suffix=".tmp"
    )
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, target)
    except BaseException:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise


def sweep_stale_partials(directory: str | Path) -> int:
    """Remove leftover partial writes from an interrupted process.

    ``atomic_write_text`` names partials ``<canonical>.part-*.tmp``; this
    removes every such file under ``directory`` and returns the count
    removed. Safe to call idempotently (for example from ``clear()``).
    """
    removed = 0
    directory_path = Path(directory)
    if not directory_path.is_dir():
        return 0
    for entry in directory_path.glob("*.part-*.tmp"):
        try:
            entry.unlink()
            removed += 1
        except OSError:
            pass
    return removed