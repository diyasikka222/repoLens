"""Persistent embedding cache abstraction and filesystem implementation.

Semantic retrieval embeds candidate documents with an
:class:`repolens.embeddings.EmbeddingProvider`. Each document is derived from
a repository file's *content* plus its identifiers; for a stable provider the
vector for a given document is fully determined by that document. That makes
document vectors cacheable across process/MCP restarts, so a restarted server
can reuse saved vectors instead of re-embedding unchanged files.

The :class:`EmbeddingCache` protocol is deliberately small and independent of
any concrete storage. A cache is keyed by three pieces of *identity*:

- the repository-relative file path (occurs per document);
- a content hash (so a modified source file invalidates its entry);
- an embedding identity (so switching model/provider never reuses stale
  vectors).

A vector is returned only when all three match; otherwise the entry behaves
as a cache miss. Corrupt, truncated, or incompatible entries are treated as
misses rather than errors, so a bad cache never breaks retrieval.

Only repository *document* vectors are persisted. Query vectors are never
stored (see requirement 5), so there is no API to persist them here.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Protocol

from repolens.atomic_write import atomic_write_text, sweep_stale_partials
from repolens.embeddings import Vector

logger = logging.getLogger("repolens.embedding_cache")


class EmbeddingCache(Protocol):
    """Minimal cache interface shared by persistent and in-memory caches.

    Implementations map ``(path, content_identity, embedding_identity)`` to a
    vector. ``content_identity`` and ``embedding_identity`` are opaque strings
    supplied by the caller; distinct values must never yield a reused vector.
    """

    def lookup(
        self,
        path: str,
        content_identity: str,
        embedding_identity: str,
    ) -> Vector | None:
        """Return the cached vector if present and valid, else ``None``."""
        ...

    def store(
        self,
        path: str,
        content_identity: str,
        embedding_identity: str,
        vector: Vector,
    ) -> None:
        """Persist ``vector`` under the given identities."""
        ...

    def clear(self) -> None:
        """Remove all cached vectors held by this cache."""
        ...


# ---------------------------------------------------------------------------
# Identity helpers
# ---------------------------------------------------------------------------


def content_identity(bytes_: bytes) -> str:
    """Return a stable identity for source bytes (content hash)."""
    return hashlib.sha256(bytes_).hexdigest()


def normalize_embedding_identity(value: object) -> str:
    """Return a stable string identity for an embedding provider/model.

    Provider objects expose their model and dimensionality in different ways;
    this folds the known attributes into a canonical, reproducible string. An
    unknown provider yields a deterministic fallback tag so the cache stays
    namespaced by what it can observe.
    """
    parts: list[str] = []
    for attr in ("_model", "model", "dimensions", "_dimensions"):
        if hasattr(value, attr):
            raw = getattr(value, attr)
            if isinstance(raw, (str, int)):
                parts.append(f"{attr}={raw}")
    if not parts:
        parts.append(f"type={type(value).__name__}")
    return hashlib.sha256(",".join(parts).encode("utf-8")).hexdigest()


def default_cache_dir(repo_root: str | Path | None = None) -> Path:
    """Return the RepoLens-controlled cache root for a repository.

    Uses ``REPOLENS_CACHE_DIR`` when set, else the platform cache directory
    (``~/.cache/repolens/embeddings`` on macOS/Linux). When ``repo_root`` is
    given, appends a stable, deterministic repository id so caches for
    different repositories never collide. The directory is always outside the
    source files being indexed.
    """
    override = os.environ.get("REPOLENS_CACHE_DIR")
    base = Path(override).expanduser() if override else (
        Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
        / "repolens"
        / "embeddings"
    )
    if repo_root is None:
        return base
    repo_id = repository_identity(repo_root)
    return base / repo_id


def repository_identity(repo_root: str | Path) -> str:
    """Return a deterministic id for a repository root path.

    The absolute, resolved path is hashed so the id is short, filesystem-safe,
    and identical for the same repository on the same machine.
    """
    resolved = str(Path(repo_root).expanduser().resolve())
    return hashlib.sha256(resolved.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Filesystem-backed cache
# ---------------------------------------------------------------------------


class FileSystemEmbeddingCache:
    """A persistent, bytes-on-disk :class:`EmbeddingCache`.

    Each document vector is stored as a single JSON file whose name is derived
    from a hash of the identity triple ``(path, content_identity,
    embedding_identity)``. Storing one file per entry keeps individual
    deletions and invalidation cheap and avoids reading a large index blob for
    a single lookup.

    Writes are atomic (temporary file + flush + :func:`os.replace`), so an
    interrupted process never leaves a half-written entry behind; the worst
    case is a stale ``.part-*.tmp`` sibling that readers ignore and
    :meth:`clear` sweeps.

    Reads fail gracefully: a missing file, a file that fails to decode, or an
    entry with a mismatched identity triple is treated as a cache miss and a
    debug log line is emitted — never an exception.
    """

    def __init__(self, directory: str | Path) -> None:
        self._directory = Path(directory)
        self._directory.mkdir(parents=True, exist_ok=True)

    @property
    def directory(self) -> Path:
        return self._directory

    def _entry_path(
        self, path: str, content_identity: str, embedding_identity: str
    ) -> Path:
        key = ",".join((path, content_identity, embedding_identity)).encode("utf-8")
        digest = hashlib.sha256(key).hexdigest()
        return self._directory / f"{digest}.json"

    def lookup(
        self,
        path: str,
        content_identity: str,
        embedding_identity: str,
    ) -> Vector | None:
        entry = self._entry_path(path, content_identity, embedding_identity)
        try:
            payload = json.loads(entry.read_text(encoding="utf-8"))
        except FileNotFoundError:
            logger.debug("cache miss: %s", entry.name)
            return None
        except (OSError, ValueError, UnicodeDecodeError):
            logger.warning("cache entry unreadable %s: treated as miss", entry.name)
            return None
        if (
            payload.get("path") != path
            or payload.get("content") != content_identity
            or payload.get("embedding") != embedding_identity
            or not isinstance(payload.get("vector"), list)
        ):
            logger.warning("cache entry identity mismatch %s: treated as miss", entry.name)
            return None
        try:
            return tuple(float(x) for x in payload["vector"])
        except (TypeError, ValueError):
            logger.warning("cache entry malformed %s: treated as miss", entry.name)
            return None

    def store(
        self,
        path: str,
        content_identity: str,
        embedding_identity: str,
        vector: Vector,
    ) -> None:
        payload = {
            "path": path,
            "content": content_identity,
            "embedding": embedding_identity,
            "vector": list(vector),
        }
        entry = self._entry_path(path, content_identity, embedding_identity)
        try:
            atomic_write_text(entry, json.dumps(payload))
        except OSError:
            logger.warning("failed to persist cache entry %s", entry.name)

    def clear(self) -> None:
        """Delete all cache entries (and any stale partial writes) under this
        cache's directory."""
        for entry in self._directory.glob("*.json"):
            try:
                entry.unlink()
            except OSError:
                logger.warning("failed to delete cache entry %s", entry.name)
        try:
            sweep_stale_partials(self._directory)
        except OSError:
            logger.warning("failed to sweep stale partials in %s", self._directory)


def make_repo_cache(
    repo_root: str | Path,
    directory: str | Path | None = None,
) -> FileSystemEmbeddingCache:
    """Create a persistent cache scoped to a repository.

    When ``directory`` is given it is used verbatim (useful for tests and
    explicit control). Otherwise the default RepoLens cache location for
    ``repo_root`` is used (see :func:`default_cache_dir`).
    """
    target = Path(directory) if directory is not None else default_cache_dir(repo_root)
    return FileSystemEmbeddingCache(target)