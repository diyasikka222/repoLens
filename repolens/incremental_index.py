"""Incremental repository indexing with persistent parser output.

Rebuilding a :class:`repolens.context.ContextEngine` currently re-parses every
Python file several times over — once for the symbol index, once for lexical
search records, once for semantic search, and once for the dependency graph.
This module makes that work incremental: per-file :class:`ModuleAnalysis`
output is persisted to a repository-scoped, stable serialized form, keyed by
the file's content hash. On a rebuild, an unchanged file's analysis is loaded
instead of re-parsed; only changed, new, or previously-unseen files are
parsed.

The result is a single immutable snapshot (:class:`RepositoryIndex`) of the
current repository — per-file ``(path, content_hash, analysis, source)`` plus
precomputed symbols and index statistics — that SymbolIndex, DependencyGraph,
CodeSearcher, and SemanticSearcher can be built from without re-parsing.

No arbitrary Python objects are pickled. Each analysis is written as a stable
JSON object and rehydrated into the frozen parser dataclasses, so the cache is
portable across processes and safe to invalidate by content hash and schema
version.

Repository isolation reuses the same scoping principle as the persistent
embedding cache: each repository gets its own subdirectory derived from a
hash of its resolved root path.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

from repolens.embedding_cache import repository_identity
from repolens.index import Symbol, SymbolIndexBuilder, SymbolKind
from repolens.parser import Argument, Class, FromImport, Function, Import, Method, ModuleAnalysis, PythonParser
from repolens.scanner import RepositoryScanner

logger = logging.getLogger("repolens.incremental_index")

#: Cache schema version. Bump to invalidate all existing cached analyses when
#: the serialized format or any representation it feeds changes.
CACHE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class IndexStats:
    """Counters describing a single incremental-index build."""

    files_discovered: int = 0
    files_parsed: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    files_removed: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "files_discovered": self.files_discovered,
            "files_parsed": self.files_parsed,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "files_removed": self.files_removed,
        }


@dataclass(frozen=True)
class IndexedFile:
    """Parsed/indexable information for one repository file."""

    path: Path
    content_hash: str
    analysis: ModuleAnalysis
    source: str


@dataclass(frozen=True)
class RepositoryIndex:
    """An immutable snapshot of a repository's parseable Python files.

    Consumers (SymbolIndex, DependencyGraph, CodeSearcher, SemanticSearcher)
    build from this single snapshot instead of re-scanning and re-parsing the
    repository themselves.
    """

    root: Path
    files: tuple[Path, ...]
    by_path: dict[Path, IndexedFile]
    symbols: tuple[Symbol, ...]
    stats: IndexStats

    def analysis_for(self, path: Path) -> ModuleAnalysis:
        return self.by_path[path].analysis

    def source_for(self, path: Path) -> str:
        return self.by_path[path].source


def content_hash(source: str) -> str:
    """Return a stable identity for one file's source text."""
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Stable serialization of ModuleAnalysis (no pickling of live objects)
# ---------------------------------------------------------------------------


def _serialize_analysis(analysis: ModuleAnalysis) -> dict:
    """Return a JSON-safe dict for an analysis. ``file_path`` is intentionally
    excluded (it is derived from the cache key, not stored)."""
    return {
        "imports": [
            {"module": i.module, "alias": i.alias, "line": i.line}
            for i in analysis.imports
        ],
        "from_imports": [
            {
                "module": f.module,
                "name": f.name,
                "alias": f.alias,
                "line": f.line,
                "level": f.level,
            }
            for f in analysis.from_imports
        ],
        "classes": [
            {
                "name": c.name,
                "base_classes": list(c.base_classes),
                "methods": [
                    {
                        "name": m.name,
                        "arguments": [
                            {"name": a.name, "annotation": a.annotation}
                            for a in m.arguments
                        ],
                        "parent_class": m.parent_class,
                        "line": m.line,
                    }
                    for m in c.methods
                ],
                "line": c.line,
            }
            for c in analysis.classes
        ],
        "functions": [
            {
                "name": f.name,
                "arguments": [
                    {"name": a.name, "annotation": a.annotation}
                    for a in f.arguments
                ],
                "line": f.line,
            }
            for f in analysis.functions
        ],
    }


def _deserialize_analysis(data: dict) -> ModuleAnalysis:
    """Rehydrate a :class:`ModuleAnalysis` from a serialized dict."""
    return ModuleAnalysis(
        file_path=None,
        imports=[
            Import(module=i["module"], alias=i.get("alias"), line=i.get("line"))
            for i in data.get("imports", [])
        ],
        from_imports=[
            FromImport(
                module=f["module"],
                name=f["name"],
                alias=f.get("alias"),
                line=f.get("line"),
                level=f.get("level", 0),
            )
            for f in data.get("from_imports", [])
        ],
        classes=[
            Class(
                name=c["name"],
                base_classes=list(c.get("base_classes", [])),
                methods=[
                    Method(
                        name=m["name"],
                        arguments=[
                            Argument(name=a["name"], annotation=a.get("annotation"))
                            for a in m.get("arguments", [])
                        ],
                        parent_class=m.get("parent_class"),
                        line=m.get("line"),
                    )
                    for m in c.get("methods", [])
                ],
                line=c.get("line"),
            )
            for c in data.get("classes", [])
        ],
        functions=[
            Function(
                name=f["name"],
                arguments=[
                    Argument(name=a["name"], annotation=a.get("annotation"))
                    for a in f.get("arguments", [])
                ],
                line=f.get("line"),
            )
            for f in data.get("functions", [])
        ],
    )


# ---------------------------------------------------------------------------
# Persistent filesystem-backed analysis cache
# ---------------------------------------------------------------------------


class AnalysisCache:
    """A repository-scoped persistent store of parsed :class:`ModuleAnalysis`.

    One JSON file per repository file, named by hash of ``(path,
    content_hash, schema_version)``. The stored payload records the content
    hash and schema version so an incompatible or tampered entry is ignored
    as a miss.
    """

    def __init__(self, directory: Path, *, persist: bool = True) -> None:
        self._directory = directory
        self._persist = persist
        self._mem: dict[str, ModuleAnalysis] = {}
        if persist:
            self._directory.mkdir(parents=True, exist_ok=True)

    @property
    def directory(self) -> Path:
        return self._directory

    @property
    def persist(self) -> bool:
        return self._persist

    def _entry_path(self, path: str, hash_: str) -> Path:
        key = f"{path}|{hash_}|{CACHE_SCHEMA_VERSION}".encode("utf-8")
        digest = hashlib.sha256(key).hexdigest()
        return self._directory / f"{digest}.json"

    def lookup(self, path: str, hash_: str) -> ModuleAnalysis | None:
        key = f"{path}|{hash_}|{CACHE_SCHEMA_VERSION}"
        if not self._persist:
            return self._mem.get(key)
        entry = self._entry_path(path, hash_)
        try:
            payload = json.loads(entry.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except (OSError, ValueError, UnicodeDecodeError):
            logger.warning("index cache entry unreadable %s: treated as miss", entry.name)
            return None
        if (
            payload.get("schema") != CACHE_SCHEMA_VERSION
            or payload.get("content_hash") != hash_
            or not isinstance(payload.get("analysis"), dict)
        ):
            logger.warning("index cache entry incompatible %s: treated as miss", entry.name)
            return None
        try:
            return _deserialize_analysis(payload["analysis"])
        except (KeyError, TypeError, ValueError):
            logger.warning("index cache entry malformed %s: treated as miss", entry.name)
            return None

    def store(self, path: str, hash_: str, analysis: ModuleAnalysis) -> None:
        key = f"{path}|{hash_}|{CACHE_SCHEMA_VERSION}"
        if not self._persist:
            self._mem[key] = analysis
            return
        payload = {
            "schema": CACHE_SCHEMA_VERSION,
            "path": path,
            "content_hash": hash_,
            "analysis": _serialize_analysis(analysis),
        }
        entry = self._entry_path(path, hash_)
        try:
            entry.write_text(json.dumps(payload), encoding="utf-8")
        except OSError:
            logger.warning("failed to persist index cache entry %s", entry.name)

    def clear(self) -> None:
        if not self._persist:
            self._mem.clear()
            return
        for entry in self._directory.glob("*.json"):
            try:
                entry.unlink()
            except OSError:
                logger.warning("failed to delete index cache entry %s", entry.name)


# ---------------------------------------------------------------------------
# Incremental index builder
# ---------------------------------------------------------------------------


class IncrementalIndexBuilder:
    """Build an incremental :class:`RepositoryIndex` for a repository.

    Scans the repository, then for each Python file decides whether a cached
    :class:`ModuleAnalysis` can be reused based on its content hash. Unchanged
    files are loaded from the analysis cache (no parser invocation); changed,
    new, or previously-unseen files are parsed and persisted. Files that no
    longer exist but are cached are dropped, so stale records never appear.
    """

    def __init__(
        self,
        root: Path | str,
        *,
        cache_dir: Path | str | None = None,
        parser: PythonParser | None = None,
        scanner: RepositoryScanner | None = None,
        persist: bool = True,
    ) -> None:
        self.root = Path(root)
        self._cache = AnalysisCache(
            self._resolve_cache_dir(cache_dir), persist=persist
        )
        self._parser = parser or PythonParser()
        self._scanner = scanner or RepositoryScanner(self.root)

    def _resolve_cache_dir(self, cache_dir: Path | str | None) -> Path:
        if cache_dir is not None:
            return Path(cache_dir)
        base = Path(home_cache_base())
        return base / "index" / repository_identity(self.root)

    def build(self) -> RepositoryIndex:
        """Scan and incrementally parse the repository into a snapshot."""
        files = self._scanner.discover_python_files()
        discovered = len(files)
        parsed = 0
        cache_hits = 0
        cache_misses = 0

        by_path: dict[Path, IndexedFile] = {}
        symbol_lists: dict[Path, list[Symbol]] = {}
        for path in files:
            source = _read_fallible(self.root / path)
            hash_ = content_hash(source)
            analysis = self._cache.lookup(path.as_posix(), hash_)
            if analysis is None:
                parsed += 1
                cache_misses += 1
                analysis = self._parser.parse_source(source, file_path=path)
                self._cache.store(path.as_posix(), hash_, analysis)
            else:
                cache_hits += 1
            by_path[path] = IndexedFile(path=path, content_hash=hash_, analysis=analysis, source=source)
            symbol_lists[path] = list(_symbols_in(path, analysis))

        # Drop cache entries for files no longer present in the repository.
        files_removed = self._prune_stale(files)

        symbols = tuple(
            sorted(
                (s for path in files for s in symbol_lists[path]),
                key=_sort_key,
            )
        )
        stats = IndexStats(
            files_discovered=discovered,
            files_parsed=parsed,
            cache_hits=cache_hits,
            cache_misses=cache_misses,
            files_removed=files_removed,
        )
        return RepositoryIndex(
            root=self.root,
            files=tuple(files),
            by_path=by_path,
            symbols=symbols,
            stats=stats,
        )

    def _prune_stale(self, files: list[Path]) -> int:
        """Remove cached entries for files no longer discovered. Returns count."""
        known_names = {path.as_posix() for path in files}
        removed = 0
        for entry in self._cache.directory.glob("*.json"):
            try:
                payload = json.loads(entry.read_text(encoding="utf-8"))
            except (OSError, ValueError, UnicodeDecodeError):
                continue
            stored_path = payload.get("path")
            if stored_path is None:
                # Older entries without an embedded path afford no safe check;
                # leave the pruning to content-hash lookups (misses) instead.
                continue
            if stored_path not in known_names:
                try:
                    entry.unlink()
                    removed += 1
                except OSError:
                    logger.warning("failed to prune index cache entry %s", entry.name)
        return removed


def _read_fallible(path: Path) -> str:
    try:
        return (path).read_text(encoding="utf-8")
    except (OSError, ValueError):
        return ""


def _symbols_in(path: Path, analysis: ModuleAnalysis) -> list[Symbol]:
    symbols: list[Symbol] = []
    for function in analysis.functions:
        symbols.append(
            Symbol(
                name=function.name,
                kind=SymbolKind.FUNCTION,
                file_path=path,
                line=function.line,
            )
        )
    for class_ in analysis.classes:
        symbols.append(
            Symbol(
                name=class_.name,
                kind=SymbolKind.CLASS,
                file_path=path,
                line=class_.line,
            )
        )
        for method in class_.methods:
            symbols.append(
                Symbol(
                    name=method.name,
                    kind=SymbolKind.METHOD,
                    file_path=path,
                    line=method.line,
                    parent_class=class_.name,
                )
            )
    return symbols


def _sort_key(symbol: Symbol) -> tuple[str, int, str]:
    return (
        symbol.file_path.as_posix(),
        symbol.line if symbol.line is not None else 0,
        symbol.name,
    )


def home_cache_base() -> Path:
    """Return the base RepoLens cache directory for indexes.

    Uses ``REPOLENS_CACHE_DIR`` for override parity with the embedding cache,
    else the platform cache directory.
    """
    override = os.environ.get("REPOLENS_CACHE_DIR")
    if override:
        return Path(override).expanduser()
    return Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "repolens"