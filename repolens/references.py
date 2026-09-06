"""Cross-file reference extraction (Milestone 22).

A conservative, fully-offline pass that records *what names a file uses* by
walking each file's AST. Unlike :class:`repolens.parser.ModuleAnalysis` (which
records structure — imports, classes, functions, methods), this layer records
usage: name loads, attribute loads, and call targets, plus the deterministic
hints (variable assignments and typed parameters) that let the call-graph
resolver establish a receiver's type *when static evidence supports it*.

Nothing is executed, nothing is imported, and nothing is guessed:

- a call whose target cannot be resolved stays *unresolved* (recorded, never
  fabricated);
- only names used in repository files produce records;
- the output is deterministic (sorted) and reuses the existing parser and
  incremental-index snapshot (``source`` + content hashes), so unchanged files
  are never re-parsed.

The extracted records are keyed per file by a sha256 content hash (mirroring
:class:`repolens.incremental_index.AnalysisCache`), so a warm rebuild parses
only the files that changed.
"""

from __future__ import annotations

import ast
import hashlib
import json
import logging
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from repolens import diagnostics
from repolens.atomic_write import atomic_write_text, sweep_stale_partials
from repolens.embedding_cache import repository_identity
from repolens.incremental_index import (
    IndexedFile,
    IncrementalIndexBuilder,
    RepositoryIndex,
    home_cache_base,
)

logger = logging.getLogger("repolens.references")

#: Cache schema version. Bump to invalidate all cached reference entries when
#: the serialized format (or anything the resolver feeds on) changes.
REFERENCE_CACHE_SCHEMA_VERSION = 1

#: Fixed evidence tags attached to extracted reference records. They describe
#: the AST pattern that produced the record so downstream consumers can weigh
#: call evidence differently from a bare attribute load.
EVIDENCE_AST_CALL = "ast_call"
EVIDENCE_AST_NAME = "ast_name"
EVIDENCE_AST_ATTRIBUTE = "ast_attribute"
EVIDENCE_BASE_CLASS = "base_class"


class ReferenceKind(str, Enum):
    """The kind of AST pattern a single reference record represents.

    Only *usage* patterns are extracted here. Import relationships are not
    re-extracted: they already live in :class:`ModuleAnalysis` (the single
    source of truth for imports), and the call-graph resolver reads them from
    there.
    """

    CALL = "call"
    NAME = "name"
    ATTRIBUTE = "attribute"


@dataclass(frozen=True)
class Reference:
    """One deterministic reference record extracted from a file's AST.

    Attributes:
        source_symbol: ``"FunctionName"``, ``"ClassName.method"``, or ``None``
            for module-level code. Nested callables are dot-joined.
        name: The name as written: a bare identifier (``"validate"``), a
            dotted attribute/call path (``"payments.charge_card"``), or a
            base-class reference (``"models.Order"``).
        kind: Which AST pattern produced the record.
        line: Source line of the reference, if known.
        evidence: Stable tags describing how the record was found.
    """

    source_symbol: str | None
    name: str
    kind: ReferenceKind
    line: int | None = None
    evidence: tuple[str, ...] = ()


@dataclass(frozen=True)
class AssignmentHint:
    """A statically-established ``name = <CallOrName>`` binding.

    Lets the resolver type a receiver: ``cart = models.Cart()``, or — for
    methods — ``self.cart = models.Cart()`` (recorded with the full method
    scope so the resolver can attribute it to the owning class).

    Attributes:
        scope: The enclosing callable (``"make_cart"`` or
            ``"CartController.__init__"``), or ``None`` at module level.
        name: The bound name (``"cart"``) or instance attribute
            (``"self.cart"``).
        target: The dotted call/name being bound (``"models.Cart"``).
        line: Source line, if known.
    """

    scope: str | None
    name: str
    target: str
    line: int | None = None


@dataclass(frozen=True)
class ParameterTypeHint:
    """A statically-established parameter annotation (``order: Order``).

    Lets the resolver type receivers used inside a callable
    (``order.total()`` → ``Order.total``).
    """

    scope: str | None
    name: str
    annotation: str
    line: int | None = None


@dataclass(frozen=True)
class ExtractedReferences:
    """The per-file result of a reference-extraction pass.

    All three collections are sorted and deduplicated so extraction output is
    deterministic regardless of AST traversal order.
    """

    references: tuple[Reference, ...] = ()
    assignments: tuple[AssignmentHint, ...] = ()
    parameter_types: tuple[ParameterTypeHint, ...] = ()


@dataclass(frozen=True)
class FileReferences:
    """Reference records for one repository file, tied to its content hash."""

    path: Path
    content_hash: str
    references: tuple[Reference, ...] = ()
    assignments: tuple[AssignmentHint, ...] = ()
    parameter_types: tuple[ParameterTypeHint, ...] = ()


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


class ReferenceExtractor:
    """Extract usage references (and type hints) from a Python source string."""

    def extract(self, source: str) -> ExtractedReferences:
        """Walk ``source`` and return deterministic, deduplicated records."""
        try:
            tree = ast.parse(source)
        except (SyntaxError, ValueError, TypeError):
            return ExtractedReferences()

        references: list[Reference] = []
        assignments: list[AssignmentHint] = []
        parameter_types: list[ParameterTypeHint] = []
        self._walk(tree, None, (), references, assignments, parameter_types)
        return ExtractedReferences(
            references=tuple(sorted(set(references), key=_reference_key)),
            assignments=tuple(sorted(set(assignments), key=_hint_key)),
            parameter_types=tuple(sorted(set(parameter_types), key=_hint_key)),
        )

    def _walk(
        self,
        node: ast.AST,
        parent: ast.AST | None,
        scope_parts: tuple[str, ...],
        references: list[Reference],
        assignments: list[AssignmentHint],
        parameter_types: list[ParameterTypeHint],
    ) -> None:
        scope = ".".join(scope_parts) if scope_parts else None

        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            nested_scope = (*scope_parts, node.name)
            for arg in _parameters(node.args):
                if arg is None:
                    continue
                annotation = _dotted(arg.annotation)
                if annotation is not None and arg.arg != "self":
                    parameter_types.append(
                        ParameterTypeHint(
                            scope=".".join(nested_scope),
                            name=arg.arg,
                            annotation=annotation,
                            line=node.lineno,
                        )
                    )
            for child in node.body:
                self._walk(
                    child, node, nested_scope, references, assignments, parameter_types
                )
            return

        if isinstance(node, ast.ClassDef):
            nested_scope = (*scope_parts, node.name)
            for base in node.bases:
                dotted = _dotted(base)
                if dotted is None:
                    continue
                kind = (
                    ReferenceKind.NAME
                    if "." not in dotted
                    else ReferenceKind.ATTRIBUTE
                )
                references.append(
                    Reference(
                        source_symbol=scope,
                        name=dotted,
                        kind=kind,
                        line=node.lineno,
                        evidence=(EVIDENCE_BASE_CLASS,),
                    )
                )
            for child in node.body:
                self._walk(
                    child, node, nested_scope, references, assignments, parameter_types
                )
            return

        if isinstance(node, ast.Assign):
            value_node = node.value
            if isinstance(value_node, ast.Call):
                value_node = value_node.func
            dotted_value = _dotted(value_node)
            if dotted_value is None:
                return
            for target in node.targets:
                if isinstance(target, ast.Name):
                    assignments.append(
                        AssignmentHint(scope, target.id, dotted_value, node.lineno)
                    )
                elif (
                    isinstance(target, ast.Attribute)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "self"
                    and scope
                ):
                    assignments.append(
                        AssignmentHint(
                            scope, f"self.{target.attr}", dotted_value, node.lineno
                        )
                    )

        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            if not _name_hidden_by_parent(node, parent):
                references.append(
                    Reference(scope, node.id, ReferenceKind.NAME, node.lineno)
                )

        if isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load):
            if isinstance(parent, ast.Attribute):
                pass  # not the outermost attribute of its chain
            elif isinstance(parent, ast.Call) and parent.func is node:
                pass  # covered by the call record below
            else:
                dotted = _dotted(node)
                if dotted is not None:
                    references.append(
                        Reference(
                            scope,
                            dotted,
                            ReferenceKind.ATTRIBUTE,
                            node.lineno,
                            (EVIDENCE_AST_ATTRIBUTE,),
                        )
                    )

        if isinstance(node, ast.Call):
            dotted = _dotted(node.func)
            if dotted is not None:
                references.append(
                    Reference(scope, dotted, ReferenceKind.CALL, node.lineno)
                )

        for child in ast.iter_child_nodes(node):
            self._walk(
                child, node, scope_parts, references, assignments, parameter_types
            )


def _parameters(arguments: ast.arguments):
    return [
        *arguments.posonlyargs,
        *arguments.args,
        *arguments.kwonlyargs,
        arguments.vararg,
        arguments.kwarg,
    ]


def _dotted(node: ast.AST | None) -> str | None:
    """Return a dotted name for Name/Attribute chains; ``None`` otherwise."""
    if node is None:
        return None
    if isinstance(node, ast.Name):
        return node.id
    parts: list[str] = []
    current = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
        return ".".join(reversed(parts))
    return None


def _name_hidden_by_parent(node: ast.Name, parent: ast.AST | None) -> bool:
    if parent is None:
        return False
    if isinstance(parent, ast.Attribute) and parent.value is node:
        return True
    if isinstance(parent, ast.Call) and parent.func is node:
        return True
    if isinstance(parent, ast.ClassDef) and any(base is node for base in parent.bases):
        return True
    return False


def _reference_key(ref: Reference) -> tuple:
    return (
        ref.source_symbol or "",
        ref.name,
        ref.kind.value,
        ref.line if ref.line is not None else 0,
        tuple(ref.evidence),
    )


def _hint_key(hint) -> tuple:
    return (
        hint.scope or "",
        hint.name,
        getattr(hint, "annotation", "") or getattr(hint, "target", ""),
        hint.line if hint.line is not None else 0,
    )


# ---------------------------------------------------------------------------
# Stable serialization
# ---------------------------------------------------------------------------


def _serialize_reference(ref: Reference) -> dict:
    return {
        "source_symbol": ref.source_symbol,
        "name": ref.name,
        "kind": ref.kind.value,
        "line": ref.line,
        "evidence": list(ref.evidence),
    }


def _deserialize_reference(data: dict) -> Reference:
    return Reference(
        source_symbol=data.get("source_symbol"),
        name=data["name"],
        kind=ReferenceKind(data["kind"]),
        line=data.get("line"),
        evidence=tuple(data.get("evidence", [])),
    )


def _serialize_hint(hint) -> dict:
    if isinstance(hint, AssignmentHint):
        return {
            "type": "assignment",
            "scope": hint.scope,
            "name": hint.name,
            "target": hint.target,
            "line": hint.line,
        }
    return {
        "type": "parameter",
        "scope": hint.scope,
        "name": hint.name,
        "annotation": hint.annotation,
        "line": hint.line,
    }


def _deserialize_hint(data: dict):
    common = dict(
        scope=data.get("scope"),
        name=data["name"],
        line=data.get("line"),
    )
    if data.get("type") == "assignment":
        return AssignmentHint(target=data["target"], **common)
    return ParameterTypeHint(annotation=data["annotation"], **common)


# ---------------------------------------------------------------------------
# Persistent per-file reference cache
# ---------------------------------------------------------------------------


class ReferenceCache:
    """Repository-scoped persistent store of extracted :class:`FileReferences`.

    Mirrors :class:`repolens.incremental_index.AnalysisCache`: one JSON file
    per repository file keyed by hash of ``(path, content_hash, schema)``, with
    atomic writes and tolerant reads (unreadable/incompatible entries are a
    miss, never an error).
    """

    def __init__(self, directory: Path, *, persist: bool = True) -> None:
        self._directory = directory
        self._persist = persist
        self._mem: dict[str, FileReferences] = {}
        if persist:
            self._directory.mkdir(parents=True, exist_ok=True)

    @property
    def directory(self) -> Path:
        return self._directory

    @property
    def persist(self) -> bool:
        return self._persist

    def _entry_path(self, path: str, hash_: str) -> Path:
        key = f"{path}|{hash_}|{REFERENCE_CACHE_SCHEMA_VERSION}".encode("utf-8")
        digest = hashlib.sha256(key).hexdigest()
        return self._directory / f"{digest}.json"

    def lookup(self, path: str, hash_: str) -> FileReferences | None:
        key = f"{path}|{hash_}|{REFERENCE_CACHE_SCHEMA_VERSION}"
        if not self._persist:
            return self._mem.get(key)
        entry = self._entry_path(path, hash_)
        try:
            payload = json.loads(entry.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except (OSError, ValueError, UnicodeDecodeError):
            logger.warning(
                "reference cache entry unreadable %s: treated as miss", entry.name
            )
            return None
        if (
            payload.get("schema") != REFERENCE_CACHE_SCHEMA_VERSION
            or payload.get("content_hash") != hash_
        ):
            logger.warning(
                "reference cache entry incompatible %s: treated as miss", entry.name
            )
            return None
        try:
            return FileReferences(
                path=Path(payload["path"]),
                content_hash=payload["content_hash"],
                references=tuple(
                    _deserialize_reference(r)
                    for r in payload.get("references", [])
                ),
                assignments=tuple(
                    _deserialize_hint(h)
                    for h in payload.get("assignments", [])
                ),
                parameter_types=tuple(
                    _deserialize_hint(h)
                    for h in payload.get("parameter_types", [])
                ),
            )
        except (KeyError, TypeError, ValueError):
            logger.warning(
                "reference cache entry malformed %s: treated as miss", entry.name
            )
            return None

    def store(self, path: str, hash_: str, refs: FileReferences) -> None:
        key = f"{path}|{hash_}|{REFERENCE_CACHE_SCHEMA_VERSION}"
        if not self._persist:
            self._mem[key] = refs
            return
        payload = {
            "schema": REFERENCE_CACHE_SCHEMA_VERSION,
            "path": path,
            "content_hash": hash_,
            "references": [_serialize_reference(r) for r in refs.references],
            "assignments": [_serialize_hint(h) for h in refs.assignments],
            "parameter_types": [
                _serialize_hint(h) for h in refs.parameter_types
            ],
        }
        entry = self._entry_path(path, hash_)
        try:
            atomic_write_text(entry, json.dumps(payload))
        except OSError:
            logger.warning("failed to persist reference cache entry %s", entry.name)

    def clear(self) -> None:
        if not self._persist:
            self._mem.clear()
            return
        for entry in self._directory.glob("*.json"):
            try:
                entry.unlink()
            except OSError:
                logger.warning("failed to delete reference cache entry %s", entry.name)
        try:
            sweep_stale_partials(self._directory)
        except OSError:
            logger.warning("failed to sweep stale partials in %s", self._directory)


# ---------------------------------------------------------------------------
# Index builder
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReferenceStats:
    """Counters describing a single reference-index build."""

    files_discovered: int = 0
    files_scanned: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    files_removed: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "files_discovered": self.files_discovered,
            "files_scanned": self.files_scanned,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "files_removed": self.files_removed,
        }


@dataclass(frozen=True)
class ReferenceIndex:
    """An immutable snapshot of the repository's extracted references."""

    root: Path
    files: tuple[Path, ...]
    by_path: dict[Path, FileReferences]
    stats: ReferenceStats

    def references_for(self, path: Path) -> FileReferences:
        return self.by_path[path]


class ReferenceIndexBuilder:
    """Build a :class:`ReferenceIndex` incrementally from an index snapshot.

    Uses the same content-hash discipline as
    :class:`repolens.incremental_index.IncrementalIndexBuilder`: unchanged
    files are restored from the persistent reference cache (no AST parse);
    only changed/new files are extracted. Stale cached entries for files that
    no longer exist are pruned.
    """

    def __init__(
        self,
        root: Path | str,
        *,
        index: RepositoryIndex | None = None,
        cache_dir: Path | str | None = None,
        extractor: ReferenceExtractor | None = None,
        persist: bool = True,
    ) -> None:
        self.root = Path(root)
        if index is None:
            index = IncrementalIndexBuilder(self.root, persist=persist).build()
        self._index: RepositoryIndex = index
        self._cache = ReferenceCache(
            self._resolve_cache_dir(cache_dir), persist=persist
        )
        self._extractor = extractor if extractor is not None else ReferenceExtractor()

    def _resolve_cache_dir(self, cache_dir: Path | str | None) -> Path:
        if cache_dir is not None:
            return Path(cache_dir)
        return home_cache_base() / "references" / repository_identity(self.root)

    def build(self) -> ReferenceIndex:
        start = time.perf_counter()
        files = list(self._index.files)
        discovered = len(files)
        scanned = 0
        hits = 0
        misses = 0

        by_path: dict[Path, FileReferences] = {}
        for path in files:
            entry: IndexedFile = self._index.by_path[path]
            cached = self._cache.lookup(path.as_posix(), entry.content_hash)
            if cached is not None:
                hits += 1
                by_path[path] = cached
                continue
            scanned += 1
            misses += 1
            extracted = self._extractor.extract(entry.source)
            refs = FileReferences(
                path=path,
                content_hash=entry.content_hash,
                references=extracted.references,
                assignments=extracted.assignments,
                parameter_types=extracted.parameter_types,
            )
            self._cache.store(path.as_posix(), entry.content_hash, refs)
            by_path[path] = refs

        removed = self._prune_stale(files)
        stats = ReferenceStats(
            files_discovered=discovered,
            files_scanned=scanned,
            cache_hits=hits,
            cache_misses=misses,
            files_removed=removed,
        )
        if diagnostics.enabled():
            diagnostics.record(
                "reference_build",
                repository=str(self.root),
                elapsed_ms=round((time.perf_counter() - start) * 1000.0, 3),
                **stats.as_dict(),
            )
        return ReferenceIndex(
            root=self.root,
            files=tuple(files),
            by_path=by_path,
            stats=stats,
        )

    def _prune_stale(self, files: list[Path]) -> int:
        known_names = {path.as_posix() for path in files}
        removed = 0
        for entry in self._cache.directory.glob("*.json"):
            try:
                payload = json.loads(entry.read_text(encoding="utf-8"))
            except (OSError, ValueError, UnicodeDecodeError):
                continue
            stored_path = payload.get("path")
            if stored_path is None or stored_path not in known_names:
                try:
                    entry.unlink()
                    removed += 1
                except OSError:
                    logger.warning(
                        "failed to prune reference cache entry %s", entry.name
                    )
        return removed