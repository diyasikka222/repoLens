"""Conservative cross-file reference/call graph (Milestone 22).

Built entirely from existing RepoLens infrastructure — the incremental index
snapshot (:class:`ModuleAnalysis` import records + source), the symbol index
(where symbols are *defined*), and the newly extracted reference records
(:mod:`repolens.references`) — this module produces a deterministic,
fully-offline call/reference graph.

The resolver is deliberately conservative. A mention (name load, attribute
load, or call) is resolved only through the following static evidence, in
order:

1. an exact local symbol (a function/class/method defined in the same file);
2. an explicitly imported symbol (``from a.b import sym``);
3. a resolved relative import (``from .validators import validate`` → an
   absolute module that exists in the repository);
4. a resolved module-qualified symbol (``import a.b; a.b.sym()``, or
   ``models.Cart``);
5. a known class/method relationship (a receiver whose type is established
   statically by ``self.attr = Cls()``, an assignment, or a typed parameter);
6. otherwise the reference is **unresolved** — recorded as such, never
   guessed.

Traversal is always bounded (:attr:`CallGraphConfig.max_transitive_nodes` and
an explicit ``max_depth`` on every transitive query); there is no unbounded
recursion anywhere. Deterministic ordering and deduplication are guaranteed.
"""

from __future__ import annotations

import builtins
from collections import deque
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from repolens.index import Symbol, SymbolIndex, SymbolIndexBuilder, SymbolKind
from repolens.incremental_index import IncrementalIndexBuilder, RepositoryIndex
from repolens.parser import FromImport, Import, ModuleAnalysis
from repolens.references import (
    FileReferences,
    Reference,
    ReferenceIndex,
    ReferenceIndexBuilder,
    ReferenceKind,
)

#: Edge evidence tags (in addition to the reference evidence produced by the
#: extractor). They record *why* an edge exists.
EVIDENCE_RESOLVED_AST_CALL = "resolved_ast_call"
EVIDENCE_IMPORT_STATEMENT = "import_statement"
EVIDENCE_FROM_IMPORT_STATEMENT = "from_import_statement"
EVIDENCE_LOCAL_SYMBOL = "local_symbol"
EVIDENCE_METHOD_RECEIVER = "method_receiver"
EVIDENCE_SELF_ATTRIBUTE = "self_attribute"
EVIDENCE_TYPED_PARAMETER = "typed_parameter"
EVIDENCE_MODULE_ALIAS = "module_alias"

#: Unresolved-reference reason tags (stable, machine-readable).
UNRESOLVED_DYNAMIC = "dynamic_call"
UNRESOLVED_NAME = "unknown_name"
UNRESOLVED_ATTRIBUTE = "unknown_attribute"
UNRESOLVED_MODULE = "unknown_module"
UNRESOLVED_RECEIVER = "unknown_receiver"
UNRESOLVED_EXTERNAL = "external_module"


class EdgeKind(str, Enum):
    """The kind of a resolved call-graph edge."""

    CALL = "call"
    METHOD_CALL = "method_call"
    INSTANTIATION = "instantiation"
    REFERENCE = "reference"
    IMPORT = "import"


class CallRelationship(str, Enum):
    """The six structured graph relationships exposed to consumers.

    Pairs map to graph directions; forward edges go from the *using* side.
    """

    CALLS = "calls"
    CALLED_BY = "called_by"
    REFERENCES = "references"
    REFERENCED_BY = "referenced_by"
    IMPORTS = "imports"
    IMPORTED_BY = "imported_by"


#: Call-like edge kinds (the source "uses" the target as a callable).
_CALL_KINDS = frozenset({EdgeKind.CALL, EdgeKind.METHOD_CALL, EdgeKind.INSTANTIATION})


def _node_key(node: "CallNode") -> tuple:
    return (
        node.file_path.as_posix(),
        node.name or "",
        node.kind.value if node.kind is not None else "",
        node.parent_class or "",
    )


def _edge_source(edge: "CallEdge") -> "CallNode":
    return edge.source


def _edge_target(edge: "CallEdge") -> "CallNode":
    return edge.target


@dataclass(frozen=True)
class CallNode:
    """A node in the call graph: a symbol (or a whole module/file).

    Attributes:
        file_path: Repository-relative file the node lives in.
        name: Symbol name, or ``None`` for a module-level node.
        kind: Symbol kind (``None`` for module-level nodes).
        parent_class: Owner class for methods.
    """

    file_path: Path
    name: str | None = None
    kind: SymbolKind | None = None
    parent_class: str | None = None

    @property
    def is_module(self) -> bool:
        return self.name is None

    @property
    def key(self) -> tuple:
        return _node_key(self)

    def display(self) -> str:
        if self.name is None:
            return self.file_path.as_posix()
        return f"{self.file_path.as_posix()}::{self.name}"

    def as_dict(self) -> dict:
        return {
            "file": self.file_path.as_posix(),
            "name": self.name,
            "kind": self.kind.value if self.kind is not None else None,
            "parent_class": self.parent_class,
        }


def module_node(path: Path) -> CallNode:
    """Return the module-level node for ``path``."""
    return CallNode(file_path=path, name=None)


@dataclass(frozen=True)
class CallEdge:
    """A directed, evidence-backed edge in the call graph.

    Attributes:
        kind: Edge kind (call / method_call / instantiation / reference /
            import).
        source: The using node.
        target: The used node.
        evidence: Stable tags describing how the edge was established.
        line: Source line of the usage, when known.
    """

    kind: EdgeKind
    source: CallNode
    target: CallNode
    evidence: tuple[str, ...] = ()
    line: int | None = None

    @property
    def relationship(self) -> CallRelationship:
        if self.kind in _CALL_KINDS:
            return CallRelationship.CALLS
        if self.kind is EdgeKind.IMPORT:
            return CallRelationship.IMPORTS
        return CallRelationship.REFERENCES

    def as_dict(self) -> dict:
        return {
            "kind": self.kind.value,
            "source": self.source.as_dict(),
            "target": self.target.as_dict(),
            "evidence": list(self.evidence),
            "line": self.line,
        }


@dataclass(frozen=True)
class UnresolvedReference:
    """A usage mention that could not be resolved conservatively.

    ``reason`` is one of the stable ``UNRESOLVED_*`` constants. These records
    represent the *known-unknowns* of the graph; they are never synthesized
    into edges.
    """

    source_file: Path
    source_symbol: str | None
    name: str
    reason: str
    line: int | None = None

    def as_dict(self) -> dict:
        return {
            "file": self.source_file.as_posix(),
            "source_symbol": self.source_symbol,
            "name": self.name,
            "reason": self.reason,
            "line": self.line,
        }


@dataclass(frozen=True)
class CallGraphConfig:
    """Bounded, deterministic limits for call-graph construction/query."""

    max_transitive_nodes: int = 5000


@dataclass(frozen=True)
class CallGraphStats:
    """Counters describing a built :class:`CallGraph`."""

    nodes: int = 0
    edges: int = 0
    calls: int = 0
    references: int = 0
    imports: int = 0
    unresolved: int = 0

    def as_dict(self) -> dict:
        return {
            "nodes": self.nodes,
            "edges": self.edges,
            "calls": self.calls,
            "references": self.references,
            "imports": self.imports,
            "unresolved": self.unresolved,
        }


# ---------------------------------------------------------------------------
# Module table / bindings
# ---------------------------------------------------------------------------


class _ModuleTable:
    """Maps dotted module names to repository files (one recipe, like graph.py)."""

    def __init__(self, files) -> None:
        self._modules: dict[str, Path] = {}
        for file_path in files:
            self._register(file_path)

    def _register(self, file_path: Path) -> None:
        if file_path.name == "__init__.py":
            parts = file_path.parts[:-1]
        else:
            parts = (*file_path.parts[:-1], file_path.stem)
        self._modules.setdefault(".".join(parts), file_path)

    def lookup(self, dotted: str) -> Path | None:
        return self._modules.get(dotted)

    def resolve(self, dotted: str) -> tuple[str, Path] | None:
        """Resolve ``dotted`` to the deepest known module: (name, file)."""
        parts = dotted.split(".") if dotted else []
        for size in range(len(parts), 0, -1):
            name = ".".join(parts[:size])
            found = self._modules.get(name)
            if found is not None:
                return name, found
        return None


@dataclass(frozen=True)
class _ModuleBinding:
    module: str
    file: Path | None


@dataclass(frozen=True)
class _SymbolBinding:
    module: str
    symbol: str
    file: Path | None


_KIND_ORDER = {
    SymbolKind.CLASS: 0,
    SymbolKind.FUNCTION: 1,
    SymbolKind.METHOD: 2,
}

_BUILTIN_NAMES = frozenset(dir(builtins)) if hasattr(builtins, "__dict__") else frozenset()


def _absolute_module(
    level: int, module: str, package_parts: tuple[str, ...]
) -> str | None:
    """Resolve a (possibly relative) import into an absolute dotted module."""
    if level == 0:
        return module or None
    depth = len(package_parts) - (level - 1)
    if depth < 0:
        return None
    base = package_parts[:depth]
    dotted = ".".join(base)
    if module:
        return f"{dotted}.{module}" if dotted else module
    return dotted or None


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------


class CallGraph:
    """A deterministic, immutable reference/call graph for one repository.

    Query methods always return deduplicated results ordered by
    ``(file, name, kind, parent_class)``; transitive queries are bounded by an
    explicit ``max_depth`` and a hard node budget.
    """

    def __init__(
        self,
        root: Path,
        nodes: list[CallNode],
        edges: list[CallEdge],
        unresolved: list[UnresolvedReference],
        config: CallGraphConfig,
    ) -> None:
        self.root = Path(root)
        self.config = config
        self._nodes = tuple(sorted(set(nodes), key=_node_key))
        self._edges = tuple(
            sorted(
                set(edges),
                key=lambda e: (
                    e.kind.value,
                    _node_key(e.source),
                    _node_key(e.target),
                    tuple(e.evidence),
                    e.line if e.line is not None else 0,
                ),
            )
        )
        self._unresolved = tuple(
            sorted(
                set(unresolved),
                key=lambda u: (
                    u.source_file.as_posix(),
                    u.source_symbol or "",
                    u.name,
                    u.reason,
                    u.line if u.line is not None else 0,
                ),
            )
        )
        self._call_out: dict[tuple, list] = {}
        self._call_in: dict[tuple, list] = {}
        self._ref_out: dict[tuple, list] = {}
        self._ref_in: dict[tuple, list] = {}
        self._import_file_out: dict[Path, list[Path]] = {}
        self._import_file_in: dict[Path, list[Path]] = {}
        for edge in self._edges:
            if edge.kind in _CALL_KINDS:
                self._call_out.setdefault(edge.source.key, []).append(edge)
                self._call_in.setdefault(edge.target.key, []).append(edge)
            elif edge.kind is EdgeKind.REFERENCE:
                self._ref_out.setdefault(edge.source.key, []).append(edge)
                self._ref_in.setdefault(edge.target.key, []).append(edge)
            elif edge.kind is EdgeKind.IMPORT:
                if edge.source.is_module and edge.target.is_module:
                    self._import_file_out.setdefault(edge.source.file_path, []).append(
                        edge.target.file_path
                    )
                    self._import_file_in.setdefault(edge.target.file_path, []).append(
                        edge.source.file_path
                    )
        self._unique_importers = {
            path: tuple(sorted(set(deps)))
            for path, deps in self._import_file_in.items()
        }

    # -- nodes --------------------------------------------------------------

    def get_nodes(self) -> list[CallNode]:
        """Return all module nodes for discovered files, sorted."""
        return list(self._nodes)

    def module_nodes(self) -> list[CallNode]:
        return [node for node in self._nodes if node.is_module]

    def symbol_node(self, symbol: Symbol) -> CallNode:
        """Return the canonical node for ``symbol`` (safe to query even when no
        edges touch it yet)."""
        return CallNode(
            file_path=symbol.file_path,
            name=symbol.name,
            kind=symbol.kind,
            parent_class=symbol.parent_class,
        )

    # -- edges --------------------------------------------------------------

    def get_edges(self) -> list[CallEdge]:
        return list(self._edges)

    def get_edges_by_relationship(self, rel: CallRelationship) -> list[CallEdge]:
        """Return edges for one of the six structured relationships."""
        if rel is CallRelationship.CALLED_BY:
            edges = [
                edge for edge in self._edges if edge.kind in _CALL_KINDS
            ]
            return [
                CallEdge(edge.kind, edge.target, edge.source, edge.evidence, edge.line)
                for edge in edges
            ]
        if rel is CallRelationship.IMPORTED_BY:
            edges = [
                edge for edge in self._edges if edge.kind is EdgeKind.IMPORT
            ]
            return [
                CallEdge(edge.kind, edge.target, edge.source, edge.evidence, edge.line)
                for edge in edges
            ]
        if rel is CallRelationship.REFERENCED_BY:
            edges = [
                edge for edge in self._edges if edge.kind is EdgeKind.REFERENCE
            ]
            return [
                CallEdge(edge.kind, edge.target, edge.source, edge.evidence, edge.line)
                for edge in edges
            ]
        if rel is CallRelationship.CALLS:
            return [edge for edge in self._edges if edge.kind in _CALL_KINDS]
        if rel is CallRelationship.IMPORTS:
            return [edge for edge in self._edges if edge.kind is EdgeKind.IMPORT]
        return [edge for edge in self._edges if edge.kind is EdgeKind.REFERENCE]

    # -- call queries -------------------------------------------------------

    def callers(self, node: CallNode, *, max_depth: int = 1) -> list[CallNode]:
        """Nodes that call ``node`` (within ``max_depth`` reverse hops)."""
        return self.bounded_transitive_callers(node, max_depth=max_depth)

    def callees(self, node: CallNode, *, max_depth: int = 1) -> list[CallNode]:
        """Nodes that ``node`` calls (within ``max_depth`` forward hops)."""
        return self.bounded_transitive_callees(node, max_depth=max_depth)

    def bounded_transitive_callers(
        self, node: CallNode, max_depth: int
    ) -> list[CallNode]:
        """BFS over reverse call edges up to ``max_depth``, deduplicated."""
        return self._reachable(self._call_in, node, max_depth)

    def bounded_transitive_callees(
        self, node: CallNode, max_depth: int
    ) -> list[CallNode]:
        """BFS over forward call edges up to ``max_depth``, deduplicated."""
        return self._reachable(self._call_out, node, max_depth, neighbor=_edge_target)

    def call_edges(self, node: CallNode) -> list[CallEdge]:
        """Forward call-like edges from ``node`` (what it calls)."""
        return sorted(
            self._call_out.get(node.key, []),
            key=lambda e: (_node_key(e.target), tuple(e.evidence)),
        )

    def called_by_edges(self, node: CallNode) -> list[CallEdge]:
        """Reverse call-like edges into ``node`` (who calls it)."""
        return sorted(
            self._call_in.get(node.key, []),
            key=lambda e: (_node_key(e.source), tuple(e.evidence)),
        )

    # -- reference queries --------------------------------------------------

    def references_to(self, node: CallNode) -> list[CallNode]:
        """Nodes with a REFERENCE edge into ``node`` (who references it)."""
        return sorted(
            {e.source for e in self._ref_in.get(node.key, [])},
            key=_node_key,
        )

    def references_from(self, node: CallNode) -> list[CallNode]:
        """Nodes that ``node`` references (its outgoing REFERENCE edges)."""
        return sorted(
            {e.target for e in self._ref_out.get(node.key, [])},
            key=_node_key,
        )

    # -- import queries -----------------------------------------------------

    def importers_of(self, path: Path) -> list[Path]:
        """Files that import ``path``'s module, sorted."""
        return list(self._unique_importers.get(path, ()))

    def imports_of(self, path: Path) -> list[Path]:
        """Modules imported by ``path``'s module, sorted."""
        targets = [
            edge.target.file_path
            for edge in self._edges
            if edge.kind is EdgeKind.IMPORT
            and edge.source.is_module
            and edge.source.file_path == path
        ]
        return sorted(set(targets), key=lambda p: p.as_posix())

    def direct_dependents(self, node: CallNode) -> list[CallNode]:
        """Files that import ``node``'s file *and* call/reference ``node``.

        A direct dependent must both pull in the module (an import edge) and
        use the specific symbol (a call or reference edge), which distinguishes
        it from a passive importer.
        """
        importers = set(self._unique_importers.get(node.file_path, ()))
        if not importers:
            return []
        incoming = [
            *(e.source for e in self._call_in.get(node.key, [])),
            *(e.source for e in self._ref_in.get(node.key, [])),
        ]
        matches = [src for src in incoming if src.file_path in importers]
        return sorted(set(matches), key=_node_key)

    # -- unresolved ---------------------------------------------------------

    def unresolved_references(self) -> list[UnresolvedReference]:
        return list(self._unresolved)

    def unresolved_in(self, path: Path) -> list[UnresolvedReference]:
        return [
            u
            for u in self._unresolved
            if u.source_file == path
        ]

    # -- stats --------------------------------------------------------------

    def stats(self) -> CallGraphStats:
        calls = sum(1 for e in self._edges if e.kind in _CALL_KINDS)
        references = sum(1 for e in self._edges if e.kind is EdgeKind.REFERENCE)
        imports = sum(1 for e in self._edges if e.kind is EdgeKind.IMPORT)
        return CallGraphStats(
            nodes=len(self._nodes),
            edges=len(self._edges),
            calls=calls,
            references=references,
            imports=imports,
            unresolved=len(self._unresolved),
        )

    # -- internals ----------------------------------------------------------

    def _reachable(
        self,
        adjacency: dict[tuple, list],
        start: CallNode,
        max_depth: int,
        *,
        neighbor: object = _edge_source,
    ) -> list[CallNode]:
        if start is None or max_depth < 1:
            return []
        distances = self._bfs(adjacency, start, max_depth, neighbor=neighbor)
        by_key = {node.key: node for node in self._nodes}
        by_key[start.key] = start
        return sorted(
            (by_key[key] for key in distances if key != start.key),
            key=_node_key,
        )

    def _bfs(
        self,
        adjacency: dict[tuple, list],
        start: CallNode,
        max_depth: int,
        neighbor: object,
    ) -> dict[tuple, int]:
        distances: dict[tuple, int] = {start.key: 0}
        queue: deque = deque([start.key])
        visited: set[tuple] = set()
        while queue and len(distances) < self.config.max_transitive_nodes:
            current = queue.popleft()
            if current in visited:
                continue
            visited.add(current)
            distance = distances[current]
            if distance >= max_depth:
                continue
            for edge in adjacency.get(current, []):
                key = neighbor(edge).key
                if key in distances:
                    continue
                if len(distances) >= self.config.max_transitive_nodes:
                    break
                distances[key] = distance + 1
                if distance + 1 < max_depth:
                    queue.append(key)
        return distances


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------


class CallGraphBuilder:
    """Build a :class:`CallGraph` incrementally from shared snapshots.

    May be handed the shared :class:`RepositoryIndex`,
    :class:`ReferenceIndex`, and :class:`SymbolIndex`; it parses no source
    itself — references are already extracted by
    :class:`ReferenceIndexBuilder` (which is incremental), so a warm build is
    pure in-memory work.
    """

    def __init__(
        self,
        root: Path | str,
        *,
        index: RepositoryIndex | None = None,
        reference_index: ReferenceIndex | None = None,
        symbol_index: SymbolIndex | None = None,
        config: CallGraphConfig | None = None,
    ) -> None:
        self.root = Path(root)
        if index is None:
            index = IncrementalIndexBuilder(self.root, persist=False).build()
        self.index: RepositoryIndex = index
        if reference_index is None:
            reference_index = ReferenceIndexBuilder(
                self.root, index=index, persist=False
            ).build()
        self.reference_index: ReferenceIndex = reference_index
        if symbol_index is None:
            symbol_index = SymbolIndexBuilder(self.root, index=index).build()
        self.symbol_index: SymbolIndex = symbol_index
        self.config = config if config is not None else CallGraphConfig()

        self._module_table = _ModuleTable(self.index.files)
        self._local_funcs: dict[Path, set[str]] = {}
        self._local_classes: dict[Path, dict[str, set[str]]] = {}
        self._symbols_by_file_name: dict[tuple, list[Symbol]] = {}
        self._bindings: dict[Path, dict[str, object]] = {}
        self._assignment_targets: dict[Path, dict[tuple, str]] = {}
        self._self_attrs: dict[Path, dict[tuple, str]] = {}
        self._typed_params: dict[Path, dict[tuple, str]] = {}

    def build(self) -> CallGraph:
        self._build_local_defs()
        self._build_symbol_lookup()
        self._build_bindings()
        self._build_hints()

        nodes: list[CallNode] = [module_node(path) for path in self.index.files]
        edges: list[CallEdge] = []
        unresolved: list[UnresolvedReference] = []

        for path in self.index.files:
            refs: FileReferences = self.reference_index.references_for(path)
            analysis: ModuleAnalysis = self.index.analysis_for(path)
            edges.extend(self._import_edges(path, analysis))
            edges.extend(self._from_import_references(path, analysis))
            for ref in refs.references:
                result = self._resolve(path, ref)
                if result is None:
                    continue
                node = result
                if ref.kind is ReferenceKind.CALL:
                    kind = self._call_edge_kind(node, ref)
                    evidence = (EVIDENCE_RESOLVED_AST_CALL, *ref.evidence)
                else:
                    kind = EdgeKind.REFERENCE
                    evidence = ref.evidence
                edges.append(
                    CallEdge(
                        kind=kind,
                        source=self._scope_node(path, ref.source_symbol),
                        target=node,
                        evidence=_unique(evidence),
                        line=ref.line,
                    )
                )
            unresolved.extend(self._unresolved_for(path, refs))

        return CallGraph(
            root=self.root,
            nodes=nodes + self._symbol_nodes(edges),
            edges=edges,
            unresolved=unresolved,
            config=self.config,
        )

    # ------------------------------------------------------------------
    # Precomputed tables
    # ------------------------------------------------------------------

    def _build_local_defs(self) -> None:
        for path in self.index.files:
            analysis = self.index.analysis_for(path)
            funcs = {f.name for f in analysis.functions}
            classes: dict[str, set[str]] = {}
            for class_ in analysis.classes:
                classes[class_.name] = {m.name for m in class_.methods}
            self._local_funcs[path] = funcs
            self._local_classes[path] = classes

    def _build_symbol_lookup(self) -> None:
        for symbol in self.symbol_index.get_all_symbols():
            self._symbols_by_file_name.setdefault(
                (symbol.file_path, symbol.name), []
            ).append(symbol)

    def _build_bindings(self) -> None:
        for path in self.index.files:
            analysis = self.index.analysis_for(path)
            binds: dict[str, object] = {}
            for imp in analysis.imports:
                binds.update(self._import_bindings(imp))
            for fi in analysis.from_imports:
                resolved = self._resolve_from_import_absolute(fi, path)
                if resolved is None or fi.name == "*":
                    continue
                module_name, module_file = resolved
                name = fi.alias or fi.name
                submodule = self._module_table.lookup(
                    f"{module_name}.{fi.name}" if module_name else fi.name
                )
                if submodule is not None:
                    # `from . import models` binds a *module* name.
                    binds[name] = _ModuleBinding(
                        module=f"{module_name}.{fi.name}" if module_name else fi.name,
                        file=submodule,
                    )
                else:
                    binds[name] = _SymbolBinding(
                        module=module_name, symbol=fi.name, file=module_file
                    )
            self._bindings[path] = binds

    def _import_bindings(self, imp: Import) -> dict[str, object]:
        resolved = self._module_table.resolve(imp.module)
        if resolved is None:
            return {}
        module_name, module_file = resolved
        if imp.alias:
            return {imp.alias: _ModuleBinding(module=module_name, file=module_file)}
        first = imp.module.split(".")[0]
        first_file = self._module_table.lookup(first)
        return {first: _ModuleBinding(module=first, file=first_file)}

    def _resolve_from_import_absolute(
        self, fi: FromImport, source: Path
    ) -> tuple[str, Path] | None:
        absolute = _absolute_module(fi.level, fi.module, source.parts[:-1])
        if not absolute:
            return None
        return self._module_table.resolve(absolute)

    def _build_hints(self) -> None:
        for path in self.index.files:
            refs = self.reference_index.references_for(path)
            assignments: dict[tuple, str] = {}
            self_attrs: dict[tuple, str] = {}
            params: dict[tuple, str] = {}
            for hint in refs.assignments:
                if hint.name.startswith("self."):
                    attr = hint.name.split(".", 1)[1]
                    class_name = (hint.scope or "").split(".")[0]
                    self_attrs[(class_name, attr)] = hint.target
                else:
                    assignments[(hint.scope or "", hint.name)] = hint.target
            for hint in refs.parameter_types:
                params[(hint.scope or "", hint.name)] = hint.annotation
            self._assignment_targets[path] = assignments
            self._self_attrs[path] = self_attrs
            self._typed_params[path] = params

    # ------------------------------------------------------------------
    # Edges from imports
    # ------------------------------------------------------------------

    def _import_edges(
        self, path: Path, analysis: ModuleAnalysis
    ) -> list[CallEdge]:
        edges: list[CallEdge] = []
        source = module_node(path)
        for imp in analysis.imports:
            resolved = self._module_table.resolve(imp.module)
            if resolved is None:
                continue
            _, module_file = resolved
            edges.append(
                CallEdge(
                    kind=EdgeKind.IMPORT,
                    source=source,
                    target=module_node(module_file),
                    evidence=(EVIDENCE_IMPORT_STATEMENT,),
                    line=imp.line,
                )
            )
        for fi in analysis.from_imports:
            resolved = self._resolve_from_import_absolute(fi, path)
            if resolved is None:
                continue
            _, module_file = resolved
            edges.append(
                CallEdge(
                    kind=EdgeKind.IMPORT,
                    source=source,
                    target=module_node(module_file),
                    evidence=(EVIDENCE_FROM_IMPORT_STATEMENT,),
                    line=fi.line,
                )
            )
        return edges

    def _from_import_references(
        self, path: Path, analysis: ModuleAnalysis
    ) -> list[CallEdge]:
        edges: list[CallEdge] = []
        source = module_node(path)
        for fi in analysis.from_imports:
            if fi.name == "*":
                continue
            binding = self._bindings[path].get(fi.alias or fi.name)
            target = self._binding_node(binding)
            if target is None:
                continue
            edges.append(
                CallEdge(
                    kind=EdgeKind.REFERENCE,
                    source=source,
                    target=target,
                    evidence=(EVIDENCE_FROM_IMPORT_STATEMENT,),
                    line=fi.line,
                )
            )
        return edges

    # ------------------------------------------------------------------
    # Resolution
    # ------------------------------------------------------------------

    def _resolve(self, path: Path, ref: Reference) -> CallNode | None:
        parts = ref.name.split(".")
        if len(parts) == 1:
            return self._resolve_simple(path, ref.source_symbol, ref.name, ref.kind)
        return self._resolve_dotted(path, ref.source_symbol, ref.name, ref.kind)

    def _resolve_simple(
        self,
        path: Path,
        scope: str | None,
        name: str,
        kind: ReferenceKind,
        *,
        depth: int = 8,
    ) -> CallNode | None:
        if depth <= 0:
            return None
        # 1. Exact local symbol.
        if name in self._local_funcs[path]:
            return CallNode(path, name, SymbolKind.FUNCTION)
        if name in self._local_classes[path]:
            return CallNode(path, name, SymbolKind.CLASS)
        # 2. Explicitly imported symbol / module alias.
        binding = self._bindings[path].get(name)
        if binding is not None:
            node = self._binding_node(binding)
            if node is not None:
                if kind is ReferenceKind.CALL and node.is_module:
                    return None  # a module is not callable
                return node
            # Internal binding that didn't resolve (e.g. `from . import mod`
            # where ``mod`` is a module, not a symbol): fall through so
            # relative/module-qualified resolution still has a chance.
        # 5. Known class/method relationship (single-part receiver).
        receiver = self._receiver_class(path, scope, name, depth=depth - 1)
        if receiver is not None:
            return receiver
        # 3. Resolved relative import: sibling module within the package.
        module_name = self._relative_module(path, name)
        if module_name is not None:
            return module_node(module_name)
        # 4. Module-qualified: a bare name that is itself a known module.
        resolved = self._module_table.resolve(name)
        if resolved is not None:
            _, module_file = resolved
            return module_node(module_file)
        return None

    def _resolve_dotted(
        self,
        path: Path,
        scope: str | None,
        name: str,
        kind: ReferenceKind,
        *,
        depth: int = 8,
    ) -> CallNode | None:
        if depth <= 0:
            return None
        parts = name.split(".")
        head = parts[0]
        rest = parts[1:]

        # 1. Local class container: ClassName.method(...).
        methods = self._local_classes[path].get(head)
        if methods is not None:
            if len(rest) >= 1 and rest[-1] in methods:
                return CallNode(path, rest[-1], SymbolKind.METHOD, head)
            return None

        # 2. Explicitly imported module alias / symbol binding.
        binding = self._bindings[path].get(head)
        if binding is not None:
            if isinstance(binding, _ModuleBinding):
                if binding.file is None:
                    return None  # external module: not an intra-repo unknown
                return self._walk_in_module(
                    binding.module, binding.file, rest, path
                )
            if isinstance(binding, _SymbolBinding):
                # Attributes on an imported symbol are conservatively unknown.
                return None
            return None

        # 3. Resolved relative import: head is a sibling submodule.
        relative = self._relative_module(path, head)
        if relative is not None:
            return self._walk_in_module(
                ".".join((*path.parts[:-1], head)),
                relative,
                rest,
                path,
            )

        # 4. Module-qualified symbol (longest known module prefix).
        module_result = self._longest_module_prefix(parts)
        if module_result is not None:
            module_name, module_file, remaining = module_result
            return self._walk_in_module(module_name, module_file, remaining, path)

        # 5. Known class/method relationship (receiver).
        return self._resolve_receiver(path, scope, parts, name, depth=depth - 1)

    def _receiver_class(
        self,
        path: Path,
        scope: str | None,
        name: str,
        *,
        depth: int = 8,
    ) -> CallNode | None:
        """Class node for a single-part name bound statically to a class."""
        if depth <= 0:
            return None
        target = self._assignment_targets[path].get((scope or "", name))
        if target is None and scope:
            target = self._typed_params[path].get((scope, name))
        if target is None or target == name:
            return None
        node = self._resolve_target_text(path, scope, target, depth=depth - 1)
        if node is not None and node.kind is SymbolKind.CLASS:
            return node
        return None

    def _resolve_target_text(
        self,
        path: Path,
        scope: str | None,
        target: str,
        *,
        depth: int = 8,
    ) -> CallNode | None:
        if depth <= 0:
            return None
        if "." in target:
            return self._resolve_dotted(path, scope, target, ReferenceKind.NAME, depth=depth)
        return self._resolve_simple(path, scope, target, ReferenceKind.NAME, depth=depth)

    def _resolve_receiver(
        self,
        path: Path,
        scope: str | None,
        parts: list[str],
        name: str,
        *,
        depth: int = 8,
    ) -> CallNode | None:
        if depth <= 0:
            return None
        head = parts[0]
        rest = parts[1:]
        if not rest:
            return self._receiver_class(path, scope, head, depth=depth - 1)

        if head == "self" and scope and "." in scope:
            class_name = scope.split(".")[0]
            receiver_hint = self._self_attrs[path].get((class_name, rest[0]))
            if receiver_hint is None:
                return None
            receiver = self._resolve_target_text(
                path, scope, receiver_hint, depth=depth - 1
            )
            if receiver is None or receiver.kind is not SymbolKind.CLASS:
                return None
            if len(rest) == 2:
                method = rest[1]
                methods = self._local_classes.get(receiver.file_path, {}).get(
                    receiver.name, set()
                )
                if method in methods:
                    return CallNode(
                        receiver.file_path, method, SymbolKind.METHOD, receiver.name
                    )
            return None

        receiver = self._receiver_class(path, scope, head, depth=depth - 1)
        if receiver is None or receiver.kind is not SymbolKind.CLASS:
            return None
        if len(rest) == 1:
            method = rest[0]
            methods = self._local_classes.get(receiver.file_path, {}).get(
                receiver.name, set()
            )
            if method in methods:
                return CallNode(
                    receiver.file_path, method, SymbolKind.METHOD, receiver.name
                )
        return None

    def _walk_in_module(
        self,
        module_name: str,
        module_file: Path | None,
        remaining: list[str],
        source: Path,
    ) -> CallNode | None:
        if module_file is None:
            return None
        segments = list(remaining)
        current_name = module_name
        current_file = module_file
        while segments:
            child = self._module_table.lookup(
                f"{current_name}.{segments[0]}" if current_name else segments[0]
            )
            if child is not None:
                current_name = f"{current_name}.{segments[0]}" if current_name else segments[0]
                current_file = child
                segments.pop(0)
                continue
            break
        if not segments:
            return module_node(current_file)
        symbol = self._resolve_symbol_in_file(current_file, segments[0])
        if symbol is not None:
            return symbol
        return None

    def _longest_module_prefix(
        self, parts: list[str]
    ) -> tuple[str, Path, list[str]] | None:
        for size in range(len(parts), 0, -1):
            name = ".".join(parts[:size])
            found = self._module_table.lookup(name)
            if found is not None:
                return name, found, parts[size:]
        return None

    def _relative_module(self, path: Path, head: str) -> Path | None:
        package = path.parts[:-1]
        if not package:
            return None
        candidate = ".".join((*package, head))
        return self._module_table.lookup(candidate)

    def _resolve_symbol_in_file(
        self, file_path: Path, name: str
    ) -> CallNode | None:
        symbols = self._symbols_by_file_name.get((file_path, name))
        if not symbols:
            return None
        pick = min(
            symbols,
            key=lambda s: (
                _KIND_ORDER[s.kind],
                s.line if s.line is not None else 0,
                s.parent_class or "",
            ),
        )
        return CallNode(
            file_path, name=pick.name, kind=pick.kind, parent_class=pick.parent_class
        )

    def _binding_node(self, binding: object | None) -> CallNode | None:
        if binding is None:
            return None
        if isinstance(binding, _ModuleBinding):
            if binding.file is None:
                return None
            return module_node(binding.file)
        if isinstance(binding, _SymbolBinding):
            if binding.file is None:
                return None
            return self._resolve_symbol_in_file(binding.file, binding.symbol)
        return None

    # ------------------------------------------------------------------
    # Node helpers
    # ------------------------------------------------------------------

    def _scope_node(self, path: Path, scope: str | None) -> CallNode:
        if not scope:
            return module_node(path)
        if "." in scope:
            class_name, _, rest = scope.partition(".")
            method = rest.split(".")[0]
            methods = self._local_classes[path].get(class_name)
            if methods is not None and method in methods:
                return CallNode(path, method, SymbolKind.METHOD, class_name)
            if class_name in self._local_funcs[path]:
                return CallNode(path, class_name, SymbolKind.FUNCTION)
            return CallNode(path, method, SymbolKind.FUNCTION)
        if scope in self._local_classes[path]:
            return CallNode(path, scope, SymbolKind.CLASS)
        return CallNode(path, scope, SymbolKind.FUNCTION)

    def _call_edge_kind(self, target: CallNode, ref: Reference) -> EdgeKind:
        if target.kind is SymbolKind.CLASS:
            return EdgeKind.INSTANTIATION
        if target.kind is SymbolKind.METHOD:
            return EdgeKind.METHOD_CALL
        return EdgeKind.CALL

    def _symbol_nodes(self, edges: list[CallEdge]) -> list[CallNode]:
        nodes: list[CallNode] = []
        for edge in edges:
            for node in (edge.source, edge.target):
                if not node.is_module:
                    nodes.append(node)
        return nodes

    # ------------------------------------------------------------------
    # Unresolved records
    # ------------------------------------------------------------------

    def _unresolved_for(
        self, path: Path, refs: FileReferences
    ) -> list[UnresolvedReference]:
        records: list[UnresolvedReference] = []
        for ref in refs.references:
            resolved = self._resolve(path, ref)
            if resolved is not None:
                continue
            head = ref.name.split(".")[0]
            if "." not in ref.name and ref.kind is ReferenceKind.NAME:
                # A bare name read that matches nothing is a local variable or
                # parameter — not an external unknown worth recording.
                continue
            binding = self._bindings[path].get(head)
            if isinstance(binding, (_ModuleBinding, _SymbolBinding)):
                if binding.file is None:
                    # External import — not an intra-repository unknown.
                    continue
            if _is_builtin(head):
                continue
            reason = _unresolved_reason(ref)
            records.append(
                UnresolvedReference(
                    source_file=path,
                    source_symbol=ref.source_symbol,
                    name=ref.name,
                    reason=reason,
                    line=ref.line,
                )
            )
        return records


def _unique(values) -> tuple:
    return tuple(dict.fromkeys(values))


def _is_builtin(name: str) -> bool:
    return name in _BUILTIN_NAMES


def _unresolved_reason(ref: Reference) -> str:
    head = ref.name.split(".")[0]
    if ref.kind is ReferenceKind.CALL:
        if len(ref.name.split(".")) > 1:
            return UNRESOLVED_RECEIVER if head in ("self",) else UNRESOLVED_MODULE
        return UNRESOLVED_NAME
    if len(ref.name.split(".")) > 1:
        return UNRESOLVED_ATTRIBUTE
    return UNRESOLVED_NAME