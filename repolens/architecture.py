"""Repository-level architecture graph (Milestone 23).

:class:`ArchitectureGraph` models the structure of a repository above
individual symbols: physical **files**, Python **modules**, and
**packages** (directories holding Python files), connected by containment and
dependency relationships.

It is a *pure projection* of existing RepoLens data — the incremental
:class:`~repolens.incremental_index.RepositoryIndex` and its per-file
:class:`~repolens.parser.ModuleAnalysis`, plus the existing
:class:`~repolens.graph.DependencyGraph` for import resolution. No file is
re-parsed here and no new parser is introduced: pass an already-built
``index`` (and optionally ``graph``) to make a build fully incremental.

Guarantees:

- deterministic node IDs, edge ordering, and serialization;
- no duplicate nodes or edges;
- repository-relative paths for files and packages, dotted names for modules;
- packages and nested packages supported (including a single root package);
- external / unresolvable imports are ignored safely and never crash;
- traversal is bounded by ``max_depth`` and a hard node cap;
- deleted files, modules, and packages vanish (and stale edges with them)
  because the graph is re-derived from the current index snapshot.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Iterable

from repolens.graph import DependencyGraph, DependencyGraphBuilder

#: Identifier of the implicit repository root package.
REPO_ROOT_PACKAGE = "."


class ArchitectureNodeKind(str, Enum):
    """The kinds of nodes that can appear in an architecture graph."""

    FILE = "file"
    MODULE = "module"
    PACKAGE = "package"


class ArchitectureRelationship(str, Enum):
    """Directed relationships between architecture nodes.

    ``CONTAINS`` / ``CONTAINED_BY`` are reciprocal views of the same edge;
    ``DEPENDS_ON`` / ``DEPENDED_ON_BY`` are reciprocal views of the same edge.
    """

    CONTAINS = "contains"
    CONTAINED_BY = "contained_by"
    DEPENDS_ON = "depends_on"
    DEPENDED_ON_BY = "depended_on_by"


@dataclass(frozen=True)
class ArchitectureNode:
    """A single node: a file, a module, or a package.

    ``id`` is stable and repository-relative: a file path (``app/api/routes.py``),
    a dotted module name (``app.api.routes``, or ``app.api`` for a package's
    ``__init__``), or a directory path for a package (``app/api``; ``.`` for the
    repository root package).
    """

    kind: ArchitectureNodeKind
    id: str

    def to_dict(self) -> dict[str, str]:
        return {"kind": self.kind.value, "id": self.id}


@dataclass(frozen=True)
class ArchitectureEdge:
    """A directed edge between two architecture nodes (referenced by id)."""

    relationship: ArchitectureRelationship
    source: ArchitectureNode
    target: ArchitectureNode

    def to_dict(self) -> dict[str, str]:
        return {
            "relationship": self.relationship.value,
            "source": self.source.to_dict(),
            "target": self.target.to_dict(),
        }


@dataclass(frozen=True)
class ArchitectureStats:
    """Counters describing a built :class:`ArchitectureGraph`."""

    nodes: int = 0
    files: int = 0
    modules: int = 0
    packages: int = 0
    edges: int = 0
    contains_edges: int = 0
    depends_on_edges: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "nodes": self.nodes,
            "files": self.files,
            "modules": self.modules,
            "packages": self.packages,
            "edges": self.edges,
            "contains_edges": self.contains_edges,
            "depends_on_edges": self.depends_on_edges,
        }


@dataclass(frozen=True)
class ArchitectureConfig:
    """Limits that keep every architecture query bounded."""

    max_transitive_nodes: int = 5000


#: node key used internally: (kind, id)
_Key = tuple[str, str]


def _key(node: ArchitectureNode) -> _Key:
    return (node.kind.value, node.id)


def _module_name(file_path: Path) -> str:
    """Dotted module name for a repository-relative file path.

    ``app/__init__.py`` is the module ``app``; ``app/api/routes.py`` is
    ``app.api.routes``.
    """
    if file_path.name == "__init__.py":
        parts = file_path.parts[:-1]
    else:
        parts = (*file_path.parts[:-1], file_path.stem)
    return ".".join(parts)


def _node_sort_key(node: ArchitectureNode) -> tuple[str, str]:
    return (node.kind.value, node.id)


class ArchitectureGraph:
    """An immutable, deterministic architecture graph for a repository.

    Build one with :class:`ArchitectureGraphBuilder`. All queries accept either
    an :class:`ArchitectureNode` or a stable identifier string: a file path, a
    package path, or a dotted module name. When a string is ambiguous between a
    package path and a package module (e.g. ``app``), the package is preferred;
    use :meth:`get_module_node` to address the module explicitly. File nodes
    delegate dependency queries to their module.
    """

    def __init__(
        self,
        root: Path,
        nodes: Iterable[ArchitectureNode],
        edges: Iterable[ArchitectureEdge],
        config: ArchitectureConfig | None = None,
    ) -> None:
        self.root = Path(root)
        self._config = config if config is not None else ArchitectureConfig()
        self._nodes: dict[_Key, ArchitectureNode] = {}
        for node in nodes:
            self._nodes[_key(node)] = node
        self._node_list = tuple(sorted(self._nodes.values(), key=_node_sort_key))

        self._files_by_path: dict[str, ArchitectureNode] = {}
        self._modules_by_name: dict[str, ArchitectureNode] = {}
        self._packages_by_path: dict[str, ArchitectureNode] = {}
        for node in self._node_list:
            if node.kind is ArchitectureNodeKind.FILE:
                self._files_by_path[node.id] = node
            elif node.kind is ArchitectureNodeKind.MODULE:
                self._modules_by_name[node.id] = node
            else:
                self._packages_by_path[node.id] = node

        self._edges: dict[tuple, ArchitectureEdge] = {}
        for edge in edges:
            self._edges[(edge.relationship.value, *_key(edge.source), *_key(edge.target))] = edge
        self._edge_list = tuple(
            sorted(
                self._edges.values(),
                key=lambda e: (
                    e.relationship.value,
                    e.source.kind.value,
                    e.source.id,
                    e.target.kind.value,
                    e.target.id,
                ),
            )
        )

        self._out: dict[ArchitectureRelationship, dict[_Key, list[ArchitectureNode]]] = {
            ArchitectureRelationship.CONTAINS: {},
            ArchitectureRelationship.DEPENDS_ON: {},
        }
        self._in: dict[ArchitectureRelationship, dict[_Key, list[ArchitectureNode]]] = {
            ArchitectureRelationship.CONTAINED_BY: {},
            ArchitectureRelationship.DEPENDED_ON_BY: {},
        }
        self._stats = self._build_stats()
        self._populate_edges()

    def _populate_edges(self) -> None:
        for relationship, outward in self._out.items():
            for edge in self._edge_list:
                if edge.relationship is not relationship:
                    continue
                outward.setdefault(_key(edge.source), []).append(edge.target)
            for targets in outward.values():
                targets.sort(key=_node_sort_key)
        reverse_rel = {
            ArchitectureRelationship.CONTAINS: ArchitectureRelationship.CONTAINED_BY,
            ArchitectureRelationship.DEPENDS_ON: ArchitectureRelationship.DEPENDED_ON_BY,
        }
        for edge in self._edge_list:
            rev = reverse_rel.get(edge.relationship)
            if rev is not None:
                self._in[rev].setdefault(_key(edge.target), []).append(edge.source)
        for targets in self._in.values():
            for values in targets.values():
                values.sort(key=_node_sort_key)

    def _build_stats(self) -> ArchitectureStats:
        files = sum(
            1 for node in self._node_list if node.kind is ArchitectureNodeKind.FILE
        )
        modules = sum(
            1 for node in self._node_list if node.kind is ArchitectureNodeKind.MODULE
        )
        packages = sum(
            1 for node in self._node_list if node.kind is ArchitectureNodeKind.PACKAGE
        )
        contains = sum(
            1 for edge in self._edge_list if edge.relationship is ArchitectureRelationship.CONTAINS
        )
        depends_on = sum(
            1 for edge in self._edge_list if edge.relationship is ArchitectureRelationship.DEPENDS_ON
        )
        return ArchitectureStats(
            nodes=len(self._node_list),
            files=files,
            modules=modules,
            packages=packages,
            edges=len(self._edge_list),
            contains_edges=contains,
            depends_on_edges=depends_on,
        )

    # ------------------------------------------------------------------
    # Identity helpers
    # ------------------------------------------------------------------

    def _require_key(self, node_or_id) -> _Key:
        node = self._resolve_node(node_or_id)
        return _key(node)

    def _or_module(self, node: ArchitectureNode) -> ArchitectureNode:
        """File nodes delegate dependency queries to their module."""
        if node.kind is not ArchitectureNodeKind.FILE:
            return node
        return self._module_of_file(node)

    def _resolve_node(self, node_or_id) -> ArchitectureNode:
        if isinstance(node_or_id, ArchitectureNode):
            if _key(node_or_id) not in self._nodes:
                raise KeyError(f"not an architecture node: {node_or_id}")
            return node_or_id
        if isinstance(node_or_id, Path):
            node_or_id = node_or_id.as_posix()
        if not isinstance(node_or_id, str):
            raise TypeError(f"expected a node or identifier, got {type(node_or_id).__name__}")
        value = node_or_id
        if value in self._files_by_path:
            return self._files_by_path[value]
        if value in self._packages_by_path:
            return self._packages_by_path[value]
        if value in self._modules_by_name:
            return self._modules_by_name[value]
        raise KeyError(f"no architecture node for {value!r}")

    # ------------------------------------------------------------------
    # Node queries
    # ------------------------------------------------------------------

    def get_file_node(self, path: Path | str) -> ArchitectureNode | None:
        """Return the FILE node for a repository-relative path, or ``None``."""
        return self._files_by_path.get(str(Path(path).as_posix()))

    def get_module_node(self, name: str) -> ArchitectureNode | None:
        """Return the MODULE node for a dotted module name, or ``None``."""
        return self._modules_by_name.get(name)

    def get_package_node(self, path: Path | str = REPO_ROOT_PACKAGE) -> ArchitectureNode | None:
        """Return the PACKAGE node for a directory path, or ``None``."""
        if isinstance(path, Path):
            path = path.as_posix()
        if path == REPO_ROOT_PACKAGE:
            path = REPO_ROOT_PACKAGE
        return self._packages_by_path.get(path)

    def files(self) -> list[ArchitectureNode]:
        """All FILE nodes, sorted."""
        return [node for node in self._node_list if node.kind is ArchitectureNodeKind.FILE]

    def modules(self) -> list[ArchitectureNode]:
        """All MODULE nodes, sorted."""
        return [node for node in self._node_list if node.kind is ArchitectureNodeKind.MODULE]

    def packages(self) -> list[ArchitectureNode]:
        """All PACKAGE nodes (including the root package), sorted."""
        return [
            node for node in self._node_list if node.kind is ArchitectureNodeKind.PACKAGE
        ]

    def get_nodes(self) -> list[ArchitectureNode]:
        """Every node, sorted deterministically."""
        return list(self._node_list)

    def get_edges(self) -> list[ArchitectureEdge]:
        """Canonical edges (CONTAINS + DEPENDS_ON), sorted deterministically."""
        return list(self._edge_list)

    def get_edges_by_relationship(
        self, relationship: ArchitectureRelationship
    ) -> list[ArchitectureEdge]:
        """All edges for one relationship (derived reverse edges included)."""
        reverse = {
            ArchitectureRelationship.CONTAINED_BY: ArchitectureRelationship.CONTAINS,
            ArchitectureRelationship.DEPENDED_ON_BY: ArchitectureRelationship.DEPENDS_ON,
        }
        canonical = reverse.get(relationship, relationship)
        edges = [edge for edge in self._edge_list if edge.relationship is canonical]
        if relationship in reverse:
            edges = [
                ArchitectureEdge(relationship, edge.target, edge.source) for edge in edges
            ]
        return sorted(
            edges,
            key=lambda e: (
                e.relationship.value,
                e.source.kind.value,
                e.source.id,
                e.target.kind.value,
                e.target.id,
            ),
        )

    # ------------------------------------------------------------------
    # Containment queries
    # ------------------------------------------------------------------

    def contains(self, node_or_id) -> list[ArchitectureNode]:
        """Nodes directly contained by ``node_or_id`` (CONTAINS targets)."""
        targets = self._out[ArchitectureRelationship.CONTAINS].get(self._require_key(node_or_id), ())
        return list(targets)

    def contained_by(self, node_or_id) -> list[ArchitectureNode]:
        """Containers of ``node_or_id`` (CONTAINED_BY sources)."""
        targets = self._in[ArchitectureRelationship.CONTAINED_BY].get(self._require_key(node_or_id), ())
        return list(targets)

    def files_in_package(self, package: Path | str | ArchitectureNode) -> list[ArchitectureNode]:
        """FILE nodes directly inside ``package`` (not nested packages)."""
        node = self._resolve_node(package)
        if node.kind is not ArchitectureNodeKind.PACKAGE:
            raise TypeError(f"expected a package, got {node.kind.value}")
        return [
            target
            for target in self._out[ArchitectureRelationship.CONTAINS].get(_key(node), ())
            if target.kind is ArchitectureNodeKind.FILE
        ]

    def modules_in_package(self, package: Path | str | ArchitectureNode) -> list[ArchitectureNode]:
        """MODULE nodes for the files directly inside ``package``."""
        node = self._resolve_node(package)
        if node.kind is not ArchitectureNodeKind.PACKAGE:
            raise TypeError(f"expected a package, got {node.kind.value}")
        result: list[ArchitectureNode] = []
        for file_node in self.files_in_package(node):
            result.append(self._module_of_file(file_node))
        result.sort(key=_node_sort_key)
        return result

    # ------------------------------------------------------------------
    # Dependency queries
    # ------------------------------------------------------------------

    def dependencies_of(self, node_or_id) -> list[ArchitectureNode]:
        """Direct DEPENDS_ON targets of ``node_or_id``.

        FILE nodes delegate to their module. For a MODULE this returns module
        targets; for a PACKAGE this returns package targets.
        """
        node = self._or_module(self._resolve_node(node_or_id))
        return list(self._out[ArchitectureRelationship.DEPENDS_ON].get(_key(node), ()))

    def dependents_of(self, node_or_id) -> list[ArchitectureNode]:
        """Direct DEPENDED_ON_BY sources of ``node_or_id``.

        FILE nodes delegate to their module.
        """
        node = self._or_module(self._resolve_node(node_or_id))
        return list(self._in[ArchitectureRelationship.DEPENDED_ON_BY].get(_key(node), ()))

    def package_dependencies(self, package: Path | str | ArchitectureNode) -> list[ArchitectureNode]:
        """Package-level DEPENDS_ON targets of ``package``."""
        node = self._resolve_node(package)
        if node.kind is not ArchitectureNodeKind.PACKAGE:
            raise TypeError(f"expected a package, got {node.kind.value}")
        return list(self._out[ArchitectureRelationship.DEPENDS_ON].get(_key(node), ()))

    def module_dependencies(self, module: str | ArchitectureNode) -> list[ArchitectureNode]:
        """Module-level DEPENDS_ON targets of ``module``."""
        node = self._resolve_node(module)
        if node.kind is not ArchitectureNodeKind.MODULE:
            raise TypeError(f"expected a module, got {node.kind.value}")
        return list(self._out[ArchitectureRelationship.DEPENDS_ON].get(_key(node), ()))

    # ------------------------------------------------------------------
    # Bounded transitive traversal
    # ------------------------------------------------------------------

    def transitive_dependencies(
        self, node_or_id, *, max_depth: int
    ) -> list[ArchitectureNode]:
        """BFS over DEPENDS_ON edges up to ``max_depth`` (deduplicated).

        ``max_depth`` is required; ``max_depth=0`` returns an empty list.
        Traversal is additionally capped by ``config.max_transitive_nodes`` so
        it can never run away on a pathological hub.
        """
        if max_depth < 0:
            raise ValueError("max_depth must be non-negative")
        root = self._or_module(self._resolve_node(node_or_id))
        return self._bfs(
            self._out[ArchitectureRelationship.DEPENDS_ON], root, max_depth
        )

    def transitive_dependents(
        self, node_or_id, *, max_depth: int
    ) -> list[ArchitectureNode]:
        """BFS over DEPENDED_ON_BY edges up to ``max_depth`` (deduplicated)."""
        if max_depth < 0:
            raise ValueError("max_depth must be non-negative")
        root = self._or_module(self._resolve_node(node_or_id))
        return self._bfs(
            self._in[ArchitectureRelationship.DEPENDED_ON_BY], root, max_depth
        )

    def _bfs(
        self, adjacency: dict[_Key, list[ArchitectureNode]], start: ArchitectureNode, max_depth: int
    ) -> list[ArchitectureNode]:
        if max_depth == 0:
            return []
        visited: dict[_Key, ArchitectureNode] = {}
        frontier = [start]
        depth = 0
        while frontier and depth < max_depth:
            depth += 1
            next_frontier: list[ArchitectureNode] = []
            for node in frontier:
                for neighbor in adjacency.get(_key(node), ()):
                    nkey = _key(neighbor)
                    if nkey in visited:
                        continue
                    visited[nkey] = neighbor
                    next_frontier.append(neighbor)
            if len(visited) >= self._config.max_transitive_nodes:
                break
            frontier = next_frontier
        return sorted(visited.values(), key=_node_sort_key)

    # ------------------------------------------------------------------
    # Internal module helpers
    # ------------------------------------------------------------------

    def _module_of_file(self, file_node: ArchitectureNode) -> ArchitectureNode:
        module = self._modules_by_name.get(_module_name(Path(file_node.id)))
        if module is None:
            raise KeyError(f"no module node for file {file_node.id!r}")
        return module

    # ------------------------------------------------------------------
    # Statistics + serialization
    # ------------------------------------------------------------------

    def stats(self) -> ArchitectureStats:
        return self._stats

    def to_dict(self) -> dict:
        """Deterministic dict (nodes, edges, stats, root) for caching/debugging."""
        return {
            "root": self.root.as_posix(),
            "nodes": [node.to_dict() for node in self._node_list],
            "edges": [edge.to_dict() for edge in self._edge_list],
            "stats": self._stats.as_dict(),
        }

    def serialize(self) -> str:
        """Deterministic, compact, stable string representation."""
        import json

        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))


class ArchitectureGraphBuilder:
    """Build an :class:`ArchitectureGraph` from existing RepoLens data.

    Pass an incremental :class:`~repolens.incremental_index.RepositoryIndex`
    (``index``) to avoid scanning or parsing; pass a prebuilt
    :class:`~repolens.graph.DependencyGraph` (``graph``) to reuse its import
    resolution outright. With neither, a cold build scans and parses the
    repository through the existing builders — never a second parser.
    """

    def __init__(
        self,
        root: Path | str,
        *,
        index: object | None = None,
        graph: DependencyGraph | None = None,
        config: ArchitectureConfig | None = None,
    ) -> None:
        self.root = Path(root)
        self._index = index
        self._graph = graph
        self._config = config if config is not None else ArchitectureConfig()

    def build(self) -> ArchitectureGraph:
        index = self._index
        if index is None:
            from repolens.incremental_index import IncrementalIndexBuilder

            index = IncrementalIndexBuilder(self.root, persist=False).build()
        files = list(index.files)

        graph = self._graph
        if graph is None:
            graph = DependencyGraphBuilder(self.root, index=index).build()

        nodes, edges = self._derive(files, graph)
        return ArchitectureGraph(self.root, nodes, edges, config=self._config)

    def _derive(self, files, graph: DependencyGraph):
        nodes: dict[_Key, ArchitectureNode] = {}
        edges: dict[tuple, ArchitectureEdge] = {}

        def add_node(kind: ArchitectureNodeKind, id_: str) -> ArchitectureNode:
            key = (kind.value, id_)
            node = nodes.get(key)
            if node is None:
                node = ArchitectureNode(kind, id_)
                nodes[key] = node
            return node

        def add_edge(relationship, source: ArchitectureNode, target: ArchitectureNode) -> None:
            key = (relationship.value, *_key(source), *_key(target))
            if key in edges:
                return
            edges[key] = ArchitectureEdge(relationship, source, target)

        file_nodes: dict[str, ArchitectureNode] = {}
        package_nodes: dict[str, ArchitectureNode] = {}
        parent_dirs = self._package_dirs(files)
        has_python = bool(files)

        # Package nodes for every directory that (recursively) holds python files,
        # including the repository root package.
        packages = {REPO_ROOT_PACKAGE} if has_python else set()
        for rel in parent_dirs:
            packages.add(rel)
        for rel in sorted(packages):
            node = add_node(ArchitectureNodeKind.PACKAGE, rel)
            package_nodes[rel] = node

        # File + module nodes; containment edges package -> file -> module.
        module_by_file: dict[str, ArchitectureNode] = {}
        for file_path in files:
            rel = file_path.as_posix()
            file_node = add_node(ArchitectureNodeKind.FILE, rel)
            file_nodes[rel] = file_node
            module_name = _module_name(file_path)
            module_node = add_node(ArchitectureNodeKind.MODULE, module_name)
            module_by_file[rel] = module_node
            add_edge(ArchitectureRelationship.CONTAINS, file_node, module_node)
            parent = self._parent_package(file_path, package_nodes)
            add_edge(ArchitectureRelationship.CONTAINS, package_nodes[parent], file_node)

        # Package -> package containment edges (nested packages).
        package_list = sorted(package_nodes.values(), key=_node_sort_key)
        for pkg in package_list:
            if pkg.id == REPO_ROOT_PACKAGE:
                continue
            parent = self._parent_package(Path(pkg.id), package_nodes)
            add_edge(
                ArchitectureRelationship.CONTAINS, package_nodes[parent], pkg
            )

        # Module -> module DEPENDS_ON edges, from the existing dependency graph.
        for edge in sorted(graph.get_all_edges(), key=lambda e: (e.source, e.target)):
            src_file = edge.source.as_posix()
            dst_file = edge.target.as_posix()
            src_module = module_by_file.get(src_file)
            dst_module = module_by_file.get(dst_file)
            if src_module is None or dst_module is None:
                continue
            if src_module.id == dst_module.id:
                continue
            add_edge(ArchitectureRelationship.DEPENDS_ON, src_module, dst_module)

        # Package -> package DEPENDS_ON edges (aggregation of module edges).
        for module_edge in sorted(edges.values(), key=lambda e: (_key(e.source), _key(e.target))):
            if module_edge.relationship is not ArchitectureRelationship.DEPENDS_ON:
                continue
            src_pkg = self._package_of_module(
                module_edge.source.id, package_nodes
            )
            dst_pkg = self._package_of_module(
                module_edge.target.id, package_nodes
            )
            if src_pkg == dst_pkg:
                continue
            add_edge(
                ArchitectureRelationship.DEPENDS_ON,
                package_nodes[src_pkg],
                package_nodes[dst_pkg],
            )

        return list(nodes.values()), list(edges.values())

    # -- package geometry helpers -----------------------------------------

    def _package_dirs(self, files) -> set[str]:
        dirs: set[str] = set()
        for file_path in files:
            parent = str(Path(file_path).parent)
            dirs.add("." if parent == Path(".").as_posix() else parent)
        # Keep only directories that are ancestors of some file.
        return {d for d in dirs if d != REPO_ROOT_PACKAGE}

    def _parent_package(self, file_path: Path, package_nodes: dict[str, ArchitectureNode]) -> str:
        parent = str(Path(file_path).parent)
        if parent == Path(".").as_posix():
            return REPO_ROOT_PACKAGE
        # Nearest ancestor directory that is itself a package node.
        parts = Path(parent).parts
        for size in range(len(parts), 0, -1):
            candidate = "/".join(parts[:size])
            if candidate in package_nodes:
                return candidate
        return REPO_ROOT_PACKAGE

    def _package_of_module(self, dotted: str, package_nodes) -> str:
        # The module's dotted name expressed as a pseudo path; map it back to
        # the deepest package that contains the module.
        pseudo = Path(dotted.replace(".", "/"))
        posix = pseudo.as_posix()
        for candidate in sorted(package_nodes, key=lambda p: p.count("/"), reverse=True):
            if candidate == REPO_ROOT_PACKAGE:
                continue
            if posix == candidate or posix.startswith(candidate + "/"):
                return candidate
        return REPO_ROOT_PACKAGE