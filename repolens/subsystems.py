"""Deterministic subsystem discovery over the repository architecture graph.

A *subsystem* is a meaningful connected architectural area of a repository:
the set of packages, modules, and files that hang off one top-level package
(everything under ``store``, everything under ``billing``, ...). Subsystems
are discovered with plain graph-based heuristics — no LLM, no project-specific
name lists, and no hard-coded clustering thresholds:

- **Connectedness** — a subsystem is one top-level package tree. Packages are
  grouped by their top-level ancestor (first path segment); the repository
  root package becomes the ``root`` subsystem.
- **Dependency structure** — cross-subsystem module-level dependency edges are
  projected onto subsystems, so each subsystem reports which other subsystems
  it depends on, which depend on it, and which of its modules are *entry
  points* (imported from other subsystems). Intra-subsystem packages are then
  stratified into deterministic *layers* by dependency depth.

Discovery is a pure function of the
:class:`~repolens.architecture.ArchitectureGraph`. It is bounded (each node is
visited a constant number of times), deterministic (sets are emitted as sorted
tuples), and additive (it never mutates the graph or touches retrieval).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from repolens.architecture import (
    REPO_ROOT_PACKAGE,
    ArchitectureGraph,
    ArchitectureNode,
    ArchitectureNodeKind,
    ArchitectureRelationship,
    _module_name,
)

#: Subsystem id for files/packages directly under the repository root.
ROOT_SUBSYSTEM = "root"


@dataclass(frozen=True)
class Subsystem:
    """A top-level architectural area of a repository (deterministic)."""

    id: str
    display_name: str
    packages: tuple[str, ...]
    modules: tuple[str, ...]
    files: tuple[str, ...]
    entry_modules: tuple[str, ...]
    dependencies: tuple[str, ...]
    dependents: tuple[str, ...]
    layers: tuple[tuple[str, ...], ...]

    def stats(self) -> dict[str, int]:
        """Deterministic size/statistics for the subsystem."""
        return {
            "packages": len(self.packages),
            "modules": len(self.modules),
            "files": len(self.files),
            "entry_modules": len(self.entry_modules),
            "dependencies": len(self.dependencies),
            "dependents": len(self.dependents),
            "layers": len(self.layers),
        }

    def contains_file(self, path) -> bool:
        return str(Path(path).as_posix()) in self.files

    def contains_module(self, name: str) -> bool:
        return name in self.modules

    def contains_package(self, path) -> bool:
        return str(Path(path).as_posix()) in self.packages


def _subsystem_of_package(package: str) -> str:
    """Subsystem id owning ``package`` (its first path segment, or root)."""
    if package == REPO_ROOT_PACKAGE or "/" not in package:
        return ROOT_SUBSYSTEM if package == REPO_ROOT_PACKAGE else package.split("/")[0]
    return package.split("/")[0]


def _package_of_file(graph: ArchitectureGraph, file_path) -> str:
    """Deepest package id containing ``file_path`` (root package involved)."""
    file_node = graph.get_file_node(file_path)
    if file_node is None:
        return REPO_ROOT_PACKAGE
    containers = graph.contained_by(file_node)
    if not containers:
        return REPO_ROOT_PACKAGE
    return containers[0].id


def _deepest_package(graph: ArchitectureGraph, pseudo_posix: str) -> str:
    """Deepest package id that contains the module at ``pseudo_posix``."""
    best = REPO_ROOT_PACKAGE
    for package in sorted(
        (p.id for p in graph.packages()), key=lambda p: p.count("/"), reverse=True
    ):
        if package == REPO_ROOT_PACKAGE:
            continue
        if pseudo_posix == package or pseudo_posix.startswith(package + "/"):
            return package
    return best


def _module_file_map(graph: ArchitectureGraph) -> dict[str, str]:
    """Map dotted module name to its defining repository-relative file path."""
    return {_module_name(Path(node.id)): node.id for node in graph.files()}


def discover_subsystems(
    graph: ArchitectureGraph, *, max_subsystems: int | None = None
) -> list[Subsystem]:
    """Return the deterministic list of subsystems for ``graph``.

    Each subsystem collects the packages, modules, and files under one
    top-level package; cross-subsystem dependencies/dependents/entry points are
    projected from module-level dependency edges; and intra-subsystem packages
    are stratified into dependency *layers*. Subsystems are returned sorted by
    id. ``max_subsystems`` bounds the number of subsystems returned (by id
    order).
    """

    # module name -> subsystem id, computed once and reused per subsystem.
    module_subsystem: dict[str, str] = {}
    file_package: dict[str, str] = {}
    for module, file_path in _module_file_map(graph).items():
        package = _package_of_file(graph, file_path)
        file_package[file_path] = package
        module_subsystem[module] = _subsystem_of_package(package)

    packages_by_subsystem: dict[str, list[str]] = {}
    for package in sorted(p.id for p in graph.packages()):
        sid = _subsystem_of_package(package)
        packages_by_subsystem.setdefault(sid, []).append(package)

    subsystems: list[Subsystem] = []
    for sid in sorted(packages_by_subsystem):
        package_names = tuple(packages_by_subsystem[sid])
        subsystems.append(
            _build_subsystem(graph, sid, package_names, module_subsystem)
        )

    if max_subsystems is not None and max_subsystems >= 0:
        subsystems = subsystems[:max_subsystems]
    return [s for s in subsystems if s.modules or s.files or s.id != ROOT_SUBSYSTEM]


def _build_subsystem(
    graph: ArchitectureGraph,
    sid: str,
    package_names: tuple[str, ...],
    module_subsystem: dict[str, str],
) -> Subsystem:
    files: set[str] = set()
    modules: set[str] = set()
    for package in package_names:
        for file_node in graph.files_in_package(package):
            files.add(file_node.id)
        for module_node in graph.modules_in_package(package):
            modules.add(module_node.id)

    dependencies: set[str] = set()
    dependents: set[str] = set()
    entry_modules: set[str] = set()
    for module in modules:
        for dep in graph.dependencies_of(module):
            other = module_subsystem.get(dep.id)
            if other is not None and other != sid:
                dependencies.add(other)
        for depender in graph.dependents_of(module):
            other = module_subsystem.get(depender.id)
            if other is not None and other != sid:
                dependents.add(other)
                entry_modules.add(module)

    layers = _package_layers(graph, package_names)

    return Subsystem(
        id=sid,
        display_name=sid,
        packages=tuple(sorted(package_names)),
        modules=tuple(sorted(modules)),
        files=tuple(sorted(files)),
        entry_modules=tuple(sorted(entry_modules)),
        dependencies=tuple(sorted(dependencies)),
        dependents=tuple(sorted(dependents)),
        layers=layers,
    )


def _package_layers(
    graph: ArchitectureGraph, package_names: tuple[str, ...]
) -> tuple[tuple[str, ...], ...]:
    """Stratify the subsystem's packages by intra-subsystem dependency depth.

    For a dependency edge ``source -> target`` (``source`` imports ``target``),
    ``layer(source) = max(layer(source), layer(target) + 1)``; packages with no
    intra-subsystem dependencies (typically the model/leaf packages) sit in
    layer 0 and higher layers accumulate dependency depth. Iteration and edge
    ordering are deterministic, so layers are stable across builds. Relaxation
    is capped at one pass per package: acyclic subsystems reach their fixpoint
    within that bound (a layer can rise at most once per pass), and the cap
    keeps pathological package cycles deterministic and terminating.
    """
    packages = sorted(package_names)
    if len(packages) < 2:
        return (tuple(packages),)

    edges: set[tuple[str, str]] = set()
    for edge in graph.get_edges_by_relationship(ArchitectureRelationship.DEPENDS_ON):
        if edge.source.kind is not ArchitectureNodeKind.PACKAGE:
            continue
        if edge.source.id not in packages or edge.target.id not in packages:
            continue
        if edge.source.id != edge.target.id:
            edges.add((edge.source.id, edge.target.id))

    layer: dict[str, int] = {package: 0 for package in packages}
    passes = 0
    while passes < len(packages):
        passes += 1
        changed = False
        for source, target in sorted(edges):
            proposed = layer[target] + 1
            if proposed > layer[source]:
                layer[source] = proposed
                changed = True
        if not changed:
            break

    grouped: dict[int, list[str]] = {}
    for package in packages:
        grouped.setdefault(layer[package], []).append(package)
    return tuple(
        tuple(sorted(grouped[level])) for level in sorted(grouped)
    )


def subsystem_of(
    graph: ArchitectureGraph,
    node_or_id,
    subsystems: list[Subsystem] | None = None,
) -> Subsystem | None:
    """Return the :class:`Subsystem` containing ``node_or_id``, or ``None``.

    ``subsystems`` may be a precomputed discovery result (to avoid recomputing
    it).
    """
    if isinstance(node_or_id, ArchitectureNode):
        node = node_or_id
    else:
        node = None
        for lookup in (graph.get_file_node, graph.get_package_node, graph.get_module_node):
            node = lookup(node_or_id)
            if node is not None:
                break
    if node is None:
        return None

    if node.kind is ArchitectureNodeKind.PACKAGE:
        sid = _subsystem_of_package(node.id)
    elif node.kind is ArchitectureNodeKind.FILE:
        sid = _subsystem_of_package(_package_of_file(graph, node.id))
    else:
        sid = _subsystem_of_package(
            _deepest_package(graph, node.id.replace(".", "/"))
        )

    candidates = subsystems if subsystems is not None else discover_subsystems(graph)
    for subsystem in candidates:
        if subsystem.id == sid:
            return subsystem
    return None