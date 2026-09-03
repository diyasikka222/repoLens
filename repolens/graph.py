"""Repository-wide dependency graph built from Python import statements.

Nodes are repository-relative Python file paths; edges point from the
importing file to the imported file. Only imports that resolve to modules
inside the analyzed repository produce edges, so standard-library and
third-party imports never appear in the graph.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from repolens.parser import FromImport, ModuleAnalysis, PythonParser
from repolens.scanner import RepositoryScanner


@dataclass(frozen=True)
class DependencyEdge:
    """A directed dependency: ``source`` imports from ``target``."""

    source: Path
    target: Path

    def __str__(self) -> str:
        return f"{self.source} -> {self.target}"


class _ModuleIndex:
    """Maps dotted module names to repository-relative Python files."""

    def __init__(self, files: Iterable[Path]) -> None:
        self._modules: dict[str, Path] = {}
        for file_path in files:
            self._register(file_path)

    def _register(self, file_path: Path) -> None:
        if file_path.name == "__init__.py":
            parts = file_path.parts[:-1]
        else:
            parts = (*file_path.parts[:-1], file_path.stem)
        self._modules.setdefault(".".join(parts), file_path)

    def lookup(self, parts: list[str]) -> Path | None:
        """Return the file for an exact dotted module name."""
        if not parts:
            return None
        return self._modules.get(".".join(parts))

    def resolve_import(self, dotted: str) -> Path | None:
        """Resolve ``import a.b.c`` to the deepest known module."""
        parts = dotted.split(".")
        for size in range(len(parts), 0, -1):
            found = self._modules.get(".".join(parts[:size]))
            if found is not None:
                return found
        return None


class DependencyGraphBuilder:
    """Builds a :class:`DependencyGraph` by scanning and parsing a repository.

    Pass an optional prebuilt snapshot (``index``, from
    :mod:`repolens.incremental_index`) to build from already parsed analyses
    without re-parsing.
    """

    def __init__(
        self, root: Path | str, *, index: object | None = None
    ) -> None:
        self.root = Path(root)
        self._scanner = RepositoryScanner(self.root) if index is None else None
        self._parser = PythonParser()
        self._index = index

    def build(self) -> DependencyGraph:
        """Scan the repository and derive all internal import edges."""
        if self._index is not None:
            files = list(self._index.files)
            index = _ModuleIndex(files)
            edges = {
                DependencyEdge(source=file_path, target=target)
                for file_path in files
                for target in self._dependencies_of(
                    file_path,
                    self._index.analysis_for(file_path),
                    index,
                )
            }
            return DependencyGraph(root=self.root, nodes=files, edges=edges)
        files = self._scanner.discover_python_files()
        index = _ModuleIndex(files)
        edges = {
            DependencyEdge(source=file_path, target=target)
            for file_path in files
            for target in self._dependencies_of(
                file_path, self._parser.parse_file(self.root / file_path), index
            )
        }
        return DependencyGraph(root=self.root, nodes=files, edges=edges)

    def _dependencies_of(
        self, file_path: Path, analysis: ModuleAnalysis, index: _ModuleIndex
    ) -> set[Path]:
        package_parts = file_path.parts[:-1]
        targets = {
            target
            for target in (
                index.resolve_import(imp.module) for imp in analysis.imports
            )
            if target is not None
        }
        for from_import in analysis.from_imports:
            target = self._resolve_from_import(from_import, package_parts, index)
            if target is not None:
                targets.add(target)
        targets.discard(file_path)
        return targets

    def _resolve_from_import(
        self,
        from_import: FromImport,
        package_parts: tuple[str, ...],
        index: _ModuleIndex,
    ) -> Path | None:
        base = self._absolute_base(from_import.level, package_parts)
        if base is None:
            return None
        module_parts = base + _split_dotted(from_import.module)
        if from_import.name != "*":
            found = index.lookup([*module_parts, from_import.name])
            if found is not None:
                return found
        return index.lookup(module_parts)

    @staticmethod
    def _absolute_base(
        level: int, package_parts: tuple[str, ...]
    ) -> list[str] | None:
        """Anchor parts for an import with ``level`` dots, or ``None`` if invalid."""
        if level == 0:
            return []
        depth = len(package_parts) - (level - 1)
        if depth < 0:
            return None
        return list(package_parts[:depth])


def _split_dotted(dotted: str) -> list[str]:
    return dotted.split(".") if dotted else []


class DependencyGraph:
    """A directed graph of import dependencies between repository files."""

    def __init__(
        self, root: Path, nodes: Iterable[Path], edges: Iterable[DependencyEdge]
    ) -> None:
        self.root = root
        self._nodes = tuple(sorted(nodes))
        self._edges = tuple(sorted(edges, key=lambda edge: (edge.source, edge.target)))
        self._dependencies: dict[Path, list[Path]] = {node: [] for node in self._nodes}
        self._dependents: dict[Path, list[Path]] = {node: [] for node in self._nodes}
        for edge in self._edges:
            self._dependencies[edge.source].append(edge.target)
            self._dependents[edge.target].append(edge.source)

    def get_all_nodes(self) -> list[Path]:
        """Return every discovered Python file as a sorted repository-relative path."""
        return list(self._nodes)

    def get_all_edges(self) -> list[DependencyEdge]:
        """Return deduplicated edges sorted by source, then target."""
        return list(self._edges)

    def get_dependencies(self, file: Path | str) -> list[Path]:
        """Return files imported by ``file``, sorted."""
        return list(self._dependencies[self._require_node(file)])

    def get_dependents(self, file: Path | str) -> list[Path]:
        """Return files that import ``file``, sorted."""
        return list(self._dependents[self._require_node(file)])

    def _require_node(self, file: Path | str) -> Path:
        node = Path(file)
        if node not in self._dependencies:
            raise ValueError(f"not a node in the graph: {node}")
        return node
