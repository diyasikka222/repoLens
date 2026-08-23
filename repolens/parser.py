"""AST-based extraction of structural information from Python source files.

This module parses Python source code with the built-in :mod:`ast` module.
Source files are never imported or executed; only their syntax tree is read.
"""

from __future__ import annotations

import ast
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Argument:
    """A single function or method parameter."""

    name: str
    annotation: str | None = None


@dataclass(frozen=True)
class Import:
    """An ``import module`` or ``import module as alias`` statement."""

    module: str
    alias: str | None = None
    line: int | None = None


@dataclass(frozen=True)
class FromImport:
    """A ``from module import name`` or ``from module import name as alias`` statement.

    ``level`` is the number of parent packages in a relative import (PEP 328);
    ``0`` marks an absolute import.
    """

    module: str
    name: str
    alias: str | None = None
    line: int | None = None
    level: int = 0


@dataclass(frozen=True)
class Function:
    """A top-level function defined in a module."""

    name: str
    arguments: list[Argument] = field(default_factory=list)
    line: int | None = None


@dataclass(frozen=True)
class Method:
    """A function defined directly inside a class body."""

    name: str
    arguments: list[Argument] = field(default_factory=list)
    parent_class: str | None = None
    line: int | None = None


@dataclass(frozen=True)
class Class:
    """A top-level class defined in a module."""

    name: str
    base_classes: list[str] = field(default_factory=list)
    methods: list[Method] = field(default_factory=list)
    line: int | None = None


@dataclass(frozen=True)
class ModuleAnalysis:
    """The structural information extracted from one Python source file."""

    file_path: Path | None
    imports: list[Import] = field(default_factory=list)
    from_imports: list[FromImport] = field(default_factory=list)
    classes: list[Class] = field(default_factory=list)
    functions: list[Function] = field(default_factory=list)


class PythonParser:
    """Extracts structure from Python source using :func:`ast.parse`."""

    def parse_source(
        self, source: str, *, file_path: Path | None = None
    ) -> ModuleAnalysis:
        """Parse source code given as a string."""
        tree = ast.parse(source, filename=str(file_path) if file_path else "<source>")
        return ModuleAnalysis(
            file_path=file_path,
            imports=self._extract_imports(tree),
            from_imports=self._extract_from_imports(tree),
            classes=self._extract_classes(tree),
            functions=self._extract_functions(tree),
        )

    def parse_file(self, path: Path | str) -> ModuleAnalysis:
        """Parse the Python file at ``path``, which must exist."""
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"source file does not exist: {file_path}")
        return self.parse_source(
            file_path.read_text(encoding="utf-8"), file_path=file_path
        )

    def _extract_imports(self, tree: ast.Module) -> list[Import]:
        return [
            Import(module=alias.name, alias=alias.asname, line=node.lineno)
            for node in self._nodes_of_type(tree, ast.Import)
            for alias in node.names
        ]

    def _extract_from_imports(self, tree: ast.Module) -> list[FromImport]:
        return [
            FromImport(
                module=node.module or "",
                name=alias.name,
                alias=alias.asname,
                line=node.lineno,
                level=node.level,
            )
            for node in self._nodes_of_type(tree, ast.ImportFrom)
            for alias in node.names
        ]

    def _extract_classes(self, tree: ast.Module) -> list[Class]:
        return [
            Class(
                name=node.name,
                base_classes=self._extract_base_classes(node),
                methods=self._extract_methods(node),
                line=node.lineno,
            )
            for node in self._top_level_nodes_of_type(tree, ast.ClassDef)
        ]

    def _extract_base_classes(self, node: ast.ClassDef) -> list[str]:
        return [ast.unparse(base) for base in node.bases]

    def _extract_methods(self, class_node: ast.ClassDef) -> list[Method]:
        return [
            Method(
                name=node.name,
                arguments=self._extract_arguments(node.args),
                parent_class=class_node.name,
                line=node.lineno,
            )
            for node in class_node.body
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
        ]

    def _extract_functions(self, tree: ast.Module) -> list[Function]:
        return [
            Function(
                name=node.name,
                arguments=self._extract_arguments(node.args),
                line=node.lineno,
            )
            for node in self._top_level_nodes_of_type(
                tree, ast.FunctionDef | ast.AsyncFunctionDef
            )
        ]

    def _extract_arguments(self, arguments: ast.arguments) -> list[Argument]:
        positional = [*arguments.posonlyargs, *arguments.args]
        keyword_only = arguments.kwonlyargs
        variadic = [arguments.vararg] if arguments.vararg else []
        keyword_variadic = [arguments.kwarg] if arguments.kwarg else []
        return [
            Argument(name=arg.arg, annotation=self._annotation(arg))
            for arg in [*positional, *variadic, *keyword_only, *keyword_variadic]
        ]

    def _annotation(self, arg: ast.arg) -> str | None:
        return ast.unparse(arg.annotation) if arg.annotation else None

    def _nodes_of_type[T: ast.AST](
        self, tree: ast.Module, node_types: type[T] | tuple[type[T], ...]
    ) -> Iterable[T]:
        return (
            node for node in ast.walk(tree) if isinstance(node, node_types)
        )

    def _top_level_nodes_of_type[T: ast.AST](
        self, tree: ast.Module, node_types: type[T] | tuple[type[T], ...]
    ) -> Iterable[T]:
        return (
            node
            for node in tree.body
            if isinstance(node, node_types)
        )
