"""Repository-wide index of symbol definitions.

The index locates where symbols are *defined* inside a repository. It is
built from the existing AST parser output, so source files are never
imported or executed here, and no AST logic is duplicated.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from repolens.parser import ModuleAnalysis, PythonParser
from repolens.scanner import RepositoryScanner


class SymbolKind(str, Enum):
    """The kind of a symbol definition."""

    CLASS = "class"
    FUNCTION = "function"
    METHOD = "method"


@dataclass(frozen=True)
class Symbol:
    """A single definition located somewhere in the repository."""

    name: str
    kind: SymbolKind
    file_path: Path
    line: int | None = None
    parent_class: str | None = None


class SymbolIndexBuilder:
    """Builds a :class:`SymbolIndex` by scanning and parsing a repository."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self._scanner = RepositoryScanner(self.root)
        self._parser = PythonParser()

    def build(self) -> SymbolIndex:
        """Scan the repository and index every definition found."""
        return SymbolIndex(
            symbol
            for file_path in self._scanner.discover_python_files()
            for symbol in self._symbols_in(
                file_path, self._parser.parse_file(self.root / file_path)
            )
        )

    def _symbols_in(
        self, file_path: Path, analysis: ModuleAnalysis
    ) -> Iterable[Symbol]:
        for function in analysis.functions:
            yield Symbol(
                name=function.name,
                kind=SymbolKind.FUNCTION,
                file_path=file_path,
                line=function.line,
            )
        for class_ in analysis.classes:
            yield Symbol(
                name=class_.name,
                kind=SymbolKind.CLASS,
                file_path=file_path,
                line=class_.line,
            )
            for method in class_.methods:
                yield Symbol(
                    name=method.name,
                    kind=SymbolKind.METHOD,
                    file_path=file_path,
                    line=method.line,
                    parent_class=class_.name,
                )


def _sort_key(symbol: Symbol) -> tuple[str, int, str]:
    return (
        symbol.file_path.as_posix(),
        symbol.line if symbol.line is not None else 0,
        symbol.name,
    )


class SymbolIndex:
    """A queryable collection of symbol definitions across a repository."""

    def __init__(self, symbols: Iterable[Symbol]) -> None:
        self._symbols = tuple(sorted(symbols, key=_sort_key))
        by_name: dict[str, list[Symbol]] = defaultdict(list)
        for symbol in self._symbols:
            by_name[symbol.name].append(symbol)
        self._by_name = dict(by_name)

    def find(self, name: str) -> list[Symbol]:
        """Return every definition named ``name``, sorted; may be empty."""
        return list(self._by_name.get(name, []))

    def find_by_kind(self, name: str, kind: SymbolKind | str) -> list[Symbol]:
        """Return definitions of ``name`` restricted to ``kind``, sorted."""
        wanted = kind if isinstance(kind, SymbolKind) else SymbolKind(kind)
        return [symbol for symbol in self.find(name) if symbol.kind is wanted]

    def get_all_symbols(self) -> list[Symbol]:
        """Return every indexed definition, sorted by file, line, then name."""
        return list(self._symbols)
