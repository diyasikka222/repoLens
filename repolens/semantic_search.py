"""Semantic code retrieval using pluggable embedding providers.

:class:`SemanticSearcher` represents every Python file in a repository and
the developer query as vectors (via an :class:`repolens.embeddings.EmbeddingProvider`)
and ranks files by cosine similarity. It is an independent capability: it
never touches the lexical ranking in :mod:`repolens.search`, and combining
the two signals is deliberately left to a future hybrid milestone.

Repository text representation
------------------------------
Each file becomes one structured, human-readable document so embeddings
capture both identifiers and free text. The layout is::

    path: billing/invoice.py
    imports: database.connection
    functions: create_invoice
    classes: InvoiceCalculator
    methods: InvoiceCalculator.calculate_total
    source:
    <raw file contents>

Path and symbol names give the provider terminology anchors; the raw source
(including comments and docstrings) contributes the prose vocabulary that
lets conceptually related but differently named code still land near the
query. Symbol kinds are conveyed by the section labels; methods carry their
parent class for context. The first character/token of every document is
structured, which keeps representations explainable and debuggable.

Ranking and determinism
-----------------------
Results are scored with :func:`cosine_similarity` against the embedded
query; only strictly positive similarities are returned (a zero vector or a
completely orthogonal query matches nothing). Results sort by descending
similarity with ties broken alphabetically by repository-relative posix
path, so a deterministic provider yields byte-identical output across runs.
Each file appears at most once.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from repolens.embeddings import EmbeddingProvider, Vector
from repolens.index import Symbol, SymbolIndexBuilder
from repolens.parser import ModuleAnalysis, PythonParser
from repolens.scanner import RepositoryScanner

DEFAULT_LIMIT = 10


def cosine_similarity(first: Vector, second: Vector) -> float:
    """Cosine of the angle between two vectors; ``0.0`` if either is zero.

    The result is clamped to ``[-1.0, 1.0]`` to absorb floating-point drift.
    """
    dot_product = sum(a * b for a, b in zip(first, second))
    first_norm = math.sqrt(sum(component * component for component in first))
    second_norm = math.sqrt(sum(component * component for component in second))
    if first_norm == 0.0 or second_norm == 0.0:
        return 0.0
    similarity = dot_product / (first_norm * second_norm)
    return max(-1.0, min(1.0, similarity))


@dataclass(frozen=True)
class SemanticResult:
    """One ranked file matching a semantic query.

    ``file_path`` is repository-relative; ``similarity`` is the cosine
    similarity between the query vector and the file's document vector.
    """

    file_path: Path
    similarity: float


class SemanticSearcher:
    """Ranks repository files against a query using embedding similarity.

    Example::

        searcher = SemanticSearcher(repo_root, FakeEmbeddingProvider())
        results = searcher.search("refund a card payment", limit=10)

    Construction scans, parses, and embeds every repository file once; each
    :meth:`search` call embeds only the query.
    """

    def __init__(self, root: Path | str, provider: EmbeddingProvider) -> None:
        self.root = Path(root)
        self._provider = provider
        self._parser = PythonParser()
        self._paths = RepositoryScanner(self.root).discover_python_files()
        symbols_by_path = self._group_symbols_by_path()
        documents = [
            self._compose_document(path, *self._read_file(path), symbols_by_path.get(path, ()))
            for path in self._paths
        ]
        self._vectors = provider.embed_texts(documents)

    def search(
        self, query: str, limit: int = DEFAULT_LIMIT
    ) -> list[SemanticResult]:
        """Return up to ``limit`` files ranked by descending cosine similarity.

        An empty or whitespace-only query, a non-positive limit, or a query
        with no positive similarity to any file yields an empty list.
        """
        if not query.strip() or limit <= 0:
            return []
        query_vector = self._provider.embed_text(query)
        scored = []
        for path, vector in zip(self._paths, self._vectors):
            similarity = cosine_similarity(query_vector, vector)
            if similarity > 0.0:
                scored.append((path, similarity))
        scored.sort(key=lambda item: (-item[1], item[0].as_posix()))
        return [
            SemanticResult(file_path=path, similarity=similarity)
            for path, similarity in scored[:limit]
        ]

    def _group_symbols_by_path(self) -> dict[Path, tuple[Symbol, ...]]:
        by_path: dict[Path, list[Symbol]] = defaultdict(list)
        index = SymbolIndexBuilder(self.root).build()
        for symbol in index.get_all_symbols():
            by_path[symbol.file_path].append(symbol)
        return {path: tuple(symbols) for path, symbols in by_path.items()}

    def _read_file(self, path: Path) -> tuple[str, ModuleAnalysis | None]:
        try:
            source = (self.root / path).read_text(encoding="utf-8")
        except (OSError, ValueError):
            return "", None
        try:
            return source, self._parser.parse_source(source, file_path=path)
        except SyntaxError:
            return source, None

    def _compose_document(
        self,
        path: Path,
        source: str,
        analysis: ModuleAnalysis | None,
        symbols: tuple[Symbol, ...],
    ) -> str:
        lines = [f"path: {path.as_posix()}"]
        if analysis is not None:
            imports = [item.module for item in analysis.imports]
            imports += [
                f"{item.module}.{item.name}" if item.module else item.name
                for item in analysis.from_imports
            ]
            if imports:
                lines.append("imports: " + " ".join(imports))
        functions = [s.name for s in symbols if s.kind.value == "function"]
        classes = [s.name for s in symbols if s.kind.value == "class"]
        methods = [
            f"{s.parent_class}.{s.name}" if s.parent_class else s.name
            for s in symbols
            if s.kind.value == "method"
        ]
        if functions:
            lines.append("functions: " + ", ".join(functions))
        if classes:
            lines.append("classes: " + ", ".join(classes))
        if methods:
            lines.append("methods: " + ", ".join(methods))
        lines.append("source:")
        lines.append(source)
        return "\n".join(lines)
