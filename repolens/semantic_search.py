"""Semantic code retrieval using pluggable embedding providers.

:class:`SemanticSearcher` represents candidate Python files in a repository
and the developer query as vectors (via an :class:`repolens.embeddings.EmbeddingProvider`)
and reranks those candidates by cosine similarity. It is an independent
capability: it never modifies the lexical ranking in :mod:`repolens.search`,
combining the two signals is left to the hybrid retriever. To keep embedding
cost bounded, the lexical :class:`~repolens.search.CodeSearcher` supplies the
candidate set that semantic ranking operates over.

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
from repolens.search import CodeSearcher

DEFAULT_LIMIT = 10
DEFAULT_CANDIDATE_LIMIT = 40


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

    Construction is cheap: it scans the repository and groups symbols, but
    never embeds anything. The expensive work of composing documents and
    running the embedding provider is deferred to :meth:`search`.

    Semantic search is *candidate-based*: the existing lexical
    :class:`~repolens.search.CodeSearcher` first retrieves a bounded set of
    candidate files, and only those candidate documents are embedded and
    reranked by cosine similarity. This keeps semantic retrieval fully
    offline and on-device without ever embedding the entire repository on a
    single query. Candidate embeddings are cached per file, so repeated or
    overlapping queries reuse already-embedded documents instead of
    re-embedding them.
    """

    def __init__(
        self,
        root: Path | str,
        provider: EmbeddingProvider,
        *,
        candidate_searcher: CodeSearcher | None = None,
        candidate_limit: int = DEFAULT_CANDIDATE_LIMIT,
    ) -> None:
        self.root = Path(root)
        self._provider = provider
        self._parser = PythonParser()
        self._candidate_searcher = (
            candidate_searcher if candidate_searcher is not None else CodeSearcher(self.root)
        )
        self._candidate_limit = candidate_limit
        self._symbols_by_path = self._group_symbols_by_path()
        self._vectors_by_path: dict[Path, Vector] = {}

    def search(
        self, query: str, limit: int = DEFAULT_LIMIT
    ) -> list[SemanticResult]:
        """Return up to ``limit`` files ranked by descending cosine similarity.

        First retrieves a bounded candidate set with the lexical
        :class:`~repolens.search.CodeSearcher`, embeds only those candidates
        (cached for reuse), then reranks them by cosine similarity.

        An empty or whitespace-only query, a non-positive limit, or a query
        whose lexical candidate set is empty yields an empty list.
        """
        if not query.strip() or limit <= 0:
            return []
        candidate_paths = [
            result.file_path
            for result in self._candidate_searcher.search(
                query, limit=self._candidate_limit
            )
        ]
        self._ensure_embeddings(candidate_paths)
        query_vector = self._provider.embed_text(query)
        if not candidate_paths:
            return []
        scored = []
        for path in candidate_paths:
            similarity = cosine_similarity(query_vector, self._vectors_by_path[path])
            if similarity > 0.0:
                scored.append((path, similarity))
        scored.sort(key=lambda item: (-item[1], item[0].as_posix()))
        return [
            SemanticResult(file_path=path, similarity=similarity)
            for path, similarity in scored[:limit]
        ]

    def _ensure_embeddings(self, candidate_paths: list[Path]) -> None:
        """Embed the candidate documents that are not already cached.

        Documents already embedded by an earlier query are reused; only the
        newly-seen candidates are handed to the provider in one batch.
        """
        missing = [path for path in candidate_paths if path not in self._vectors_by_path]
        if not missing:
            return
        documents = [
            self._compose_document(
                path,
                *self._read_file(path),
                self._symbols_by_path.get(path, ()),
            )
            for path in missing
        ]
        for path, vector in zip(
            missing, self._provider.embed_texts(documents)
        ):
            self._vectors_by_path[path] = vector

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
