"""Deterministic lexical code search over a repository.

This module implements the RepoLens retrieval baseline: given a
natural-language query such as ``"invoice calculation"`` it returns the most
relevant Python files and symbols. The search is entirely local and lexical —
no LLMs, embeddings, vector databases, network access, or caching are
involved. The same repository state and the same query always produce the
same results.

Architecture
------------
``CodeSearcher`` builds an in-memory search index once at construction time
by reusing the existing RepoLens analysis components:

1. :class:`repolens.scanner.RepositoryScanner` discovers all Python files as
   repository-relative paths.
2. :class:`repolens.index.SymbolIndexBuilder` indexes every class, function,
   and method definition (name, kind, line, parent class). No AST logic is
   duplicated here.
3. :class:`repolens.parser.PythonParser` supplies import/module names per
   file; each file's raw source text is read locally from disk in the same
   pass so files are parsed at most twice overall.

Searchable content (one token set per layer, per file)
------------------------------------------------------
- **path tokens**: the file path without its ``.py`` suffix
  (``billing/invoice.py`` → ``billing``, ``invoice``).
- **symbol tokens**: subtokens of every symbol name defined in the file,
  plus its parent class for methods (``InvoiceService.calculate_total`` →
  ``invoice``, ``service``, ``calculate``, ``total``).
- **import tokens**: subtokens of imported module names, ``from ... import``
  names, and aliases.
- **source tokens**: every word of the file's source text, including
  comments and docstrings.

Tokenization / normalization
----------------------------
A single tokenizer is applied uniformly to queries and content. Text is
split into alphanumeric runs (so spaces, underscores, hyphens, slashes,
dots, and punctuation all act as separators), then each run is split at
camelCase boundaries, then lowercased::

    "invoice_calculation"  -> ["invoice", "calculation"]
    "InvoiceCalculation"   -> ["invoice", "calculation"]
    "HTTPServerError"      -> ["https", "server", "error"]

There is no stemming, stop-word removal, or any other NLP: matching is exact
token equality, which keeps the behavior explainable.

Scoring formula
---------------
The score of a file is a non-negative integer summed over the *distinct*
query terms (repeated terms count once). For each term, the file earns the
weight of every layer it appears in:

========================================  =======
Match kind                                Weight
========================================  =======
Exact full symbol-name match              +100 per query term
Symbol name/parent-class token match      +10 per term
File-path token match                     +5 per term
Import/module name token match            +3 per term
Source-code token match                   +1 per term
========================================  ========

An *exact* symbol match means the whole normalized query equals one symbol's
full normalized name (e.g. ``"invoice service"``, ``"invoice_service"`` and
``"InvoiceService"`` all equal ``InvoiceService``), so a precise definition
lookup always outranks scattered token hits. Because weights strictly
decrease by tier, a file whose symbols embody the query outranks one that
merely mentions it in a comment, which in turn outranks unrelated files.

Deterministic ordering
----------------------
Results are sorted by ``(-score, repository-relative posix path)``. Scores
are integers computed from set membership only, so ties are broken purely
alphabetically and the output order is fully reproducible.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from repolens.index import Symbol, SymbolIndexBuilder
from repolens.parser import PythonParser
from repolens.scanner import RepositoryScanner

DEFAULT_LIMIT = 10

WEIGHT_EXACT_SYMBOL_NAME = 100
WEIGHT_SYMBOL_NAME_TOKEN = 10
WEIGHT_PATH_TOKEN = 5
WEIGHT_IMPORT_TOKEN = 3
WEIGHT_SOURCE_TOKEN = 1

_WORD_RUN = re.compile(r"[A-Za-z0-9]+")
_CAMEL_BOUNDARY = re.compile(r"[A-Z]+(?=[A-Z][a-z0-9])|[A-Z]?[a-z0-9]+|[A-Z]+")


def tokenize(text: str) -> list[str]:
    """Split text into lowercase comparison tokens.

    Any non-alphanumeric character acts as a separator, and camelCase or
    digit boundaries split identifier runs before lowercasing, so queries
    and identifiers written with different conventions normalize to the
    same tokens.
    """
    return [
        piece.lower()
        for run in _WORD_RUN.findall(text)
        for piece in _CAMEL_BOUNDARY.findall(run)
    ]


def _unique_in_order(tokens: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for token in tokens:
        if token not in seen:
            seen.add(token)
            unique.append(token)
    return unique


@dataclass(frozen=True)
class SearchResult:
    """One ranked file matching a search query.

    ``file_path`` is repository-relative. ``matched_terms`` lists the query
    terms that matched anywhere in the file, in original query order.
    ``symbols`` holds the definitions in this file whose names matched,
    ordered by file position.
    """

    file_path: Path
    score: int
    matched_terms: tuple[str, ...]
    symbols: tuple[Symbol, ...]


@dataclass(frozen=True)
class _SymbolEntry:
    """Precomputed lookup data for one indexed symbol."""

    symbol: Symbol
    tokens: frozenset[str]
    compact_name: str


@dataclass(frozen=True)
class _FileRecord:
    """All searchable content for a single Python file."""

    path: Path
    path_tokens: frozenset[str]
    import_tokens: frozenset[str]
    source_tokens: frozenset[str]
    symbols: tuple[_SymbolEntry, ...]


class CodeSearcher:
    """Deterministic lexical search over a repository.

    Example::

        searcher = CodeSearcher(repo_root)
        results = searcher.search("invoice calculation", limit=10)

    Construction scans, parses, and indexes the repository once; each
    :meth:`search` call is pure computation over that index.
    """

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self._parser = PythonParser()
        self._paths = RepositoryScanner(self.root).discover_python_files()
        self._symbols_by_path = self._group_symbols_by_path()
        self._records = tuple(self._build_records())

    def search(self, query: str, limit: int = DEFAULT_LIMIT) -> list[SearchResult]:
        """Return up to ``limit`` results ranked by deterministic relevance.

        An empty or separator-only query, a non-positive limit, or a
        repository without matches yields an empty list.
        """
        terms = _unique_in_order(tokenize(query))
        if not terms or limit <= 0:
            return []
        candidates = []
        for record in self._records:
            score, matched_terms, symbols = self._score(record, terms)
            if score > 0:
                candidates.append(
                    SearchResult(
                        file_path=record.path,
                        score=score,
                        matched_terms=matched_terms,
                        symbols=symbols,
                    )
                )
        candidates.sort(key=lambda result: (-result.score, result.file_path.as_posix()))
        return candidates[:limit]

    def _group_symbols_by_path(self) -> dict[Path, list[Symbol]]:
        by_path: dict[Path, list[Symbol]] = defaultdict(list)
        index = SymbolIndexBuilder(self.root).build()
        for symbol in index.get_all_symbols():
            by_path[symbol.file_path].append(symbol)
        return by_path

    def _build_records(self) -> Iterator[_FileRecord]:
        for path in self._paths:
            source_tokens, import_tokens = self._source_and_import_tokens(path)
            yield _FileRecord(
                path=path,
                path_tokens=frozenset(tokenize(path.with_suffix("").as_posix())),
                import_tokens=import_tokens,
                source_tokens=source_tokens,
                symbols=tuple(self._symbol_entries(path)),
            )

    def _symbol_entries(self, path: Path) -> Iterator[_SymbolEntry]:
        for symbol in self._symbols_by_path.get(path, []):
            tokens = tokenize(symbol.name)
            if symbol.parent_class:
                tokens.extend(tokenize(symbol.parent_class))
            yield _SymbolEntry(
                symbol=symbol,
                tokens=frozenset(tokens),
                compact_name="".join(tokenize(symbol.name)),
            )

    def _source_and_import_tokens(self, path: Path) -> tuple[frozenset[str], frozenset[str]]:
        try:
            source = (self.root / path).read_text(encoding="utf-8")
            analysis = self._parser.parse_source(source, file_path=path)
        except (OSError, SyntaxError, ValueError):
            return frozenset(), frozenset()
        import_tokens: set[str] = set()
        for imported in analysis.imports:
            import_tokens.update(tokenize(imported.module))
            if imported.alias:
                import_tokens.update(tokenize(imported.alias))
        for from_import in analysis.from_imports:
            import_tokens.update(tokenize(from_import.module))
            import_tokens.update(tokenize(from_import.name))
            if from_import.alias:
                import_tokens.update(tokenize(from_import.alias))
        return frozenset(tokenize(source)), frozenset(import_tokens)

    def _score(
        self, record: _FileRecord, terms: list[str]
    ) -> tuple[int, tuple[str, ...], tuple[Symbol, ...]]:
        score = 0
        hits: set[str] = set()
        matched_symbols: list[Symbol] = []
        compact_query = "".join(terms)
        for entry in record.symbols:
            symbol_hit = False
            for term in terms:
                if term in entry.tokens:
                    score += WEIGHT_SYMBOL_NAME_TOKEN
                    symbol_hit = True
                    hits.add(term)
            if entry.compact_name and entry.compact_name == compact_query:
                score += WEIGHT_EXACT_SYMBOL_NAME * len(terms)
                symbol_hit = True
                hits.update(terms)
            if symbol_hit:
                matched_symbols.append(entry.symbol)
        for term in terms:
            if term in record.path_tokens:
                score += WEIGHT_PATH_TOKEN
                hits.add(term)
            if term in record.import_tokens:
                score += WEIGHT_IMPORT_TOKEN
                hits.add(term)
            if term in record.source_tokens:
                score += WEIGHT_SOURCE_TOKEN
                hits.add(term)
        matched_terms = tuple(term for term in terms if term in hits)
        return score, matched_terms, tuple(matched_symbols)
