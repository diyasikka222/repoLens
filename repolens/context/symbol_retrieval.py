"""Symbol-aware retrieval for the context engine (M19).

When a developer query names a known function, class, method, or module, the
engine should give that file priority. This layer re-uses the *existing*
symbol system (:class:`~repolens.index.SymbolIndex`), built either from a
:class:`~repolens.incremental_index.RepositoryIndex` snapshot or by scanning
the repository, and does not introduce a second symbol representation.

It provides:

- :class:`SymbolMatch` — a matched symbol plus the reason it matched;
- :func:`match_symbols` — find symbols in a query, preferring exact identifier
  matches and falling back to near-exact (shared-token) matches.

Nothing here performs lexical/semantic retrieval; it only identifies *which*
known symbols a query references so the engine can boost those files.
"""

from __future__ import annotations

from dataclasses import dataclass

from repolens.context.intent import extract_symbol_tokens
from repolens.index import Symbol, SymbolIndex, SymbolIndexBuilder

#: Category value reported for files boosted by symbol matching.
SYMBOL_MATCH = "symbol_match"


@dataclass(frozen=True)
class SymbolMatch:
    """A known symbol referenced by a query."""

    symbol: Symbol
    exact: bool

    @property
    def path(self):
        return self.symbol.file_path


def _normalize(name: str) -> str:
    """Return the tokens of ``name`` joined without separators (for equality)."""
    return name.replace("_", "").lower()


def build_symbol_index(root, index: object | None) -> SymbolIndex:
    """Return the existing :class:`SymbolIndex` for ``root``.

    When an incremental snapshot is provided its precomputed symbols are used,
    so no re-parsing happens; otherwise the index is built by scanning.
    """
    return SymbolIndexBuilder(root, index=index).build()


def _exact_matches(
    symbol_index: SymbolIndex, identifiers: list[str]
) -> dict[str, list[SymbolMatch]]:
    """Map an identifier token to its exact symbol matches (by name).

    ``identifiers`` are raw query tokens (already lowercased); we compare
    against the normalised (lowercased) symbol names so case and underscore
    variance are tolerated, analogous to the lexical tokenizer.
    """
    results: dict[str, list[SymbolMatch]] = {}
    for token in identifiers:
        for symbol in symbol_index.get_all_symbols():
            if _normalize(symbol.name) == _normalize(token):
                results.setdefault(token, []).append(SymbolMatch(symbol, exact=True))
    return results


def _near_matches(
    symbol_index: SymbolIndex, identifiers: list[str]
) -> dict[str, list[SymbolMatch]]:
    """Map an identifier token to near-exact matches (shared token overlap).

    A symbol is a near match for a candidate identifier when every token of the
    (compact) identifier appears among the symbol's own name tokens. This lets
    e.g. ``invoice_service`` match both ``InvoiceService`` and an
    ``invoice_service`` helper without a strict substring requirement.
    """
    results: dict[str, list[SymbolMatch]] = {}
    for token in identifiers:
        token_tokens = set(_compact_tokens(token))
        if not token_tokens:
            continue
        for symbol in symbol_index.get_all_symbols():
            symbol_tokens = set(_compact_tokens(symbol.name))
            symbol_tokens = {t for t in symbol_tokens if len(t) > 1} or symbol_tokens
            if symbol_tokens and token_tokens <= symbol_tokens:
                # Avoid duplicating an already-exact match for this token.
                if any(m.symbol is symbol for m in results.get(token, [])):
                    continue
                results.setdefault(token, []).append(SymbolMatch(symbol, exact=False))
    return results


def _compact_tokens(name: str) -> list[str]:
    import re

    parts = re.split(r"[_ ]+", name)
    tokens: list[str] = []
    for part in parts:
        if not part:
            continue
        tokens.extend(re.findall(r"[a-z0-9]+|[A-Z][a-z0-9]*|[A-Z]+", part))
    return sorted({t.lower() for t in tokens})


def match_symbols(
    query: str,
    root,
    index: object | None = None,
    *,
    symbol_index: SymbolIndex | None = None,
) -> list[SymbolMatch]:
    """Return known symbols referenced by ``query``, exact matches first.

    ``symbol_index`` may be supplied to avoid rebuilding the index; otherwise
    it is derived from ``root`` (and the optional incremental ``index``).
    Results are ordered deterministically: exact matches before near matches,
    then by file path, then by symbol name.
    """
    if symbol_index is None:
        symbol_index = build_symbol_index(root, index)
    identifiers = extract_symbol_tokens(query)
    if not identifiers:
        return []

    seen: set[Symbol] = set()
    ordered: list[SymbolMatch] = []

    def _add(token: str, match: SymbolMatch) -> None:
        if match.symbol in seen:
            return
        seen.add(match.symbol)
        ordered.append(match)

    # Exact first, then near.
    exact = _exact_matches(symbol_index, identifiers)
    for token in identifiers:
        for match in exact.get(token, []):
            _add(token, match)
    near = _near_matches(symbol_index, identifiers)
    for token in identifiers:
        for match in near.get(token, []):
            _add(token, match)

    ordered.sort(
        key=lambda m: (
            0 if m.exact else 1,
            m.symbol.file_path.as_posix(),
            m.symbol.name,
        )
    )
    return ordered


def symbol_file_paths(matches: list[SymbolMatch]) -> set:
    """Return the set of file paths referenced by ``matches``."""
    return {m.symbol.file_path for m in matches}