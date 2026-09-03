"""Deterministic query-intent classification for the context engine (M19).

A small, lightweight, rule-based classifier that labels a developer's
repository question so the engine can tailor dependency expansion. It never
uses an LLM and never calls any external service: classification is a pure,
deterministic function of the query text.

The classifier distinguishes four practical intents:

- :attr:`QueryIntent.IMPLEMENTATION` — "where is X implemented / defined"
- :attr:`QueryIntent.EXPLANATION` — "how does X work / flow"
- :attr:`QueryIntent.DEPENDENCY` — "what depends on X", "what uses X",
  "impact of changing X"
- :attr:`QueryIntent.MODIFICATION` — "where should I modify X", "what files
  should I change"

Ambiguous or uninteresting queries fall back to :attr:`QueryIntent.UNKNOWN`.

The same module exposes :func:`extract_symbol_tokens`, a tiny helper that
pulls likely identifier tokens (camelCase / snake_case runs) out of a query so
the symbol-aware retrieval layer can look them up in the existing symbol index.
"""

from __future__ import annotations

import re
from enum import Enum

#: A distinct run of (alphanumeric) characters that may be a code identifier.
_WORD_RUN = re.compile(r"[A-Za-z0-9_]+")

#: Intent rule set. Keys are integer specificity (higher wins); each rule is a
#: (pattern, intent) pair. Patterns are matched as word boundaries on the
#: lowercased query, earliest registered order preserves determinism.
_IMPLEMENTATION_TERMS = (
    "where is", "where's", "where can i find", "implemented", "implementation",
    "implement", "defined", "definition", "defines", "located where",
)
_EXPLANATION_TERMS = (
    "how does", "how do", "how is", "how are", "explain", "work", "works",
    "flow", "flows", "functioning", "what happens when", "what does",
)
_DEPENDENCY_TERMS = (
    "what depends", "what uses", "uses", "depends on", "dependency",
    "dependencies", "impact of", "impact", "affected by", "affected",
    "who imports", "imports", "called by", "callers", "references",
)
_MODIFICATION_TERMS = (
    "where should i modify", "where should i change", "where do i modify",
    "how do i change", "how should i change", "what files should i change",
    "what files do i need to change", "modify", "modification", "edit",
    "change", "update to change", "add the feature", "fix the bug",
)


class QueryIntent(str, Enum):
    """A deterministic classification of a repository question."""

    IMPLEMENTATION = "implementation"
    EXPLANATION = "explanation"
    DEPENDENCY = "dependency"
    MODIFICATION = "modification"
    UNKNOWN = "unknown"


def _tokens(query: str) -> list[str]:
    return [t.lower() for t in _WORD_RUN.findall(query)]


def _contains_any(query: str, terms: tuple[str, ...]) -> bool:
    lowered = " " + query.lower() + " "
    return any(f" {term.rstrip()} " in lowered or lowered.startswith(f" {term}")
                for term in terms)


def classify_intent(query: str) -> QueryIntent:
    """Return the rule-based intent for ``query`` (deterministic).

    Order of checks: modification, dependency/impact, implementation,
    explanation. The first matching, most specific intent wins; a query that
    matches none is :attr:`QueryIntent.UNKNOWN`.
    """
    if not query or not query.strip():
        return QueryIntent.UNKNOWN

    if _contains_any(query, _MODIFICATION_TERMS):
        return QueryIntent.MODIFICATION
    if _contains_any(query, _DEPENDENCY_TERMS):
        return QueryIntent.DEPENDENCY
    if _contains_any(query, _IMPLEMENTATION_TERMS):
        return QueryIntent.IMPLEMENTATION
    if _contains_any(query, _EXPLANATION_TERMS):
        return QueryIntent.EXPLANATION
    return QueryIntent.UNKNOWN


def extract_symbol_tokens(query: str) -> list[str]:
    """Return candidate identifier tokens from ``query``, in order, deduplicated.

    Any non-stopword alphanumeric token that contains a letter is considered a
    symbol candidate. This intentionally includes plain lowercase words (e.g.
    ``login``, ``refund``) because real modules are routinely referenced that
    way in prose; candidates that do not match any real symbol are simply
    ignored downstream. The result is deterministic and conservative upstream
    of the symbol index, which is the actual authority on what exists.
    """
    words = _tokens(query)
    stop = {
        "where", "what", "how", "which", "when", "who", "is", "are", "was",
        "does", "do", "did", "i", "the", "a", "an", "of", "to", "in", "and",
        "or", "that", "this", "it", "its", "for", "with", "from", "on", "at",
        "by", "should", "can", "could", "would", "me", "you", "not", "have",
        "has", "be", "so", "as", "if", "then", "than", "which", "there",
        "explain", "implemented", "implementation", "implement", "defined",
        "definition", "modify", "modification", "change", "edited", "edit",
        "depend", "depends", "dependency", "dependencies", "impact", "work",
        "works", "flow", "function", "found", "find", "file", "files", "code",
        "where's", "how's", "many", "much", "about", "into", "onto", "over",
        "under", "doesn't", "don't", "cannot", "before", "after", "during",
        "whenever", "why", "ever", "need", "use", "used", "using", "update",
    }
    seen: set[str] = set()
    result: list[str] = []
    for word in words:
        if word in stop:
            continue
        if not any(ch.isalpha() for ch in word):
            continue
        if word not in seen:
            seen.add(word)
            result.append(word)
    return result