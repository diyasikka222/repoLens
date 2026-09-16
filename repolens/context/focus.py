"""Focused source representation for oversized files (P26.2 feasibility).

When a single file is larger than the remaining context budget, the only way
to surface it without raising the budget is to expose a *focused portion* of
its source instead of the whole file.  This module is the additive, internal
representation for that idea.  It is deliberately not wired into the default
``get_context`` pipeline — nothing here changes ranking, selection, the token
budget, the firewall, or MCP output unless a producer opts in by turning a
full-file :class:`~repolens.context.ContextCandidate` into focused items.

Core model:

- :class:`SourceSpan` — a 1-based, inclusive line range; knows how to slice a
  source string deterministically and how to map a firewall-relative line back
  to the original file.
- :class:`SymbolSpan` — a parsed definition span (``path``, ``name``, kind,
  optional ``parent_class``, ``start_line``/``end_line``).
- :func:`extract_symbol_spans` — deterministic AST-based extraction that
  mirrors the existing parser's scope rule (top-level functions/classes plus
  class-body methods) and captures the end line that
  :class:`~repolens.index.Symbol` does not persist.
- :func:`focus_candidate` — turn an existing full-file candidate into a
  focused one: ``source`` is the symbol's line range, ``estimated_tokens`` is
  recomputed for that slice, and every provenance field (role, retrieval
  signals, graph distance, inclusion/selection reasons, architecture and
  change-plan metadata) is preserved verbatim.
- :func:`focused_items_for_file` — the full-file candidate followed by one
  focused candidate per symbol span, in deterministic order.  This is how a
  full-file item and multiple focused symbols from one file coexist.
- :func:`focus_package` — opt-in expansion of a whole
  :class:`~repolens.context.ContextPackage` into full + focused items
  (a no-op until called; never invoked by the engine, MCP, or CLI).

A focused item is still a :class:`~repolens.context.ContextCandidate`, so it
flows through the existing ranking, budget, and firewall machinery unchanged.

Firewall finding lines are relative to the focused slice (the firewall scans
``candidate.source``); :meth:`SourceSpan.original_line` maps them back to the
original file for provenance.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from repolens.context.candidate import ContextCandidate
from repolens.context.package import ContextPackage
from repolens.context.tokens import estimate_tokens
from repolens.index import SymbolKind


@dataclass(frozen=True, order=True)
class SourceSpan:
    """A 1-based, inclusive line range inside one source file."""

    start_line: int
    end_line: int

    def __post_init__(self) -> None:
        if self.start_line < 1:
            raise ValueError(
                f"start_line must be >= 1, got {self.start_line}"
            )
        if self.end_line < self.start_line:
            raise ValueError(
                f"end_line ({self.end_line}) must be >= start_line "
                f"({self.start_line})"
            )

    @property
    def line_count(self) -> int:
        """Number of lines covered by this span."""
        return self.end_line - self.start_line + 1

    def slice_source(self, source: str) -> str:
        """Return lines ``[start_line, end_line]`` of ``source`` inclusive.

        Deterministic: always keeps the original line endings via
        ``splitlines(keepends=True)``.  A span reaching past the end of the
        source simply yields the lines that exist.
        """
        lines = source.splitlines(keepends=True)
        return "".join(lines[self.start_line - 1 : self.end_line])

    def original_line(self, relative_line: int) -> int | None:
        """Map a firewall-relative ``relative_line`` back to the original file.

        Returns ``None`` when the relative line falls outside the span.
        """
        if relative_line < 1 or relative_line > self.line_count:
            return None
        return self.start_line + relative_line - 1


@dataclass(frozen=True, order=True)
class SymbolSpan:
    """The parsed definition span of one symbol in one file."""

    path: Path
    name: str
    kind: SymbolKind
    parent_class: str | None = None
    start_line: int = 0
    end_line: int = 0

    @property
    def span(self) -> SourceSpan:
        """The 1-based inclusive line range of this symbol."""
        return SourceSpan(self.start_line, self.end_line)

    def slice_source(self, source: str) -> str:
        """Return the source lines this symbol occupies."""
        return self.span.slice_source(source)

    def original_line(self, relative_line: int) -> int | None:
        """Map a relative line back to the original file (see
        :meth:`SourceSpan.original_line`)."""
        return self.span.original_line(relative_line)


def extract_symbol_spans(source: str, path: Path) -> tuple[SymbolSpan, ...]:
    """Parse ``source`` and return its deterministic definition spans.

    Scope mirrors :class:`~repolens.parser.PythonParser`: top-level functions
    and classes, plus methods defined directly in a class body.  Nested
    defs are intentionally ignored, exactly like the existing parser.

    The end line comes from the AST ``end_lineno`` (never persisted by the
    symbol index), falling back to the start line when the node reports no
    end.  For decorated definitions the span starts at the first decorator
    line so the focused slice keeps its surrounding annotations.  Results are
    ordered deterministically by ``(path, start line, name)``.  Unparseable
    source yields an empty tuple (never an exception).
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return ()

    def _start(node) -> int:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.decorator_list:
            return node.decorator_list[0].lineno
        return node.lineno

    spans: list[SymbolSpan] = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            spans.append(
                SymbolSpan(
                    path=path,
                    name=node.name,
                    kind=SymbolKind.CLASS,
                    start_line=_start(node),
                    end_line=node.end_lineno or node.lineno,
                )
            )
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    spans.append(
                        SymbolSpan(
                            path=path,
                            name=child.name,
                            kind=SymbolKind.METHOD,
                            parent_class=node.name,
                            start_line=_start(child),
                            end_line=child.end_lineno or child.lineno,
                        )
                    )
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            spans.append(
                SymbolSpan(
                    path=path,
                    name=node.name,
                    kind=SymbolKind.FUNCTION,
                    start_line=_start(node),
                    end_line=node.end_lineno or node.lineno,
                )
            )
    return tuple(
        sorted(spans, key=lambda s: (s.path.as_posix(), s.start_line, s.name))
    )


def focus_candidate(candidate: ContextCandidate, span: SymbolSpan) -> ContextCandidate:
    """Return ``candidate`` focused to ``span``.

    The focused candidate keeps the original candidate's identity and every
    provenance field (path, role, retrieval signals, graph distance,
    inclusion/selection reasons, architecture and change-plan metadata), but
    its ``source`` is the symbol's line range and ``estimated_tokens`` are
    recomputed for that slice.  The new focus fields
    (``focus_name``/``focus_kind``/``focus_parent_class``/``focus_start_line``/
    ``focus_end_line``) record which symbol the slice came from.
    """
    if span.path != candidate.path:
        raise ValueError(
            f"span path {span.path.as_posix()!r} does not match candidate "
            f"path {candidate.path.as_posix()!r}"
        )
    focused_source = span.slice_source(candidate.source)
    return ContextCandidate(
        path=candidate.path,
        source=focused_source,
        role=candidate.role,
        estimated_tokens=estimate_tokens(focused_source),
        selection_reason=candidate.selection_reason,
        retrieval_rank=candidate.retrieval_rank,
        retrieval_score=candidate.retrieval_score,
        lexical_rank=candidate.lexical_rank,
        semantic_rank=candidate.semantic_rank,
        graph_distance=candidate.graph_distance,
        inclusion_reason=candidate.inclusion_reason,
        architecture_rank=candidate.architecture_rank,
        architecture_metadata=candidate.architecture_metadata,
        module=candidate.module,
        symbol=candidate.symbol,
        change_score=candidate.change_score,
        change_category=candidate.change_category,
        change_confidence=candidate.change_confidence,
        change_relationship=candidate.change_relationship,
        change_priority=candidate.change_priority,
        focus_name=span.name,
        focus_kind=span.kind.value,
        focus_parent_class=span.parent_class,
        focus_start_line=span.start_line,
        focus_end_line=span.end_line,
    )


def focused_items_for_file(
    candidate: ContextCandidate,
) -> tuple[ContextCandidate, ...]:
    """Return ``candidate`` plus one focused item per symbol span.

    The full-file item comes first and is byte-for-byte the original
    candidate; focused items follow in deterministic ``(start line, name)``
    order.  This is the representation that lets a full-file item and several
    focused symbols from the *same* file coexist in one package.  A file with
    no parseable symbols (or an already-focused candidate) yields just the
    full-file item.
    """
    if candidate.focus_start_line is not None:
        return (candidate,)
    spans = extract_symbol_spans(candidate.source, candidate.path)
    return (candidate,) + tuple(focus_candidate(candidate, span) for span in spans)


def focus_package(
    package: ContextPackage,
    *,
    paths: Sequence[Path] | None = None,
) -> ContextPackage:
    """Return ``package`` expanded into full + focused items (opt-in only).

    ``paths`` limits which selected files get focused expansions (``None``
    means every selected file).  Every selected file keeps its full-file item
    first, then its focused symbol items in deterministic order, so the
    derived package preserves deterministic ordering.  This function is never
    called by the engine, MCP, CLI, or any default path — it is the explicit
    opt-in the final budget-selection fix would use.
    """
    wanted = frozenset(paths) if paths is not None else None
    expanded: list[ContextCandidate] = []
    for candidate in package.selected_files:
        if wanted is not None and candidate.path not in wanted:
            expanded.append(candidate)
            continue
        expanded.extend(focused_items_for_file(candidate))
    return ContextPackage(
        query=package.query,
        budget=package.budget,
        selected_files=tuple(expanded),
        primary_candidates=package.primary_candidates,
        dependency_candidates=package.dependency_candidates,
        excluded_candidates=package.excluded_candidates,
        intent=package.intent,
        matched_symbols=package.matched_symbols,
        change_candidates=package.change_candidates,
    )