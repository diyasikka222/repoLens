"""Focused-context substitution during budget selection (P26.2 Step 4).

When a full-file candidate cannot fit in the remaining context budget, budget
selection tries a *focused-symbol* representation of the same file before
permanently rejecting it.  This module is the opt-in production wiring for the
additive focused model already built in :mod:`repolens.context.focus`.

Design contract (all deterministic):

- :class:`FocusedSelection` synthesizes a *bounded* set of focused candidates
  for one oversized candidate using the existing :func:`focus_candidate` API.
- Symbols are preferred using existing evidence, in this order of determinism:
  (1) names recorded per path in ``evidence`` (query symbol matches, change-plan
  target symbols), then (2) the candidate's own ``symbol`` field when present
  (change-plan candidates carry it).  When no symbol-level evidence names any
  symbol in the file, a bounded fallback takes the first ``fallback_limit``
  spans in deterministic ``(start line, name)`` order — never every symbol in a
  large file.
- Every focused candidate keeps the full-file candidate's provenance verbatim
  (role, retrieval signals, inclusion reason, architecture/change-plan
  metadata) via :func:`focus_candidate`, and recomputes ``estimated_tokens``
  for the actual focused slice.
- The full-file candidate is rejected exactly as before (``exceeds_total_budget``
  when it alone exceeds the whole budget, else ``over_budget``); the focused
  items are then elected against the *same* remaining budget.  This preserves
  ranking semantics (only the higher-ranked file's focused slice can consume
  budget before lower-ranked candidates are considered).
- Each synthesized focused item emits a JSON-safe *focus event* through the
  optional ``on_event`` sink recording why focus was attempted
  (``symbol_evidence`` / ``fallback``) and why the full representation was
  rejected.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Sequence

from repolens.context.candidate import ContextCandidate
from repolens.context.focus import extract_symbol_spans, focus_candidate

#: Bounded fallback when a file carries no symbol-level evidence.  Never
#: expands every symbol in a large file.
DEFAULT_FALLBACK_LIMIT = 3

#: Overall cap of focused candidates synthesized for one oversized file, even
#: when evidence names more symbols.
DEFAULT_MAX_FOCUSED = 8

#: Machine-readable reason a fallback of symbols was chosen over evidence.
FOCUS_REASON_EVIDENCE = "symbol_evidence"
FOCUS_REASON_FALLBACK = "fallback"

#: All keys a focus event may carry (JSON-safe).
_FOCUS_EVENT_KEYS = (
    "path",
    "focus_name",
    "focus_kind",
    "focus_start_line",
    "focus_end_line",
    "focused_estimated_tokens",
    "full_estimated_tokens",
    "focus_reason",
    "full_rejection_reason",
)


def snapshot_focus_event(event: dict) -> dict:
    """Return a JSON-safe copy of a focus event (unknown keys dropped)."""
    return {key: event.get(key) for key in _FOCUS_EVENT_KEYS}


@dataclass(frozen=True)
class FocusedSelection:
    """Deterministic synthesizer of bounded focused candidates for selection.

    ``evidence`` maps a repository-relative path to the symbol names known to be
    relevant for that file (query symbol matches, change-plan targets). When a
    name matches a symbol span, that span is preferred. ``fallback_limit`` caps
    how many spans are synthesized for a file with no evidence;
    ``max_focused`` caps how many focused candidates any single oversized file
    may contribute. ``on_event`` (optional) receives one JSON-safe dict per
    synthesized focused candidate.
    """

    evidence: Mapping[Path, Sequence[str]] | None = None
    fallback_limit: int = DEFAULT_FALLBACK_LIMIT
    max_focused: int = DEFAULT_MAX_FOCUSED
    on_event: Callable[[dict], None] | None = None

    def focused_items_for(
        self,
        candidate: ContextCandidate,
        *,
        full_reason: str,
    ) -> tuple[ContextCandidate, ...]:
        """Return the bounded, evidence-prioritized focused items for ``candidate``.

        ``full_reason`` is the deterministic rejection reason for the full-file
        representation (``exceeds_total_budget`` / ``over_budget``) and is
        recorded on every emitted focus event.  An already-focused candidate
        yields nothing; a file with no parseable symbols yields nothing.
        Ordering is deterministic: evidence names in ``evidence`` order first,
        matching the deterministic span order.
        """
        if candidate.is_focused:
            return ()
        spans = extract_symbol_spans(candidate.source, candidate.path)
        if not spans:
            return ()
        names = self._names_for(candidate)
        if names:
            preferred = [span for span in spans if span.name in names]
            chosen_spans = preferred[: self.max_focused]
            focus_reason = FOCUS_REASON_EVIDENCE
        else:
            chosen_spans = spans[: self.fallback_limit]
            focus_reason = FOCUS_REASON_FALLBACK
        if not chosen_spans:
            return ()
        items = tuple(focus_candidate(candidate, span) for span in chosen_spans)
        for item in items:
            self._emit(item, candidate, focus_reason, full_reason)
        return items

    def _names_for(self, candidate: ContextCandidate) -> tuple[str, ...]:
        names: list[str] = []
        if self.evidence is not None:
            for name in self.evidence.get(candidate.path, ()):
                if name and name not in names:
                    names.append(name)
        if candidate.symbol and candidate.symbol not in names:
            names.append(candidate.symbol)
        return tuple(names)

    def _emit(
        self,
        item: ContextCandidate,
        candidate: ContextCandidate,
        focus_reason: str,
        full_reason: str,
    ) -> None:
        if self.on_event is None:
            return
        self.on_event(
            {
                "path": item.path.as_posix(),
                "focus_name": item.focus_name,
                "focus_kind": item.focus_kind,
                "focus_start_line": item.focus_start_line,
                "focus_end_line": item.focus_end_line,
                "focused_estimated_tokens": item.estimated_tokens,
                "full_estimated_tokens": candidate.estimated_tokens,
                "focus_reason": focus_reason,
                "full_rejection_reason": full_reason,
            }
        )