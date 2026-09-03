"""The final context package produced by the dependency-aware context engine.

A :class:`ContextPackage` is the complete, serializable result of answering a
developer query: the query itself, the files selected within the configured
budget, the retrieval and dependency candidates that fed the selection, the
candidates that were excluded and why, and the total estimated token cost.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from repolens.context.candidate import ContextCandidate, ExcludedCandidate
from repolens.context.config import ContextBudget


@dataclass(frozen=True)
class ContextPackage:
    """A serializable context package for one query."""

    query: str
    budget: ContextBudget
    selected_files: tuple[ContextCandidate, ...] = ()
    primary_candidates: tuple[ContextCandidate, ...] = ()
    dependency_candidates: tuple[ContextCandidate, ...] = ()
    excluded_candidates: tuple[ExcludedCandidate, ...] = ()
    intent: str | None = None
    matched_symbols: tuple[str, ...] = ()

    @property
    def total_estimated_tokens(self) -> int:
        """Sum of estimated tokens across the selected files."""
        return sum(item.estimated_tokens for item in self.selected_files)

    def to_dict(self) -> dict:
        """Return a JSON-serializable dict (paths as posix strings)."""
        data = {
            "query": self.query,
            "budget": {
                "max_tokens": self.budget.max_tokens,
                "truncate_oversized": self.budget.truncate_oversized,
            },
            "total_estimated_tokens": self.total_estimated_tokens,
            "intent": self.intent,
            "matched_symbols": list(self.matched_symbols),
            "selected_files": [_candidate_dict(c) for c in self.selected_files],
            "primary_candidates": [_candidate_dict(c) for c in self.primary_candidates],
            "dependency_candidates": [_candidate_dict(c) for c in self.dependency_candidates],
            "excluded_candidates": [_excluded_dict(c) for c in self.excluded_candidates],
        }
        return data

    def to_json(self, **json_kwargs) -> str:
        """Return the package serialized as a JSON string."""
        return json.dumps(self.to_dict(), **json_kwargs)


def _candidate_dict(candidate: ContextCandidate) -> dict:
    return {
        "path": candidate.path.as_posix(),
        "role": candidate.role.value,
        "estimated_tokens": candidate.estimated_tokens,
        "selection_reason": candidate.selection_reason,
        "inclusion_reason": candidate.inclusion_reason,
        "retrieval_rank": candidate.retrieval_rank,
        "retrieval_score": candidate.retrieval_score,
        "lexical_rank": candidate.lexical_rank,
        "semantic_rank": candidate.semantic_rank,
        "graph_distance": candidate.graph_distance,
    }


def _excluded_dict(candidate: ExcludedCandidate) -> dict:
    return {
        "path": candidate.path.as_posix(),
        "estimated_tokens": candidate.estimated_tokens,
        "reason": candidate.reason,
    }
