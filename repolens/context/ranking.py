"""Deterministic context-candidate ranking.

Primary (directly retrieved) files rank above files discovered only through
dependency expansion.

Primary-ranking policy (in order):
1. retrieval rank (1-based, lower is better; unranked last),
2. retrieval score (higher is better),
3. repository-relative path (alphabetical).

Dependency-expanded ranking policy (in order):
1. graph distance (closer first),
2. relationship strength (dependents — reverse dependencies / callers — rank
   before dependencies at equal distance),
3. repository-relative path (alphabetical, tie-break).

The ranking is a fixed, explainable policy. It is not learned and does not
use an LLM, and it introduces no tunable coefficients.
"""

from __future__ import annotations

from repolens.context.candidate import (
    INCLUSION_SYMBOL_MATCH,
    CandidateRole,
    ContextCandidate,
)


def _candidate_key(candidate: ContextCandidate) -> tuple:
    """Return the deterministic sort key for a single candidate."""
    if candidate.role is CandidateRole.PRIMARY:
        primary_rank = (
            candidate.retrieval_rank
            if candidate.retrieval_rank is not None
            else 10**9
        )
        score = candidate.retrieval_score if candidate.retrieval_score is not None else 0.0
        # Symbol-matched primaries outrank equally-retrieved non-symbol matches,
        # *after* the existing retrieval-rank order is respected. Defaults
        # (inclusion_reason is None) leave the historical ordering unchanged.
        symbol_boost = (
            0 if candidate.inclusion_reason == INCLUSION_SYMBOL_MATCH else 1
        )
        return (
            0,
            primary_rank,
            symbol_boost,
            -score,
            candidate.path.as_posix(),
        )

    # Dependency-expanded candidate.
    role_order = {
        CandidateRole.DEPENDENT: 0,
        CandidateRole.DEPENDENCY: 1,
    }
    distance = candidate.graph_distance if candidate.graph_distance is not None else 10**9
    return (
        1,
        distance,
        role_order[candidate.role],
        candidate.path.as_posix(),
    )


def rank_candidates(candidates: list[ContextCandidate]) -> list[ContextCandidate]:
    """Return ``candidates`` sorted by the deterministic context-ranking policy.

    The input list is not modified.
    """
    return sorted(candidates, key=_candidate_key)
