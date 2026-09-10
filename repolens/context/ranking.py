"""Deterministic context-candidate ranking.

Primary (directly retrieved) files rank above files discovered only through
dependency expansion.

Primary-ranking policy (in order):
1. retrieval rank (1-based, lower is better; unranked last),
2. retrieval score (higher is better),
3. repository-relative path (alphabetical).

Symbol-discovered files that retrieval never surfaced (P26.1 Step 2) join the
primary tier as ``PRIMARY``-role candidates without retrieval signals: they
rank after every retrieved primary and before every dependency/architecture/
change-plan candidate, ordered by path. Only files already present as
primaries/dependencies are re-ranked — existing orderings are preserved.

Dependency-expanded ranking policy (in order):
1. graph distance (closer first),
2. relationship strength (dependents — reverse dependencies / callers — rank
   before dependencies at equal distance),
3. repository-relative path (alphabetical, tie-break).

Change-plan-only ranking policy (Milestone 24.2). Candidates introduced
exclusively by the change-plan layer (``inclusion_reason == "change_plan"``)
form a final tier *below* every retrieval primary and every
dependency-expanded/architecture candidate — change-plan signals never
outrank direct query matches:

1. tier 4 — strictly after all non-change tiers (0 direct match, 1 neighbor,
   2 proximity, 3 generic/unranked dependency);
2. change-plan inspection priority (lower inspection priority number first);
3. repository-relative path (alphabetical, tie-break).

A file that is both a retrieval/dependency/architecture candidate *and* a
change-plan candidate keeps its higher tier: the engine deduplicates on first
occurrence with primary/dependency/architecture candidates added before
change-plan candidates.

The ranking is a fixed, explainable policy. It is not learned and does not
use an LLM, and it introduces no tunable coefficients.
"""

from __future__ import annotations

from repolens.context.candidate import (
    INCLUSION_CHANGE_PLAN,
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

    # Change-plan-only candidates: the final, clearly-separated tier. They
    # always sort below every dependency/architecture candidate (whose arch
    # bucket is at most 3) and never above retrieval primaries.
    if candidate.inclusion_reason == INCLUSION_CHANGE_PLAN:
        return (
            1,
            4,
            candidate.change_priority if candidate.change_priority is not None else 10**9,
            candidate.path.as_posix(),
        )

    # Dependency-expanded candidate.
    role_order = {
        CandidateRole.DEPENDENT: 0,
        CandidateRole.DEPENDENCY: 1,
    }
    distance = candidate.graph_distance if candidate.graph_distance is not None else 10**9
    # Architecture rank creates a pre-distance bucket: 0 = direct match,
    # 1 = neighbour, 2 = generic proximity; bucket 3 = no architecture signal
    # (the historical behaviour, unchanged when architecture is disabled).
    arch_bucket = (
        min(candidate.architecture_rank, 2)
        if candidate.architecture_rank is not None
        else 3
    )
    return (
        1,
        arch_bucket,
        distance,
        role_order[candidate.role],
        candidate.path.as_posix(),
    )


def rank_candidates(candidates: list[ContextCandidate]) -> list[ContextCandidate]:
    """Return ``candidates`` sorted by the deterministic context-ranking policy.

    The input list is not modified.
    """
    return sorted(candidates, key=_candidate_key)
