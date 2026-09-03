"""Token-budget-aware selection of context candidates.

The engine sorts the full candidate set with :func:`repolens.context.ranking.rank_candidates`
and then walks it in order, adding candidates to the final package while the
accumulated *estimated* token count stays within the configured budget.

Behaviour guarantees:

- deterministic (given the ranked input and budget, output is fixed);
- never intentionally exceeds the requested budget (``budget.max_tokens``);
- preserves highest-ranked candidates first (only a candidate that fits is
  dropped, and only to make room for higher-ranked ones is not the mechanic —
  ranking is fixed and respected);
- a single candidate larger than the entire budget is excluded with reason
  ``exceeds_total_budget`` and the walk continues so smaller candidates can
  still be included;
- a zero budget yields an empty selection;
- ``budget.max_tokens=None`` means unlimited (every candidate is included).

Candidates that are skipped because they would exceed the *remaining* budget
are returned with reason ``over_budget``.
"""

from __future__ import annotations

from dataclasses import replace

from repolens.context.candidate import ContextCandidate, ExcludedCandidate
from repolens.context.config import ContextBudget
from repolens.context.tokens import estimate_tokens

#: Rough characters per estimated token; mirrors :mod:`repolens.context.tokens`.
_CHARS_PER_TOKEN = 4


def select_within_budget(
    ranked: list[ContextCandidate],
    budget: ContextBudget,
) -> tuple[list[ContextCandidate], list[ExcludedCandidate]]:
    """Select candidates subject to ``budget``, preserving rank order.

    Returns ``(selected, excluded)`` where ``selected`` is in the same order
    as ``ranked`` and ``excluded`` lists the candidates that did not fit, each
    with the reason it was dropped.

    When ``budget.truncate_oversized`` is ``True``, a candidate larger than
    the *remaining* budget is truncated to its head so it still fits and is
    returned (with its source and estimated-token count recomputed) instead of
    being excluded. When ``False`` (the default) an oversized candidate is
    excluded exactly as before.
    """
    selected: list[ContextCandidate] = []
    excluded: list[ExcludedCandidate] = []
    remaining = budget.max_tokens

    for candidate in ranked:
        tokens = candidate.estimated_tokens

        if remaining is None:
            selected.append(candidate)
            continue

        if tokens <= remaining:
            selected.append(candidate)
            remaining -= tokens
            continue

        # Candidate does not fit in the remaining budget.
        if budget.truncate_oversized and tokens > 0 and remaining > 0:
            allowed_chars = remaining * _CHARS_PER_TOKEN
            truncated_source = candidate.source[:allowed_chars]
            truncated = replace(
                candidate,
                source=truncated_source,
                estimated_tokens=estimate_tokens(truncated_source),
            )
            selected.append(truncated)
            remaining -= truncated.estimated_tokens
            continue

        if remaining == budget.max_tokens:
            # Nothing fits at all (candidate alone exceeds the whole budget).
            excluded.append(
                ExcludedCandidate(
                    path=candidate.path,
                    estimated_tokens=tokens,
                    reason="exceeds_total_budget",
                )
            )
        else:
            excluded.append(
                ExcludedCandidate(
                    path=candidate.path,
                    estimated_tokens=tokens,
                    reason="over_budget",
                )
            )

    return selected, excluded
