"""Hybrid retrieval combining lexical and semantic search.

:class:`HybridSearcher` merges the ranked results from a lexical
:class:`~repolens.search.CodeSearcher` and a semantic
:class:`~repolens.semantic_search.SemanticSearcher` into a single ranking
using one of two configurable fusion strategies.

Fusion strategies
-----------------
**Weighted** (default):
    Scores are min-max normalised independently per signal, then combined
    as a weighted sum::

        hybrid = w_lex * norm_lex + w_sem * norm_sem

    Weights must be non-negative with at least one positive; they are
    normalised internally so ``w_lex + w_sem == 1.0``.

**Reciprocal Rank Fusion (RRF)**:
    Each document's score is the weighted sum of reciprocal ranks::

        rrf_score(d) = w_lex / (k + rank_lex(d)) + w_sem / (k + rank_sem(d))

    ``k`` is a configurable constant (default 60).  Documents absent from
    one list receive no contribution from that signal.

Normalisation
-------------
Each sub-retriever produces results in its own score space:

- Lexical: non-negative integers (token-match weights, typically 0–215).
- Semantic: cosine similarities in ``[-1.0, 1.0]`` (only positive values
  are returned by ``SemanticSearcher``).

For the weighted strategy, scores are **min-max normalised** independently
within the set of candidates each retriever actually returns.  The RRF
strategy uses rank-based scoring and does not require normalisation.

A file that appears in only one retriever's results receives a normalised
score of ``0.0`` for the missing signal (weighted) or no contribution (RRF).
This means a file found by only lexical search is penalised relative to a
file found by both signals.

Ranking
-------
Results are sorted by ``(-hybrid_score, file_path.posix())``.  Ties are
broken alphabetically by repository-relative path.  Each file appears at
most once.  Results respect the caller-supplied ``limit``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from repolens.search import CodeSearcher, SearchResult
from repolens.semantic_search import SemanticSearcher, SemanticResult

DEFAULT_LIMIT = 10
DEFAULT_LEXICAL_WEIGHT = 0.5
DEFAULT_SEMANTIC_WEIGHT = 0.5
DEFAULT_RRF_K = 60

# Fallback pool size for weighted fusion when the semantic searcher does not
# expose a candidate limit. The pool must be larger than any realistic result
# cap so that min-max normalisation is computed over a stable set and the
# ranking does not shift when a caller asks for fewer results.
DEFAULT_SCORE_POOL_SIZE = 1000


class FusionStrategy(Enum):
    """Available hybrid fusion strategies."""

    WEIGHTED = "weighted"
    RRF = "rrf"


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------


def min_max_normalise(
    values: dict[Path, float],
) -> dict[Path, float]:
    """Min-max normalise a ``{path: score}`` mapping to ``[0.0, 1.0]``.

    If all values are equal the result is ``1.0`` for every entry.
    An empty mapping returns an empty mapping.
    """
    if not values:
        return {}
    scores = list(values.values())
    min_score = min(scores)
    max_score = max(scores)
    span = max_score - min_score
    if span == 0.0:
        return {path: 1.0 for path in values}
    return {path: (score - min_score) / span for path, score in values.items()}


def _rank_map(results: list[Any], key: str = "file_path") -> dict[Path, int]:
    """Build a ``{path: 1-based rank}`` from an ordered result list."""
    return {getattr(r, key): rank for rank, r in enumerate(results, start=1)}


def _score_map(results: list[Any], key: str, score_attr: str) -> dict[Path, float]:
    """Build a ``{path: score}`` from an ordered result list."""
    return {getattr(r, key): float(getattr(r, score_attr)) for r in results}


# ---------------------------------------------------------------------------
# Weight validation
# ---------------------------------------------------------------------------


def _validate_weights(lexical_weight: float, semantic_weight: float) -> tuple[float, float]:
    """Validate and normalise weights to sum to 1.0.

    Requirements:
    - Both weights must be >= 0.
    - At least one weight must be > 0.

    Returns the normalised ``(lexical_weight, semantic_weight)`` pair.
    """
    if lexical_weight < 0 or semantic_weight < 0:
        raise ValueError(
            f"Weights must be non-negative, got lexical={lexical_weight}, semantic={semantic_weight}"
        )
    total = lexical_weight + semantic_weight
    if total == 0.0:
        raise ValueError("At least one weight must be positive")
    return (lexical_weight / total, semantic_weight / total)


# ---------------------------------------------------------------------------
# Result model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HybridResult:
    """One ranked file from the hybrid retriever.

    ``file_path`` is repository-relative.  ``hybrid_score`` is the fusion
    score (weighted or RRF).  The contribution and rank fields allow callers
    to understand *why* a file ranked highly.
    """

    file_path: Path
    hybrid_score: float

    # Weighted fusion details
    lexical_contribution: float = 0.0
    semantic_contribution: float = 0.0

    # Rank and raw-score provenance
    lexical_rank: int | None = None
    semantic_rank: int | None = None
    lexical_raw_score: float | None = None
    semantic_raw_score: float | None = None

    # Metadata
    fusion_strategy: str = FusionStrategy.WEIGHTED.value


# ---------------------------------------------------------------------------
# HybridSearcher
# ---------------------------------------------------------------------------


class HybridSearcher:
    """Combines lexical and semantic retrieval into one ranking.

    Supports two fusion strategies via the ``strategy`` parameter:

    - ``"weighted"`` (default): min-max normalised scores combined with
      configurable weights.
    - ``"rrf"``: Reciprocal Rank Fusion combining rank positions.

    Example::

        searcher = HybridSearcher(
            repo_root,
            lexical_searcher=CodeSearcher(repo_root),
            semantic_searcher=SemanticSearcher(repo_root, provider),
        )
        results = searcher.search("refund card payment", limit=5)

    Both sub-searchers are constructed externally and injected; this class
    never scans or embeds the repository itself.
    """

    def __init__(
        self,
        root: Path | str,
        *,
        lexical_searcher: CodeSearcher,
        semantic_searcher: SemanticSearcher,
        lexical_weight: float = DEFAULT_LEXICAL_WEIGHT,
        semantic_weight: float = DEFAULT_SEMANTIC_WEIGHT,
        strategy: FusionStrategy | str = FusionStrategy.WEIGHTED,
        rrf_k: int = DEFAULT_RRF_K,
    ) -> None:
        self.root = Path(root)
        self._lexical = lexical_searcher
        self._semantic = semantic_searcher
        self._strategy = (
            strategy if isinstance(strategy, FusionStrategy)
            else FusionStrategy(strategy)
        )
        self._rrf_k = rrf_k

        # Validate and store normalised weights
        self._lexical_weight, self._semantic_weight = _validate_weights(
            lexical_weight, semantic_weight
        )

    def search(
        self, query: str, limit: int = DEFAULT_LIMIT
    ) -> list[HybridResult]:
        """Return up to ``limit`` files ranked by hybrid score.

        An empty or whitespace-only query, a non-positive limit, or
        repository with no matches yields an empty list.

        The hybrid ranking is computed over a stable candidate pool (the
        semantic searcher's candidate limit) so that requesting a smaller
        ``limit`` never changes which file ranks first. Each retriever is
        asked for the pool in full, the fusion scores are computed over it,
        and only the final ranked list is truncated to ``limit``.
        """
        if not query.strip() or limit <= 0:
            return []

        pool_size = self._semantic_pool_size()
        lexical_results = self._lexical.search(query, limit=pool_size)
        semantic_results = self._semantic.search(query, limit=pool_size)

        if self._strategy == FusionStrategy.RRF:
            return self._search_rrf(lexical_results, semantic_results, limit)
        return self._search_weighted(lexical_results, semantic_results, limit)

    def _semantic_pool_size(self) -> int:
        """Return the stable candidate pool size for fusion.

        Uses the semantic searcher's own candidate limit when available (this
        is the frame its embeddings are computed over), falling back to a
        large fixed constant so normalisation is never truncated by ``limit``.
        """
        limit = getattr(self._semantic, "_candidate_limit", None)
        if limit is None or limit < 1:
            return DEFAULT_SCORE_POOL_SIZE
        return limit

    # -- weighted strategy ---------------------------------------------------

    def _search_weighted(
        self,
        lexical_results: list[SearchResult],
        semantic_results: list[SemanticResult],
        limit: int,
    ) -> list[HybridResult]:
        """Combine results using min-max normalised weighted scores."""
        lexical_scores = _score_map(lexical_results, "file_path", "score")
        semantic_scores = _score_map(semantic_results, "file_path", "similarity")

        norm_lexical = min_max_normalise(lexical_scores)
        norm_semantic = min_max_normalise(semantic_scores)

        lexical_ranks = _rank_map(lexical_results, "file_path")
        semantic_ranks = _rank_map(semantic_results, "file_path")

        all_paths = set(lexical_scores) | set(semantic_scores)

        scored: list[HybridResult] = []
        for path in all_paths:
            lex_norm = norm_lexical.get(path, 0.0)
            sem_norm = norm_semantic.get(path, 0.0)
            lex_contrib = self._lexical_weight * lex_norm
            sem_contrib = self._semantic_weight * sem_norm
            hybrid = lex_contrib + sem_contrib
            scored.append(
                HybridResult(
                    file_path=path,
                    hybrid_score=hybrid,
                    lexical_contribution=lex_contrib,
                    semantic_contribution=sem_contrib,
                    lexical_rank=lexical_ranks.get(path),
                    semantic_rank=semantic_ranks.get(path),
                    lexical_raw_score=lexical_scores.get(path),
                    semantic_raw_score=semantic_scores.get(path),
                    fusion_strategy=FusionStrategy.WEIGHTED.value,
                )
            )

        scored.sort(key=lambda r: (-r.hybrid_score, r.file_path.as_posix()))
        return scored[:limit]

    # -- RRF strategy --------------------------------------------------------

    def _search_rrf(
        self,
        lexical_results: list[SearchResult],
        semantic_results: list[SemanticResult],
        limit: int,
    ) -> list[HybridResult]:
        """Combine results using Reciprocal Rank Fusion."""
        k = self._rrf_k
        lexical_ranks = _rank_map(lexical_results, "file_path")
        semantic_ranks = _rank_map(semantic_results, "file_path")
        lexical_scores = _score_map(lexical_results, "file_path", "score")
        semantic_scores = _score_map(semantic_results, "file_path", "similarity")

        all_paths = set(lexical_ranks) | set(semantic_ranks)

        scored: list[HybridResult] = []
        for path in all_paths:
            lex_rank = lexical_ranks.get(path)
            sem_rank = semantic_ranks.get(path)

            lex_rrf = self._lexical_weight / (k + lex_rank) if lex_rank is not None else 0.0
            sem_rrf = self._semantic_weight / (k + sem_rank) if sem_rank is not None else 0.0
            hybrid = lex_rrf + sem_rrf

            scored.append(
                HybridResult(
                    file_path=path,
                    hybrid_score=hybrid,
                    lexical_contribution=lex_rrf,
                    semantic_contribution=sem_rrf,
                    lexical_rank=lex_rank,
                    semantic_rank=sem_rank,
                    lexical_raw_score=lexical_scores.get(path),
                    semantic_raw_score=semantic_scores.get(path),
                    fusion_strategy=FusionStrategy.RRF.value,
                )
            )

        scored.sort(key=lambda r: (-r.hybrid_score, r.file_path.as_posix()))
        return scored[:limit]
