"""Hybrid retrieval combining lexical and semantic search.

:class:`HybridSearcher` merges the ranked results from a lexical
:class:`~repolens.search.CodeSearcher` and a semantic
:class:`~repolens.semantic_search.SemanticSearcher` into a single ranking
using a configurable weighted score over min-max normalised sub-scores.

Normalisation
-------------
Each sub-retriever produces results in its own score space:

- Lexical: non-negative integers (weights sum per query term, typically
  0–215).
- Semantic: cosine similarities in ``[-1.0, 1.0]`` (only positive values
  are returned by ``SemanticSearcher``).

To combine them safely, scores are **min-max normalised** independently
within the set of candidates each retriever actually returns:

::

    normalised_score(x) =
        (score(x) - min_score) / (max_score - min_score)
        if max_score != min_score else 1.0

A file that appears in only one retriever's results receives a normalised
score of ``0.0`` for the missing signal. This means a file found by only
lexical search is penalised relative to a file found by both, which
encourages the hybrid system to prefer results validated by multiple
signals.

Hybrid score
------------
::

    hybrid_score = lexical_weight * norm_lexical + semantic_weight * norm_semantic

Default weights are ``0.5`` each. Weights must be non-negative and need
not sum to 1 (normalisation handles the scale).

Ranking
-------
Results are sorted by ``(-hybrid_score, file_path.posix())``. Ties are
broken alphabetically by repository-relative path. Each file appears at
most once. Results respect the caller-supplied ``limit``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from repolens.search import CodeSearcher, SearchResult
from repolens.semantic_search import SemanticSearcher, SemanticResult

DEFAULT_LIMIT = 10
DEFAULT_LEXICAL_WEIGHT = 0.5
DEFAULT_SEMANTIC_WEIGHT = 0.5


def _min_max_normalise(
    values: dict[Path, float],
) -> dict[Path, float]:
    """Min-max normalise a {path: score} mapping to [0.0, 1.0].

    If all values are equal the result is ``1.0`` for every entry.
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


@dataclass(frozen=True)
class HybridResult:
    """One ranked file from the hybrid retriever.

    ``file_path`` is repository-relative. ``hybrid_score`` is the weighted
    combination of the normalised lexical and semantic sub-scores.
    ``lexical_contribution`` and ``semantic_contribution`` record each
    sub-score after normalisation, weighted by the configured weight.
    """

    file_path: Path
    hybrid_score: float
    lexical_contribution: float
    semantic_contribution: float


class HybridSearcher:
    """Combines lexical and semantic retrieval into one ranking.

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
    ) -> None:
        self.root = Path(root)
        self._lexical = lexical_searcher
        self._semantic = semantic_searcher
        self._lexical_weight = lexical_weight
        self._semantic_weight = semantic_weight

    def search(
        self, query: str, limit: int = DEFAULT_LIMIT
    ) -> list[HybridResult]:
        """Return up to ``limit`` files ranked by hybrid score.

        An empty or whitespace-only query, a non-positive limit, or
        repository with no matches yields an empty list.
        """
        if not query.strip() or limit <= 0:
            return []

        lexical_results = self._lexical.search(query, limit=limit)
        semantic_results = self._semantic.search(query, limit=limit)

        # Collect raw scores per path
        lexical_scores: dict[Path, float] = {
            r.file_path: float(r.score) for r in lexical_results
        }
        semantic_scores: dict[Path, float] = {
            r.file_path: r.similarity for r in semantic_results
        }

        # Normalise independently
        norm_lexical = _min_max_normalise(lexical_scores)
        norm_semantic = _min_max_normalise(semantic_scores)

        # Union of all candidate paths
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
                )
            )

        scored.sort(key=lambda r: (-r.hybrid_score, r.file_path.as_posix()))
        return scored[:limit]
