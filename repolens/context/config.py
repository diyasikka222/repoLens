"""Configuration objects for the dependency-aware context engine.

Each configuration is a small frozen dataclass that the :class:`~repolens.context.engine.ContextEngine`
consumes. They capture the three knobs named in the milestone: how retrieval
is performed, how far dependency expansion reaches, and what token budget to
respect.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from repolens.embedding_cache import EmbeddingCache
from repolens.embeddings import EmbeddingProvider
from repolens.evaluation import Searcher

DEFAULT_RETRIEVAL_LIMIT = 8
DEFAULT_LEXICAL_WEIGHT = 0.5
DEFAULT_SEMANTIC_WEIGHT = 0.5
DEFAULT_RRF_K = 60

DEFAULT_TOTAL_BUDGET = 8000
DEFAULT_DEPTH = 1


@dataclass(frozen=True)
class RetrievalConfig:
    """Describes which retrieval strategy the context engine should use.

    ``strategy`` selects the existing retrieval algorithm (RRF hybrid by
    default); the weights and RRF constant match the production defaults and
    are never changed here. The engine consumes the chosen searcher's results;
    it does not re-implement retrieval.
    """

    strategy: str = "rrf"
    limit: int = DEFAULT_RETRIEVAL_LIMIT
    lexical_weight: float = DEFAULT_LEXICAL_WEIGHT
    semantic_weight: float = DEFAULT_SEMANTIC_WEIGHT
    rrf_k: int = DEFAULT_RRF_K

    def build_searcher(
        self,
        root: Path | str,
        *,
        embedding_provider: Optional[EmbeddingProvider] = None,
        embedding_cache: Optional[EmbeddingCache] = None,
        index: Optional[object] = None,
    ) -> Searcher:
        """Construct the configured Searcher from existing RepoLens components."""
        root_path = Path(root)
        if self.strategy == "lexical":
            from repolens.search import CodeSearcher

            return CodeSearcher(root_path, index=index)
        if self.strategy == "semantic":
            if embedding_provider is None:
                from repolens.embeddings import FakeEmbeddingProvider

                embedding_provider = FakeEmbeddingProvider()
            from repolens.search import CodeSearcher
            from repolens.semantic_search import SemanticSearcher

            return SemanticSearcher(
                root_path,
                embedding_provider,
                candidate_searcher=CodeSearcher(root_path, index=index),
                cache=embedding_cache,
                index=index,
            )
        if self.strategy in ("rrf", "weighted"):
            from repolens.retrieval import FusionStrategy, HybridSearcher
            from repolens.search import CodeSearcher
            from repolens.semantic_search import SemanticSearcher

            if embedding_provider is None:
                from repolens.embeddings import FakeEmbeddingProvider

                embedding_provider = FakeEmbeddingProvider()
            lexical = CodeSearcher(root_path, index=index)
            semantic = SemanticSearcher(
                root_path,
                embedding_provider,
                candidate_searcher=lexical,
                cache=embedding_cache,
                index=index,
            )
            strategy = (
                FusionStrategy.RRF if self.strategy == "rrf" else FusionStrategy.WEIGHTED
            )
            return HybridSearcher(
                root_path,
                lexical_searcher=lexical,
                semantic_searcher=semantic,
                lexical_weight=self.lexical_weight,
                semantic_weight=self.semantic_weight,
                strategy=strategy,
                rrf_k=self.rrf_k,
            )
        raise ValueError(f"unknown retrieval strategy: {self.strategy!r}")


@dataclass(frozen=True)
class DependencyExpansionConfig:
    """Controls dependency-aware context expansion.

    ``depth`` is the number of graph hops to traverse beyond the retrieved
    primary files:

    - ``0`` — no expansion (retrieved files only);
    - ``1`` — retrieved files plus their direct dependencies and dependents;
    - ``2`` — one additional graph hop.

    ``include_dependencies`` and ``include_dependents`` independently toggle
    forward (files a candidate imports) and reverse (files that import a
    candidate) edges. Cycles are broken by breadth-first traversal that never
    revisits a file.
    """

    depth: int = DEFAULT_DEPTH
    include_dependencies: bool = True
    include_dependents: bool = True


@dataclass(frozen=True)
class ContextBudget:
    """A token budget the context package must respect.

    ``max_tokens`` is an approximate budget in *estimated* tokens (see
    :mod:`repolens.context.tokens`). The engine never intentionally exceeds
    it. Set ``max_tokens=None`` for an unlimited budget.
    """

    max_tokens: Optional[int] = DEFAULT_TOTAL_BUDGET
