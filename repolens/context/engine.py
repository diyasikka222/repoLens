"""The dependency-aware context engine (Milestone 12).

:class:`ContextEngine` turns a developer query into a :class:`~repolens.context.package.ContextPackage`:

    query → retrieval → candidates → dependency expansion → ranking →
    budget → final context package

It is the public entry point of the context engine and is intentionally
decoupled from any CLI, server, agent, or MCP layer. It composes the existing
RepoLens pieces (retrieval searcher and dependency graph) and adds the
dependency expansion, ranking, budgeting, packaging, and rendering stages.

The engine consumes retrieval results through the generic
:class:`repolens.evaluation.Searcher` protocol — it never re-implements
retrieval. A searcher may be injected directly, or built on demand from a
:class:`~repolens.context.config.RetrievalConfig`.
"""

from __future__ import annotations

from pathlib import Path

from repolens.context.budget import select_within_budget
from repolens.context.candidate import (
    CandidateRole,
    ContextCandidate,
    ExcludedCandidate,
)
from repolens.context.config import (
    ContextBudget,
    DependencyExpansionConfig,
    RetrievalConfig,
)
from repolens.context.expansion import expand_dependencies
from repolens.context.package import ContextPackage
from repolens.context.ranking import rank_candidates
from repolens.context.tokens import estimate_tokens
from repolens.evaluation import Searcher
from repolens.graph import DependencyGraphBuilder

DEFAULT_PRIMARY_LIMIT = 8


class ContextEngine:
    """Build context packages for developer queries against a repository.

    Example::

        engine = ContextEngine(repo_root)
        package = engine.build_context("Where is authentication handled?")

    The recommended retrieval strategy is the existing RRF hybrid, produced
    with the default weights and RRF constant. Pass a pre-built
    :class:`~repolens.evaluation.Searcher` to use any retrieval strategy::

        rrf = RetrievalConfig().build_searcher(repo_root)
        engine = ContextEngine(repo_root, searcher=rrf)

    ``primary_limit`` is how many top retrieval results are treated as primary
    (directly retrieved) candidates before dependency expansion.
    """

    def __init__(
        self,
        root: Path | str,
        *,
        searcher: Searcher | None = None,
        retrieval: RetrievalConfig | None = None,
        dependency: DependencyExpansionConfig | None = None,
        budget: ContextBudget | None = None,
        embedding_provider=None,
        index: object | None = None,
        primary_limit: int = DEFAULT_PRIMARY_LIMIT,
    ) -> None:
        self.root = Path(root)
        if not self.root.is_dir():
            raise NotADirectoryError(f"repository root is not a directory: {self.root}")

        if searcher is not None:
            self._searcher = searcher
            self._primary_limit = primary_limit
        else:
            cfg = retrieval if retrieval is not None else RetrievalConfig()
            self._searcher = cfg.build_searcher(
                self.root,
                embedding_provider=embedding_provider,
                index=index,
            )
            self._primary_limit = cfg.limit

        self._dep_config = dependency if dependency is not None else DependencyExpansionConfig()
        self._budget = budget if budget is not None else ContextBudget()
        self._graph = DependencyGraphBuilder(self.root, index=index).build()

    def build_context(self, query: str) -> ContextPackage:
        """Compute a context package for ``query``."""
        results = self._searcher.search(query, limit=self._primary_limit)
        primary_meta = [
            _retrieval_metadata(result, rank)
            for rank, result in enumerate(results, start=1)
        ]

        primary_candidates = self._build_primary_candidates(primary_meta)
        primary_set = {candidate.path for candidate in primary_candidates}

        dependency_nodes = expand_dependencies(
            self._graph,
            seeds=[candidate.path for candidate in primary_candidates],
            config=self._dep_config,
        )
        dependency_candidates = self._build_dependency_candidates(dependency_nodes)

        all_candidates = list(primary_candidates) + list(dependency_candidates)
        ranked = rank_candidates(all_candidates)
        selected, excluded = select_within_budget(ranked, self._budget)

        return ContextPackage(
            query=query,
            budget=self._budget,
            selected_files=tuple(selected),
            primary_candidates=tuple(primary_candidates),
            dependency_candidates=tuple(dependency_candidates),
            excluded_candidates=tuple(excluded),
        )

    def render(self, package: ContextPackage) -> str:
        """Render ``package`` to deterministic text for an agent."""
        from repolens.context.render import render_context

        return render_context(package)

    # -- candidate construction ---------------------------------------------

    def _build_primary_candidates(self, primary_meta: list) -> list[ContextCandidate]:
        candidates: list[ContextCandidate] = []
        for meta in primary_meta:
            path = meta["file_path"]
            source = self._read_source(path)
            reason = _primary_reason(meta)
            candidates.append(
                ContextCandidate(
                    path=path,
                    source=source,
                    role=CandidateRole.PRIMARY,
                    estimated_tokens=estimate_tokens(source),
                    selection_reason=reason,
                    retrieval_rank=meta.get("retrieval_rank"),
                    retrieval_score=meta.get("retrieval_score"),
                    lexical_rank=meta.get("lexical_rank"),
                    semantic_rank=meta.get("semantic_rank"),
                )
            )
        return candidates

    def _build_dependency_candidates(
        self, dependency_nodes,
    ) -> list[ContextCandidate]:
        candidates: list[ContextCandidate] = []
        for node in dependency_nodes:
            source = self._read_source(node.path)
            candidates.append(
                ContextCandidate(
                    path=node.path,
                    source=source,
                    role=node.role,
                    estimated_tokens=estimate_tokens(source),
                    selection_reason=_dependency_reason(node),
                    graph_distance=node.distance,
                )
            )
        return candidates

    def _read_source(self, path: Path) -> str:
        try:
            return (self.root / path).read_text(encoding="utf-8")
        except (OSError, ValueError):
            return ""


# ---------------------------------------------------------------------------
# Retrieval metadata adapter
# ---------------------------------------------------------------------------

def _retrieval_metadata(result, rank: int) -> dict:
    """Extract generic retrieval metadata from any searcher result object.

    The retrieval layer returns several result types (code search, semantic
    search, hybrid). This adapter reads only attributes that exist, so the
    engine works across all of them without duplicating retrieval logic.
    """
    meta: dict = {
        "file_path": result.file_path,
        "retrieval_rank": rank,
        "retrieval_score": None,
        "lexical_rank": None,
        "semantic_rank": None,
    }
    for attr in ("hybrid_score", "score", "similarity"):
        if hasattr(result, attr):
            meta["retrieval_score"] = getattr(result, attr)
            break
    if hasattr(result, "lexical_rank"):
        meta["lexical_rank"] = result.lexical_rank
    if hasattr(result, "semantic_rank"):
        meta["semantic_rank"] = result.semantic_rank
    return meta


def _primary_reason(meta: dict) -> str:
    rank = meta.get("retrieval_rank")
    if rank is not None:
        return f"retrieved as primary result at rank {rank}"
    return "retrieved as primary result"


def _dependency_reason(node) -> str:
    direction = "imports it" if node.role is CandidateRole.DEPENDENT else "it imports"
    return (
        f"{'dependent' if node.role is CandidateRole.DEPENDENT else 'dependency'}: "
        f"graph distance {node.distance} ({direction})"
    )
