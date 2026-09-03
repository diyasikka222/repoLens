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
    INCLUSION_DEPENDENCY,
    INCLUSION_DEPENDENT,
    INCLUSION_HYBRID_MATCH,
    INCLUSION_LEXICAL_MATCH,
    INCLUSION_SEMANTIC_MATCH,
    INCLUSION_SYMBOL_MATCH,
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
from repolens.context.intent import QueryIntent, classify_intent
from repolens.context.package import ContextPackage
from repolens.context.ranking import rank_candidates
from repolens.context.symbol_retrieval import match_symbols, symbol_file_paths
from repolens.context.tokens import estimate_tokens
from repolens.evaluation import Searcher
from repolens.graph import DependencyGraphBuilder
from repolens.index import SymbolIndexBuilder

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
        # Re-use the existing symbol index (from the incremental snapshot when
        # available, otherwise by scanning) — never a second symbol system.
        self._symbol_index = SymbolIndexBuilder(self.root, index=index).build()

    def build_context(self, query: str) -> ContextPackage:
        """Compute a context package for ``query``."""
        intent = classify_intent(query)
        symbol_matches = match_symbols(
            query, self.root, index=None, symbol_index=self._symbol_index
        )
        symbol_paths = symbol_file_paths(symbol_matches)

        results = self._searcher.search(query, limit=self._primary_limit)
        primary_meta = [
            _retrieval_metadata(result, rank)
            for rank, result in enumerate(results, start=1)
        ]

        primary_candidates = self._build_primary_candidates(
            primary_meta, symbol_paths
        )
        primary_set = {candidate.path for candidate in primary_candidates}

        eff_config = self._effective_expansion_config(intent)
        # Anchor expansion on the symbol-matched file(s) when the query names
        # one (a precise dependency/impact or implementation question), so the
        # expanded files are the true neighbours of the referenced symbol
        # rather than unrelated co-retrieved primaries.
        expansion_seeds = (
            sorted(symbol_paths)
            if symbol_paths
            else [candidate.path for candidate in primary_candidates]
        )
        dependency_nodes = expand_dependencies(
            self._graph,
            seeds=expansion_seeds,
            config=eff_config,
        )
        dependency_candidates = self._build_dependency_candidates(dependency_nodes)

        all_candidates = list(primary_candidates) + list(dependency_candidates)
        # Never include the same file twice: a file that is both a retrieved
        # primary and a dependency-expanded node keeps its higher-priority
        # (primary) role and retains its retrieval signals.
        all_candidates = _dedupe_candidates(all_candidates)
        ranked = rank_candidates(all_candidates)
        selected, excluded = select_within_budget(ranked, self._budget)

        return ContextPackage(
            query=query,
            budget=self._budget,
            selected_files=tuple(selected),
            primary_candidates=tuple(primary_candidates),
            dependency_candidates=tuple(dependency_candidates),
            excluded_candidates=tuple(excluded),
            intent=intent,
            matched_symbols=tuple(s.symbol.name for s in symbol_matches),
        )

    def render(self, package: ContextPackage) -> str:
        """Render ``package`` to deterministic text for an agent."""
        from repolens.context.render import render_context

        return render_context(package)

    # -- candidate construction ---------------------------------------------

    def _build_primary_candidates(
        self, primary_meta: list, symbol_paths: set
    ) -> list[ContextCandidate]:
        candidates: list[ContextCandidate] = []
        for meta in primary_meta:
            path = meta["file_path"]
            source = self._read_source(path)
            reason = _primary_reason(meta)
            inclusion = _primary_inclusion(meta, symbol_paths)
            candidates.append(
                ContextCandidate(
                    path=path,
                    source=source,
                    role=CandidateRole.PRIMARY,
                    estimated_tokens=estimate_tokens(source),
                    selection_reason=reason,
                    inclusion_reason=inclusion,
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
            inclusion = (
                INCLUSION_DEPENDENT
                if node.role is CandidateRole.DEPENDENT
                else INCLUSION_DEPENDENCY
            )
            candidates.append(
                ContextCandidate(
                    path=node.path,
                    source=source,
                    role=node.role,
                    estimated_tokens=estimate_tokens(source),
                    selection_reason=_dependency_reason(node),
                    inclusion_reason=inclusion,
                    graph_distance=node.distance,
                )
            )
        return candidates

    def _effective_expansion_config(
        self, intent: QueryIntent,
    ) -> DependencyExpansionConfig:
        """Narrow dependency expansion to the direction that suits ``intent``.

        Intent selects the *preferred* direction; the user's explicit flags
        always win, so this never widens what the caller allowed.

        - implementation / explanation → a file's dependencies (what it uses);
        - dependency / impact → its dependents (what uses it);
        - modification → its dependents (what would be affected);
        - unknown → both directions (the historical default).
        """
        cfg = self._dep_config
        if intent is QueryIntent.IMPLEMENTATION:
            want_deps, want_dependents = True, False
        elif intent is QueryIntent.EXPLANATION:
            want_deps, want_dependents = True, False
        elif intent is QueryIntent.DEPENDENCY:
            want_deps, want_dependents = False, True
        elif intent is QueryIntent.MODIFICATION:
            want_deps, want_dependents = False, True
        else:  # UNKNOWN — historical default: both directions.
            return cfg

        return DependencyExpansionConfig(
            depth=cfg.depth,
            include_dependencies=want_deps and cfg.include_dependencies,
            include_dependents=want_dependents and cfg.include_dependents,
            max_expanded=cfg.max_expanded,
        )

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
        "signals": [],
    }
    for attr in ("hybrid_score", "score", "similarity"):
        if hasattr(result, attr):
            meta["retrieval_score"] = getattr(result, attr)
            break
    if hasattr(result, "lexical_rank"):
        meta["lexical_rank"] = result.lexical_rank
        meta["signals"].append("lexical")
    if hasattr(result, "semantic_rank"):
        meta["semantic_rank"] = result.semantic_rank
        meta["signals"].append("semantic")
    return meta


def _dedupe_candidates(candidates: list[ContextCandidate]) -> list[ContextCandidate]:
    """Drop later candidates with a path already seen, keeping the first.

    Primary candidates appear before dependency candidates, so a file that is
    both keeps its primary role and retrieval metadata.
    """
    seen: set[Path] = set()
    result: list[ContextCandidate] = []
    for candidate in candidates:
        if candidate.path in seen:
            continue
        seen.add(candidate.path)
        result.append(candidate)
    return result


def _primary_reason(meta: dict) -> str:
    rank = meta.get("retrieval_rank")
    if rank is not None:
        return f"retrieved as primary result at rank {rank}"
    return "retrieved as primary result"


def _primary_inclusion(meta: dict, symbol_paths: set) -> str:
    """Return the machine-readable inclusion category for a primary file."""
    if meta["file_path"] in symbol_paths:
        return INCLUSION_SYMBOL_MATCH
    signals = meta.get("signals") or []
    if "lexical" in signals and "semantic" in signals:
        return INCLUSION_HYBRID_MATCH
    if "semantic" in signals:
        return INCLUSION_SEMANTIC_MATCH
    return INCLUSION_LEXICAL_MATCH


def _dependency_reason(node) -> str:
    direction = "imports it" if node.role is CandidateRole.DEPENDENT else "it imports"
    return (
        f"{'dependent' if node.role is CandidateRole.DEPENDENT else 'dependency'}: "
        f"graph distance {node.distance} ({direction})"
    )
