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

import time
from pathlib import Path

from repolens import diagnostics
from repolens.context.budget import select_within_budget
from repolens.context.candidate import (
    INCLUSION_API_CONSUMER,
    INCLUSION_ARCHITECTURE,
    INCLUSION_CONFIGURATION,
    INCLUSION_DEPENDENCY,
    INCLUSION_DEPENDENT,
    INCLUSION_HYBRID_MATCH,
    INCLUSION_LEXICAL_MATCH,
    INCLUSION_SEMANTIC_MATCH,
    INCLUSION_SYMBOL_MATCH,
    INCLUSION_TEST,
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
        reference_graph: object | None = None,
        primary_limit: int = DEFAULT_PRIMARY_LIMIT,
        architecture: object | None = None,
    ) -> None:
        self.root = Path(root)
        if not self.root.is_dir():
            raise NotADirectoryError(f"repository root is not a directory: {self.root}")

        # M23.2 architecture-aware retrieval is opt-in via ``architecture=``.
        # When the caller supplies a configuration (and does not pass an
        # incremental index), a single index snapshot is materialised here so
        # the DependencyGraphBuilder, SymbolIndexBuilder, and
        # ArchitectureGraphBuilder share exactly one scan/parse pass.
        self._arch_config = architecture
        if (
            getattr(self._arch_config, "enabled", False)
            and index is None
        ):
            from repolens.incremental_index import IncrementalIndexBuilder
            index = IncrementalIndexBuilder(self.root, persist=False).build()

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
        # Keep the incremental snapshot so the impact analysis (Milestone 21)
        # can reuse analyses instead of re-parsing the repository.
        self._index = index
        self._graph = DependencyGraphBuilder(self.root, index=index).build()
        # Re-use the existing symbol index (from the incremental snapshot when
        # available, otherwise by scanning) — never a second symbol system.
        self._symbol_index = SymbolIndexBuilder(self.root, index=index).build()
        # Optional M22 static call graph: when present, change-aware contexts
        # also surface statically resolved callers/callees of the target symbol.
        self._reference_graph = reference_graph
        # Architecture graph and subsystems are built lazily on first use.
        self._arch_graph = None
        self._arch_subsystems = None

    def build_context(self, query: str) -> ContextPackage:
        """Compute a context package for ``query``."""
        start = time.perf_counter()
        intent = classify_intent(query)
        symbol_matches = match_symbols(
            query, self.root, index=None, symbol_index=self._symbol_index
        )
        symbol_paths = symbol_file_paths(symbol_matches)

        # M23.2 architecture enrichment happens *before* dependency expansion so
        # its direct matches can anchor the expansion seeds.
        arch_matches = ()
        arch_direct_paths: set = set()
        if getattr(self._arch_config, "enabled", False):
            from repolens.architecture_retrieval import architecture_candidates

            self._ensure_architecture()
            arch_matches = architecture_candidates(
                query,
                self._arch_graph,
                subsystems=self._arch_subsystems,
                symbol_matches=symbol_matches,
                config=self._arch_config,
            )
            arch_direct_paths = {
                Path(candidate.path)
                for candidate in arch_matches
                if candidate.rank == 0
            }

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
        # rather than unrelated co-retrieved primaries. When architecture
        # enrichment found direct matches (and no symbol was named), anchor on
        # those instead.
        combined_seeds = (
            sorted(set(symbol_paths) | arch_direct_paths)
            if (symbol_paths or arch_direct_paths)
            else []
        )
        expansion_seeds = (
            combined_seeds
            if combined_seeds
            else [candidate.path for candidate in primary_candidates]
        )
        dependency_nodes = expand_dependencies(
            self._graph,
            seeds=expansion_seeds,
            config=eff_config,
        )
        dependency_candidates = self._build_dependency_candidates(dependency_nodes)
        architecture_candidates_ctx = self._build_architecture_candidates(
            arch_matches
        )

        all_candidates = (
            list(primary_candidates)
            + list(dependency_candidates)
            + list(architecture_candidates_ctx)
        )
        # Never include the same file twice: a file that is both a retrieved
        # primary and a dependency-expanded node keeps its higher-priority
        # (primary) role and retains its retrieval signals.
        all_candidates = _dedupe_candidates(all_candidates)
        ranked = rank_candidates(all_candidates)
        selected, excluded = select_within_budget(ranked, self._budget)

        package = ContextPackage(
            query=query,
            budget=self._budget,
            selected_files=tuple(selected),
            primary_candidates=tuple(primary_candidates),
            dependency_candidates=tuple(dependency_candidates)
            + tuple(architecture_candidates_ctx),
            excluded_candidates=tuple(excluded),
            intent=intent,
            matched_symbols=tuple(s.symbol.name for s in symbol_matches),
        )
        if diagnostics.enabled():
            diagnostics.record(
                "context_build",
                repository=str(self.root),
                elapsed_ms=round((time.perf_counter() - start) * 1000.0, 3),
                candidates=len(all_candidates),
                selected=len(package.selected_files),
                context_size=package.total_estimated_tokens,
                budget=self._budget.max_tokens,
                intent=intent.value,
                architecture_candidates=len(architecture_candidates_ctx),
            )
        return package

    def render(self, package: ContextPackage) -> str:
        """Render ``package`` to deterministic text for an agent."""
        from repolens.context.render import render_context

        return render_context(package)

    # -- architecture-aware retrieval (M23.2) --------------------------------

    def _ensure_architecture(self) -> None:
        """Lazily build the architecture graph and subsystems (once)."""
        if self._arch_graph is not None:
            return
        from repolens.architecture import ArchitectureGraphBuilder
        from repolens.subsystems import discover_subsystems

        self._arch_graph = ArchitectureGraphBuilder(
            self.root, index=self._index, graph=self._graph
        ).build()
        self._arch_subsystems = discover_subsystems(self._arch_graph)

    def _effective_arch_config(self):
        """Return the architecture retrieval config actually in effect.

        An explicit configuration is respected verbatim (so ``enabled=False``
        disables architecture queries); without one, the default limits are
        used so the public architecture query surface works out of the box.
        """
        if self._arch_config is not None:
            return self._arch_config
        from repolens.architecture_retrieval import ArchitectureRetrievalConfig

        return ArchitectureRetrievalConfig()

    def architecture_candidates(self, query: str) -> list[ArchitectureCandidate]:
        """Return the bounded architecture candidates for ``query``.

        The pipeline is: architecture signals -> direct matches (rank 0) ->
        bounded neighbourhood expansion (rank 1) -> proximity/subsystem context
        (rank 2). Returns ``[]`` when the query yields no architecture signal
        or when architecture retrieval is explicitly disabled.
        """
        from repolens.architecture_retrieval import architecture_candidates

        config = self._effective_arch_config()
        if not config.enabled:
            return []
        self._ensure_architecture()
        return architecture_candidates(
            query,
            self._arch_graph,
            subsystems=self._arch_subsystems,
            config=config,
        )

    def discover_subsystems(self):
        """Return the deterministic subsystem list for the repository."""
        self._ensure_architecture()
        return self._arch_subsystems

    def explain_architecture_match(self, query: str, node_id) -> dict | None:
        """Explain whether/why ``query`` selected ``node_id`` as an
        architecture match (deterministic dict, or ``None``).
        """
        from repolens.architecture_retrieval import explain_architecture_match

        config = self._effective_arch_config()
        self._ensure_architecture()
        return explain_architecture_match(
            query,
            self._arch_graph,
            node_id,
            subsystems=self._arch_subsystems,
            config=config,
        )

    # -- change-aware impact context (Milestone 21) --------------------------

    def build_impact_context(
        self,
        target: str,
        *,
        max_depth: int | None = None,
        budget: ContextBudget | None = None,
    ) -> ContextPackage:
        """Build a change-aware context package for ``target``.

        The package prioritizes, in order: the changed target itself, direct
        dependents, linked tests, indirect dependents, then configuration/API
        consumers — all subject to ``budget`` (the engine's budget by default).
        Deterministic; never a ``get_context`` behavior change.
        """
        from repolens.impact import (
            ImpactAnalyzer,
            ImpactConfig,
            Relationship,
            RiskLevel,
        )

        config = (
            ImpactConfig(max_depth=max_depth)
            if max_depth is not None
            else ImpactConfig()
        )
        analyzer = ImpactAnalyzer(
            self.root,
            index=self._index,
            graph=self._graph,
            symbol_index=self._symbol_index,
            reference_graph=self._reference_graph,
            config=config,
        )
        result = analyzer.analyze(target)
        return self._package_impact(result, budget)

    def build_change_context(self, query: str) -> ContextPackage:
        """Build impact-aware context from a change question (``query``).

        A deterministic symbol is extracted from the query; when none is found
        an :class:`repolens.impact.ImpactTargetError` is raised rather than
        silently returning a generic package.
        """
        from repolens.context.budget import select_within_budget
        from repolens.impact import ImpactAnalyzer, ImpactConfig, ImpactTargetError

        matches = match_symbols(
            query, self.root, index=None, symbol_index=self._symbol_index
        )
        if not matches:
            raise ImpactTargetError(
                "No likely impact target could be determined from the query."
            )
        match = matches[0]
        analyzer = ImpactAnalyzer(
            self.root,
            index=self._index,
            graph=self._graph,
            symbol_index=self._symbol_index,
            reference_graph=self._reference_graph,
            config=ImpactConfig(),
        )
        result = analyzer.analyze(f"{match.path.as_posix()}::{match.symbol.name}")
        return self._package_impact(result, None)

    def _package_impact(self, result, budget: ContextBudget | None) -> ContextPackage:
        """Turn an :class:`ImpactResult` into a prioritized context package."""
        from repolens.context.budget import select_within_budget
        from repolens.impact import Relationship, RiskLevel
        from repolens.impact import TargetKind

        effective_budget = budget if budget is not None else self._budget
        ordered = self._impact_candidates(result)
        selected, excluded = select_within_budget(ordered, effective_budget)

        return ContextPackage(
            query=result.target,
            budget=effective_budget,
            selected_files=tuple(selected),
            primary_candidates=tuple(
                c for c in selected if c.role is CandidateRole.PRIMARY
            ),
            dependency_candidates=tuple(
                c for c in selected if c.role is not CandidateRole.PRIMARY
            ),
            excluded_candidates=tuple(excluded),
            intent="impact",
            matched_symbols=(result.symbol,) if result.symbol else (),
        )

    def _impact_candidates(self, result):
        """Map an impact result to prioritized :class:`ContextCandidate` objects.

        Ordering is fixed: changed target, direct callers, direct callees,
        direct dependents, tests, indirect callers, indirect callees, indirect
        dependents, then API consumers, configuration, and finally reverse
        dependencies. Tie-breaks are repository-relative paths, so the order is
        deterministic.
        """
        from repolens.impact import Relationship, TargetKind

        symbol = result.symbol
        candidates: list[ContextCandidate] = []

        def add(path, *, role, inclusion, reason, distance=None):
            source = self._read_source(path)
            candidates.append(
                ContextCandidate(
                    path=path,
                    source=source,
                    role=role,
                    estimated_tokens=estimate_tokens(source),
                    selection_reason=reason,
                    inclusion_reason=inclusion,
                    graph_distance=distance,
                )
            )

        # 1. The changed target itself.
        if result.target_path is not None:
            add(
                result.target_path,
                role=CandidateRole.PRIMARY,
                inclusion=INCLUSION_SYMBOL_MATCH,
                reason=(
                    f"changed target: {result.target}"
                    + (f" (symbol {symbol})" if symbol else "")
                ),
            )

        buckets = {
            Relationship.DIRECT_CALLER: (INCLUSION_DEPENDENT, CandidateRole.PRIMARY),
            Relationship.DIRECT_CALLEE: (INCLUSION_DEPENDENCY, CandidateRole.DEPENDENCY),
            Relationship.DIRECT_DEPENDENCY: (INCLUSION_DEPENDENT, CandidateRole.DEPENDENT),
            Relationship.TEST: (INCLUSION_TEST, CandidateRole.PRIMARY),
            Relationship.INDIRECT_CALLER: (INCLUSION_DEPENDENT, CandidateRole.DEPENDENT),
            Relationship.INDIRECT_CALLEE: (INCLUSION_DEPENDENCY, CandidateRole.DEPENDENCY),
            Relationship.INDIRECT_DEPENDENCY: (INCLUSION_DEPENDENT, CandidateRole.DEPENDENT),
            Relationship.API_CONSUMER: (INCLUSION_API_CONSUMER, CandidateRole.PRIMARY),
            Relationship.CONFIGURATION: (INCLUSION_CONFIGURATION, CandidateRole.PRIMARY),
            Relationship.REVERSE_DEPENDENCY: (INCLUSION_DEPENDENCY, CandidateRole.DEPENDENCY),
        }
        for relationship, (inclusion, role) in buckets.items():
            items = [
                item for item in result.items if item.relationship is relationship
            ]
            items.sort(key=lambda item: (item.depth, item.path.as_posix()))
            for item in items:
                add(
                    item.path,
                    role=role,
                    inclusion=inclusion,
                    reason=item.reason,
                    distance=item.depth or None,
                )
        return candidates

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

    def _build_architecture_candidates(
        self, architecture_matches,
    ) -> list[ContextCandidate]:
        """Turn raw architecture matches into :class:`ContextCandidate` objects.

        Architecture candidates join the pipeline as dependency-role candidates
        carrying their architecture rank and metadata, so the ranking policy
        interleaves them deterministically without disturbing the primary layer.
        """
        candidates: list[ContextCandidate] = []
        for match in architecture_matches:
            path = Path(match.path)
            source = self._read_source(path)
            candidates.append(
                ContextCandidate(
                    path=path,
                    source=source,
                    role=CandidateRole.DEPENDENCY,
                    estimated_tokens=estimate_tokens(source),
                    selection_reason=match.reason,
                    inclusion_reason=INCLUSION_ARCHITECTURE,
                    architecture_rank=match.rank,
                    architecture_metadata={
                        "node_kind": match.node.kind.value,
                        "node_id": match.node.id,
                        "package": match.package,
                        "subsystem": match.subsystem,
                        "direction": match.direction.value,
                    },
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
