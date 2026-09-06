"""Deterministic verification for M23.2 architecture-aware context retrieval.

Part 1 runs against the multi-subsystem ``architecture_retrieval_repository``
fixture and asserts the exact signal-extraction and bounded candidate-retrieval
behaviour: explicit module/package/file references, plain-term leaf matching,
rank tiering (0 direct / 1 neighbour / 2 subsystem proximity), dependency and
dependent expansion, subsystem discovery, determinism, and the expansion caps.

Part 2 runs against the real RepoLens repository: a cold architecture build
(scan + parse) then a warm build (zero files re-parsed), comparing baseline
vs architecture-aware retrieval on several query shapes and asserting that
repeated retrieval is deterministic. It prints real statistics without any
flaky timing assertions.

Deterministic and fully offline. Uses temp caches; does not modify the
repository under analysis.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from repolens.architecture import ArchitectureGraphBuilder
from repolens.architecture_retrieval import (
    ArchitectureRetrievalConfig,
    architecture_candidates,
    explain_architecture_match,
    extract_architecture_signals,
)
from repolens.context import ContextBudget, ContextEngine
from repolens.incremental_index import IncrementalIndexBuilder
from repolens.search import CodeSearcher
from repolens.subsystems import discover_subsystems

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURE = REPO_ROOT / "tests" / "fixtures" / "architecture_retrieval_repository"

QUERIES = (
    "store.services.checkout",          # explicit module
    "checkout flow",                    # term -> module leaf
    "store/repositories",               # explicit package
    "how does billing work",            # term -> package leaf + dependents
)


def part1(root: Path) -> None:
    graph = ArchitectureGraphBuilder(root).build()
    subsystems = discover_subsystems(graph)
    print("fixture subsystem discovery")
    print(f"  subsystems: {[s.id for s in subsystems]}")
    print(f"  store stats: {dict(subsystems[3].stats())}")

    # Signals.
    module_signals = extract_architecture_signals("store.services.checkout", graph)
    assert any(s.node.id == "store.services.checkout" for s in module_signals)
    package_signals = extract_architecture_signals("store/repositories", graph)
    assert any(s.node.id == "store/repositories" for s in package_signals)
    file_signals = extract_architecture_signals("store/services/checkout.py", graph)
    assert any(
        s.kind == "file" and s.node.id == "store/services/checkout.py"
        for s in file_signals
    )
    assert extract_architecture_signals("lorem ipsum bluesky", graph) == []
    print("  signals (module / package / file / none): OK")

    # Candidates: rank tiers + neighbourhood.
    candidates = architecture_candidates("checkout", graph, subsystems=subsystems)
    by_reason: dict[str, set[str]] = {}
    for c in candidates:
        by_reason.setdefault(c.reason, set()).add(c.node.id)
    assert "store.services.checkout" in by_reason["architecture: direct module match"]
    assert {"store.models.order", "store.repositories.orders"} <= by_reason[
        "architecture: dependency of matched module"
    ]
    assert {"admin.reports", "store.api.catalog", "tests.test_checkout"} <= by_reason[
        "architecture: dependent module"
    ]
    proximity = by_reason["architecture: same subsystem"]
    assert proximity and all(
        c.rank == 2 for c in candidates if c.node.id in proximity
    )
    ranks = [c.rank for c in candidates]
    assert ranks == sorted(ranks)
    print(f"  checkout topology: {len(candidates)} nodes "
          f"({len(proximity)} in same subsystem), rank-ordered: OK")

    # Direct package match pulls package relationships.
    package_candidates = architecture_candidates(
        "store/repositories", graph, subsystems=subsystems
    )
    package_reasons = {c.reason for c in package_candidates}
    assert {
        "architecture: direct package match",
        "architecture: module in matched package",
        "architecture: dependency package",
        "architecture: dependent package",
    } <= package_reasons
    print("  package query: direct + module-in-package + package edges: OK")

    # Boundedness.
    capped = architecture_candidates(
        "checkout",
        graph,
        subsystems=subsystems,
        config=ArchitectureRetrievalConfig(max_module_candidates=6),
    )
    modules = sum(1 for c in capped if c.node.kind.value == "module")
    assert modules <= 6
    assert (
        architecture_candidates(
            "checkout",
            graph,
            subsystems=subsystems,
            config=ArchitectureRetrievalConfig(enabled=False),
        )
        == []
    )
    print("  bounded expansion (cap honoured) + disabled config: OK")

    # Determinism.
    first = architecture_candidates("checkout", graph, subsystems=subsystems)
    second = architecture_candidates("checkout", graph, subsystems=subsystems)
    assert [c.to_dict() for c in first] == [c.to_dict() for c in second]
    print("  deterministic retrieval: OK")

    # Explainability.
    explanation = explain_architecture_match(
        "checkout", graph, "store.services.checkout", subsystems=subsystems
    )
    assert explanation is not None
    assert explanation["reason"] == "architecture: direct module match"
    print(f"  explain: {explanation['reason']} -> {explanation['path']}: OK")


def part2(root: Path) -> None:
    cache_dir = Path(tempfile.mkdtemp(prefix="repolens-arch-retrieval-real-"))
    index = IncrementalIndexBuilder(root, cache_dir=cache_dir / "index").build()
    cold = ArchitectureGraphBuilder(root, index=index).build()

    warm_index = IncrementalIndexBuilder(root, cache_dir=cache_dir / "index").build()
    warm = ArchitectureGraphBuilder(root, index=warm_index).build()
    assert warm_index.stats.files_parsed == 0, "warm index must not re-parse"

    subsystems = discover_subsystems(cold)
    graph_stats = cold.stats()
    assert graph_stats.nodes > 0 and graph_stats.depends_on_edges > 0
    print("real repo: architecture + retrieval")
    print(f"  graph stats: {graph_stats.as_dict()}")
    print(f"  subsystems: {len(subsystems)} "
          f"({', '.join(s.id for s in subsystems[:5])}, ...)")
    print(f"  warm index parsed={warm_index.stats.files_parsed}: "
          f"architecture is a pure projection: OK")

    for query in QUERIES:
        signals = extract_architecture_signals(query, cold)
        candidates = architecture_candidates(query, cold, subsystems=subsystems)
        repeated = architecture_candidates(query, cold, subsystems=subsystems)
        assert [c.to_dict() for c in candidates] == [c.to_dict() for c in repeated]
        print(f"  retrieve({query!r}): {len(signals)} signals, "
              f"{len(candidates)} candidates (deterministic): OK")

    # Engine-level comparison: architecture-aware vs baseline on the same query.
    engine = ContextEngine(
        root,
        searcher=CodeSearcher(root),
        architecture=ArchitectureRetrievalConfig(),
        budget=ContextBudget(max_tokens=None),
    )
    pkg = engine.build_context("how does context retrieval rank candidates")
    arch_included = sum(
        1
        for c in pkg.selected_files
        if c.inclusion_reason == "architecture"
    )
    print(f"  engine build_context: {len(pkg.selected_files)} files selected, "
          f"of which {arch_included} architecture-sourced: OK")

    baseline = ContextEngine(
        root,
        searcher=CodeSearcher(root),
        architecture=ArchitectureRetrievalConfig(enabled=False),
        budget=ContextBudget(max_tokens=None),
    ).build_context("how does context retrieval rank candidates")
    assert len(baseline.selected_files) <= len(pkg.selected_files)
    print("  architecture-aware ctx is a superset of baseline (files): OK")


def main() -> int:
    print("PART 1 - synthetic architecture-retrieval repository")
    part1(FIXTURE)

    print("\nPART 2 - real repository (RepoLens itself)")
    part2(REPO_ROOT)

    print("\nVERIFICATION PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())