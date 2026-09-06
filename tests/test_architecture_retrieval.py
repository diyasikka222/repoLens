"""Tests for architecture-aware context retrieval (Milestone 23.2).

:mod:`repolens.architecture_retrieval` extracts architecture signals from a
developer query, expands them deterministically and *boundedly* across the
architecture graph, and explains each match. These tests pin the exact
behaviour against the multi-subsystem ``architecture_retrieval_repository``
fixture.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from repolens.architecture import ArchitectureGraphBuilder, ArchitectureNodeKind
from repolens.architecture_retrieval import (
    ArchitectureDirection,
    ArchitectureRetrievalConfig,
    architecture_candidates,
    explain_architecture_match,
    extract_architecture_signals,
)
from repolens.context.symbol_retrieval import SymbolMatch
from repolens.index import Symbol
from repolens.subsystems import discover_subsystems

ROOT = Path(__file__).parent / "fixtures" / "architecture_retrieval_repository"


@pytest.fixture(scope="module")
def graph():
    return ArchitectureGraphBuilder(ROOT).build()


@pytest.fixture(scope="module")
def subsystems(graph):
    return discover_subsystems(graph)


def _node_ids(candidates):
    return [c.node.id for c in candidates]


def _paths(candidates):
    return [c.path for c in candidates]


def _order_symbol_match(name: str, path: str) -> SymbolMatch:
    return SymbolMatch(
        Symbol(
            name=name,
            kind="class",
            file_path=Path(path),
            line=1,
            parent_class=None,
        ),
        exact=True,
    )


# ---------------------------------------------------------------------------
# Signal extraction
# ---------------------------------------------------------------------------


def test_explicit_module_signal(graph) -> None:
    signals = extract_architecture_signals("store.services.checkout", graph)
    assert any(
        s.kind == "module" and s.node.id == "store.services.checkout"
        for s in signals
    )
    assert signals[0].reason == "architecture: direct module match"


def test_explicit_package_signal_graph_style(graph) -> None:
    signals = extract_architecture_signals("store/repositories", graph)
    assert any(
        s.kind == "package" and s.node.id == "store/repositories"
        for s in signals
    )


def test_explicit_file_signal(graph) -> None:
    signals = extract_architecture_signals("store/services/checkout.py", graph)
    assert any(
        s.kind == "file" and s.node.id == "store/services/checkout.py"
        for s in signals
    )


def test_term_tagging_module_leaf(graph) -> None:
    signals = extract_architecture_signals("how does checkout work", graph)
    assert any(
        s.kind == "module" and s.node.id == "store.services.checkout"
        for s in signals
    )


def test_term_tagging_package_leaf(graph) -> None:
    signals = extract_architecture_signals("billing", graph)
    assert any(
        s.kind == "package" and s.node.id == "billing" for s in signals
    )


def test_plural_tolerance_for_role_terms(graph) -> None:
    signals = extract_architecture_signals("show me the repositories", graph)
    assert any(
        s.kind == "package" and s.node.id == "store/repositories"
        for s in signals
    )


def test_symbol_match_links_to_module(graph) -> None:
    match = _order_symbol_match("Order", "store/models/order.py")
    signals = extract_architecture_signals(
        "order class", graph, symbol_matches=[match]
    )
    assert any(
        s.kind == "symbol" and s.node.id == "store.models.order" for s in signals
    )


def test_extraction_is_deterministic_and_capped(graph) -> None:
    query = "store checkout orders cart refund invoices charges catalog reports graph"
    first = extract_architecture_signals(query, graph)
    second = extract_architecture_signals(query, graph)
    assert first == second
    assert len(first) <= 24


def test_no_signal_for_unrelated_query(graph) -> None:
    assert extract_architecture_signals("green sky lorem ipsum", graph) == []


# ---------------------------------------------------------------------------
# Candidate retrieval: direct matches
# ---------------------------------------------------------------------------


def test_module_query_yields_direct_match(graph, subsystems) -> None:
    candidates = architecture_candidates(
        "store.services.checkout", graph, subsystems=subsystems
    )
    direct = [c for c in candidates if c.rank == 0]
    assert any(c.node.id == "store.services.checkout" for c in direct)
    assert all(c.direction is ArchitectureDirection.MATCHED for c in direct)
    assert all(c.path == "store/services/checkout.py" for c in direct)


def test_package_query_yields_package_and_contained_modules(graph, subsystems) -> None:
    candidates = architecture_candidates(
        "store/repositories", graph, subsystems=subsystems
    )
    package_matches = [c for c in candidates if c.node.kind is ArchitectureNodeKind.PACKAGE]
    assert any(c.node.id == "store/repositories" and c.rank == 0 for c in package_matches)
    # Modules in the matched package come with a rank-1 reason.
    in_package = [c for c in candidates if c.reason == "architecture: module in matched package"]
    assert {c.node.id for c in in_package} == {
        "store.repositories",
        "store.repositories.carts",
        "store.repositories.orders",
    }


def test_file_query_yields_file_direct_match(graph, subsystems) -> None:
    candidates = architecture_candidates(
        "store/services/checkout.py", graph, subsystems=subsystems
    )
    assert any(
        c.node.kind is ArchitectureNodeKind.FILE
        and c.node.id == "store/services/checkout.py"
        and c.rank == 0
        for c in candidates
    )


# ---------------------------------------------------------------------------
# Bounded neighbourhood expansion
# ---------------------------------------------------------------------------


def test_dependency_and_dependent_expansion(graph, subsystems) -> None:
    candidates = architecture_candidates(
        "checkout", graph, subsystems=subsystems
    )
    deps = [
        c for c in candidates
        if c.reason == "architecture: dependency of matched module"
    ]
    dependents = [
        c for c in candidates
        if c.reason == "architecture: dependent module"
    ]
    assert {c.node.id for c in deps} == {
        "store.models.order",
        "store.repositories.orders",
    }
    assert {"admin.reports", "store.api.catalog", "tests.test_checkout"} <= {
        c.node.id for c in dependents
    }


def test_neighbor_inside_package(graph, subsystems) -> None:
    candidates = architecture_candidates(
        "checkout", graph, subsystems=subsystems
    )
    neighbors = [
        c for c in candidates if c.reason == "architecture: neighboring module"
    ]
    assert {c.node.id for c in neighbors} == {
        "store.services",
        "store.services.refund",
    }


def test_same_subsystem_proximity_tier(graph, subsystems) -> None:
    candidates = architecture_candidates(
        "checkout", graph, subsystems=subsystems
    )
    proximity = [
        c for c in candidates if c.reason == "architecture: same subsystem"
    ]
    assert proximity
    assert all(c.rank == 2 for c in proximity)
    assert all(c.direction is ArchitectureDirection.SUBSYSTEM for c in proximity)
    assert {c.node.id for c in proximity} <= set(
        m for m in _by_id(subsystems)["store"].modules
    )


def test_package_to_package_relationships(graph, subsystems) -> None:
    candidates = architecture_candidates(
        "store/repositories", graph, subsystems=subsystems
    )
    reasons = {c.reason for c in candidates}
    assert "architecture: dependency package" in reasons  # store/repositories -> store/models
    assert "architecture: dependent package" in reasons  # store/api, store/services depend on it


def _by_id(subsystems):
    return {s.id: s for s in subsystems}


# ---------------------------------------------------------------------------
# Boundedness
# ---------------------------------------------------------------------------


def test_expansion_is_bounded_by_module_candidates(graph, subsystems) -> None:
    cfg = ArchitectureRetrievalConfig(max_module_candidates=6)
    candidates = architecture_candidates(
        "checkout", graph, subsystems=subsystems, config=cfg
    )
    module_count = sum(
        1 for c in candidates if c.node.kind is ArchitectureNodeKind.MODULE
    )
    assert module_count <= 6


def test_expansion_is_bounded_by_expanded_nodes(graph, subsystems) -> None:
    cfg = ArchitectureRetrievalConfig(max_expanded_nodes=4)
    candidates = architecture_candidates(
        "checkout", graph, subsystems=subsystems, config=cfg
    )
    # Direct matches are not counted against the expansion cap.
    assert sum(1 for c in candidates if c.reason == "architecture: same subsystem") <= 4


def test_disabled_config_returns_empty(graph, subsystems) -> None:
    cfg = ArchitectureRetrievalConfig(enabled=False)
    assert (
        architecture_candidates(
            "checkout", graph, subsystems=subsystems, config=cfg
        )
        == []
    )


def test_no_signal_returns_empty(graph, subsystems) -> None:
    assert architecture_candidates("lorem ipsum", graph, subsystems=subsystems) == []


def test_never_returns_entire_connected_component(graph, subsystems) -> None:
    candidates = architecture_candidates(
        "checkout", graph, subsystems=subsystems
    )
    assert len(candidates) < graph.stats().modules


# ---------------------------------------------------------------------------
# Determinism + metadata
# ---------------------------------------------------------------------------


def test_candidate_retrieval_is_deterministic(graph, subsystems) -> None:
    query = "checkout"
    first = architecture_candidates(query, graph, subsystems=subsystems)
    second = architecture_candidates(query, graph, subsystems=subsystems)
    assert [c.to_dict() for c in first] == [c.to_dict() for c in second]


def test_candidates_are_deduplicated_by_node(graph, subsystems) -> None:
    candidates = architecture_candidates(
        "checkout", graph, subsystems=subsystems
    )
    keys = [(c.node.kind.value, c.node.id) for c in candidates]
    assert len(keys) == len(set(keys))


def test_candidate_metadata_reports_package_and_subsystem(graph, subsystems) -> None:
    candidates = architecture_candidates(
        "checkout", graph, subsystems=subsystems
    )
    checkout = next(
        c for c in candidates if c.node.id == "store.services.checkout"
    )
    assert checkout.package == "store/services"
    assert checkout.subsystem == "store"
    assert checkout.rank == 0


def test_candidate_order_is_rank_then_direction(graph, subsystems) -> None:
    candidates = architecture_candidates(
        "checkout", graph, subsystems=subsystems
    )
    ranks = [c.rank for c in candidates]
    assert ranks == sorted(ranks)


# ---------------------------------------------------------------------------
# Explainability
# ---------------------------------------------------------------------------


def test_explain_architecture_match_by_module(graph, subsystems) -> None:
    result = explain_architecture_match(
        "checkout", graph, "store.services.checkout", subsystems=subsystems
    )
    assert result is not None
    assert result["reason"] == "architecture: direct module match"
    assert result["rank"] == 0
    assert result["direction"] == "matched"
    assert result["path"] == "store/services/checkout.py"


def test_explain_architecture_match_by_file_path(graph, subsystems) -> None:
    result = explain_architecture_match(
        "checkout", graph, "store/services/checkout.py", subsystems=subsystems
    )
    assert result is not None
    assert result["node"]["id"] in {
        "store.services.checkout",
        "store/services/checkout.py",
    }


def test_explain_returns_none_for_unmatched(graph, subsystems) -> None:
    assert (
        explain_architecture_match(
            "checkout", graph, "store/models/cart.py", subsystems=subsystems
        )
        is None
    )


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_empty_repository_has_no_candidates(tmp_path: Path) -> None:
    graph = ArchitectureGraphBuilder(tmp_path).build()
    assert architecture_candidates("checkout", graph) == []


def test_single_file_repository(tmp_path: Path) -> None:
    (tmp_path / "main.py").write_text("print(1)\n", encoding="utf-8")
    graph = ArchitectureGraphBuilder(tmp_path).build()
    candidates = architecture_candidates("main", graph)
    assert any(
        c.node.kind is ArchitectureNodeKind.MODULE
        and c.node.id == "main"
        and c.path == "main.py"
        for c in candidates
    )