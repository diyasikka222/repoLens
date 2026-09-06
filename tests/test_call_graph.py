"""Tests for the conservative, offline call graph (Milestone 22).

The call graph is built entirely from the parser, symbol index, dependency
graph and the incremental reference index — no LLM, no network, and no AST
parsing during a warm build. Residual unknowns are recorded as unresolved with
a stable, machine-readable reason and never guessed.

The fixture repository under ``tests/fixtures/callgraph_repository`` exercises
every resolution path: exact locals, imported symbols, aliased imports,
relative imports, module-qualified calls, class instantiations, method
receivers (assignment, ``self`` attribute and typed parameter), plus dynamic
(unknown) calls that must stay unresolved.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from repolens.call_graph import (
    CallGraphBuilder,
    CallRelationship,
    EdgeKind,
    UNRESOLVED_MODULE,
)
from repolens.incremental_index import IncrementalIndexBuilder
from repolens.index import SymbolIndexBuilder
from repolens.references import ReferenceIndexBuilder

ROOT = Path(__file__).parent / "fixtures" / "callgraph_repository"


@pytest.fixture(scope="module")
def graph():
    index = IncrementalIndexBuilder(ROOT, persist=False).build()
    ref_index = ReferenceIndexBuilder(ROOT, index=index, persist=False).build()
    sym_index = SymbolIndexBuilder(ROOT, index=index).build()
    return CallGraphBuilder(
        ROOT,
        index=index,
        reference_index=ref_index,
        symbol_index=sym_index,
    ).build()


def _node(graph, name: str, parent: str | None = None):
    for node in graph.get_nodes():
        if node.name == name and (parent is None or node.parent_class == parent):
            return node
    return None


def _names(nodes) -> set[str]:
    return {n.name for n in nodes}


# ---------------------------------------------------------------------------
# Resolution matrix
# ---------------------------------------------------------------------------


def test_local_function_call(graph) -> None:
    node = _node(graph, "charge_card")
    assert _names(graph.callers(node)) == {
        "buy",
        "process_payment",
        "test_buy",
    }


def test_aliased_import_resolves_to_original(graph) -> None:
    node = _node(graph, "sanitize")
    assert _names(graph.callers(node)) == {"buy"}


def test_relative_import_resolves(graph) -> None:
    node = _node(graph, "validate")
    assert _names(graph.callers(node)) == {"buy", "test_validate"}


def test_module_qualified_call_resolves(graph) -> None:
    charge_card = _node(graph, "charge_card")
    edges = [
        e for e in graph.call_edges(_node(graph, "buy", "CartController"))
        if e.target.name == "charge_card"
    ]
    assert any(e.target.name == "charge_card" for e in edges)


def test_class_instantiation_is_an_instantiation_edge(graph) -> None:
    cart = _node(graph, "Cart")
    kinds = {e.kind for e in graph._call_in[cart.key]}
    assert EdgeKind.INSTANTIATION in kinds


def test_self_attribute_receiver_resolves_method(graph) -> None:
    cart_add = _node(graph, "add", "Cart")
    assert _names(graph.callers(cart_add)) == {"buy"}


def test_typed_parameter_receiver_resolves_method(graph) -> None:
    order_total = _node(graph, "total", "Order")
    assert _names(graph.callers(order_total)) == {"process_payment"}


def test_bounded_transitive_callers(graph) -> None:
    process_payment = _node(graph, "process_payment")
    callers = graph.bounded_transitive_callers(process_payment, max_depth=3)
    assert {n.name for n in callers} == {"test_process"}
    assert graph.bounded_transitive_callers(process_payment, max_depth=0) == []


def test_bounded_transitive_callees(graph) -> None:
    test_process = _node(graph, "test_process")
    callees = graph.bounded_transitive_callees(test_process, max_depth=4)
    names = {n.name for n in callees}
    assert "process_payment" in names
    assert "total" in names  # reached transitively via the typed parameter


# ---------------------------------------------------------------------------
# Unresolved references
# ---------------------------------------------------------------------------


def test_dynamic_call_remains_unresolved_never_guessed(graph) -> None:
    records = graph.unresolved_in(Path("app/checkout.py"))
    dynamic = [r for r in records if r.name == "bookings.make"]
    assert len(dynamic) == 1
    assert dynamic[0].reason == UNRESOLVED_MODULE
    assert _node(graph, "make") is None  # no invented node


def test_untyped_parameter_remains_unresolved(graph) -> None:
    records = graph.unresolved_in(Path("app/validators.py"))
    assert any(r.name == "value.strip" for r in records)


def test_unresolved_references_are_deterministic(graph) -> None:
    records = graph.unresolved_references()
    assert records == sorted(
        records,
        key=lambda u: (
            u.source_file.as_posix(),
            u.source_symbol or "",
            u.name,
            u.reason,
            u.line or 0,
        ),
    )


# ---------------------------------------------------------------------------
# Relationships
# ---------------------------------------------------------------------------

_REL = {
    CallRelationship.CALLS,
    CallRelationship.CALLED_BY,
    CallRelationship.REFERENCES,
    CallRelationship.REFERENCED_BY,
    CallRelationship.IMPORTS,
    CallRelationship.IMPORTED_BY,
}


def test_all_six_relationships_are_supported(graph) -> None:
    by_rel = {r: len(graph.get_edges_by_relationship(r)) for r in _REL}
    for relationship in _REL:
        assert by_rel[relationship] >= 1
    assert by_rel[CallRelationship.CALLS] >= by_rel[CallRelationship.CALLED_BY]


def test_import_edges_flow_both_directions(graph) -> None:
    imports_checkout = graph.imports_of(Path("app/checkout.py"))
    assert Path("app/payments.py") in imports_checkout
    assert Path("app/validators.py") in imports_checkout
    assert Path("app/checkout.py") in graph.importers_of(Path("app/payments.py"))


def test_direct_dependents_require_import_plus_usage(graph) -> None:
    cart_controller = _node(graph, "CartController")
    dep_names = {
        (n.file_path.as_posix(), n.name or "")
        for n in graph.direct_dependents(cart_controller)
    }
    assert ("tests/test_checkout.py", "test_buy") in dep_names
    assert ("app/checkout.py", "buy") not in dep_names  # same file import baseline


# ---------------------------------------------------------------------------
# Static behavior of the builder
# ---------------------------------------------------------------------------


def test_warm_build_does_not_parse_any_source(monkeypatch) -> None:
    index = IncrementalIndexBuilder(ROOT, persist=False).build()
    ref_index = ReferenceIndexBuilder(ROOT, index=index, persist=False).build()
    sym_index = SymbolIndexBuilder(ROOT, index=index).build()

    def boom(*args, **kwargs):
        raise AssertionError("snapshot-backed build must not instantiate a builder")

    monkeypatch.setattr("repolens.call_graph.IncrementalIndexBuilder", boom)
    monkeypatch.setattr("repolens.call_graph.ReferenceIndexBuilder", boom)
    monkeypatch.setattr("repolens.call_graph.SymbolIndexBuilder", boom)
    result = CallGraphBuilder(
        ROOT,
        index=index,
        reference_index=ref_index,
        symbol_index=sym_index,
    ).build()
    assert result.stats().nodes >= 1


def test_graph_build_is_deterministic(monkeypatch) -> None:
    index = IncrementalIndexBuilder(ROOT, persist=False).build()
    ref_index = ReferenceIndexBuilder(ROOT, index=index, persist=False).build()
    sym_index = SymbolIndexBuilder(ROOT, index=index).build()

    def boom(*args, **kwargs):
        raise AssertionError("snapshot-backed build must not instantiate a builder")

    monkeypatch.setattr("repolens.call_graph.IncrementalIndexBuilder", boom)
    monkeypatch.setattr("repolens.call_graph.ReferenceIndexBuilder", boom)
    monkeypatch.setattr("repolens.call_graph.SymbolIndexBuilder", boom)

    first = CallGraphBuilder(
        ROOT,
        index=index,
        reference_index=ref_index,
        symbol_index=sym_index,
    ).build()
    second = CallGraphBuilder(
        ROOT,
        index=index,
        reference_index=ref_index,
        symbol_index=sym_index,
    ).build()
    assert first.get_nodes() == second.get_nodes()
    assert first.get_edges() == second.get_edges()


def test_cold_full_build_matches_snapshot_backed_build() -> None:
    cold = CallGraphBuilder(ROOT).build()
    index = IncrementalIndexBuilder(ROOT, persist=False).build()
    ref_index = ReferenceIndexBuilder(ROOT, index=index, persist=False).build()
    sym_index = SymbolIndexBuilder(ROOT, index=index).build()
    warm = CallGraphBuilder(
        ROOT,
        index=index,
        reference_index=ref_index,
        symbol_index=sym_index,
    ).build()
    assert cold.get_edges() == warm.get_edges()
    assert cold.get_nodes() == warm.get_nodes()
    assert cold.unresolved_references() == warm.unresolved_references()


def test_self_assignment_does_not_loop() -> None:
    from pathlib import Path as _Path

    import tempfile

    root = _Path(tempfile.mkdtemp())
    (root / "m.py").write_text(
        "def f():\n    x = x\n    return x\n\ndef g():\n    return f()\n",
        encoding="utf-8",
    )
    graph = CallGraphBuilder(root).build()
    assert graph.stats().nodes >= 2


def test_stats_are_consistent(graph) -> None:
    stats = graph.stats()
    assert stats.edges == len(graph.get_edges())
    assert stats.nodes == len(graph.get_nodes())
    assert stats.unresolved == len(graph.unresolved_references())
    assert stats.calls >= 1
    assert stats.imports >= 1
    assert stats.references >= 1