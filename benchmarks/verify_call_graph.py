"""Deterministic verification for M22 offline call-graph analysis.

Part 1 builds the small ``callgraph_repository``-style synthetic repository and
asserts the exact resolution matrix of :class:`repolens.call_graph.CallGraph`:
local calls, aliased and relative imports, module-qualified calls, class
instantiations, method receivers (assignment, ``self`` attribute, typed
parameter), bounded reverse/forward traversal, importers, direct dependents,
and residual unresolved references (dynamic calls that must never be guessed).

It then proves the M21/M22 impact integration: symbol targets analyzed with
``reference_graph`` report ``DIRECT_CALLER`` / ``DIRECT_CALLEE`` relationships
with ``confidence="static"`` while the plain analyzer (no reference graph) is
unchanged.

Part 2 runs against the real RepoLens repository itself: the graph builds,
is deterministic across runs, reference re-indexing is warm (no source
re-parsing after the first indexed build), and a real public symbol
(``classify_intent``) surfaces statically resolved callers.

Deterministic and fully offline. Prints exact counts. Does not modify the
repository under analysis (index and reference builds use temp caches).
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from repolens.call_graph import (
    CallGraphBuilder,
    CallRelationship,
    EdgeKind,
    UNRESOLVED_MODULE,
)
from repolens.impact import (
    EVIDENCE_RESOLVED_CALL,
    ImpactAnalyzer,
    Relationship,
)
from repolens.incremental_index import IncrementalIndexBuilder
from repolens.index import SymbolIndexBuilder
from repolens.references import ReferenceIndexBuilder


def write_file(root: Path, relative: str, source: str) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")


def build_synthetic(root: Path) -> None:
    """Recreate the ``tests/fixtures/callgraph_repository`` source tree.

    Every resolution path is exercised: exact locals, relative and aliased
    from-imports, ``from . import`` bindings, module-qualified imports, class
    instantiations, method receivers (assignment, ``self`` attribute, typed
    parameter), and dynamic (unknown) calls that must stay unresolved.
    """
    write_file(root, "app/__init__.py", "")
    write_file(
        root,
        "app/models.py",
        "class Order:\n"
        "    def total(self):\n"
        "        return 100\n"
        "\n"
        "class Cart:\n"
        "    def add(self, item):\n"
        "        return True\n"
        "    def checkout(self, order):\n"
        "        return order.total()\n"
        "\n"
        "def tax(amount):\n"
        "    return amount * 0.08\n",
    )
    write_file(
        root,
        "app/payments.py",
        "from .models import Order, tax\n"
        "\n"
        "def charge_card(card):\n"
        "    return True\n"
        "\n"
        "def refund(order):\n"
        "    return True\n"
        "\n"
        "def process_payment(order: Order, card):\n"
        "    total = tax(order.total())\n"
        "    result = charge_card(card)\n"
        "    return result\n",
    )
    write_file(
        root,
        "app/validators.py",
        "def validate(value):\n"
        "    return value is not None\n"
        "\n"
        "def sanitize(value):\n"
        "    return value.strip()\n",
    )
    write_file(
        root,
        "app/checkout.py",
        "from .payments import charge_card, refund\n"
        "from .validators import sanitize as clean\n"
        "from .validators import validate\n"
        "from . import models\n"
        "import app.payments as payments\n"
        "\n"
        "class CartController:\n"
        "    def __init__(self):\n"
        "        self.cart = models.Cart()\n"
        "    def buy(self, card):\n"
        "        order = models.Cart()\n"
        "        self.cart.add(card)\n"
        "        valid = validate(card)\n"
        "        cleaned = clean(str(card))\n"
        "        self.cart.checkout(order)\n"
        "        charge_card(card)\n"
        "        payments.charge_card(card)\n"
        "        refund(order)\n"
        "        bookings = globals().get(\"bookings\")\n"
        "        return bookings.make()\n",
    )
    write_file(
        root,
        "tests/test_checkout.py",
        "from app import payments\n"
        "from app.checkout import CartController\n"
        "from app.payments import process_payment\n"
        "\n"
        "def test_buy():\n"
        "    c = CartController()\n"
        "    c.buy(\"1234\")\n"
        "    payments.charge_card(\"1234\")\n"
        "\n"
        "def test_process():\n"
        "    process_payment(None, \"1234\")\n",
    )
    write_file(
        root,
        "tests/test_payments.py",
        "from app.payments import refund\n"
        "from app.validators import validate\n"
        "\n"
        "def test_refund():\n"
        "    refund(None)\n"
        "\n"
        "def test_validate():\n"
        "    assert validate(1) is True\n",
    )


def _node(graph, name: str, parent: str | None = None):
    for node in graph.get_nodes():
        if node.name == name and (parent is None or node.parent_class == parent):
            return node
    return None


def _names(nodes) -> set[str]:
    return {n.name for n in nodes}


def build_graph(root: Path, cache_dir: Path):
    index = IncrementalIndexBuilder(root, cache_dir=cache_dir / "index").build()
    ref_index = ReferenceIndexBuilder(
        root, index=index, cache_dir=cache_dir / "ref", persist=True
    ).build()
    sym_index = SymbolIndexBuilder(root, index=index).build()
    return (
        CallGraphBuilder(
            root,
            index=index,
            reference_index=ref_index,
            symbol_index=sym_index,
        ).build(),
        ref_index,
        index,
    )


def serialize_graph(graph) -> list[str]:
    nodes = sorted(
        (
            n.file_path.as_posix(),
            n.name or "",
            n.kind.value if n.kind else "",
            n.parent_class or "",
        )
        for n in graph.get_nodes()
    )
    edges = sorted(
        (
            e.kind.value,
            e.source.file_path.as_posix(),
            e.source.name or "",
            e.target.file_path.as_posix(),
            e.target.name or "",
        )
        for e in graph.get_edges()
    )
    return nodes + edges


def part1(root: Path) -> None:
    cache_dir = Path(tempfile.mkdtemp(prefix="repolens-cg-bench-"))
    graph, ref_index, index = build_graph(root, cache_dir)

    # Exact resolution matrix.
    assert _names(graph.callers(_node(graph, "charge_card"))) == {
        "buy",
        "process_payment",
        "test_buy",
    }, "local + aliased import resolution"
    assert _names(graph.callers(_node(graph, "sanitize"))) == {"buy"}
    assert _names(graph.callers(_node(graph, "validate"))) == {"buy", "test_validate"}
    print("callers(charge_card) =", sorted(_names(graph.callers(_node(graph, "charge_card")))))
    print("callers(validate)    =", sorted(_names(graph.callers(_node(graph, "validate")))))

    # Receiver resolution: self-attribute and typed parameter.
    assert _names(graph.callers(_node(graph, "add", "Cart"))) == {"buy"}
    assert _names(graph.callers(_node(graph, "total", "Order"))) == {"process_payment"}
    print("callers(Cart.add)    =", sorted(_names(graph.callers(_node(graph, "add", "Cart")))))
    print("callers(Order.total) =", sorted(_names(graph.callers(_node(graph, "total", "Order")))))

    # Instantiations are call edges into the class node.
    cart = _node(graph, "Cart")
    assert EdgeKind.INSTANTIATION in {e.kind for e in graph.called_by_edges(cart)}
    print("instantiation edges into Cart: OK")

    # Forward callees of the buy() method (including the in-method
    # instantiation of models.Cart()).
    buy = _node(graph, "buy", "CartController")
    callees = _names(graph.callees(buy, max_depth=3))
    assert {
        "Cart",
        "add",
        "checkout",
        "validate",
        "sanitize",
        "charge_card",
        "refund",
    } <= callees
    print("callees(buy, depth=3) =", len(callees), "nodes")

    # Bounded transitive queries (reverse and forward), depth 0 -> empty.
    pp = _node(graph, "process_payment")
    assert graph.bounded_transitive_callers(pp, max_depth=0) == []
    assert _names(graph.bounded_transitive_callers(pp, max_depth=3)) == {"test_process"}
    tp = _node(graph, "test_process")
    deep = _names(graph.bounded_transitive_callees(tp, max_depth=4))
    assert {"process_payment", "total", "tax", "charge_card"} <= deep
    print("bounded transitive reverse/forward: OK (depth-bounded)")

    # Module import + direct-dependency queries.
    importers = {p.as_posix() for p in graph.importers_of(Path("app/payments.py"))}
    assert {
        "app/checkout.py",
        "tests/test_checkout.py",
        "tests/test_payments.py",
    } <= importers
    imported = {p.as_posix() for p in graph.imports_of(Path("app/checkout.py"))}
    assert {
        "app/payments.py",
        "app/validators.py",
        "app/__init__.py",
    } <= imported
    direct_files = {
        n.file_path.as_posix() for n in graph.direct_dependents(_node(graph, "charge_card"))
    }
    assert direct_files == {"app/checkout.py", "tests/test_checkout.py"}
    print("importers_of(app/payments.py) =", sorted(importers))
    print("imports_of(app/checkout.py)   =", sorted(imported))

    # Residual unknowns: recorded with a stable reason, never guessed.
    unresolved = graph.unresolved_in(Path("app/checkout.py"))
    dynamic = {u.name: u.reason for u in unresolved}
    assert any("bookings" in expr for expr in dynamic), "globals() dynamic access"
    assert any(reason is UNRESOLVED_MODULE for reason in dynamic.values())
    order_total = [
        u for u in graph.unresolved_in(Path("app/models.py"))
        if "total" in u.name
    ]
    assert order_total, "untyped receiver must stay unresolved"
    print("unresolved (dynamic / untyped):", len(graph.unresolved_references()))
    print("  checkout.py:", dynamic)

    # Stats self-consistency.
    stats = graph.stats()
    assert stats.nodes == len(graph.get_nodes())
    assert stats.edges == len(graph.get_edges())
    assert stats.calls == len(graph.get_edges_by_relationship(CallRelationship.CALLS))
    assert stats.imports == len(graph.get_edges_by_relationship(CallRelationship.IMPORTS))
    assert stats.references == len(
        graph.get_edges_by_relationship(CallRelationship.REFERENCES)
    )
    assert stats.unresolved == len(graph.unresolved_references())
    print("stats:", stats.as_dict())

    # Determinism.
    second, _, _ = build_graph(root, cache_dir)
    assert serialize_graph(graph) == serialize_graph(second)
    print("graph deterministic across builds: OK")

    # Impact integration with and without the reference graph.
    plain = ImpactAnalyzer(root)
    enhanced = ImpactAnalyzer(root, reference_graph=second)
    base = plain.analyze("charge_card")
    assert not {
        item.relationship
        for item in base.items
    } & {Relationship.DIRECT_CALLER, Relationship.DIRECT_CALLEE}
    assert all(item.confidence is None for item in base.items)
    print("plain analyze: no call relationships, no confidence: OK")

    callers = [
        item.path.as_posix()
        for item in enhanced.analyze("charge_card").items
        if item.relationship is Relationship.DIRECT_CALLER
    ]
    assert "app/checkout.py" in callers
    item = [
        it
        for it in enhanced.analyze("buy").items
        if it.path.as_posix() == "app/models.py"
    ][0]
    assert item.relationship is Relationship.DIRECT_CALLEE
    assert item.confidence == "static"
    assert EVIDENCE_RESOLVED_CALL in item.evidence
    print("enhanced analyze: DIRECT_CALLER app/checkout.py; buy -> DIRECT_CALLEE app/models.py (static)")

    print("warm reference build: files_scanned =", ref_index.stats.files_scanned)


def part2(root: Path) -> None:
    cache_dir = Path(tempfile.mkdtemp(prefix="repolens-cg-real-"))
    index = IncrementalIndexBuilder(root, cache_dir=cache_dir / "index").build()
    ref_index = ReferenceIndexBuilder(
        root, index=index, cache_dir=cache_dir / "ref", persist=True
    ).build()
    cold_stats = ref_index.stats
    warm_index = IncrementalIndexBuilder(root, cache_dir=cache_dir / "index").build()
    warm_ref = ReferenceIndexBuilder(
        root, index=warm_index, cache_dir=cache_dir / "ref", persist=True
    ).build()
    sym_index = SymbolIndexBuilder(root, index=index).build()
    graph = CallGraphBuilder(
        root,
        index=index,
        reference_index=ref_index,
        symbol_index=sym_index,
    ).build()

    stats = graph.stats()
    assert stats.nodes > 0 and stats.calls > 0
    print("real repo: call graph")
    print(f"  nodes={stats.nodes} calls={stats.calls} "
          f"imports={stats.imports} references={stats.references}")
    print(f"  unresolved={stats.unresolved}")

    # Warm reference re-indexing must not re-parse the sources.
    assert warm_ref.stats.files_scanned == 0, "warm reference build must scan 0 files"
    print(f"  cold reference scan={cold_stats.files_scanned} "
          f"warm reference scan={warm_ref.stats.files_scanned}: OK")

    # Determinism on the real repository.
    assert serialize_graph(graph) == serialize_graph(
        CallGraphBuilder(
            root,
            index=index,
            reference_index=ref_index,
            symbol_index=sym_index,
        ).build()
    )
    print("  deterministic across builds: OK")

    # A real public symbol is called by other modules.
    node = _node(graph, "classify_intent")
    assert node is not None, "expected classify_intent node in real RepoLens"
    callers = graph.callers(node, max_depth=1)
    assert callers, "expected at least one real caller of classify_intent"
    print("  callers(classify_intent):", len(callers))
    for n in sorted(callers, key=lambda n: (n.file_path.as_posix(), n.name)):
        print(f"    {n.file_path.as_posix()} :: {n.name}")

    # The impact integration works on the real repository too.
    analyzer = ImpactAnalyzer(root, reference_graph=graph)
    result = analyzer.analyze("classify_intent")
    call_rel = sorted(
        {item.relationship.value for item in result.items}
        & {"direct_caller", "indirect_caller", "direct_callee", "indirect_callee"}
    )
    assert call_rel, "expected at least one call relationship for classify_intent"
    print("  analyze(classify_intent) call relationships:", call_rel)
    print("  summary:", {
        k: result.summary[k]
        for k in ("direct_callers", "indirect_callers", "direct_callees")
    })


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "repo"
        build_synthetic(root)
        print("PART 1 - synthetic repository")
        part1(root)

    real_root = Path(__file__).resolve().parent.parent
    print("\nPART 2 - real repository (RepoLens itself)")
    part2(real_root)

    print("\nVERIFICATION PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())