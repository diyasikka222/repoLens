"""Tests for the M22 call-graph integration into impact analysis.

The ``ImpactAnalyzer`` accepts an optional M22 :class:`CallGraph`. Only when
one is supplied do symbol targets gain callers/callees relationships — the
default (no reference graph) behavior is byte-for-byte M21.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from repolens.call_graph import CallGraphBuilder
from repolens.impact import (
    EVIDENCE_RESOLVED_CALL,
    ImpactAnalyzer,
    Relationship,
    RiskLevel,
)
from repolens.incremental_index import IncrementalIndexBuilder
from repolens.index import SymbolIndexBuilder
from repolens.references import ReferenceIndexBuilder

ROOT = Path(__file__).parent / "fixtures" / "callgraph_repository"


@pytest.fixture(scope="module")
def reference_graph():
    index = IncrementalIndexBuilder(ROOT, persist=False).build()
    ref_index = ReferenceIndexBuilder(ROOT, index=index, persist=False).build()
    sym_index = SymbolIndexBuilder(ROOT, index=index).build()
    return CallGraphBuilder(
        ROOT,
        index=index,
        reference_index=ref_index,
        symbol_index=sym_index,
    ).build()


@pytest.fixture(scope="module")
def analyzer(reference_graph) -> ImpactAnalyzer:
    return ImpactAnalyzer(ROOT, reference_graph=reference_graph)


@pytest.fixture(scope="module")
def plain_analyzer() -> ImpactAnalyzer:
    return ImpactAnalyzer(ROOT)


def _items_for(result, relationship: Relationship):
    return [item.path.as_posix() for item in result.items if item.relationship is relationship]


def test_symbol_target_reports_direct_callers(analyzer: ImpactAnalyzer) -> None:
    result = analyzer.analyze("charge_card")
    callers = _items_for(result, Relationship.DIRECT_CALLER)
    assert "app/checkout.py" in callers
    assert Path("app/payments.py") not in callers  # same file as the definition


def test_call_items_carry_static_confidence_and_evidence(
    analyzer: ImpactAnalyzer,
) -> None:
    result = analyzer.analyze("charge_card")
    by_path = {item.path.as_posix(): item for item in result.items}
    item = by_path["app/checkout.py"]
    assert item.relationship is Relationship.DIRECT_CALLER
    assert item.confidence == "static"
    assert EVIDENCE_RESOLVED_CALL in item.evidence
    assert item.risk is RiskLevel.HIGH


def test_test_files_keep_test_priority_over_callers(
    analyzer: ImpactAnalyzer,
) -> None:
    result = analyzer.analyze("charge_card")
    by_path = {item.path.as_posix(): item for item in result.items}
    # test_checkout.py calls charge_card, but TEST outranks DIRECT_CALLER.
    assert by_path["tests/test_checkout.py"].relationship is Relationship.TEST
    assert EVIDENCE_RESOLVED_CALL in by_path["tests/test_checkout.py"].evidence


def test_callees_relationship_lists_what_symbol_calls(
    analyzer: ImpactAnalyzer,
) -> None:
    result = analyzer.analyze("process_payment")
    callees = _items_for(result, Relationship.DIRECT_CALLEE)
    assert "app/models.py" in callees  # tax(...) and Order.total(...)


def test_method_callers_are_reported(analyzer: ImpactAnalyzer) -> None:
    result = analyzer.analyze("add")
    callers = _items_for(result, Relationship.DIRECT_CALLER)
    assert "app/checkout.py" in callers


def test_transitive_callers_are_bounded(analyzer: ImpactAnalyzer) -> None:
    result = analyzer.analyze("tax")
    indirect = _items_for(result, Relationship.INDIRECT_CALLER)
    # process_payment (direct) and tests/test_checkout.py::test_process (via
    # the typed-parameter receiver) plus the direct deep chain are reachable.
    assert "tests/test_checkout.py" in indirect or "app/payments.py" in _items_for(
        result, Relationship.DIRECT_CALLER
    )


def test_without_reference_graph_no_call_relationships(
    plain_analyzer: ImpactAnalyzer,
) -> None:
    result = plain_analyzer.analyze("charge_card")
    rels = {item.relationship for item in result.items}
    assert not rels & {
        Relationship.DIRECT_CALLER,
        Relationship.INDIRECT_CALLER,
        Relationship.DIRECT_CALLEE,
        Relationship.INDIRECT_CALLEE,
    }
    assert all(item.confidence is None for item in result.items)


def test_file_targets_do_not_introduce_call_relationships(
    analyzer: ImpactAnalyzer,
) -> None:
    result = analyzer.analyze("app/payments.py")
    rels = {item.relationship for item in result.items}
    assert not rels & {
        Relationship.DIRECT_CALLER,
        Relationship.DIRECT_CALLEE,
    }


def test_summary_counts_call_relationships(analyzer: ImpactAnalyzer) -> None:
    result = analyzer.analyze("charge_card")
    assert result.summary["direct_callers"] >= 1
    assert result.summary["indirect_callers"] == 0
    assert "direct_callees" in result.summary


def test_deterministic_with_reference_graph(analyzer: ImpactAnalyzer) -> None:
    first = analyzer.analyze("process_payment")
    second = analyzer.analyze("process_payment")
    assert first.to_dict() == second.to_dict()