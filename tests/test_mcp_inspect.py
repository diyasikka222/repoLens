"""Offline tests for the M22 call-graph inspection MCP tool.

Exercises argument validation, the structured response for known symbols,
bounded-depth traversal, safe rejection of unknown symbols, and the additive
registration of ``inspect_symbol`` on the MCP server (never altering
``get_context``). Fully offline and deterministic.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from repolens.call_graph import CallGraphBuilder
from repolens.context import ContextBudget, ContextEngine, ContextFirewall
from repolens.incremental_index import IncrementalIndexBuilder
from repolens.index import SymbolIndexBuilder
from repolens.mcp import build_mcp_server
from repolens.mcp.errors import InvalidArgumentsError
from repolens.mcp.inspect_tool import (
    parse_inspect_arguments,
    run_inspect_symbol,
    validate_max_depth,
    validate_name,
)
from repolens.references import ReferenceIndexBuilder
from repolens.search import CodeSearcher

ROOT = Path(__file__).parent / "fixtures" / "callgraph_repository"


@pytest.fixture(scope="module")
def call_graph():
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
def inspect_factory(call_graph):
    def factory(**kwargs):
        return call_graph

    return factory


# --------------------------------------------------------------------------
# argument validation
# --------------------------------------------------------------------------


def test_validate_name_accepts_symbols_and_dotted_paths() -> None:
    assert validate_name("charge_card") == "charge_card"
    assert validate_name(" models.Cart ") == "models.Cart"
    assert validate_name("a::b") == "a::b"


def test_validate_name_rejects_non_symbols() -> None:
    for bad in ("", "   ", "app/payments.py", "../secret.py", 42, None):
        with pytest.raises(InvalidArgumentsError):
            validate_name(bad)


def test_parse_inspect_arguments() -> None:
    assert parse_inspect_arguments({"name": "buy"}) == {
        "name": "buy",
        "max_depth": None,
    }
    parsed = parse_inspect_arguments({"name": "buy", "max_depth": 2})
    assert parsed["max_depth"] == 2


def test_parse_inspect_arguments_rejects_unknown_and_missing() -> None:
    with pytest.raises(InvalidArgumentsError):
        parse_inspect_arguments({"name": "buy", "symbol": "x"})
    with pytest.raises(InvalidArgumentsError):
        parse_inspect_arguments({})


def test_validate_max_depth() -> None:
    assert validate_max_depth(0) == 0
    assert validate_max_depth(4) == 4
    assert validate_max_depth(None) is None
    with pytest.raises(InvalidArgumentsError):
        validate_max_depth(-1)


# --------------------------------------------------------------------------
# structured inspection
# --------------------------------------------------------------------------


def test_inspect_symbol_reports_callers(inspect_factory) -> None:
    response = run_inspect_symbol(inspect_factory, "charge_card")
    assert response["status"] == "ok"
    assert response["node_count"] >= 1
    caller_files = {c["file"] for c in response["callers"]}
    assert "app/checkout.py" in caller_files
    assert "tests/test_checkout.py" in caller_files


def test_inspect_symbol_reports_callees(inspect_factory) -> None:
    response = run_inspect_symbol(inspect_factory, "buy")
    callee_names = {c["name"] for c in response["callees"]}
    assert {
        "charge_card",
        "refund",
        "validate",
        "sanitize",
        "Cart",
        "add",
        "checkout",
    } <= callee_names


def test_inspect_symbol_is_bounded_by_depth(inspect_factory) -> None:
    shallow = run_inspect_symbol(inspect_factory, "charge_card", max_depth=1)
    deep = run_inspect_symbol(inspect_factory, "charge_card", max_depth=3)
    assert deep["caller_count"] >= shallow["caller_count"]
    deep_names = {c["name"] for c in deep["callers"]}
    assert "test_buy" in deep_names


def test_inspect_symbol_unknown_is_safe(inspect_factory) -> None:
    with pytest.raises(InvalidArgumentsError) as exc:
        run_inspect_symbol(inspect_factory, "does_not_exist")
    assert "does_not_exist" in exc.value.safe_message
    assert exc.value.diagnostic is None


def test_inspect_symbol_deterministic(inspect_factory) -> None:
    first = run_inspect_symbol(inspect_factory, "buy")
    second = run_inspect_symbol(inspect_factory, "buy")
    assert first == second


# --------------------------------------------------------------------------
# MCP server registration
# --------------------------------------------------------------------------


def _engine_factory(path: Path):
    def factory(max_tokens=None, dependency_depth=None, **kwargs):
        return ContextEngine(
            path,
            searcher=CodeSearcher(path),
            budget=ContextBudget(max_tokens=max_tokens) if max_tokens else ContextBudget(),
        )

    return factory


def _has_tool(server, name: str) -> bool:
    tools = asyncio.run(server.list_tools())
    return name in {t.name for t in tools}


def test_inspect_symbol_registered_only_with_factory(
    inspect_factory,
) -> None:
    firewall = ContextFirewall()
    base = build_mcp_server(_engine_factory(ROOT), firewall)
    assert not _has_tool(base, "inspect_symbol")
    assert _has_tool(base, "get_context")
    with_inspect = build_mcp_server(
        _engine_factory(ROOT), firewall, inspect_factory=inspect_factory
    )
    assert _has_tool(with_inspect, "inspect_symbol")
    assert _has_tool(with_inspect, "get_context")


def test_inspect_and_impact_factories_registered_together(
    inspect_factory,
) -> None:
    firewall = ContextFirewall()

    def impact_factory(**kwargs):
        from repolens.impact import ImpactAnalyzer

        return ImpactAnalyzer(ROOT, reference_graph=inspect_factory())

    server = build_mcp_server(
        _engine_factory(ROOT),
        firewall,
        impact_factory=impact_factory,
        inspect_factory=inspect_factory,
    )
    tools = asyncio.run(server.list_tools())
    names = [t.name for t in tools]
    assert "get_context" in names
    assert "analyze_impact" in names
    assert "inspect_symbol" in names


def test_protocol_level_inspect_symbol(inspect_factory) -> None:
    server = build_mcp_server(
        _engine_factory(ROOT), ContextFirewall(), inspect_factory=inspect_factory
    )

    async def run():
        result = await server.call_tool(
            "inspect_symbol", {"name": "charge_card"}
        )
        return result

    result = asyncio.run(run())
    assert result.is_error is False
    data = json.loads(result.content[0].text)
    assert data["name"] == "charge_card"
    assert data["caller_count"] >= 2


def test_protocol_level_inspect_symbol_unknown_is_error(
    inspect_factory,
) -> None:
    server = build_mcp_server(
        _engine_factory(ROOT), ContextFirewall(), inspect_factory=inspect_factory
    )

    async def run():
        return await server.call_tool("inspect_symbol", {"name": "missing_sym"})

    result = asyncio.run(run())
    assert result.is_error is True
    text = result.content[0].text
    assert "missing_sym" in text
    assert "traceback" not in text.lower()