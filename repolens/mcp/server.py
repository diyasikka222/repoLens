"""Build the RepoLens MCP server (Milestones 14/21/22).

This module constructs an :class:`~mcp.server.mcpserver.MCPServer` (the MCP
2.x SDK) that exposes a ``get_context`` tool, plus additive ``analyze_impact``
and ``inspect_symbol`` tools when their factories are wired in.  It is a thin
adapter: it injects the existing
:class:`~repolens.context.ContextEngine` factory,
:class:`~repolens.context.ContextFirewall`, and (optionally) the M21 impact
analyzer / M22 call-graph factories, then calls their public methods.  No
retrieval, ranking, budgeting, or security logic lives here.

Transport
---------
The server uses **stdio** for local use by an IDE or terminal agent.  Because
MCP communicates over stdout, *all* diagnostics go to stderr (through the
standard :mod:`logging` module), never to stdout.
"""

from __future__ import annotations

import logging

from mcp.server.mcpserver import MCPServer
from mcp import types

from repolens.context import ContextFirewall
from repolens.mcp.errors import McpError
from repolens.mcp.impact_tool import (
    ImpactFactory,
    parse_impact_arguments,
    run_analyze_impact,
)
from repolens.mcp.inspect_tool import (
    InspectFactory,
    parse_inspect_arguments,
    run_inspect_symbol,
)
from repolens.mcp.tool import EngineFactory, parse_arguments, run_get_context

logger = logging.getLogger("repolens.mcp")

SERVER_NAME = "repolens"
SERVER_VERSION = "0.1.0"

TOOL_NAME = "get_context"
IMPACT_TOOL_NAME = "analyze_impact"
INSPECT_TOOL_NAME = "inspect_symbol"

TOOL_DESCRIPTION = (
    "Search the repository and return a safe context package for the given "
    "developer query. RepoLens finds the most relevant files, follows "
    "relevant dependency relationships, respects a context budget, and "
    "filters potentially sensitive content before returning it. "
    "The tool does NOT guarantee perfect secret detection; it is a "
    "defense-in-depth layer. Returns structured context: query, selected "
    "files with selection reasons, estimated token count, budget, firewall "
    "decisions, and rendered safe context."
)

IMPACT_TOOL_DESCRIPTION = (
    "Analyze the blast radius of changing a repository target. Accepts a file "
    "path, a dotted module, a symbol name, or 'path/to/file.py::symbol'.\n\n"
    "Returns structured output: the resolved target, a deterministic risk "
    "classification (low/medium/high), a summary of affected files grouped by "
    "relationship (direct_dependency, indirect_dependency, reverse_dependency, "
    "test, configuration, api_consumer, direct_caller, indirect_caller, "
    "direct_callee, indirect_callee), and the list of impacted files with "
    "the reason and evidence for each. Reverse traversal is bounded by "
    "max_depth (default 4) and is never unbounded. Ambiguous or unknown "
    "targets are rejected with a safe message.\n\n"
    "This is NOT a full language-server call graph: symbol-level findings are "
    "conservative import/naming evidence, clearly separated from module "
    "dependency edges. Caller/callee relationships (Milestone 22) are "
    "statically resolved and clearly flagged with a 'static' confidence "
    "value; when a workflow lacks the call graph they are simply absent."
)

INSPECT_TOOL_DESCRIPTION = (
    "Inspect the statically resolved call relationships of a symbol using the "
    "offline call graph. Accepts a bare symbol name (e.g. 'charge_card', "
    "'Cart') or a dotted module path.\n\n"
    "Returns structured output: the matching call-graph nodes (file, kind, "
    "parent class), and for each the direct callers and direct callees along "
    "with their counts. Traversal is bounded by max_depth (default 1) and is "
    "never unbounded. Symbols that cannot be resolved are rejected with a safe "
    "message.\n\n"
    "This is offline and deterministic: no external resolution or language "
    "server is consulted. Findings reflect what the code itself references."
)


def build_mcp_server(
    engine_factory: EngineFactory,
    firewall: ContextFirewall,
    *,
    impact_factory: ImpactFactory | None = None,
    inspect_factory: InspectFactory | None = None,
    server_name: str = SERVER_NAME,
    server_version: str = SERVER_VERSION,
) -> MCPServer:
    """Build and configure an :class:`MCPServer` exposing tools.

    Args:
        engine_factory: A callable ``(max_tokens=..., dependency_depth=...)``
            that returns a configured :class:`ContextEngine`.
        firewall: A :class:`ContextFirewall` used to guarantee safe output.
        impact_factory: Optional callable returning an
            :class:`repolens.impact.ImpactAnalyzer`; when provided, the
            ``analyze_impact`` tool is registered alongside ``get_context``
            (which is unchanged and always registered).
        inspect_factory: Optional callable returning a
            :class:`repolens.call_graph.CallGraph`; when provided, the
            additive ``inspect_symbol`` tool is registered. Independent of
            ``impact_factory``.
        server_name: MCP server name.
        server_version: MCP server version.
    """
    server = MCPServer(name=server_name, version=server_version)

    def get_context(
        query: str,
        max_tokens: int | None = None,
        dependency_depth: int | None = None,
    ) -> types.CallToolResult:
        try:
            parsed = parse_arguments(
                {"query": query, "max_tokens": max_tokens,
                 "dependency_depth": dependency_depth}
            )
        except McpError as exc:
            _log_diagnostic(exc)
            return _error_result(exc.safe_message)

        try:
            response = run_get_context(
                engine_factory,
                firewall,
                parsed["query"],
                max_tokens=parsed["max_tokens"],
                dependency_depth=parsed["dependency_depth"],
            )
        except McpError as exc:
            _log_diagnostic(exc)
            return _error_result(exc.safe_message)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Unexpected failure in get_context: %s", type(exc).__name__
            )
            return _error_result(
                "An unexpected internal error occurred while processing the "
                "query."
            )

        # The response is a JSON-safe dict; MCP serializes it as text content.
        return response

    server.add_tool(
        get_context,
        name=TOOL_NAME,
        description=TOOL_DESCRIPTION,
    )

    if impact_factory is not None:
        def analyze_impact(
            target: str,
            max_depth: int | None = None,
            limit: int | None = None,
        ) -> types.CallToolResult:
            try:
                parsed = parse_impact_arguments(
                    {"target": target, "max_depth": max_depth, "limit": limit}
                )
            except McpError as exc:
                _log_diagnostic(exc)
                return _error_result(exc.safe_message)

            try:
                response = run_analyze_impact(
                    impact_factory,
                    parsed["target"],
                    max_depth=parsed["max_depth"],
                    limit=parsed["limit"],
                )
            except McpError as exc:
                _log_diagnostic(exc)
                return _error_result(exc.safe_message)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Unexpected failure in analyze_impact: %s",
                    type(exc).__name__,
                )
                return _error_result(
                    "An unexpected internal error occurred while analyzing "
                    "impact."
                )

            return response

        server.add_tool(
            analyze_impact,
            name=IMPACT_TOOL_NAME,
            description=IMPACT_TOOL_DESCRIPTION,
        )

    if inspect_factory is not None:
        def inspect_symbol(
            name: str,
            max_depth: int | None = None,
        ) -> types.CallToolResult:
            try:
                parsed = parse_inspect_arguments(
                    {"name": name, "max_depth": max_depth}
                )
            except McpError as exc:
                _log_diagnostic(exc)
                return _error_result(exc.safe_message)

            try:
                response = run_inspect_symbol(
                    inspect_factory,
                    parsed["name"],
                    max_depth=parsed["max_depth"],
                )
            except McpError as exc:
                _log_diagnostic(exc)
                return _error_result(exc.safe_message)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "Unexpected failure in inspect_symbol: %s",
                    type(exc).__name__,
                )
                return _error_result(
                    "An unexpected internal error occurred while inspecting "
                    "the symbol."
                )

            return response

        server.add_tool(
            inspect_symbol,
            name=INSPECT_TOOL_NAME,
            description=INSPECT_TOOL_DESCRIPTION,
        )

    return server


def _error_result(safe_message: str) -> types.CallToolResult:
    """Return an MCP error result with a safe message."""
    return types.CallToolResult(
        content=[types.TextContent(type="text", text=safe_message)],
        isError=True,
    )


def _log_diagnostic(exc: McpError) -> None:
    """Log a private diagnostic to stderr (never to stdout)."""
    if exc.diagnostic:
        logger.warning("MCP error diagnostic: %s", exc.diagnostic)
    else:
        logger.warning("MCP error: %s", exc.safe_message)
