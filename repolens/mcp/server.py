"""Build the RepoLens MCP server (Milestone 14).

This module constructs an :class:`~mcp.server.mcpserver.MCPServer` (the MCP
2.x SDK) that exposes a single ``get_context`` tool.  It is a thin adapter: it
injects the existing :class:`~repolens.context.ContextEngine` factory and
:class:`~repolens.context.ContextFirewall`, then calls their public methods.
No retrieval, ranking, budgeting, or security logic lives here.

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
from repolens.mcp.tool import EngineFactory, parse_arguments, run_get_context

logger = logging.getLogger("repolens.mcp")

SERVER_NAME = "repolens"
SERVER_VERSION = "0.1.0"

TOOL_NAME = "get_context"

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


def build_mcp_server(
    engine_factory: EngineFactory,
    firewall: ContextFirewall,
    *,
    server_name: str = SERVER_NAME,
    server_version: str = SERVER_VERSION,
) -> MCPServer:
    """Build and configure an :class:`MCPServer` exposing ``get_context``.

    Args:
        engine_factory: A callable ``(max_tokens=..., dependency_depth=...)``
            that returns a configured :class:`ContextEngine`.
        firewall: A :class:`ContextFirewall` used to guarantee safe output.
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
