"""Model Context Protocol (MCP) integration for RepoLens (Milestone 14).

Exposes RepoLens to AI coding agents through the Model Context Protocol.
This layer is a *thin adapter*: it calls the existing public RepoLens APIs
(:class:`~repolens.context.ContextEngine` and
:class:`~repolens.context.ContextFirewall`) and never re-implements retrieval,
ranking, budgeting, or security logic.

Architecture::

    Agent
      ↓
    MCP (get_context)
      ↓
    Context Firewall
      ↓
    Context Engine
      ↓
    Retrieval / Graph

The MCP server uses the **stdio** transport for local use by an IDE or
terminal agent.  The only tool exposed is ``get_context``, which returns only
firewall-cleared, safe context.

This package depends on the higher-level public RepoLens APIs; core
components do not depend on this package, so MCP remains optional.
"""

from repolens.mcp.errors import (
    ConfigurationError,
    ContextEngineError,
    FirewallError,
    InternalError,
    InvalidArgumentsError,
    McpError,
    RepositoryError,
)
from repolens.mcp.server import (
    SERVER_NAME,
    SERVER_VERSION,
    TOOL_DESCRIPTION,
    TOOL_NAME,
    build_mcp_server,
)
from repolens.mcp.tool import (
    parse_arguments,
    run_get_context,
    validate_dependency_depth,
    validate_max_tokens,
    validate_query,
)

__all__ = [
    "ConfigurationError",
    "ContextEngineError",
    "FirewallError",
    "InternalError",
    "InvalidArgumentsError",
    "McpError",
    "RepositoryError",
    "SERVER_NAME",
    "SERVER_VERSION",
    "TOOL_DESCRIPTION",
    "TOOL_NAME",
    "build_mcp_server",
    "parse_arguments",
    "run_get_context",
    "validate_dependency_depth",
    "validate_max_tokens",
    "validate_query",
]
