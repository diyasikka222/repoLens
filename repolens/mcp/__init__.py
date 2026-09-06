"""Model Context Protocol (MCP) integration for RepoLens (Milestones 14/21/22).

Exposes RepoLens to AI coding agents through the Model Context Protocol.
This layer is a *thin adapter*: it calls the existing public RepoLens APIs
(:class:`~repolens.context.ContextEngine`,
:class:`~repolens.context.ContextFirewall`) and never re-implements retrieval,
ranking, budgeting, or security logic.

Architecture::

    Agent
      ↓
    MCP (get_context / analyze_impact / inspect_symbol)
      ↓
    Context Firewall / Impact Analysis / Call Graph
      ↓
    Context Engine
      ↓
    Retrieval / Graph

The MCP server uses the **stdio** transport for local use by an IDE or
terminal agent.  The primary tool is ``get_context``, which returns only
firewall-cleared, safe context.  When an impact analyzer factory is wired in,
the ``analyze_impact`` tool is exposed as well (deterministic change impact
analysis; see :mod:`repolens.impact`).  When an M22 call-graph factory is
wired in, the additive ``inspect_symbol`` tool exposes statically resolved
call relationships (see :mod:`repolens.call_graph`).

This package depends on the higher-level public RepoLens APIs; core
components do not depend on this package, so MCP remains optional.
"""

from repolens.mcp.architecture_tool import (
    ArchitectureFactory,
    parse_architecture_candidates_arguments,
    parse_discover_subsystems_arguments,
    parse_explain_architecture_match_arguments,
    parse_inspect_architecture_arguments,
    run_architecture_candidates,
    run_discover_subsystems,
    run_explain_architecture_match,
    run_inspect_architecture,
    validate_arch_target,
    validate_architecture_config as validate_arch_config,
    validate_bool_flag as validate_arch_bool_flag,
    validate_limit as validate_arch_limit,
    validate_max_depth as validate_arch_max_depth,
    validate_max_subsystems as validate_arch_max_subsystems,
)
from repolens.mcp.errors import (
    ArchitectureError,
    ConfigurationError,
    ContextEngineError,
    FirewallError,
    InternalError,
    InvalidArgumentsError,
    McpError,
    RepositoryError,
)
from repolens.mcp.impact_tool import (
    parse_impact_arguments,
    run_analyze_impact,
    validate_limit,
    validate_max_depth,
    validate_target,
)
from repolens.mcp.inspect_tool import (
    parse_inspect_arguments,
    run_inspect_symbol,
    validate_max_depth as validate_inspect_max_depth,
    validate_name,
)
from repolens.mcp.server import (
    ARCHITECTURE_TOOL_DESCRIPTION,
    ARCHITECTURE_TOOL_NAME,
    CANDIDATES_TOOL_DESCRIPTION,
    CANDIDATES_TOOL_NAME,
    EXPLAIN_ARCH_TOOL_DESCRIPTION,
    EXPLAIN_ARCH_TOOL_NAME,
    IMPACT_TOOL_DESCRIPTION,
    IMPACT_TOOL_NAME,
    INSPECT_TOOL_DESCRIPTION,
    INSPECT_TOOL_NAME,
    SERVER_NAME,
    SERVER_VERSION,
    SUBSYSTEMS_TOOL_DESCRIPTION,
    SUBSYSTEMS_TOOL_NAME,
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
    "ARCHITECTURE_TOOL_DESCRIPTION",
    "ARCHITECTURE_TOOL_NAME",
    "ArchitectureError",
    "ArchitectureFactory",
    "CANDIDATES_TOOL_DESCRIPTION",
    "CANDIDATES_TOOL_NAME",
    "ConfigurationError",
    "ContextEngineError",
    "EXPLAIN_ARCH_TOOL_DESCRIPTION",
    "EXPLAIN_ARCH_TOOL_NAME",
    "FirewallError",
    "IMPACT_TOOL_DESCRIPTION",
    "IMPACT_TOOL_NAME",
    "INSPECT_TOOL_DESCRIPTION",
    "INSPECT_TOOL_NAME",
    "InternalError",
    "InvalidArgumentsError",
    "McpError",
    "RepositoryError",
    "SERVER_NAME",
    "SERVER_VERSION",
    "SUBSYSTEMS_TOOL_DESCRIPTION",
    "SUBSYSTEMS_TOOL_NAME",
    "TOOL_DESCRIPTION",
    "TOOL_NAME",
    "build_mcp_server",
    "parse_arguments",
    "parse_architecture_candidates_arguments",
    "parse_discover_subsystems_arguments",
    "parse_explain_architecture_match_arguments",
    "parse_impact_arguments",
    "parse_inspect_arguments",
    "parse_inspect_architecture_arguments",
    "run_analyze_impact",
    "run_architecture_candidates",
    "run_discover_subsystems",
    "run_explain_architecture_match",
    "run_get_context",
    "run_inspect_architecture",
    "run_inspect_symbol",
    "validate_arch_bool_flag",
    "validate_arch_config",
    "validate_arch_limit",
    "validate_arch_max_depth",
    "validate_arch_max_subsystems",
    "validate_arch_target",
    "validate_dependency_depth",
    "validate_inspect_max_depth",
    "validate_limit",
    "validate_max_depth",
    "validate_max_tokens",
    "validate_name",
    "validate_query",
    "validate_target",
]
