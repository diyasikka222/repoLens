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
)
from repolens.mcp.change_plan_tool import (
    ChangePlanFactory,
    parse_change_context_arguments,
    parse_change_plan_arguments,
    run_change_context,
    run_change_plan,
)
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
from repolens import __version__
from repolens.mcp.tool import EngineFactory, parse_arguments, run_get_context

logger = logging.getLogger("repolens.mcp")

SERVER_NAME = "repolens"
SERVER_VERSION = __version__

TOOL_NAME = "get_context"
IMPACT_TOOL_NAME = "analyze_impact"
INSPECT_TOOL_NAME = "inspect_symbol"
ARCHITECTURE_TOOL_NAME = "inspect_architecture"
SUBSYSTEMS_TOOL_NAME = "discover_subsystems"
CANDIDATES_TOOL_NAME = "architecture_candidates"
EXPLAIN_ARCH_TOOL_NAME = "explain_architecture_match"
CHANGE_PLAN_TOOL_NAME = "change_plan"
CHANGE_CONTEXT_TOOL_NAME = "change_context"

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

ARCHITECTURE_TOOL_DESCRIPTION = (
    "Inspect the repository architecture around a single target. Accepts a "
    "repo-relative file path (e.g. 'store/services/checkout.py'), a dotted "
    "module (e.g. 'store.services.checkout'), or a package path (e.g. "
    "'store/repositories').\n\n"
    "Returns structured output: the resolved node type, its package / module / "
    "file identities, containing package, subsystem, direct dependencies and "
    "dependents, and a bounded transitive neighborhood (max_depth, default 2). "
    "Statistics include direct/transitive dependency and dependent counts, "
    "package file/module membership, and optional subsystem stats. Depth is "
    "capped and the response is deterministic. Unknown targets are rejected "
    "with a safe message; it never dumps the whole repository."
)

SUBSYSTEMS_TOOL_DESCRIPTION = (
    "List the deterministic architectural subsystems of the repository. "
    "Subsystems are discovered with plain graph heuristics (top-level package "
    "trees) — no LLM is used for naming.\n\n"
    "Returns each subsystem's id and display name, its packages, modules, "
    "files, entry modules, optional dependency/dependent subsystem ids, and "
    "optional statistics. Ordering is deterministic (by id). Controllable via "
    "max_subsystems (capped)."
)

CANDIDATES_TOOL_DESCRIPTION = (
    "Generate architecture-aware candidates for a query, independent of full "
    "context retrieval. Accepts a free-text query and returns the bounded, "
    "deterministic set of architecture candidates (direct matches, "
    "dependencies/dependents, neighbors, package and subsystem proximity).\n\n"
    "Each candidate reports its file, module, package, score, architecture "
    "score (rank), inclusion reason (e.g. 'architecture: direct package "
    "match', 'architecture: dependency of matched module'), node type, "
    "subsystem, and dependency direction. The query's matched signals are "
    "returned too. Bounded by limit (default 20) and max_depth (default 1); "
    "an optional 'architecture' configuration object further caps expansion. "
    "Deterministic."
)

EXPLAIN_ARCH_TOOL_DESCRIPTION = (
    "Explain why a specific file, module, or package was (or was not) "
    "considered architecturally relevant to a query.\n\n"
    "Returns the query's architecture signals, the matched nodes for the "
    "query, the target's inclusion reason and architecture score, its "
    "subsystem relationship, and bounded dependency path(s) from a matched "
    "module to the target. When the target was not architecturally relevant a "
    "structured explanation is returned (never an exception). Unknown or "
    "unsafe targets produce a safe validation error."
)

CHANGE_PLAN_TOOL_DESCRIPTION = (
    "Produce a deterministic change plan for a change you want to make in the "
    "repository. Accepts a natural-language change request (e.g. 'make "
    "checkout reject empty carts'), and optionally an explicit target (a file "
    "path, dotted module, package, or symbol name) and its kind.\n\n"
    "Returns structured output: the request analysis/signals, the resolved "
    "primary change target (with confidence and reasons), all target "
    "candidates, the affected files and symbols, statically resolved callers "
    "and callees, dependency and dependent groups, the architecture/subsystem "
    "context, affected tests, a bounded deterministic inspection order, a "
    "risk classification (low/medium/high) with risk factors, and statistics. "
    "Bounded by max_targets (<=50), max_files (<=500), max_tests (<=100), and "
    "max_depth (<=6). Deterministic; never modifies the repository."
)

CHANGE_CONTEXT_TOOL_DESCRIPTION = (
    "Build a change-aware context package for a change you want to make. "
    "Accepts the same change request and optional target as change_plan, plus "
    "a context token budget, and folds the plan's affected files into the "
    "existing ranking and budget pipeline (change-plan files form a final "
    "inspection tier — they never outrank direct query matches).\n\n"
    "Returns only firewall-cleared safe context: the selected files with "
    "roles, decisions, selection reasons, and a deterministic per-file "
    "explanation of change-plan provenance, plus blocked files, the compact "
    "change plan (primary target, counts, risk, confidence, summary), and the "
    "rendered safe context. Category flags (include_tests, "
    "include_dependencies, include_callers, include_callees, "
    "include_architecture) control which plan categories are folded in."
)


def build_mcp_server(
    engine_factory: EngineFactory,
    firewall: ContextFirewall,
    *,
    impact_factory: ImpactFactory | None = None,
    inspect_factory: InspectFactory | None = None,
    architecture_factory: ArchitectureFactory | None = None,
    change_plan_factory: ChangePlanFactory | None = None,
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
        architecture_factory: Optional callable returning the shared
            :class:`~repolens.mcp.architecture_tool.ArchitectureState`; when
            provided, the four additive M23.3 architecture tools are
            registered (``inspect_architecture``, ``discover_subsystems``,
            ``architecture_candidates``, ``explain_architecture_match``).
            Independent of the other factories.
        change_plan_factory: Optional callable returning the shared
            :class:`~repolens.mcp.change_plan_tool.ChangePlanState`; when
            provided, the additive M24.2 ``change_plan`` and ``change_context``
            tools are registered. Independent of the other factories.
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

    if architecture_factory is not None:
        _register_architecture_tools(server, architecture_factory)

    if change_plan_factory is not None:
        _register_change_plan_tools(
            server, engine_factory, firewall, change_plan_factory
        )

    return server


def _register_architecture_tools(server: MCPServer, architecture_factory: ArchitectureFactory) -> None:
    """Register the four additive M23.3 architecture tools (unchanged core).

    Each tool declares an explicit parameter signature so the MCP SDK derives a
    precise input schema (same pattern as the core, impact, and inspect tools)
    and all values are validated by the tool-specific ``parse_*`` function.
    """
    from mcp import types

    def _invoke(parse, run, **kwargs) -> types.CallToolResult:
        try:
            parsed = parse(dict(kwargs))
        except McpError as exc:
            _log_diagnostic(exc)
            return _error_result(exc.safe_message)
        try:
            return run(architecture_factory, **parsed)
        except McpError as exc:
            _log_diagnostic(exc)
            return _error_result(exc.safe_message)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Unexpected failure in an architecture tool: %s",
                type(exc).__name__,
            )
            return _error_result(
                "An unexpected internal error occurred while processing "
                "the architecture query."
            )

    def inspect_architecture(
        target: str,
        max_depth: int | None = None,
        include_dependencies: bool | None = None,
        include_dependents: bool | None = None,
        include_subsystems: bool | None = None,
    ) -> types.CallToolResult:
        return _invoke(
            parse_inspect_architecture_arguments,
            run_inspect_architecture,
            target=target,
            max_depth=max_depth,
            include_dependencies=include_dependencies,
            include_dependents=include_dependents,
            include_subsystems=include_subsystems,
        )

    def discover_subsystems(
        max_subsystems: int | None = None,
        include_dependencies: bool | None = None,
        include_dependents: bool | None = None,
        include_stats: bool | None = None,
    ) -> types.CallToolResult:
        return _invoke(
            parse_discover_subsystems_arguments,
            run_discover_subsystems,
            max_subsystems=max_subsystems,
            include_dependencies=include_dependencies,
            include_dependents=include_dependents,
            include_stats=include_stats,
        )

    def architecture_candidates(
        query: str,
        limit: int | None = None,
        max_depth: int | None = None,
        architecture: dict | None = None,
    ) -> types.CallToolResult:
        return _invoke(
            parse_architecture_candidates_arguments,
            run_architecture_candidates,
            query=query,
            limit=limit,
            max_depth=max_depth,
            architecture=architecture,
        )

    def explain_architecture_match(
        query: str,
        target: str,
    ) -> types.CallToolResult:
        return _invoke(
            parse_explain_architecture_match_arguments,
            run_explain_architecture_match,
            query=query,
            target=target,
        )

    server.add_tool(
        inspect_architecture,
        name=ARCHITECTURE_TOOL_NAME,
        description=ARCHITECTURE_TOOL_DESCRIPTION,
    )
    server.add_tool(
        discover_subsystems,
        name=SUBSYSTEMS_TOOL_NAME,
        description=SUBSYSTEMS_TOOL_DESCRIPTION,
    )
    server.add_tool(
        architecture_candidates,
        name=CANDIDATES_TOOL_NAME,
        description=CANDIDATES_TOOL_DESCRIPTION,
    )
    server.add_tool(
        explain_architecture_match,
        name=EXPLAIN_ARCH_TOOL_NAME,
        description=EXPLAIN_ARCH_TOOL_DESCRIPTION,
    )


def _register_change_plan_tools(
    server: MCPServer,
    engine_factory: EngineFactory,
    firewall: ContextFirewall,
    change_plan_factory: ChangePlanFactory,
) -> None:
    """Register the two additive M24.2 change-plan tools (unchanged core).

    Each tool declares an explicit parameter signature so the MCP SDK derives
    a precise input schema, and all values are validated by the tool-specific
    ``parse_*`` function.
    """
    def _invoke(parse, run, **kwargs) -> types.CallToolResult:
        try:
            parsed = parse(dict(kwargs))
        except McpError as exc:
            _log_diagnostic(exc)
            return _error_result(exc.safe_message)
        try:
            return run(**parsed)
        except McpError as exc:
            _log_diagnostic(exc)
            return _error_result(exc.safe_message)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Unexpected failure in a change-plan tool: %s",
                type(exc).__name__,
            )
            return _error_result(
                "An unexpected internal error occurred while processing "
                "the change-plan request."
            )

    def change_plan(
        request: str,
        target: str | None = None,
        target_kind: str | None = None,
        max_targets: int | None = None,
        max_files: int | None = None,
        max_tests: int | None = None,
        max_depth: int | None = None,
    ) -> types.CallToolResult:
        return _invoke(
            parse_change_plan_arguments,
            lambda **kw: run_change_plan(change_plan_factory, **kw),
            request=request,
            target=target,
            target_kind=target_kind,
            max_targets=max_targets,
            max_files=max_files,
            max_tests=max_tests,
            max_depth=max_depth,
        )

    def change_context(
        request: str,
        target: str | None = None,
        target_kind: str | None = None,
        max_tokens: int | None = None,
        max_targets: int | None = None,
        max_files: int | None = None,
        max_tests: int | None = None,
        max_depth: int | None = None,
        include_tests: bool | None = None,
        include_dependencies: bool | None = None,
        include_callers: bool | None = None,
        include_callees: bool | None = None,
        include_architecture: bool | None = None,
    ) -> types.CallToolResult:
        return _invoke(
            parse_change_context_arguments,
            lambda **kw: run_change_context(
                engine_factory, firewall, change_plan_factory, **kw
            ),
            request=request,
            target=target,
            target_kind=target_kind,
            max_tokens=max_tokens,
            max_targets=max_targets,
            max_files=max_files,
            max_tests=max_tests,
            max_depth=max_depth,
            include_tests=include_tests,
            include_dependencies=include_dependencies,
            include_callers=include_callers,
            include_callees=include_callees,
            include_architecture=include_architecture,
        )

    server.add_tool(
        change_plan,
        name=CHANGE_PLAN_TOOL_NAME,
        description=CHANGE_PLAN_TOOL_DESCRIPTION,
    )
    server.add_tool(
        change_context,
        name=CHANGE_CONTEXT_TOOL_NAME,
        description=CHANGE_CONTEXT_TOOL_DESCRIPTION,
    )


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
