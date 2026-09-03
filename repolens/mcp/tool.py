"""The ``get_context`` MCP tool (Milestone 14).

This module defines the single primary tool exposed by the RepoLens MCP
server.  It is a thin adapter that:

1. validates the tool arguments;
2. calls :meth:`repolens.context.ContextEngine.build_context`;
3. passes the result through :class:`repolens.context.ContextFirewall`;
4. returns ONLY the safe result.

The original :class:`~repolens.context.package.ContextPackage` is never
returned to the caller; only the firewall-cleared
:class:`~repolens.context.firewall.safe_package.SafeContextPackage` ever
reaches the MCP response.
"""

from __future__ import annotations

from typing import Callable

from repolens.context import ContextEngine, ContextFirewall, ContextPackage
from repolens.context.firewall import FirewallResult, SafeContextPackage
from repolens.mcp.errors import (
    ContextEngineError,
    FirewallError,
    InternalError,
    InvalidArgumentsError,
)

#: A factory that builds a :class:`ContextEngine` for the given options.
#: Injected so tests can provide fakes; the launcher provides the real one.
EngineFactory = Callable[..., ContextEngine]


def validate_query(query) -> str:
    """Validate the ``query`` argument (must be non-empty, non-whitespace)."""
    if not isinstance(query, str):
        raise InvalidArgumentsError(
            "The 'query' argument must be a string.",
            diagnostic=f"query had type {type(query).__name__}",
        )
    stripped = query.strip()
    if not stripped:
        raise InvalidArgumentsError("The 'query' argument must not be empty.")
    return stripped


def validate_max_tokens(value) -> int | None:
    """Validate optional ``max_tokens`` (must be a positive integer)."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise InvalidArgumentsError(
            "The 'max_tokens' argument must be a positive integer."
        )
    if value <= 0:
        raise InvalidArgumentsError("The 'max_tokens' argument must be positive.")
    return value


def validate_dependency_depth(value) -> int | None:
    """Validate optional ``dependency_depth`` (must be non-negative int)."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise InvalidArgumentsError(
            "The 'dependency_depth' argument must be a non-negative integer."
        )
    if value < 0:
        raise InvalidArgumentsError(
            "The 'dependency_depth' argument must be non-negative."
        )
    return value


def parse_arguments(arguments) -> dict:
    """Coerce raw MCP tool arguments into validated engine options.

    Returns ``{"query": str, "max_tokens": int|None, "dependency_depth": int|None}``.
    Raises :class:`InvalidArgumentsError` on any problem.
    """
    if arguments is None:
        raise InvalidArgumentsError("No arguments were provided to get_context.")
    if not isinstance(arguments, dict):
        raise InvalidArgumentsError(
            "get_context arguments must be an object.",
            diagnostic=f"arguments had type {type(arguments).__name__}",
        )
    if "query" not in arguments:
        raise InvalidArgumentsError("The 'query' argument is required.")

    query = validate_query(arguments.get("query"))
    max_tokens = validate_max_tokens(arguments.get("max_tokens"))
    dependency_depth = validate_dependency_depth(arguments.get("dependency_depth"))

    # Reject unknown keys to keep the surface tight and predictable.
    allowed = {"query", "max_tokens", "dependency_depth"}
    unknown = set(arguments) - allowed
    if unknown:
        names = ", ".join(sorted(unknown))
        raise InvalidArgumentsError(
            f"Unsupported argument(s): {names}."
        )

    return {
        "query": query,
        "max_tokens": max_tokens,
        "dependency_depth": dependency_depth,
    }


def run_get_context(
    engine_factory: EngineFactory,
    firewall: ContextFirewall,
    query: str,
    *,
    max_tokens: int | None = None,
    dependency_depth: int | None = None,
) -> dict:
    """Execute ``get_context`` and return a safe, JSON-serializable response.

    ``engine_factory`` builds a :class:`ContextEngine` honoring the requested
    options; ``firewall`` is the (already constructed) security boundary.

    Raises an :class:`~repolens.mcp.errors.McpError` subclass on failure.
    """
    if not callable(engine_factory):
        raise InternalError(
            "The context service is unavailable.",
            diagnostic="engine factory is not callable",
        )
    if not isinstance(firewall, ContextFirewall):
        raise InternalError(
            "The context security service is unavailable.",
            diagnostic="injected firewall is not a ContextFirewall",
        )

    # 1. Build a ContextEngine honoring the requested options.
    try:
        engine = engine_factory(max_tokens=max_tokens, dependency_depth=dependency_depth)
    except InvalidArgumentsError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise ContextEngineError(
            "The context engine could not be configured for this query.",
            diagnostic=f"engine factory failed: {type(exc).__name__}",
        ) from exc

    # 2. Build the ContextPackage via the engine (never bypassed).
    try:
        package: ContextPackage = engine.build_context(query)
    except Exception as exc:  # noqa: BLE001
        raise ContextEngineError(
            "The context engine could not process the query.",
            diagnostic=f"build_context failed: {type(exc).__name__}",
        ) from exc

    # 3. Inspect the package with the firewall.
    try:
        result: FirewallResult = firewall.inspect(package)
        safe: SafeContextPackage = firewall.safe_package(package, result)
    except Exception as exc:  # noqa: BLE001
        raise FirewallError(
            "The context firewall could not inspect the result.",
            diagnostic=f"firewall failed: {type(exc).__name__}",
        ) from exc

    # 4. Render ONLY the safe result.
    try:
        rendered = _render_safe(safe)
    except Exception as exc:  # noqa: BLE001
        raise InternalError(
            "The response could not be rendered.",
            diagnostic=f"rendering failed: {type(exc).__name__}",
        ) from exc

    return _build_response(safe, result, rendered)


def _build_response(safe: SafeContextPackage, result: FirewallResult, rendered: str) -> dict:
    """Assemble the structured, safe response dictionary."""
    return {
        "status": "ok",
        "query": safe.query,
        "budget": {"max_tokens": safe.budget.max_tokens},
        "total_estimated_tokens": safe.total_estimated_tokens,
        "intent": safe.intent,
        "matched_symbols": list(safe.matched_symbols),
        "selected_files": [
            {
                "path": c.path,
                "role": c.role,
                "decision": c.decision,
                "estimated_tokens": c.estimated_tokens,
                "selection_reason": c.selection_reason,
                "inclusion_reason": c.inclusion_reason,
            }
            for c in safe.safe_files
        ],
        "blocked_files": [
            {"path": c.path, "reason": "blocked by firewall"}
            for c in safe.blocked_files
        ],
        "firewall": {
            "enabled": safe.firewall_enabled,
            "policy_version": safe.policy_version,
            "findings": [
                {
                    "path": f.path,
                    "line": f.line,
                    "type": f.type,
                    "severity": f.severity,
                    "decision": f.decision,
                    "reason": f.reason,
                }
                for f in safe.findings
            ],
        },
        "rendered_safe_context": rendered,
    }


def _render_safe(safe: SafeContextPackage) -> str:
    """Render the safe context to deterministic text."""
    from repolens.context.firewall.render import render_safe_context

    return render_safe_context(safe)
