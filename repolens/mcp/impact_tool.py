"""The ``analyze_impact`` MCP tool (Milestone 21).

A thin adapter over :class:`repolens.impact.ImpactAnalyzer`. Like the
``get_context`` tool it:

1. validates the tool arguments;
2. builds the impact analyzer via an injected factory (lazy);
3. resolves and analyzes the target;
4. returns ONLY structured, safe, JSON-serializable output.

No retrieval, graph, or evidence logic lives here.
"""

from __future__ import annotations

from typing import Callable

from repolens.impact import ImpactAnalyzer, ImpactTargetError, ImpactResult
from repolens.mcp.errors import (
    ImpactAnalysisError,
    InternalError,
    InvalidArgumentsError,
)


#: A factory that builds (once) an :class:`ImpactAnalyzer` for the repository.
#: Injected so tests can provide fakes; the launcher provides the real one.
ImpactFactory = Callable[..., ImpactAnalyzer]


def validate_target(target) -> str:
    """Validate the ``target`` argument (non-empty string)."""
    if not isinstance(target, str):
        raise InvalidArgumentsError(
            "The 'target' argument must be a string.",
            diagnostic=f"target had type {type(target).__name__}",
        )
    stripped = target.strip()
    if not stripped:
        raise InvalidArgumentsError("The 'target' argument must not be empty.")
    return stripped


def validate_max_depth(value) -> int | None:
    """Validate optional ``max_depth`` (non-negative integer)."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise InvalidArgumentsError(
            "The 'max_depth' argument must be a non-negative integer."
        )
    if value < 0:
        raise InvalidArgumentsError("The 'max_depth' argument must be non-negative.")
    return value


def validate_limit(value) -> int | None:
    """Validate optional ``limit`` (positive integer)."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise InvalidArgumentsError(
            "The 'limit' argument must be a positive integer."
        )
    if value <= 0:
        raise InvalidArgumentsError("The 'limit' argument must be positive.")
    return value


def parse_impact_arguments(arguments) -> dict:
    """Coerce raw MCP tool arguments into validated impact options.

    Returns ``{"target": str, "max_depth": int|None, "limit": int|None}``.
    Raises :class:`InvalidArgumentsError` on any problem.
    """
    if arguments is None:
        raise InvalidArgumentsError("No arguments were provided to analyze_impact.")
    if not isinstance(arguments, dict):
        raise InvalidArgumentsError(
            "analyze_impact arguments must be an object.",
            diagnostic=f"arguments had type {type(arguments).__name__}",
        )
    if "target" not in arguments:
        raise InvalidArgumentsError("The 'target' argument is required.")

    target = validate_target(arguments.get("target"))
    max_depth = validate_max_depth(arguments.get("max_depth"))
    limit = validate_limit(arguments.get("limit"))

    allowed = {"target", "max_depth", "limit"}
    unknown = set(arguments) - allowed
    if unknown:
        names = ", ".join(sorted(unknown))
        raise InvalidArgumentsError(
            f"Unsupported argument(s): {names}."
        )

    return {
        "target": target,
        "max_depth": max_depth,
        "limit": limit,
    }


def run_analyze_impact(
    impact_factory: ImpactFactory,
    target: str,
    *,
    max_depth: int | None = None,
    limit: int | None = None,
) -> dict:
    """Execute ``analyze_impact`` and return a structured, safe response.

    ``impact_factory`` lazily builds the repository :class:`ImpactAnalyzer`.
    Ambiguous or unknown targets produce a safe :class:`InvalidArgumentsError`
    instead of leaking internal paths beyond the repository.
    """
    if not callable(impact_factory):
        raise InternalError(
            "The impact analysis service is unavailable.",
            diagnostic="impact factory is not callable",
        )

    try:
        analyzer: ImpactAnalyzer = impact_factory()
    except Exception as exc:  # noqa: BLE001
        raise ImpactAnalysisError(
            "The impact analyzer could not be initialized.",
            diagnostic=f"impact factory failed: {type(exc).__name__}",
        ) from exc

    try:
        result: ImpactResult = analyzer.analyze(target, max_depth=max_depth, limit=limit)
    except ImpactTargetError as exc:
        raise InvalidArgumentsError(str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise ImpactAnalysisError(
            "The impact analysis could not be completed.",
            diagnostic=f"analyze failed at runtime: {type(exc).__name__}",
        ) from exc

    return _build_response(result, limit)


def _build_response(result: ImpactResult, limit: int | None) -> dict:
    """Assemble the structured impact-analysis response dictionary."""
    items = result.to_dict()["items"]
    if limit is not None:
        items = items[:limit]
    return {
        "status": "ok",
        "target": result.target,
        "kind": result.kind,
        "target_path": result.target_path.as_posix() if result.target_path else None,
        "symbol": result.symbol,
        "module": result.module,
        "risk": result.risk.value,
        "max_depth_reached": result.max_depth_reached,
        "summary": dict(result.summary),
        "item_count": len(items),
        "items": items,
    }