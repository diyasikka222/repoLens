"""The ``inspect_symbol`` MCP tool (Milestone 22).

A thin, additive adapter over the M22 :class:`repolens.call_graph.CallGraph`.
It exposes the statically resolved call relationships of a symbol rather than
re-implementing any resolution logic. Following the established MCP tool
pattern it:

1. validates the tool arguments;
2. builds the call graph via an injected factory (lazy);
3. looks up the symbol nodes and their direct callers/callees;
4. returns ONLY structured, safe, JSON-serializable output.

Like ``analyze_impact`` the tool is *additive*: it is registered only when a
factory is provided, and it never changes ``get_context``.
"""

from __future__ import annotations

import re
from typing import Callable

from repolens.call_graph import CallGraph
from repolens.mcp.errors import (
    ImpactAnalysisError,
    InternalError,
    InvalidArgumentsError,
)

#: Matches bare symbol names (underscores allowed) and dotted module paths.
_NAME_RE = re.compile(r"[\w.]+")

#: A factory that builds (once) a :class:`CallGraph` for the repository.
#: Injected so tests can provide fakes; the launcher provides the real one.
InspectFactory = Callable[..., CallGraph]


def validate_name(value) -> str:
    """Validate the ``name`` argument (non-empty symbol name)."""
    if not isinstance(value, str):
        raise InvalidArgumentsError(
            "The 'name' argument must be a string.",
            diagnostic=f"name had type {type(value).__name__}",
        )
    stripped = value.strip()
    if not stripped:
        raise InvalidArgumentsError("The 'name' argument must not be empty.")
    if not (_NAME_RE.fullmatch(stripped) or "::" in stripped):
        # Symbols may be dotted module paths (e.g. ``models.Cart``); refuse
        # anything with separators so untrusted clients cannot smuggle paths.
        raise InvalidArgumentsError(
            "The 'name' argument must be a symbol name.",
            diagnostic=f"unsupported name: {stripped!r}",
        )
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


def parse_inspect_arguments(arguments) -> dict:
    """Coerce raw MCP tool arguments into validated inspect options.

    Returns ``{"name": str, "max_depth": int|None}``.
    Raises :class:`InvalidArgumentsError` on any problem.
    """
    if arguments is None:
        raise InvalidArgumentsError("No arguments were provided to inspect_symbol.")
    if not isinstance(arguments, dict):
        raise InvalidArgumentsError(
            "inspect_symbol arguments must be an object.",
            diagnostic=f"arguments had type {type(arguments).__name__}",
        )
    if "name" not in arguments:
        raise InvalidArgumentsError("The 'name' argument is required.")

    name = validate_name(arguments.get("name"))
    max_depth = validate_max_depth(arguments.get("max_depth"))

    allowed = {"name", "max_depth"}
    unknown = set(arguments) - allowed
    if unknown:
        names = ", ".join(sorted(unknown))
        raise InvalidArgumentsError(f"Unsupported argument(s): {names}.")

    return {"name": name, "max_depth": max_depth}


def run_inspect_symbol(
    inspect_factory: InspectFactory,
    name: str,
    *,
    max_depth: int | None = None,
) -> dict:
    """Execute ``inspect_symbol`` and return a structured, safe response.

    ``inspect_factory`` lazily builds the repository :class:`CallGraph`.
    Unknown symbols produce a safe :class:`InvalidArgumentsError` instead of
    leaking internal details.
    """
    if not callable(inspect_factory):
        raise InternalError(
            "The symbol inspection service is unavailable.",
            diagnostic="inspect factory is not callable",
        )

    try:
        graph: CallGraph = inspect_factory()
    except Exception as exc:  # noqa: BLE001
        raise ImpactAnalysisError(
            "The call graph could not be initialized.",
            diagnostic=f"inspect factory failed: {type(exc).__name__}",
        ) from exc

    try:
        return _build_response(graph, name, max_depth)
    except InvalidArgumentsError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise ImpactAnalysisError(
            "The symbol inspection could not be completed.",
            diagnostic=f"inspect failed at runtime: {type(exc).__name__}",
        ) from exc


def query_symbols(call_graph: CallGraph, name: str) -> list[dict]:
    """Return matching call nodes for ``name`` as safe, dict-shaped entries.

    A node matches when its ``name`` equals ``name`` (the final component of a
    dotted form is also accepted for convenience). Entries are ordered by file
    path and node kind for deterministic output.
    """
    wanted = name.rsplit(".", 1)[-1].rsplit("::", 1)[-1]
    nodes = [node for node in call_graph.get_nodes() if node.name == wanted]
    nodes.sort(key=lambda node: (node.file_path.as_posix(), _kind(node), node.parent_class or ""))
    return [
        {
            "file": node.file_path.as_posix(),
            "name": node.name,
            "kind": _kind(node),
            "parent_class": node.parent_class,
        }
        for node in nodes
    ]


def _build_response(call_graph: CallGraph, name: str, max_depth: int | None) -> dict:
    """Assemble the structured symbol-inspection response dictionary."""
    if max_depth is None:
        max_depth = 1
    nodes = query_symbols(call_graph, name)
    if not nodes:
        raise InvalidArgumentsError(
            f"No symbol named {name!r} was found in the repository."
        )

    wanted = name.rsplit(".", 1)[-1].rsplit("::", 1)[-1]
    matched_keys = {
        _node_key(node) for node in call_graph.get_nodes() if node.name == wanted
    }

    callers: list[dict] = []
    callees: list[dict] = []
    seen_callers: set = set()
    seen_callees: set = set()

    def add_node(target, registry: list[dict], seen: set) -> None:
        key = _node_key(target)
        if key not in seen:
            seen.add(key)
            registry.append(
                {
                    "file": target.file_path.as_posix(),
                    "name": target.name,
                    "kind": _kind(target),
                    "parent_class": target.parent_class,
                }
            )

    for node in call_graph.get_nodes():
        if _node_key(node) not in matched_keys:
            continue
        for caller in call_graph.callers(node, max_depth=max_depth):
            add_node(caller, callers, seen_callers)
        for callee in call_graph.callees(node, max_depth=max_depth):
            add_node(callee, callees, seen_callees)

    callers.sort(key=lambda e: (e["file"], e["name"]))
    callees.sort(key=lambda e: (e["file"], e["name"]))

    return {
        "status": "ok",
        "name": name,
        "max_depth": max_depth,
        "node_count": len(nodes),
        "nodes": nodes,
        "caller_count": len(callers),
        "callers": callers,
        "callee_count": len(callees),
        "callees": callees,
    }


def _kind(node) -> str | None:
    return node.kind.value if node.kind else None


def _node_key(node) -> tuple:
    return (node.file_path.as_posix(), node.name, _kind(node), node.parent_class or "")