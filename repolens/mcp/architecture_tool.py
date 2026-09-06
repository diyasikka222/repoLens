"""Architecture-intelligence MCP tools (Milestone 23.3).

A thin, additive adapter over the existing RepoLens architecture layers:

- ``inspect_architecture`` — structured overview of one repository node
  (file / module / package): its package/module/file identities, subsystem,
  bounded dependency/dependent lists, transitive neighborhood, and stats.
- ``discover_subsystems`` — the deterministic subsystem discovery result.
- ``architecture_candidates`` — architecture-aware candidate generation
  independent of full context retrieval.
- ``explain_architecture_match`` — why a node was (or was not) considered
  architecturally relevant to a query.

Following the established MCP tool pattern each tool:

1. validates its arguments strictly;
2. builds the shared architecture state via an injected factory (lazy, once);
3. calls only existing public RepoLens APIs
   (:class:`~repolens.architecture.ArchitectureGraph`,
   :func:`repolens.subsystems.discover_subsystems`,
   :mod:`repolens.architecture_retrieval`);
4. returns only structured, safe, JSON-serializable output.

No retrieval, ranking, expansion, or discovery logic is reimplemented here, and
no internal Python object is ever returned to the caller.
"""

from __future__ import annotations

import re
import threading
from typing import Callable

from repolens.architecture import ArchitectureGraph, ArchitectureNodeKind
from repolens.architecture_retrieval import (
    ArchitectureRetrievalConfig,
    architecture_candidates,
    extract_architecture_signals,
)
from repolens.mcp.errors import (
    ArchitectureError,
    InternalError,
    InvalidArgumentsError,
)
from repolens.mcp.tool import validate_query
from repolens.subsystems import discover_subsystems

#: Boundedness caps imposed by the MCP layer on top of the existing limits.
MAX_MAX_DEPTH = 6
MAX_CANDIDATE_LIMIT = 200
MAX_SUBSYSTEMS_LIMIT = 500
MAX_ARCH_CONFIG_PACKAGES = 100
MAX_ARCH_CONFIG_MODULES = 500
MAX_ARCH_CONFIG_EXPANDED = 1000
MAX_ARCH_CONFIG_DEPTH = MAX_MAX_DEPTH

#: Per-side cap on direct dependency / dependent lists in one response.
NODE_NEIGHBOR_CAP = 200

#: Reject path traversal and escape attempts in target arguments.
_TARGET_RE = re.compile(r"[A-Za-z0-9_./,-]+")
_FORBIDDEN_PART = re.compile(r"(^\.\.$|/\.\./|^\.\./|/\.\.$|^/|^\\|:)")


def _target_is_safe(value: str) -> bool:
    """True when ``value`` cannot point outside the repository.

    Rejects absolute paths, Windows drive/UNC prefixes, ``..`` path
    components (anywhere), and characters outside the safe target alphabet.
    """
    if "\x00" in value:
        return False
    if not _TARGET_RE.fullmatch(value):
        return False
    return not _FORBIDDEN_PART.search(value)


def validate_arch_target(value) -> str:
    """Validate the shared ``target`` argument across architecture tools."""
    if not isinstance(value, str):
        raise InvalidArgumentsError(
            "The 'target' argument must be a string.",
            diagnostic=f"target had type {type(value).__name__}",
        )
    stripped = value.strip()
    if not stripped:
        raise InvalidArgumentsError("The 'target' argument must not be empty.")
    if not _target_is_safe(stripped):
        raise InvalidArgumentsError(
            "The 'target' argument must be a repository-relative file, module, "
            "or package.",
            diagnostic=f"unsafe target: {stripped!r}",
        )
    return stripped


def validate_max_depth(value, *, name: str = "max_depth") -> int | None:
    """Validate optional bounded ``max_depth`` (non-negative, capped)."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise InvalidArgumentsError(
            f"The '{name}' argument must be a non-negative integer."
        )
    if value < 0:
        raise InvalidArgumentsError(
            f"The '{name}' argument must be non-negative."
        )
    if value > MAX_MAX_DEPTH:
        raise InvalidArgumentsError(
            f"The '{name}' argument must not exceed {MAX_MAX_DEPTH}."
        )
    return value


def validate_limit(value) -> int | None:
    """Validate optional ``limit`` (positive integer, capped)."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise InvalidArgumentsError(
            "The 'limit' argument must be a positive integer."
        )
    if value <= 0:
        raise InvalidArgumentsError("The 'limit' argument must be positive.")
    if value > MAX_CANDIDATE_LIMIT:
        raise InvalidArgumentsError(
            f"The 'limit' argument must not exceed {MAX_CANDIDATE_LIMIT}."
        )
    return value


def validate_max_subsystems(value) -> int | None:
    """Validate optional ``max_subsystems`` (positive integer, capped)."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise InvalidArgumentsError(
            "The 'max_subsystems' argument must be a positive integer."
        )
    if value <= 0:
        raise InvalidArgumentsError(
            "The 'max_subsystems' argument must be positive."
        )
    if value > MAX_SUBSYSTEMS_LIMIT:
        raise InvalidArgumentsError(
            f"The 'max_subsystems' argument must not exceed "
            f"{MAX_SUBSYSTEMS_LIMIT}."
        )
    return value


def validate_bool_flag(value, *, name: str) -> bool:
    """Validate an optional boolean flag argument."""
    if value is None:
        return True
    if not isinstance(value, bool):
        raise InvalidArgumentsError(
            f"The '{name}' argument must be true or false."
        )
    return value


def validate_architecture_config(value) -> ArchitectureRetrievalConfig | None:
    """Validate the optional architecture configuration object.

    Accepts a dict with any of the four documented caps; every key must be a
    positive integer within the MCP bounds.
    """
    if value is None:
        return None
    if not isinstance(value, dict):
        raise InvalidArgumentsError(
            "The 'architecture' argument must be an object.",
            diagnostic=f"architecture had type {type(value).__name__}",
        )
    allowed = {
        "max_package_candidates",
        "max_module_candidates",
        "max_neighbor_depth",
        "max_expanded_nodes",
    }
    unknown = set(value) - allowed
    if unknown:
        names = ", ".join(sorted(unknown))
        raise InvalidArgumentsError(f"Unsupported architecture option(s): {names}.")
    try:
        return ArchitectureRetrievalConfig(
            max_package_candidates=(
                _validate_arch_int(
                    value.get("max_package_candidates"),
                    "max_package_candidates",
                    MAX_ARCH_CONFIG_PACKAGES,
                )
            ),
            max_module_candidates=(
                _validate_arch_int(
                    value.get("max_module_candidates"),
                    "max_module_candidates",
                    MAX_ARCH_CONFIG_MODULES,
                )
            ),
            max_neighbor_depth=(
                _validate_arch_int(
                    value.get("max_neighbor_depth"),
                    "max_neighbor_depth",
                    MAX_ARCH_CONFIG_DEPTH,
                )
            ),
            max_expanded_nodes=(
                _validate_arch_int(
                    value.get("max_expanded_nodes"),
                    "max_expanded_nodes",
                    MAX_ARCH_CONFIG_EXPANDED,
                )
            ),
        )
    except InvalidArgumentsError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise InvalidArgumentsError(
            "The 'architecture' argument is invalid.",
            diagnostic=f"architecture config: {type(exc).__name__}",
        ) from exc


def _validate_arch_int(value, name: str, maximum: int) -> int:
    if value is None:
        return ArchitectureRetrievalConfig().__dict__[name]
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise InvalidArgumentsError(
            f"The 'architecture.{name}' option must be a positive integer."
        )
    if value > maximum:
        raise InvalidArgumentsError(
            f"The 'architecture.{name}' option must not exceed {maximum}."
        )
    return value


# ---------------------------------------------------------------------------
# Argument validation
# ---------------------------------------------------------------------------


def parse_inspect_architecture_arguments(arguments) -> dict:
    """Coerce raw ``inspect_architecture`` arguments into validated options."""
    if arguments is None:
        raise InvalidArgumentsError("No arguments were provided to inspect_architecture.")
    if not isinstance(arguments, dict):
        raise InvalidArgumentsError(
            "inspect_architecture arguments must be an object.",
            diagnostic=f"arguments had type {type(arguments).__name__}",
        )
    if "target" not in arguments:
        raise InvalidArgumentsError("The 'target' argument is required.")

    allowed = {
        "target",
        "max_depth",
        "include_dependencies",
        "include_dependents",
        "include_subsystems",
    }
    unknown = set(arguments) - allowed
    if unknown:
        names = ", ".join(sorted(unknown))
        raise InvalidArgumentsError(f"Unsupported argument(s): {names}.")

    max_depth = validate_max_depth(arguments.get("max_depth"))
    if max_depth is None:
        max_depth = 2

    return {
        "target": validate_arch_target(arguments.get("target")),
        "max_depth": max_depth,
        "include_dependencies": validate_bool_flag(
            arguments.get("include_dependencies"), name="include_dependencies"
        ),
        "include_dependents": validate_bool_flag(
            arguments.get("include_dependents"), name="include_dependents"
        ),
        "include_subsystems": validate_bool_flag(
            arguments.get("include_subsystems"), name="include_subsystems"
        ),
    }


def parse_discover_subsystems_arguments(arguments) -> dict:
    """Coerce raw ``discover_subsystems`` arguments into validated options."""
    if arguments is None:
        raise InvalidArgumentsError("No arguments were provided to discover_subsystems.")
    if not isinstance(arguments, dict):
        raise InvalidArgumentsError(
            "discover_subsystems arguments must be an object.",
            diagnostic=f"arguments had type {type(arguments).__name__}",
        )

    allowed = {
        "max_subsystems",
        "include_stats",
        "include_dependencies",
        "include_dependents",
    }
    unknown = set(arguments) - allowed
    if unknown:
        names = ", ".join(sorted(unknown))
        raise InvalidArgumentsError(f"Unsupported argument(s): {names}.")

    return {
        "max_subsystems": validate_max_subsystems(
            arguments.get("max_subsystems")
        ),
        "include_stats": validate_bool_flag(
            arguments.get("include_stats"), name="include_stats"
        ),
        "include_dependencies": validate_bool_flag(
            arguments.get("include_dependencies"), name="include_dependencies"
        ),
        "include_dependents": validate_bool_flag(
            arguments.get("include_dependents"), name="include_dependents"
        ),
    }


def parse_architecture_candidates_arguments(arguments) -> dict:
    """Coerce raw ``architecture_candidates`` arguments into validated options."""
    if arguments is None:
        raise InvalidArgumentsError("No arguments were provided to architecture_candidates.")
    if not isinstance(arguments, dict):
        raise InvalidArgumentsError(
            "architecture_candidates arguments must be an object.",
            diagnostic=f"arguments had type {type(arguments).__name__}",
        )
    if "query" not in arguments:
        raise InvalidArgumentsError("The 'query' argument is required.")

    max_depth = validate_max_depth(arguments.get("max_depth"))
    if max_depth is None:
        max_depth = 1

    limit = validate_limit(arguments.get("limit"))
    if limit is None:
        limit = 20

    allowed = {"query", "limit", "max_depth", "architecture"}
    unknown = set(arguments) - allowed
    if unknown:
        names = ", ".join(sorted(unknown))
        raise InvalidArgumentsError(f"Unsupported argument(s): {names}.")

    return {
        "query": validate_query(arguments.get("query")),
        "limit": limit,
        "max_depth": max_depth,
        "architecture": validate_architecture_config(
            arguments.get("architecture")
        ),
    }


def parse_explain_architecture_match_arguments(arguments) -> dict:
    """Coerce raw ``explain_architecture_match`` arguments into options."""
    if arguments is None:
        raise InvalidArgumentsError("No arguments were provided to explain_architecture_match.")
    if not isinstance(arguments, dict):
        raise InvalidArgumentsError(
            "explain_architecture_match arguments must be an object.",
            diagnostic=f"arguments had type {type(arguments).__name__}",
        )
    if "query" not in arguments:
        raise InvalidArgumentsError("The 'query' argument is required.")
    if "target" not in arguments:
        raise InvalidArgumentsError("The 'target' argument is required.")

    allowed = {"query", "target"}
    unknown = set(arguments) - allowed
    if unknown:
        names = ", ".join(sorted(unknown))
        raise InvalidArgumentsError(f"Unsupported argument(s): {names}.")

    return {
        "query": validate_query(arguments.get("query")),
        "target": validate_arch_target(arguments.get("target")),
    }


# ---------------------------------------------------------------------------
# Shared lazy architecture state
# ---------------------------------------------------------------------------

#: A factory that builds (once) the shared architecture state for the repo.
#: Injected so tests can provide fakes; the launcher provides the real one.
ArchitectureFactory = Callable[..., "ArchitectureState"]


class ArchitectureState:
    """Lazily-built, shared architecture state for the MCP tools.

    The incremental index, the architecture graph, the subsystem discovery, and
    the lookup maps used by retrieval are all computed **once** on first use and
    then reused for every subsequent call, so architecture tool calls after the
    first perform no scanning or parsing at all.
    """

    def __init__(self, root: str | Path) -> None:
        from repolens.mcp.deps import _build_index

        self._root = root
        self._index = None
        self._parsed = None
        self._graph = None
        self._subsystems = None
        self._graph_basis = None
        self._subsystem_basis = None
        self._build_index = _build_index

    @property
    def root(self) -> Path:
        from pathlib import Path

        return Path(self._root)

    @property
    def index(self):
        if self._index is None:
            self._index = self._build_index(self.root)
            self._parsed = self._index.stats.files_parsed
        return self._index

    @property
    def parsed_file_count(self) -> int | None:
        """Files parsed by the incremental index (``None`` until first build)."""
        return self._parsed

    @property
    def graph(self) -> ArchitectureGraph:
        """The shared :class:`ArchitectureGraph` (a pure index projection)."""
        if self._graph is None:
            from repolens.architecture import ArchitectureGraphBuilder

            self._graph = ArchitectureGraphBuilder(
                self.root, index=self.index
            ).build()
        return self._graph

    @property
    def subsystems(self):
        if self._subsystems is None:
            self._subsystems = discover_subsystems(self.graph)
        return self._subsystems

    @property
    def graph_basis(self) -> dict:
        """Shared lookup maps used by architecture-aware retrieval."""
        if self._graph_basis is None:
            from repolens.architecture_retrieval import _index_graph

            self._graph_basis = _index_graph(self.graph)
        return self._graph_basis

    @property
    def subsystem_basis(self) -> dict:
        """Shared subsystem lookup maps used by architecture-aware retrieval."""
        if self._subsystem_basis is None:
            from repolens.architecture_retrieval import _index_subsystems

            self._subsystem_basis = _index_subsystems(self.subsystems)
        return self._subsystem_basis


# ---------------------------------------------------------------------------
# Tool execution
# ---------------------------------------------------------------------------


def _require_state(factory: ArchitectureFactory) -> ArchitectureState:
    if not callable(factory):
        raise InternalError(
            "The architecture service is unavailable.",
            diagnostic="architecture factory is not callable",
        )
    try:
        return factory()
    except Exception as exc:  # noqa: BLE001
        raise ArchitectureError(
            "The architecture graph could not be initialized.",
            diagnostic=f"architecture factory failed: {type(exc).__name__}",
        ) from exc


def run_inspect_architecture(
    architecture_factory: ArchitectureFactory,
    target: str,
    *,
    max_depth: int = 2,
    include_dependencies: bool = True,
    include_dependents: bool = True,
    include_subsystems: bool = True,
) -> dict:
    """Execute ``inspect_architecture`` and return a structured, safe response.

    Unknown targets produce a safe :class:`InvalidArgumentsError` instead of
    leaking internal details.
    """
    state = _require_state(architecture_factory)
    try:
        return _build_inspect_response(
            state,
            target,
            max_depth=max_depth,
            include_dependencies=include_dependencies,
            include_dependents=include_dependents,
            include_subsystems=include_subsystems,
        )
    except InvalidArgumentsError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise ArchitectureError(
            "The architecture inspection could not be completed.",
            diagnostic=f"inspect failed at runtime: {type(exc).__name__}",
        ) from exc


def run_discover_subsystems(
    architecture_factory: ArchitectureFactory,
    *,
    max_subsystems: int | None = None,
    include_stats: bool = True,
    include_dependencies: bool = True,
    include_dependents: bool = True,
) -> dict:
    """Execute ``discover_subsystems`` and return a structured, safe response."""
    state = _require_state(architecture_factory)
    try:
        return _build_subsystems_response(
            state,
            max_subsystems=max_subsystems,
            include_stats=include_stats,
            include_dependencies=include_dependencies,
            include_dependents=include_dependents,
        )
    except Exception as exc:  # noqa: BLE001
        raise ArchitectureError(
            "The subsystem discovery could not be completed.",
            diagnostic=f"discover_subsystems failed at runtime: {type(exc).__name__}",
        ) from exc


def run_architecture_candidates(
    architecture_factory: ArchitectureFactory,
    query: str,
    *,
    limit: int = 20,
    max_depth: int = 1,
    architecture: ArchitectureRetrievalConfig | None = None,
) -> dict:
    """Execute ``architecture_candidates`` and return a structured, safe response."""
    state = _require_state(architecture_factory)
    try:
        return _build_candidates_response(
            state,
            query,
            limit=limit,
            max_depth=max_depth,
            architecture=architecture,
        )
    except Exception as exc:  # noqa: BLE001
        raise ArchitectureError(
            "The architecture candidates could not be generated.",
            diagnostic=f"architecture_candidates failed at runtime: {type(exc).__name__}",
        ) from exc


def run_explain_architecture_match(
    architecture_factory: ArchitectureFactory,
    query: str,
    target: str,
) -> dict:
    """Execute ``explain_architecture_match`` and return a safe, structured response.

    A target that was not architecturally relevant returns a *structured*
    explanation with ``is_architecturally_relevant: false`` rather than raising.
    """
    state = _require_state(architecture_factory)
    try:
        return _build_explain_response(state, query, target)
    except Exception as exc:  # noqa: BLE001
        raise ArchitectureError(
            "The architecture explanation could not be completed.",
            diagnostic=f"explain failed at runtime: {type(exc).__name__}",
        ) from exc


# ---------------------------------------------------------------------------
# Response builders
# ---------------------------------------------------------------------------


def resolve_node(state: ArchitectureState, target: str):
    """Resolve ``target`` to a file / package / module node, or ``None``."""
    graph = state.graph
    return (
        graph.get_file_node(target)
        or graph.get_package_node(target)
        or graph.get_module_node(target)
    )


def _edge_nodes(nodes, cap: int = NODE_NEIGHBOR_CAP) -> list[dict]:
    result: list[dict] = []
    for node in nodes:
        if len(result) >= cap:
            break
        result.append({"id": node.id, "kind": node.kind.value})
    return result


def _build_inspect_response(
    state: ArchitectureState,
    target: str,
    *,
    max_depth: int,
    include_dependencies: bool,
    include_dependents: bool,
    include_subsystems: bool,
) -> dict:
    graph = state.graph
    basis = state.graph_basis
    subsystem_basis = state.subsystem_basis

    node = resolve_node(state, target)
    if node is None:
        raise InvalidArgumentsError(
            f"No repository node named {target!r} was found.",
            diagnostic=f"inspecting unknown architecture target: {target!r}",
        )

    info: dict = {
        "status": "ok",
        "target": target,
        "kind": node.kind.value,
        "id": node.id,
    }

    package: str | None = None
    module: str | None = None
    file_path: str | None = None
    if node.kind is ArchitectureNodeKind.PACKAGE:
        package = node.id
        file_path = basis["package_rep"].get(node.id)
    elif node.kind is ArchitectureNodeKind.MODULE:
        module = node.id
        file_path = basis["file_by_module"].get(node.id)
        package = basis["module_package"].get(node.id)
    else:
        file_path = node.id
        module = basis["module_by_file"].get(node.id)
        package = basis["package_by_file"].get(node.id)

    info["package"] = package
    info["containing_package"] = package
    info["module"] = module
    info["file"] = file_path

    subsystem_id = None
    if package is not None:
        subsystem_id = subsystem_basis["package"].get(package)
    if subsystem_id is None and node.kind is ArchitectureNodeKind.MODULE:
        subsystem_id = subsystem_basis["module"].get(node.id)
    info["subsystem"] = subsystem_id

    if include_dependencies:
        info["dependencies"] = _edge_nodes(graph.dependencies_of(node))
    if include_dependents:
        info["dependents"] = _edge_nodes(graph.dependents_of(node))

    neighborhood: dict = {"max_depth": max_depth}
    if include_dependencies:
        neighborhood["dependencies"] = [
            n.id for n in graph.transitive_dependencies(node, max_depth=max_depth)
        ][:NODE_NEIGHBOR_CAP]
    if include_dependents:
        neighborhood["dependents"] = [
            n.id for n in graph.transitive_dependents(node, max_depth=max_depth)
        ][:NODE_NEIGHBOR_CAP]
    info["neighborhood"] = neighborhood

    statistics: dict = {
        "direct_dependency_count": len(graph.dependencies_of(node)),
        "direct_dependent_count": len(graph.dependents_of(node)),
    }
    if include_dependencies:
        statistics["transitive_dependency_count"] = len(
            graph.transitive_dependencies(node, max_depth=max_depth)
        )
    if include_dependents:
        statistics["transitive_dependent_count"] = len(
            graph.transitive_dependents(node, max_depth=max_depth)
        )
    target_package = package if node.kind is not ArchitectureNodeKind.PACKAGE else node.id
    if target_package is not None:
        statistics["package_file_count"] = len(graph.files_in_package(target_package))
        statistics["package_module_count"] = len(graph.modules_in_package(target_package))
    if include_subsystems and subsystem_id is not None:
        subsystem = subsystem_basis["by_id"].get(subsystem_id)
        if subsystem is not None:
            statistics["subsystem_stats"] = subsystem.stats()
    info["statistics"] = statistics
    return info


def _build_subsystems_response(
    state: ArchitectureState,
    *,
    max_subsystems: int | None,
    include_stats: bool,
    include_dependencies: bool,
    include_dependents: bool,
) -> dict:
    subsystems = state.subsystems
    if max_subsystems is not None:
        subsystems = subsystems[:max_subsystems]

    entries: list[dict] = []
    for subsystem in subsystems:
        entry: dict = {
            "id": subsystem.id,
            "display_name": subsystem.display_name,
            "packages": list(subsystem.packages),
            "modules": list(subsystem.modules),
            "files": list(subsystem.files),
            "entry_modules": list(subsystem.entry_modules),
        }
        if include_dependencies:
            entry["dependencies"] = list(subsystem.dependencies)
        if include_dependents:
            entry["dependents"] = list(subsystem.dependents)
        if include_stats:
            entry["statistics"] = subsystem.stats()
        entries.append(entry)

    return {
        "status": "ok",
        "subsystem_count": len(entries),
        "max_subsystems": max_subsystems,
        "subsystems": entries,
    }


def _build_candidates_response(
    state: ArchitectureState,
    query: str,
    *,
    limit: int,
    max_depth: int,
    architecture: ArchitectureRetrievalConfig | None,
) -> dict:
    config = architecture if architecture is not None else ArchitectureRetrievalConfig()
    config = ArchitectureRetrievalConfig(
        enabled=config.enabled,
        max_package_candidates=config.max_package_candidates,
        max_module_candidates=config.max_module_candidates,
        max_neighbor_depth=max_depth,
        max_expanded_nodes=config.max_expanded_nodes,
    )

    signals = extract_architecture_signals(query, state.graph)
    candidates = architecture_candidates(
        query,
        state.graph,
        subsystems=state.subsystems,
        config=config,
    )[:limit]

    basis = state.graph_basis
    entries: list[dict] = []
    for candidate in candidates:
        node = candidate.node
        module_name = None
        if node.kind is ArchitectureNodeKind.MODULE:
            module_name = node.id
        elif node.kind is ArchitectureNodeKind.FILE:
            module_name = basis["module_by_file"].get(node.id)
        entries.append(
            {
                "file": candidate.path,
                "module": module_name,
                "package": candidate.package,
                "score": round(1 / (candidate.rank + 1), 3),
                "architecture_score": candidate.rank,
                "inclusion_reason": candidate.reason,
                "architecture_node_type": node.kind.value,
                "architecture_node_id": node.id,
                "subsystem": candidate.subsystem,
                "dependency_direction": candidate.direction.value,
            }
        )

    return {
        "status": "ok",
        "query": query,
        "max_depth": max_depth,
        "max_neighbor_depth": max_depth,
        "limit": limit,
        "signal_count": len(signals),
        "signals": [
            {
                "kind": s.kind,
                "value": s.value,
                "node_kind": s.node.kind.value,
                "node_id": s.node.id,
                "reason": s.reason,
            }
            for s in signals
        ],
        "candidate_count": len(entries),
        "candidates": entries,
    }


def _build_explain_response(
    state: ArchitectureState,
    query: str,
    target: str,
) -> dict:
    from repolens.architecture_retrieval import explain_architecture_match as _explain

    graph = state.graph
    result = _explain(
        query,
        graph,
        target,
        subsystems=state.subsystems,
    )

    signals = extract_architecture_signals(query, graph)

    if result is None:
        return {
            "status": "ok",
            "query": query,
            "target": target,
            "is_architecturally_relevant": False,
            "explanation": (
                "The target was not selected as an architecture candidate for "
                "this query. It may still be relevant to lexical retrieval; "
                "architecture explanations cover only graph-structural matches."
            ),
            "signal_count": len(signals),
            "signals": [
                {
                    "kind": s.kind,
                    "value": s.value,
                    "node_kind": s.node.kind.value,
                    "node_id": s.node.id,
                    "reason": s.reason,
                }
                for s in signals
            ],
        }

    candidates = architecture_candidates(
        query,
        graph,
        subsystems=state.subsystems,
    )
    matched_nodes = [
        {
            "id": c.node.id,
            "kind": c.node.kind.value,
            "reason": c.reason,
            "rank": c.rank,
            "direction": c.direction.value,
        }
        for c in candidates
    ]

    subsystem_relationship = _subsystem_relationship(
        state, result["subsystem"], target
    )

    dependency_paths = _dependency_paths(
        state, query, candidates, result["node"]["id"], max_depth=6
    )

    return {
        "status": "ok",
        "query": query,
        "target": target,
        "is_architecturally_relevant": True,
        "node": {
            "id": result["node"]["id"],
            "kind": result["node"]["kind"],
        },
        "path": result["path"],
        "inclusion_reason": result["reason"],
        "architecture_score": result["rank"],
        "direction": result["direction"],
        "package": result["package"],
        "subsystem": result["subsystem"],
        "subsystem_relationship": subsystem_relationship,
        "signal_count": len(signals),
        "signals": [
            {
                "kind": s.kind,
                "value": s.value,
                "node_kind": s.node.kind.value,
                "node_id": s.node.id,
                "reason": s.reason,
            }
            for s in signals
        ],
        "matched_node_count": len(matched_nodes),
        "matched_nodes": matched_nodes,
        "dependency_path_count": len(dependency_paths),
        "dependency_paths": dependency_paths,
        "affects_context_ranking": True,
    }


def _subsystem_relationship(state: ArchitectureState, subsystem_id, target: str) -> dict:
    if subsystem_id is None:
        return {"target_subsystem": None, "note": "target has no subsystem"}
    return {
        "target_subsystem": subsystem_id,
        "display_name": subsystem_id,
    }


def _dependency_paths(state: ArchitectureState, query, candidates, target_id: str, *, max_depth: int) -> list[list[dict]]:
    """Return bounded, deterministic dependency chain(s) connecting matched
    modules to ``target_id`` (always including the trivial direct path)."""
    graph = state.graph
    start_ids = [c.node.id for c in candidates if c.rank == 0]
    if target_id in start_ids:
        return [[{"id": target_id, "kind": "matched"}]]
    path = _find_dependency_chain(graph, start_ids, target_id, max_depth=max_depth)
    if path is None:
        return []
    return [[{"id": nid, "kind": "module"} for nid in path]]


def _find_dependency_chain(graph, start_ids, target_id, *, max_depth: int) -> list[str] | None:
    """Bounded BFS over module dependency edges from ``start_ids`` to
    ``target_id`` (parents tracked); returns the module-id chain or ``None``."""
    if target_id in start_ids:
        return [target_id]
    parents: dict[str, str | None] = {sid: None for sid in start_ids}
    frontier = list(start_ids)
    depth = 0
    while frontier and depth < max_depth:
        depth += 1
        next_frontier: list[str] = []
        for node_id in frontier:
            node = graph.get_module_node(node_id) or graph.get_package_node(node_id)
            if node is None:
                continue
            neighbors = graph.dependencies_of(node)
            for neighbor in neighbors:
                key = neighbor.id
                if key in parents:
                    continue
                parents[key] = node_id
                if key == target_id:
                    chain = [key]
                    cursor = node_id
                    while cursor is not None:
                        chain.append(cursor)
                        cursor = parents[cursor]
                    return list(reversed(chain))
                next_frontier.append(key)
        frontier = next_frontier
    return None