"""Dependency-aware context expansion over the repository dependency graph.

Given the set of *primary* (directly retrieved) files, expansion walks the
existing repository dependency graph to discover related files that an agent
would need to understand the task. It supports:

- forward edges: files a candidate imports (its dependencies);
- reverse edges: files that import a candidate (its dependents / reverse
  dependencies).

Traversal is breadth-first up to a configurable depth, never revisits a file,
and produces a deterministic result. Only relationships the graph actually
contains are used; no new relationships are invented.

Each expanded file is classified by the direction of the edge along which it
was first reached:

- ``DEPENDENCY`` — reached by following a candidate's imports;
- ``DEPENDENT`` — reached because it imports a candidate.

Because a file may be reachable both as a dependency and as a dependent, the
role recorded is the one from the shortest path; ties are resolved
deterministically by the BFS visit order.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path

from repolens.context.candidate import CandidateRole
from repolens.context.config import DependencyExpansionConfig
from repolens.graph import DependencyGraph


@dataclass(frozen=True)
class ExpandedNode:
    """A graph-reachable file discovered during expansion."""

    path: Path
    role: CandidateRole
    distance: int


def expand_dependencies(
    graph: DependencyGraph,
    seeds: list[Path],
    config: DependencyExpansionConfig,
) -> list[ExpandedNode]:
    """Expand from ``seeds`` over ``graph`` up to ``config.depth``.

    ``seeds`` is the list of primary (retrieved) file paths. Returns the
    non-primary files reached, each with its role and graph distance, ordered
    deterministically by (distance, role, path). Breadth-first traversal
    prevents cycles and duplicate files.
    """
    depth = config.depth
    if depth < 0:
        raise ValueError(f"dependency depth must be >= 0, got {depth}")
    if depth == 0:
        return []

    max_expanded = getattr(config, "max_expanded", None)
    if max_expanded is not None and max_expanded < 0:
        raise ValueError(f"max_expanded must be >= 0, got {max_expanded}")

    seed_set = set(seeds)
    result: dict[Path, ExpandedNode] = {}
    # A cheap order of discovery; used to enforce max_expanded deterministically.
    discovered: list[Path] = []

    # (path, role, distance) frontier. Each seed is visited at distance 0.
    frontier: deque[tuple[Path, CandidateRole, int]] = deque(
        (seed, CandidateRole.PRIMARY, 0) for seed in seeds
    )
    visited: set[Path] = set()

    while frontier:
        path, _role, distance = frontier.popleft()
        if path in visited:
            continue
        visited.add(path)

        if distance >= depth:
            continue

        # Forward: dependencies of `path`.
        if config.include_dependencies:
            for neighbor in graph.get_dependencies(path):
                if neighbor in visited or neighbor in seed_set:
                    continue
                _record_and_enqueue(
                    result, frontier, neighbor,
                    role=CandidateRole.DEPENDENCY, distance=distance + 1,
                    discovered=discovered, max_expanded=max_expanded,
                )
        # Reverse: dependents of `path` (files that import it).
        if config.include_dependents:
            for neighbor in graph.get_dependents(path):
                if neighbor in visited or neighbor in seed_set:
                    continue
                _record_and_enqueue(
                    result, frontier, neighbor,
                    role=CandidateRole.DEPENDENT, distance=distance + 1,
                    discovered=discovered, max_expanded=max_expanded,
                )

    return _ordered(result)


def _record_and_enqueue(
    result: dict[Path, ExpandedNode],
    frontier: deque[tuple[Path, CandidateRole, int]],
    path: Path,
    *,
    role: CandidateRole,
    distance: int,
    discovered: list[Path],
    max_expanded: int | None,
) -> None:
    if path in result:
        return
    if max_expanded is not None and len(discovered) >= max_expanded:
        return
    node = ExpandedNode(path=path, role=role, distance=distance)
    result[path] = node
    discovered.append(path)
    frontier.append((path, role, distance))


def _ordered(nodes: dict[Path, ExpandedNode]) -> list[ExpandedNode]:
    """Return nodes sorted deterministically by (distance, role, path)."""
    role_order = {
        CandidateRole.DEPENDENT: 0,
        CandidateRole.DEPENDENCY: 1,
        CandidateRole.PRIMARY: 2,
    }
    return sorted(
        nodes.values(),
        key=lambda node: (
            node.distance,
            role_order[node.role],
            node.path.as_posix(),
        ),
    )
