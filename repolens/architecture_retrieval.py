"""Architecture-aware context retrieval for the context engine (M23.2).

This module makes RepoLens retrieve *architecture*: when a developer query
names a package, module, file, subsystem, or architectural role (``api``,
``repository``, ``checkout``, ...), retrieval is enriched with the surrounding
structure — the matched nodes plus their bounded neighbourhood of internal
dependencies, dependents, sibling modules, and same-subsystem files.

Pieces:

- :class:`ArchitectureRetrievalConfig` — the boundedness limits.
- :class:`ArchitectureSignal` — one *signal* that a query points at an
  architecture node: a dotted module name, a package path, a file path, a
  term matching a node's leaf component, or a symbol whose file resolves to a
  module.
- :func:`extract_architecture_signals` — deterministic, generic signal
  extraction (no LLM, no hard-coded project names).
- :class:`ArchitectureCandidate` — a concrete file-addressable candidate
  produced by expanding signals across the architecture graph.
- :func:`architecture_candidates` — bounded signal expansion; never walks the
  whole connected component.
- :func:`explain_architecture_match` — human/agent-readable explanation of one
  match.

Architecture-aware retrieval is *optional* and additive: callers opt in by
passing an :class:`ArchitectureRetrievalConfig` to the engine, and all limits
here keep both runtime and result size deterministic. When architecture yields
no signal, the historical (M22) retrieval path is unchanged.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from repolens.architecture import (
    ArchitectureGraph,
    ArchitectureNode,
    ArchitectureNodeKind,
)
from repolens.context.intent import extract_symbol_tokens
from repolens.subsystems import (
    Subsystem,
    _deepest_package,
    _module_file_map,
    _package_of_file,
    discover_subsystems,
)

#: The repository root package id has no meaningful leaf component.
ROOT_SUBSYSTEM_LEAF = "."

#: Upper bound on the number of architecture signals extracted per query.
MAX_SIGNALS = 24

#: Upper bound on signals contributed by one plain word token.
MAX_TOKEN_SIGNALS = 4

#: Deterministic reason strings (stable tokens, prefixed for machine parsers).
REASON_DIRECT_FILE = "architecture: direct file match"
REASON_DIRECT_MODULE = "architecture: direct module match"
REASON_DIRECT_PACKAGE = "architecture: direct package match"
REASON_DIRECT_TERM = "architecture: architectural term match"
REASON_SYMBOL = "architecture: module containing matched symbol"
REASON_DEPENDENCY = "architecture: dependency of matched module"
REASON_DEPENDENT = "architecture: dependent module"
REASON_DEEPER_DEPENDENCY = "architecture: deeper dependency"
REASON_DEEPER_DEPENDENT = "architecture: deeper dependent"
REASON_NEIGHBOR = "architecture: neighboring module"
REASON_IN_PACKAGE = "architecture: module in matched package"
REASON_DEP_PACKAGE = "architecture: dependency package"
REASON_DEPENDENT_PACKAGE = "architecture: dependent package"
REASON_SUBSYSTEM = "architecture: same subsystem"


class ArchitectureDirection(str, Enum):
    """Why a candidate sits relative to the matched architecture node(s)."""

    MATCHED = "matched"
    DEPENDENCY = "dependency"
    DEPENDENT = "dependent"
    NEIGHBOR = "neighbor"
    SUBSYSTEM = "subsystem"


#: Dotted identifier run: ``app.api.routes``, ``checkout_controller``, ...
_DOTTED_RUN = re.compile(r"[A-Za-z_][A-Za-z0-9_.]*")

#: Path-ish run containing at least one slash, optionally ending in ``.py``.
_PATH_RUN = re.compile(r"[A-Za-z0-9_.]+(?:/[A-Za-z0-9_.]+)+(?:\.py)?")


@dataclass(frozen=True)
class ArchitectureRetrievalConfig:
    """Boundedness limits for architecture-aware retrieval.

    All values are caps on candidate counts or traversal depth; they make every
    architecture query finish in predictable time and never explode the result
    set. ``enabled=False`` behaves exactly like the historical retrieval path.
    """

    enabled: bool = True
    max_package_candidates: int = 6
    max_module_candidates: int = 24
    max_neighbor_depth: int = 1
    max_expanded_nodes: int = 50


@dataclass(frozen=True)
class ArchitectureSignal:
    """One indication that ``query`` targets architecture node ``node``."""

    kind: str  # "file" | "module" | "package" | "symbol" | "term"
    value: str
    node: ArchitectureNode
    reason: str

    def to_dict(self) -> dict[str, str]:
        return {
            "kind": self.kind,
            "value": self.value,
            "node": self.node.to_dict(),
            "reason": self.reason,
        }


@dataclass(frozen=True)
class ArchitectureCandidate:
    """A file-addressable candidate selected from architecture structure."""

    path: str
    node: ArchitectureNode
    reason: str
    rank: int
    direction: ArchitectureDirection
    package: str | None
    subsystem: str | None

    def to_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "node": self.node.to_dict(),
            "reason": self.reason,
            "rank": self.rank,
            "direction": self.direction.value,
            "package": self.package,
            "subsystem": self.subsystem,
        }


def _norm(name: str) -> str:
    """Case- and trailing-``s``-tolerant comparison key (applied to both sides)."""
    key = name.lower()
    if len(key) > 2 and key.endswith("s"):
        key = key[:-1]
    return key


def extract_architecture_signals(
    query: str,
    graph: ArchitectureGraph,
    *,
    symbol_matches=(),
) -> list[ArchitectureSignal]:
    """Return the deterministic architecture signals for ``query``.

    Signals come from, in order: explicit dotted/file-path mentions, plain word
    tokens matching node leaf components, and ``symbol_matches`` whose defining
    file resolves to an architecture module. Results are sorted by
    (kind, value) and capped at :data:`MAX_SIGNALS`. The extraction is generic:
    it never consults a project-specific name list.
    """
    signals: list[ArchitectureSignal] = []
    seen: set[tuple[str, str]] = set()

    def add(kind: str, value: str, node: ArchitectureNode, reason: str) -> None:
        key = (kind, value)
        if key in seen or len(signals) >= MAX_SIGNALS:
            return
        seen.add(key)
        signals.append(ArchitectureSignal(kind, value, node, reason))

    # 1. Explicit dotted / path references (exact node lookups). Only runs that
    # contain a separator are treated as exact mentions; plain words go through
    # leaf-component matching in step 2.
    dotted = _DOTTED_RUN.findall(query)
    paths = _PATH_RUN.findall(query)
    for run in dotted + paths:
        if "." not in run and "/" not in run:
            continue
        _emit_path_signal(run, graph, add)

    # 2. Plain word tokens matching node leaf components. Components of an
    # already-resolved dotted/path mention are consumed so they cannot produce
    # redundant broader signals (e.g. ``store`` inside ``store/repositories``).
    separated = [r for r in dotted if "." in r or "/" in r] + list(paths)
    excluded = _consumed_words(separated)
    for token in extract_symbol_tokens(query):
        if token in excluded:
            continue
        _emit_token_signals(token, graph, add)

    # 3. Symbol matches -> their defining module.
    ordered_matches = sorted(
        symbol_matches, key=lambda m: (m.symbol.file_path.as_posix(), m.symbol.name)
    )
    module_by_file = {v: k for k, v in _module_file_map(graph).items()}
    for match in ordered_matches:
        module_name = module_by_file.get(match.symbol.file_path.as_posix())
        if module_name is None:
            continue
        module_node = graph.get_module_node(module_name)
        if module_node is not None:
            add("symbol", module_name, module_node, REASON_SYMBOL)

    signals.sort(key=lambda s: (s.kind, s.value))
    return signals


def _consumed_words(runs: list[str]) -> set[str]:
    """Word tokens already covered by exact dotted-path signal extraction.

    Only runs that *contain* a separator (``.`` or ``/``) are real dotted/path
    mentions; plain words like ``checkout`` are not consumed here and are still
    matched against node leaf components.
    """
    words: set[str] = set()
    for run in runs:
        if "." not in run and "/" not in run:
            continue
        words.update(re.split(r"[._/]+", run))
    return {w.lower() for w in words if w}


def _emit_path_signal(run: str, graph: ArchitectureGraph, add) -> None:
    """Emit a signal for an exact dotted module / package / file mention."""
    stripped = run.rstrip(".")
    if not stripped:
        return
    for candidate in (stripped, stripped.lower()):
        # file (exact, or with `.py` when the query omitted it)
        file_node = graph.get_file_node(candidate) or graph.get_file_node(
            candidate + ".py"
        )
        if file_node is not None:
            add("file", file_node.id, file_node, REASON_DIRECT_FILE)
            return
        # package (dotted path or slash path)
        package_node = (
            graph.get_package_node(candidate)
            or graph.get_package_node(candidate.replace(".", "/"))
        )
        if package_node is not None:
            add("package", package_node.id, package_node, REASON_DIRECT_PACKAGE)
            return
        # module (dotted name)
        module_node = graph.get_module_node(candidate)
        if module_node is not None:
            add("module", module_node.id, module_node, REASON_DIRECT_MODULE)
            return


def _emit_token_signals(token: str, graph: ArchitectureGraph, add) -> None:
    """Emit signals for a word token matching node leaf components."""
    if len(token) < 2:
        return
    key = _norm(token)
    contributed = 0
    for node, leaf in _leaf_iter(graph):
        if _norm(leaf) == key:
            add(node.kind.value, node.id, node, _direct_reason(node.kind))
            contributed += 1
            if contributed >= MAX_TOKEN_SIGNALS:
                return


def _direct_reason(kind: ArchitectureNodeKind) -> str:
    return {
        ArchitectureNodeKind.FILE: REASON_DIRECT_FILE,
        ArchitectureNodeKind.MODULE: REASON_DIRECT_MODULE,
        ArchitectureNodeKind.PACKAGE: REASON_DIRECT_PACKAGE,
    }[kind]


def _leaf_iter(graph: ArchitectureGraph):
    """Yield (node, leaf-component) pairs deterministically.

    Leaf order is modules, packages, then files — each sorted by id — so token
    matching prefers the module interpretation and stays deterministic.
    """
    for module in graph.modules():
        yield module, module.id.rsplit(".", 1)[-1]
    for package in graph.packages():
        if package.id == ROOT_SUBSYSTEM_LEAF:
            continue
        yield package, package.id.rsplit("/", 1)[-1]
    for file_node in graph.files():
        yield file_node, Path(file_node.id).stem


def _index_graph(graph: ArchitectureGraph) -> dict:
    """Precompute cheap lookup maps for one retrieval call."""
    file_by_module = _module_file_map(graph)
    module_by_file = {v: k for k, v in file_by_module.items()}
    package_by_file: dict[str, str] = {}
    for file_node in graph.files():
        package_by_file[file_node.id] = _package_of_file(graph, file_node.id)
    package_rep: dict[str, str] = {}
    for file_path, package in sorted(package_by_file.items()):
        if file_path.endswith("/__init__.py"):
            package_rep.setdefault(package, file_path)
    for file_path, package in sorted(package_by_file.items()):
        package_rep.setdefault(package, file_path)
    module_package: dict[str, str] = {}
    for module in graph.modules():
        module_package[module.id] = _deepest_package(
            graph, module.id.replace(".", "/")
        )
    return {
        "module_by_file": module_by_file,
        "file_by_module": file_by_module,
        "package_by_file": package_by_file,
        "package_rep": package_rep,
        "module_package": module_package,
    }


def _index_subsystems(subsystems: list[Subsystem]) -> dict:
    module_subsystem: dict[str, str] = {}
    package_subsystem: dict[str, str] = {}
    by_id = {s.id: s for s in subsystems}
    for subsystem in subsystems:
        for module in subsystem.modules:
            module_subsystem[module] = subsystem.id
        for package in subsystem.packages:
            package_subsystem[package] = subsystem.id
    return {
        "by_id": by_id,
        "module": module_subsystem,
        "package": package_subsystem,
    }


def architecture_candidates(
    query: str,
    graph: ArchitectureGraph,
    *,
    symbol_matches=(),
    subsystems: list[Subsystem] | None = None,
    config: ArchitectureRetrievalConfig | None = None,
) -> list[ArchitectureCandidate]:
    """Return the bounded, deterministic architecture candidates for ``query``.

    Pipeline: signals → direct matches (rank 0) → matched package contents,
    neighbours, dependencies/dependents (rank 1) → deeper traversal and
    same-subsystem context (rank 2). Every loop is capped by ``config`` limits,
    so a pathological repository can never produce a runaway expansion.
    Candidates are deduplicated by node and ordered by (rank, direction, node).
    When architecture is disabled or yields no signal, an empty list is
    returned so callers can fall back to the historical path unchanged.
    """
    cfg = config if config is not None else ArchitectureRetrievalConfig()
    if not cfg.enabled:
        return []
    signals = extract_architecture_signals(query, graph, symbol_matches=symbol_matches)
    if not signals:
        return []

    discovered = subsystems if subsystems is not None else discover_subsystems(graph)
    basis = _index_graph(graph)
    sub = _index_subsystems(discovered)

    candidates: dict[tuple, ArchitectureCandidate] = {}
    matched_modules: set[str] = set()
    matched_packages: set[str] = set()
    expanded = 0

    def add_node(
        node: ArchitectureNode,
        reason: str,
        rank: int,
        direction: ArchitectureDirection,
        *,
        count_expansion: bool = False,
    ) -> None:
        nonlocal expanded
        if count_expansion:
            if expanded >= cfg.max_expanded_nodes:
                return
            expanded += 1
        path = _candidate_path(node, basis)
        if path is None:
            return
        package: str | None = None
        if node.kind is ArchitectureNodeKind.PACKAGE:
            package = node.id
        elif node.kind is ArchitectureNodeKind.MODULE:
            package = basis["module_package"].get(node.id)
        else:
            package = basis["package_by_file"].get(node.id)
        subsystem_id = None
        if package is not None:
            subsystem_id = sub["package"].get(package)
        if subsystem_id is None and node.kind is ArchitectureNodeKind.MODULE:
            subsystem_id = sub["module"].get(node.id)
        key = (node.kind.value, node.id)
        existing = candidates.get(key)
        if existing is not None and existing.rank <= rank:
            return
        candidates[key] = ArchitectureCandidate(
            path=path,
            node=node,
            reason=reason,
            rank=rank,
            direction=direction,
            package=package,
            subsystem=subsystem_id,
        )

    # Direct matches from signals.
    for signal in signals:
        node = signal.node
        if node.kind is ArchitectureNodeKind.PACKAGE:
            matched_packages.add(node.id)
        elif node.kind is ArchitectureNodeKind.MODULE:
            matched_modules.add(node.id)
        add_node(node, signal.reason, 0, ArchitectureDirection.MATCHED)

    # Rank-1 / rank-2 expansion, bounded throughout.
    for module_id in sorted(matched_modules)[: cfg.max_module_candidates]:
        _expand_module(graph, module_id, basis, cfg, add_node)
        if expanded >= cfg.max_expanded_nodes:
            break
    for package_id in sorted(matched_packages)[: cfg.max_package_candidates]:
        _expand_package(graph, package_id, cfg, add_node)

    _expand_subsystems(graph, matched_modules, matched_packages, sub, cfg, add_node)

    result = sorted(
        candidates.values(),
        key=lambda c: (c.rank, c.direction.value, c.node.kind.value, c.node.id),
    )
    packages = [c for c in result if c.node.kind is ArchitectureNodeKind.PACKAGE]
    modules = [c for c in result if c.node.kind is ArchitectureNodeKind.MODULE]
    files = [c for c in result if c.node.kind is ArchitectureNodeKind.FILE]
    capped = (
        packages[: cfg.max_package_candidates]
        + modules[: cfg.max_module_candidates]
        + files[: cfg.max_module_candidates]
    )
    return sorted(capped, key=lambda c: (c.rank, c.direction.value, c.node.kind.value, c.node.id))


def _candidate_path(node: ArchitectureNode, basis: dict) -> str | None:
    if node.kind is ArchitectureNodeKind.FILE:
        return node.id
    if node.kind is ArchitectureNodeKind.MODULE:
        return basis["file_by_module"].get(node.id)
    return basis["package_rep"].get(node.id)


def _expand_module(graph, module_id, basis, cfg, add_node) -> None:
    """Expand a matched module: neighbours, deps, dependents, deeper levels."""
    package = basis["module_package"].get(module_id)
    if package is not None:
        for other in graph.modules_in_package(package):
            if other.id == module_id:
                continue
            add_node(
                other,
                REASON_NEIGHBOR,
                1,
                ArchitectureDirection.NEIGHBOR,
                count_expansion=True,
            )
    for dep in graph.dependencies_of(module_id):
        add_node(
            dep,
            REASON_DEPENDENCY,
            1,
            ArchitectureDirection.DEPENDENCY,
            count_expansion=True,
        )
    for depender in graph.dependents_of(module_id):
        add_node(
            depender,
            REASON_DEPENDENT,
            1,
            ArchitectureDirection.DEPENDENT,
            count_expansion=True,
        )
    if cfg.max_neighbor_depth >= 1:
        for deeper in graph.transitive_dependencies(
            module_id, max_depth=cfg.max_neighbor_depth
        ):
            if deeper.id == module_id:
                continue
            add_node(
                deeper,
                REASON_DEEPER_DEPENDENCY,
                2,
                ArchitectureDirection.DEPENDENCY,
                count_expansion=True,
            )
        for deeper in graph.transitive_dependents(
            module_id, max_depth=cfg.max_neighbor_depth
        ):
            if deeper.id == module_id:
                continue
            add_node(
                deeper,
                REASON_DEEPER_DEPENDENT,
                2,
                ArchitectureDirection.DEPENDENT,
                count_expansion=True,
            )


def _expand_package(graph, package_id, cfg, add_node) -> None:
    """Expand a matched package: its modules, dependency packages, dependents."""
    for module_node in graph.modules_in_package(package_id):
        add_node(
            module_node,
            REASON_IN_PACKAGE,
            1,
            ArchitectureDirection.NEIGHBOR,
            count_expansion=True,
        )
    for dep in graph.package_dependencies(package_id):
        if dep.kind is not ArchitectureNodeKind.PACKAGE:
            continue
        add_node(
            dep,
            REASON_DEP_PACKAGE,
            1,
            ArchitectureDirection.DEPENDENCY,
            count_expansion=True,
        )
    for depender in graph.dependents_of(package_id):
        if depender.kind is not ArchitectureNodeKind.PACKAGE:
            continue
        add_node(
            depender,
            REASON_DEPENDENT_PACKAGE,
            1,
            ArchitectureDirection.DEPENDENT,
            count_expansion=True,
        )


def _expand_subsystems(graph, matched_modules, matched_packages, sub, cfg, add_node) -> None:
    """Add same-subsystem context modules (rank 2, proximity tier)."""
    wanted: set[str] = set()
    for module in matched_modules:
        sid = sub["module"].get(module)
        if sid is not None:
            wanted.add(sid)
    for package in matched_packages:
        sid = sub["package"].get(package)
        if sid is not None:
            wanted.add(sid)
    expanded = 0
    for sid in sorted(wanted):
        subsystem = sub["by_id"].get(sid)
        if subsystem is None:
            continue
        for module in subsystem.modules:
            if module in matched_modules:
                continue
            node = graph.get_module_node(module)
            if node is None:
                continue
            add_node(
                node,
                REASON_SUBSYSTEM,
                2,
                ArchitectureDirection.SUBSYSTEM,
                count_expansion=True,
            )
            expanded += 1
            if expanded >= cfg.max_expanded_nodes:
                return


def explain_architecture_match(
    query: str,
    graph: ArchitectureGraph,
    node_id,
    *,
    symbol_matches=(),
    subsystems: list[Subsystem] | None = None,
    config: ArchitectureRetrievalConfig | None = None,
) -> dict | None:
    """Explain why ``query`` selected ``node_id`` as an architecture match.

    Returns a deterministic dict (reason, rank, direction, package, subsystem,
    path) when the node is among the architecture candidates for ``query``, or
    ``None`` otherwise. ``node_id`` may be a file path, a package path, a
    dotted module name, or a stored :class:`ArchitectureNode`.
    """
    candidates = architecture_candidates(
        query,
        graph,
        symbol_matches=symbol_matches,
        subsystems=subsystems,
        config=config,
    )
    if isinstance(node_id, ArchitectureNode):
        target = (node_id.kind, node_id.id)
        path = None
    else:
        resolved = (
            graph.get_file_node(node_id)
            or graph.get_package_node(node_id)
            or graph.get_module_node(node_id)
        )
        target = (resolved.kind, resolved.id) if resolved is not None else None
        path = str(Path(node_id).as_posix()) if resolved is None else None

    for candidate in candidates:
        if target is not None and (candidate.node.kind, candidate.node.id) == target:
            return _explanation(candidate, query)
        if path is not None and candidate.path == path:
            return _explanation(candidate, query)
    return None


def _explanation(candidate: ArchitectureCandidate, query: str) -> dict:
    return {
        "query": query,
        "node": candidate.node.to_dict(),
        "path": candidate.path,
        "reason": candidate.reason,
        "rank": candidate.rank,
        "direction": candidate.direction.value,
        "package": candidate.package,
        "subsystem": candidate.subsystem,
    }