"""Deterministic change-surface reasoning (Milestone 23, step 1).

A *change surface* is the structured set of repository objects that a change
to one or more files, modules, or symbols touches — directly or transitively —
in both directions:

- *incoming* — files that depend on, use, call, or subclass the target
  (changing the target affects them);
- *outgoing* — files the target depends on, uses, calls, or subclasses
  (changes to those affect the target).

This module builds that surface exclusively on the existing RepoLens
infrastructure — it is not a second analysis model:

- :class:`~repolens.graph.DependencyGraph` for file-level import edges
  (traversed here with an explicit, bounded, cycle-safe BFS);
- the Milestone 22 :class:`~repolens.call_graph.CallGraph` for statically
  resolved call edges (``bounded_transitive_callers``/``callees``);
- :class:`~repolens.impact.ImpactAnalyzer` for conservative symbol-level
  evidence (symbol imports, base-class relationships, tests, configuration);
- :class:`~repolens.index.SymbolIndex` and ``ModuleAnalysis`` for symbol
  resolution and base-class lookup.

Everything runs fully offline and deterministically: the same repository
snapshot and targets always produce the same :class:`ChangeSurfaceResult`,
with stable ordering, explicit traversal depth, and cycle-safety.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from repolens.call_graph import CallGraph, CallGraphBuilder
from repolens.impact import (
    EVIDENCE_BASE_CLASS,
    ImpactAnalyzer,
    ImpactTarget,
    Relationship,
    TargetKind,
)
from repolens.parser import ModuleAnalysis

#: Impact relationships mapped to the :data:`ChangeRelationship.OTHER` kind.
_OTHER_IMPACT_RELATIONSHIPS = frozenset(
    {Relationship.TEST, Relationship.CONFIGURATION}
)


# ---------------------------------------------------------------------------
# Relationship taxonomy
# ---------------------------------------------------------------------------


class ChangeRelationship(str, Enum):
    """How one surface item relates to a change target.

    The five kinds are deliberately coarse so the surface stays small and
    consumable; evidence and reasons carry the detail.
    """

    IMPORT = "import"
    CALL = "call"
    SYMBOL_REFERENCE = "symbol_reference"
    INHERITANCE = "inheritance"
    OTHER = "other"


class SurfaceDirection(str, Enum):
    """The direction of a surface edge relative to the change target."""

    INCOMING = "incoming"
    OUTGOING = "outgoing"


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChangeSurfaceConfig:
    """Explicit, deterministic bounds for surface analysis.

    Every traversal is capped by ``max_depth`` (edge count from the target)
    and ``max_nodes`` (a hard budget on visited files), so a pathological or
    enormous dependency graph cannot be walked past a fixed cost.
    """

    max_depth: int = 2
    max_nodes: int = 2000
    include_calls: bool = True
    include_symbol_refs: bool = True
    include_inheritance: bool = True
    include_tests: bool = True
    include_config: bool = True


# ---------------------------------------------------------------------------
# Result model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SurfaceTarget:
    """One resolved, canonical change target of a surface analysis."""

    display: str
    kind: str
    file_path: Path
    module: str | None = None
    symbol: str | None = None

    def to_dict(self) -> dict:
        return {
            "display": self.display,
            "kind": self.kind,
            "file_path": self.file_path.as_posix(),
            "module": self.module,
            "symbol": self.symbol,
        }


@dataclass(frozen=True)
class ChangeSurfaceItem:
    """One affected file, with the relationship that puts it on the surface.

    Attributes:
        path: Repository-relative path of the affected file.
        relationship: Coarse relationship kind (see
            :class:`ChangeRelationship`).
        direction: Whether the target affects this file (:attr:`INCOMING`) or
            the target uses this file (:attr:`OUTGOING`).
        depth: Edge distance from the target (1 = direct, 2+ = transitive).
        reason: Human-readable explanation of *why* the file is listed.
        symbol: Symbol involved (caller/callee symbol, referenced symbol,
            base class), when symbol-level evidence was found.
        source: Display label of the change target this item comes from.
        module: Dotted module name of ``path``.
        evidence: Stable evidence tags describing how the relationship was
            established.
    """

    path: Path
    relationship: ChangeRelationship
    direction: SurfaceDirection
    depth: int
    reason: str
    symbol: str | None = None
    source: str | None = None
    module: str | None = None
    evidence: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return {
            "path": self.path.as_posix(),
            "relationship": self.relationship.value,
            "direction": self.direction.value,
            "depth": self.depth,
            "reason": self.reason,
            "symbol": self.symbol,
            "source": self.source,
            "module": self.module,
            "evidence": list(self.evidence),
        }


@dataclass(frozen=True)
class ChangeSurfaceResult:
    """The complete, deterministic surface of one (or more) change targets.

    ``items`` never contains the targets themselves; they are exposed as
    ``targets`` so downstream consumers can prioritize them separately.
    """

    targets: tuple[SurfaceTarget, ...]
    items: tuple[ChangeSurfaceItem, ...]
    max_depth_reached: int
    summary: dict

    @property
    def directly_affected(self) -> tuple[ChangeSurfaceItem, ...]:
        """Files touched at depth 1 (direct dependencies/callers/etc.)."""
        return tuple(item for item in self.items if item.depth == 1)

    @property
    def transitively_affected(self) -> tuple[ChangeSurfaceItem, ...]:
        """Files touched at depth >= 2 (indirect relationships)."""
        return tuple(item for item in self.items if item.depth >= 2)

    @property
    def files(self) -> tuple[Path, ...]:
        """Every distinct affected path, in first-appearance order."""
        return tuple(dict.fromkeys(item.path for item in self.items))

    def to_dict(self) -> dict:
        return {
            "targets": [target.to_dict() for target in self.targets],
            "items": [item.to_dict() for item in self.items],
            "max_depth_reached": self.max_depth_reached,
            "summary": dict(self.summary),
        }

    def to_json(self, **json_kwargs) -> str:
        import json

        return json.dumps(self.to_dict(), **json_kwargs)


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------


class ChangeSurfaceAnalyzer:
    """Compute the deterministic change surface of one or more targets.

    Reuses a single shared :class:`~repolens.impact.ImpactAnalyzer` (and, when
    calls are enabled, a shared :class:`~repolens.call_graph.CallGraph`) so
    nothing is re-parsed across repeated :meth:`analyze` calls.
    """

    def __init__(
        self,
        root: Path | str,
        *,
        analyzer: ImpactAnalyzer | None = None,
        reference_graph: CallGraph | None = None,
        config: ChangeSurfaceConfig | None = None,
    ) -> None:
        self.root = Path(root)
        self.config = config if config is not None else ChangeSurfaceConfig()
        self.analyzer = analyzer if analyzer is not None else ImpactAnalyzer(self.root)
        self.reference_graph = reference_graph
        if (
            self.reference_graph is None
            and self.config.include_calls
            and self.config.max_depth >= 1
        ):
            self.reference_graph = CallGraphBuilder(
                self.root,
                index=self.analyzer.index,
                symbol_index=self.analyzer.symbol_index,
            ).build()

    # ------------------------------------------------------------------
    # Target resolution
    # ------------------------------------------------------------------

    def resolve(self, target: str | ImpactTarget) -> SurfaceTarget:
        """Resolve ``target`` to a canonical :class:`SurfaceTarget`.

        Accepts the same forms as
        :meth:`ImpactAnalyzer.resolve_target <repolens.impact.ImpactAnalyzer.resolve_target>`
        (``path/to/file.py::symbol``, a file path, a dotted module name, a
        uniquely defined symbol, or a uniquely named module leaf) as well as
        pre-resolved :class:`~repolens.impact.ImpactTarget` instances.

        Raises :class:`~repolens.impact.ImpactTargetError` for unknown or
        ambiguous targets.
        """
        resolved = (
            target
            if isinstance(target, ImpactTarget)
            else self.analyzer.resolve_target(str(target))
        )
        return SurfaceTarget(
            display=resolved.label,
            kind=resolved.kind.value,
            file_path=resolved.file_path,
            module=resolved.module,
            symbol=resolved.symbol,
        )

    # ------------------------------------------------------------------
    # Public analysis entry point
    # ------------------------------------------------------------------

    def analyze(
        self,
        targets: str | ImpactTarget | list[object] | tuple[object, ...],
        *,
        max_depth: int | None = None,
        direction: str = "both",
        limit: int | None = None,
    ) -> ChangeSurfaceResult:
        """Compute the change surface of ``targets``.

        ``targets`` may be a single target string/:class:`ImpactTarget` or a
        sequence of them; the results are merged and de-duplicated.

        ``max_depth`` overrides the constructor's default for this call only;
        ``direction`` is ``"incoming"``, ``"outgoing"``, or ``"both"``;
        ``limit`` caps the number of returned items (after deterministic
        ordering).

        Deterministic: the same repository snapshot and targets always produce
        the same :class:`ChangeSurfaceResult`.
        """
        depth = self.config.max_depth if max_depth is None else max_depth
        if depth < 0:
            raise ValueError("max_depth must be >= 0")
        directions = _parse_direction(direction)
        if limit is not None and limit < 0:
            raise ValueError("limit must be >= 0")

        sequence = targets if isinstance(targets, (list, tuple)) else [targets]
        resolved: list[SurfaceTarget] = []
        seen: set[tuple[str, str | None]] = set()
        for target in sequence:
            surface_target = self.resolve(target)
            key = (surface_target.file_path.as_posix(), surface_target.symbol)
            if key in seen:
                continue
            seen.add(key)
            resolved.append(surface_target)

        merged: dict[tuple, _PendingItem] = {}
        for surface_target in resolved:
            for item in self._surface_for_target(surface_target, depth, directions):
                _merge_item(merged, item)

        items = sorted(
            (_finalize_item(item) for item in merged.values()),
            key=_item_sort_key,
        )
        if limit is not None:
            items = items[:limit]

        max_reached = max((item.depth for item in items), default=0)
        summary = _build_summary(resolved, items)
        return ChangeSurfaceResult(
            targets=tuple(resolved),
            items=tuple(items),
            max_depth_reached=max_reached,
            summary=summary,
        )

    # ------------------------------------------------------------------
    # Per-target traversal
    # ------------------------------------------------------------------

    def _surface_for_target(
        self,
        target: SurfaceTarget,
        depth: int,
        directions: list[SurfaceDirection],
    ) -> list[ChangeSurfaceItem]:
        items: list[ChangeSurfaceItem] = []
        incoming = SurfaceDirection.INCOMING in directions
        outgoing = SurfaceDirection.OUTGOING in directions

        if self.config.max_depth >= 1 and depth >= 1:
            if incoming:
                items.extend(
                    _import_item(path, distance, target, incoming=True)
                    for path, distance in self._reach_imports(
                        target.file_path, depth, incoming=True
                    ).items()
                )
            if outgoing:
                items.extend(
                    _import_item(path, distance, target, incoming=False)
                    for path, distance in self._reach_imports(
                        target.file_path, depth, incoming=False
                    ).items()
                )

        if target.symbol is not None:
            if (
                self.config.include_calls
                and self.reference_graph is not None
                and depth >= 1
            ):
                items.extend(self._call_items(target, depth, directions))
            if (
                self.config.include_symbol_refs
                or self.config.include_inheritance
                or self.config.include_tests
                or self.config.include_config
            ):
                items.extend(self._impact_analysis_items(target))

        if self.config.include_inheritance and outgoing:
            items.extend(self._outgoing_inheritance_items(target))
        return items

    def _reach_imports(
        self, start: Path, max_depth: int, *, incoming: bool
    ) -> dict[Path, int]:
        """Bounded, cycle-safe BFS over import edges.

        ``incoming=True`` walks who *imports* the target (reverse edges);
        ``incoming=False`` walks what the target *imports* (forward edges).
        Returns ``{relative_path: distance}`` without the start node.
        """
        if max_depth <= 0:
            return {}
        neighbor = self.analyzer.graph.get_dependents if incoming else (
            self.analyzer.graph.get_dependencies
        )
        distances: dict[Path, int] = {start: 0}
        queue: deque[Path] = deque([start])
        visited: set[Path] = set()
        while queue and len(distances) <= self.config.max_nodes:
            node = queue.popleft()
            if node in visited:
                continue
            visited.add(node)
            distance = distances[node]
            if distance >= max_depth:
                continue
            for dependent in neighbor(node):
                if dependent in distances:
                    continue
                if len(distances) >= self.config.max_nodes:
                    break
                distances[dependent] = distance + 1
                queue.append(dependent)
        distances.pop(start, None)
        return distances

    def _call_items(
        self,
        target: SurfaceTarget,
        max_depth: int,
        directions: list[SurfaceDirection],
    ) -> list[ChangeSurfaceItem]:
        """Caller/callee items for a symbol target via the static call graph.

        Depth attribution is exact: each hop band (1..max_depth) is queried
        separately and only freshly reached nodes are tagged with that depth.
        """
        node = self._call_node(target)
        if node is None:
            return []
        items: list[ChangeSurfaceItem] = []
        prior_in: set[tuple] = set()
        prior_out: set[tuple] = set()
        for depth in range(1, max_depth + 1):
            if SurfaceDirection.INCOMING in directions:
                reached = self.reference_graph.bounded_transitive_callers(
                    node, max_depth=depth
                )
                for found in sorted(reached, key=lambda n: (n.file_path.as_posix(), n.name or "")):
                    if found.key in prior_in or found.file_path == target.file_path:
                        continue
                    prior_in.add(found.key)
                    items.append(
                        ChangeSurfaceItem(
                            path=found.file_path,
                            relationship=ChangeRelationship.CALL,
                            direction=SurfaceDirection.INCOMING,
                            depth=depth,
                            reason=(
                                f"{found.name or found.file_path.as_posix()} calls "
                                f"{target.display} directly"
                                if depth == 1
                                else
                                f"{found.name or found.file_path.as_posix()} calls "
                                f"{target.display} transitively (depth {depth})"
                            ),
                            symbol=found.name,
                            source=target.display,
                            module=_module_of(found.file_path),
                        )
                    )
            if SurfaceDirection.OUTGOING in directions:
                reached = self.reference_graph.bounded_transitive_callees(
                    node, max_depth=depth
                )
                for found in sorted(reached, key=lambda n: (n.file_path.as_posix(), n.name or "")):
                    if found.key in prior_out or found.file_path == target.file_path:
                        continue
                    prior_out.add(found.key)
                    items.append(
                        ChangeSurfaceItem(
                            path=found.file_path,
                            relationship=ChangeRelationship.CALL,
                            direction=SurfaceDirection.OUTGOING,
                            depth=depth,
                            reason=(
                                f"{target.display} calls "
                                f"{found.name or found.file_path.as_posix()} directly"
                                if depth == 1
                                else
                                f"{target.display} calls "
                                f"{found.name or found.file_path.as_posix()} "
                                f"transitively (depth {depth})"
                            ),
                            symbol=found.name,
                            source=target.display,
                            module=_module_of(found.file_path),
                        )
                    )
        return items

    def _impact_analysis_items(
        self, target: SurfaceTarget,
    ) -> list[ChangeSurfaceItem]:
        """Impact-derived items for a target via the impact analyzer.

        One :meth:`~repolens.impact.ImpactAnalyzer.analyze` pass yields:
        symbol-reference / base-class evidence (mapped to
        :data:`SYMBOL_REFERENCE` / :data:`INHERITANCE`) and test /
        configuration links (mapped to :data:`OTHER`). Dependency edges and
        call relationships are computed by this module's own traversals, so
        those impact relationships are ignored here.
        """
        result = self.analyzer.analyze(
            _as_impact_target(target), max_depth=self.config.max_depth
        )
        items: list[ChangeSurfaceItem] = []
        for item in result.items:
            if item.relationship is Relationship.API_CONSUMER:
                inherited = EVIDENCE_BASE_CLASS in item.evidence
                if inherited and not self.config.include_inheritance:
                    continue
                if not inherited and not self.config.include_symbol_refs:
                    continue
                items.append(
                    ChangeSurfaceItem(
                        path=item.path,
                        relationship=(
                            ChangeRelationship.INHERITANCE
                            if inherited
                            else ChangeRelationship.SYMBOL_REFERENCE
                        ),
                        direction=SurfaceDirection.INCOMING,
                        depth=max(item.depth, 1),
                        reason=item.reason,
                        symbol=item.symbol,
                        source=target.display,
                        module=_module_of(item.path),
                        evidence=item.evidence,
                    )
                )
                continue
            if item.relationship not in _OTHER_IMPACT_RELATIONSHIPS:
                continue
            if (
                item.relationship is Relationship.CONFIGURATION
                and not self.config.include_config
            ):
                continue
            if item.relationship is Relationship.TEST and not self.config.include_tests:
                continue
            items.append(
                ChangeSurfaceItem(
                    path=item.path,
                    relationship=ChangeRelationship.OTHER,
                    direction=SurfaceDirection.INCOMING,
                    depth=max(item.depth, 1),
                    reason=item.reason,
                    symbol=item.symbol,
                    source=target.display,
                    module=_module_of(item.path),
                    evidence=item.evidence,
                )
            )
        return items

    def _outgoing_inheritance_items(
        self, target: SurfaceTarget,
    ) -> list[ChangeSurfaceItem]:
        """Base classes of a class target, resolved to repository files.

        Only conservative: a base class contributes an outgoing inheritance
        item when its (dotted) name resolves to a symbol defined somewhere in
        the repository.
        """
        if target.symbol is None:
            return []
        analysis = self._analysis(target.file_path)
        if analysis is None:
            return []
        classes = [cls for cls in analysis.classes if cls.name == target.symbol]
        if not classes:
            return []
        items: list[ChangeSurfaceItem] = []
        emitted: set[tuple] = set()
        for cls in classes:
            for base in cls.base_classes:
                leaf = base.rsplit(".", 1)[-1]
                if leaf == target.symbol:
                    continue
                for match in self.analyzer.symbol_index.find(leaf):
                    if (
                        match.file_path == target.file_path
                        or (match.file_path, leaf) in emitted
                    ):
                        continue
                    emitted.add((match.file_path, leaf))
                    items.append(
                        ChangeSurfaceItem(
                            path=match.file_path,
                            relationship=ChangeRelationship.INHERITANCE,
                            direction=SurfaceDirection.OUTGOING,
                            depth=1,
                            reason=(
                                f"{target.display} subclasses {base}"
                            ),
                            symbol=leaf,
                            source=target.display,
                            module=_module_of(match.file_path),
                            evidence=(EVIDENCE_BASE_CLASS,),
                        )
                    )
        return items

    def _call_node(self, target: SurfaceTarget):
        """The call-graph node closest to a symbol target, or ``None``.

        A module-level definition is preferred over a same-named method so
        callers of the definition are reported.
        """
        if target.symbol is None or self.reference_graph is None:
            return None
        matches = [
            node
            for node in self.reference_graph.get_nodes()
            if node.file_path == target.file_path and node.name == target.symbol
        ]
        if not matches:
            return None
        return min(
            matches,
            key=lambda node: (node.parent_class is not None, node.parent_class or ""),
        )

    def _analysis(self, path: Path) -> ModuleAnalysis | None:
        try:
            return self.analyzer.index.analysis_for(path)
        except KeyError:
            return None


# ---------------------------------------------------------------------------
# Item helpers
# ---------------------------------------------------------------------------


@dataclass
class _PendingItem:
    path: Path
    relationship: ChangeRelationship
    direction: SurfaceDirection
    depth: int
    reason: str
    symbol: str | None = None
    source: str | None = None
    module: str | None = None
    evidence: set[str] | None = None


def _merge_item(
    merged: dict[tuple, _PendingItem], item: ChangeSurfaceItem,
) -> None:
    key = (
        item.path.as_posix(),
        item.direction.value,
        item.relationship.value,
    )
    existing = merged.get(key)
    if existing is None:
        merged[key] = _PendingItem(
            path=item.path,
            relationship=item.relationship,
            direction=item.direction,
            depth=item.depth,
            reason=item.reason,
            symbol=item.symbol,
            source=item.source,
            module=item.module,
            evidence=set(item.evidence),
        )
        return
    if item.depth < existing.depth:
        existing.depth = item.depth
        existing.reason = item.reason
        existing.source = item.source
    if not existing.symbol and item.symbol:
        existing.symbol = item.symbol
    if not existing.module and item.module:
        existing.module = item.module
    if existing.evidence is None:
        existing.evidence = set()
    existing.evidence.update(item.evidence)


def _finalize_item(pending: _PendingItem) -> ChangeSurfaceItem:
    return ChangeSurfaceItem(
        path=pending.path,
        relationship=pending.relationship,
        direction=pending.direction,
        depth=pending.depth,
        reason=pending.reason,
        symbol=pending.symbol,
        source=pending.source,
        module=pending.module,
        evidence=tuple(sorted(pending.evidence or ())),
    )


def _import_item(
    path: Path, distance: int, target: SurfaceTarget, *, incoming: bool,
) -> ChangeSurfaceItem:
    anchor = target.file_path.as_posix()
    if incoming:
        reason = (
            f"{path.as_posix()} imports {anchor} directly"
            if distance == 1
            else f"{path.as_posix()} imports {anchor} transitively (depth {distance})"
        )
    else:
        reason = (
            f"{anchor} imports {path.as_posix()} directly"
            if distance == 1
            else f"{anchor} imports {path.as_posix()} transitively (depth {distance})"
        )
    return ChangeSurfaceItem(
        path=path,
        relationship=ChangeRelationship.IMPORT,
        direction=(
            SurfaceDirection.INCOMING if incoming else SurfaceDirection.OUTGOING
        ),
        depth=distance,
        reason=reason,
        source=target.display,
        module=_module_of(path),
    )


def _item_sort_key(item: ChangeSurfaceItem) -> tuple:
    return (
        item.direction.value,
        item.relationship.value,
        item.depth,
        item.path.as_posix(),
        item.symbol or "",
        item.reason,
    )


def _build_summary(
    targets: list[SurfaceTarget], items: list[ChangeSurfaceItem],
) -> dict:
    by_relationship: dict[str, int] = {}
    by_direction: dict[str, int] = {}
    for item in items:
        by_relationship[item.relationship.value] = (
            by_relationship.get(item.relationship.value, 0) + 1
        )
        by_direction[item.direction.value] = (
            by_direction.get(item.direction.value, 0) + 1
        )
    target_kinds: dict[str, int] = {}
    for target in targets:
        target_kinds[target.kind] = target_kinds.get(target.kind, 0) + 1
    return {
        "targets": len(targets),
        "target_kinds": target_kinds,
        "direct_affected": sum(1 for item in items if item.depth == 1),
        "transitive_affected": sum(1 for item in items if item.depth >= 2),
        "by_relationship": by_relationship,
        "by_direction": by_direction,
    }


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _parse_direction(value: str) -> list[SurfaceDirection]:
    cleaned = value.strip().lower()
    if cleaned in ("both", ""):
        return [SurfaceDirection.INCOMING, SurfaceDirection.OUTGOING]
    if cleaned == "incoming":
        return [SurfaceDirection.INCOMING]
    if cleaned == "outgoing":
        return [SurfaceDirection.OUTGOING]
    raise ValueError(
        f"direction must be 'incoming', 'outgoing', or 'both'; got {value!r}"
    )


def _as_impact_target(target: SurfaceTarget) -> ImpactTarget:
    return ImpactTarget(
        kind=TargetKind(target.kind),
        file_path=target.file_path,
        symbol=target.symbol,
        module=target.module,
        display=target.display,
    )


def _module_of(path: Path) -> str:
    if path.name == "__init__.py":
        return ".".join(path.parts[:-1])
    return ".".join((*path.parts[:-1], path.stem))