"""Change impact analysis (Milestone 21).

Deterministic, fully-offline analysis of *what changing a repository target
could affect*, built exclusively on the existing RepoLens infrastructure:

- the incremental :class:`~repolens.incremental_index.RepositoryIndex` snapshot
  (per-file parsed analysis and source, without re-reading/parsing);
- the :class:`~repolens.graph.DependencyGraph` (import edges between files);
- the :class:`~repolens.index.SymbolIndex` (where symbols are *defined*).

It never executes repository code, never calls a language server, an LLM, or
the network. Evidence is *conservative*: a relationship is reported only when
the existing static analysis can support it, and symbol-level claims are
clearly separated from file/module dependency claims.

If a relationship cannot be established reliably it is not labelled as
certain. Multi-hop symbol *call* resolution (a true language-server call
graph) is explicitly out of scope; see :doc:`/docs/impact-analysis`.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from repolens.graph import DependencyGraph, DependencyGraphBuilder
from repolens.index import SymbolIndex, SymbolIndexBuilder
from repolens.incremental_index import IncrementalIndexBuilder, RepositoryIndex
from repolens.parser import ModuleAnalysis, PythonParser

# ---------------------------------------------------------------------------
# Relationship taxonomy
# ---------------------------------------------------------------------------


class Relationship(str, Enum):
    """How an impacted file relates to the changed target.

    Every relationship is grounded in repository evidence:

    - :attr:`DIRECT_DEPENDENCY` — the file imports the target directly
      (reverse dependency edge at depth 1);
    - :attr:`INDIRECT_DEPENDENCY` — the file imports the target transitively
      through at least one other importer (depth >= 2);
    - :attr:`REVERSE_DEPENDENCY` — the *target* imports the file (changing the
      target changes how the target uses this file);
    - :attr:`TEST` — the file is a test directly linked to the target;
    - :attr:`CONFIGURATION` — the file is a configuration/declarative file
      referencing the target's module;
    - :attr:`API_CONSUMER` — the file touches a *symbol* defined by the target
      (an import of that symbol, a base-class relationship, or a conservative
      source-text reference), separate from the module-level dependency.
    """

    DIRECT_DEPENDENCY = "direct_dependency"
    INDIRECT_DEPENDENCY = "indirect_dependency"
    REVERSE_DEPENDENCY = "reverse_dependency"
    TEST = "test"
    CONFIGURATION = "configuration"
    API_CONSUMER = "api_consumer"


#: Stable, machine-readable evidence tags attached to :class:`ImpactItem`.
#: They record *how* each relationship was established so downstream consumers
#: can weigh string/import evidence differently from graph edges.
EVIDENCE_DEPENDENCY_EDGE = "dependency_edge"
EVIDENCE_SYMBOL_IMPORT = "symbol_import"
EVIDENCE_SYMBOL_TEXT = "symbol_text_reference"
EVIDENCE_BASE_CLASS = "base_class"
EVIDENCE_PACKAGE_EXPORT = "package_export"
EVIDENCE_TEST_NAME = "test_name_match"
EVIDENCE_TEST_IMPORT = "test_import"
EVIDENCE_TEST_SYMBOL = "test_symbol_reference"
EVIDENCE_CONFIG_IMPORT = "config_file_imports_target"
EVIDENCE_CONFIG_TEXT = "config_text_reference"

#: Priority used when a file matches several relationships (higher wins).
#: A file gets exactly one ``relationship`` (its most specific pairing); the
#: remaining evidence is preserved in :attr:`ImpactItem.evidence`.
_RELATIONSHIP_PRIORITY = {
    Relationship.TEST: 6,
    Relationship.CONFIGURATION: 5,
    Relationship.API_CONSUMER: 4,
    Relationship.DIRECT_DEPENDENCY: 3,
    Relationship.INDIRECT_DEPENDENCY: 2,
    Relationship.REVERSE_DEPENDENCY: 1,
}

#: Deterministic sort bucket per relationship (used for stable output order).
_RELATIONSHIP_BUCKET = {
    Relationship.DIRECT_DEPENDENCY: 0,
    Relationship.INDIRECT_DEPENDENCY: 1,
    Relationship.TEST: 2,
    Relationship.API_CONSUMER: 3,
    Relationship.CONFIGURATION: 4,
    Relationship.REVERSE_DEPENDENCY: 5,
}


class RiskLevel(str, Enum):
    """Deterministic risk classification for an impact item or its aggregate."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


# ---------------------------------------------------------------------------
# Result model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ImpactItem:
    """One file affected by a change to the analyzed target.

    Attributes:
        path: Repository-relative path of the affected file.
        relationship: How the file relates to the target (see
            :class:`Relationship`).
        risk: This file's individual risk level.
        reason: Human-readable explanation of *why* this file is listed.
        symbol: Symbol involved, when symbol-level evidence was found.
        evidence: Stable evidence tags describing how the relationship was
            established.
        depth: Reverse-traversal depth (1 = direct dependent; 2+ = indirect),
            graph distance for reverse dependencies, or 0 for evidence-based
            relationships without a graph edge.
    """

    path: Path
    relationship: Relationship
    risk: RiskLevel
    reason: str
    symbol: str | None = None
    evidence: tuple[str, ...] = ()
    depth: int = 0

    def to_dict(self) -> dict:
        return {
            "path": self.path.as_posix(),
            "relationship": self.relationship.value,
            "risk": self.risk.value,
            "reason": self.reason,
            "symbol": self.symbol,
            "evidence": list(self.evidence),
            "depth": self.depth,
        }


@dataclass(frozen=True)
class ImpactResult:
    """The complete, deterministic result of one impact analysis.

    ``items`` never contains the changed target itself — the target is exposed
    as ``target_path``/``symbol`` and prioritised separately by the context
    integration, not listed as an *affected* file.
    """

    target: str
    kind: str | None
    target_path: Path | None
    symbol: str | None
    module: str | None
    items: tuple[ImpactItem, ...]
    risk: RiskLevel
    max_depth_reached: int
    summary: dict

    @property
    def files(self) -> tuple[Path, ...]:
        return tuple(item.path for item in self.items)

    def to_dict(self) -> dict:
        return {
            "target": self.target,
            "kind": self.kind,
            "target_path": self.target_path.as_posix() if self.target_path else None,
            "symbol": self.symbol,
            "module": self.module,
            "risk": self.risk.value,
            "max_depth_reached": self.max_depth_reached,
            "summary": dict(self.summary),
            "items": [item.to_dict() for item in self.items],
        }

    def to_json(self, **json_kwargs) -> str:
        import json

        return json.dumps(self.to_dict(), **json_kwargs)


class ImpactTargetError(ValueError):
    """Raised when a target cannot be resolved deterministically or at all."""


# ---------------------------------------------------------------------------
# Target model
# ---------------------------------------------------------------------------


class TargetKind(str, Enum):
    """What kind of repository object a target string denotes."""

    FILE = "file"
    MODULE = "module"
    SYMBOL = "symbol"
    SYMBOL_IN_FILE = "symbol_in_file"


@dataclass(frozen=True)
class ImpactTarget:
    """A resolved, canonical impact target."""

    kind: TargetKind
    file_path: Path
    symbol: str | None = None
    module: str | None = None
    display: str | None = None

    @property
    def label(self) -> str:
        if self.display is not None:
            return self.display
        if self.kind in (TargetKind.SYMBOL, TargetKind.SYMBOL_IN_FILE):
            return f"{self.file_path.as_posix()}::{self.symbol}"
        if self.module:
            return self.module
        return self.file_path.as_posix()


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ImpactConfig:
    """Bounded, deterministic limits for impact analysis.

    The reverse traversal is *always* bounded by ``max_depth`` (no unbounded
    recursion) and by ``max_nodes`` (a hard cap on visited dependents, so a
    pathological or enormous dependency graph cannot be walked past a fixed
    budget).
    """

    max_depth: int = 4
    max_nodes: int = 2000
    include_reverse: bool = True
    include_tests: bool = True
    include_config: bool = True
    #: Non-Python declarative files scanned for module references.
    config_filenames: tuple[str, ...] = ("pyproject.toml", "setup.cfg", "tox.ini")
    #: Python files considered "configuration" when they import the target.
    config_stems: tuple[str, ...] = ("config", "settings", "setup", "conf")


#: Test-like directory names.
_TEST_DIRS = frozenset({"test", "tests"})


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------


class ImpactAnalyzer:
    """Analyze the blast radius of a change to a file, module, or symbol.

    Reuses the existing incremental index, dependency graph, and symbol index
    whenever supplied — nothing is re-parsed on repeated :meth:`analyze` calls.
    """

    def __init__(
        self,
        root: Path | str,
        *,
        index: RepositoryIndex | None = None,
        graph: DependencyGraph | None = None,
        symbol_index: SymbolIndex | None = None,
        config: ImpactConfig | None = None,
    ) -> None:
        self.root = Path(root)
        self.config = config if config is not None else ImpactConfig()
        if index is None:
            index = IncrementalIndexBuilder(self.root, persist=False).build()
        self.index: RepositoryIndex = index
        self.graph = graph if graph is not None else DependencyGraphBuilder(
            self.root, index=index
        ).build()
        self.symbol_index = (
            symbol_index
            if symbol_index is not None
            else SymbolIndexBuilder(self.root, index=index).build()
        )
        self._parser = PythonParser()
        self._modules: dict[str, Path] = self._build_module_map(index.files)

    # ------------------------------------------------------------------
    # Target resolution
    # ------------------------------------------------------------------

    def resolve_target(self, target: str) -> ImpactTarget:
        """Resolve ``target`` to a canonical :class:`ImpactTarget`.

        Accepted forms (checked in order):

        - ``path/to/file.py::symbol`` — symbol defined in that file;
        - ``path/to/file.py`` (or ``path/to/file``) — repository file;
        - ``dotted.module.name`` — module that resolves to a repository file;
        - ``symbol_name`` — a uniquely defined symbol (ambiguous symbols are
          rejected with a message pointing at the candidate files);
        - ``leaf_name`` — a uniquely named module (e.g. ``processor``).

        Raises :class:`ImpactTargetError` for unknown or ambiguous targets.
        """
        raw = str(target).strip().replace("\\", "/")
        if not raw:
            raise ImpactTargetError("Impact target must not be empty.")

        if "::" in raw:
            return self._resolve_file_symbol(raw)

        if raw.endswith(".py") or "/" in raw:
            path = _clean_path(raw)
            if path not in self.index.files:
                raise ImpactTargetError(
                    f"No repository file matches {raw!r}."
                )
            return ImpactTarget(
                kind=TargetKind.FILE,
                file_path=path,
                module=self._module_of(path),
                display=path.as_posix(),
            )

        # Dotted module name (exact match wins).
        module_file = self._modules.get(raw)
        if module_file is not None:
            return ImpactTarget(
                kind=TargetKind.MODULE,
                file_path=module_file,
                module=raw,
                display=raw,
            )

        # Symbol defined in exactly one file.
        symbols = self.symbol_index.find(raw)
        if symbols:
            distinct = {s.file_path for s in symbols}
            if len(distinct) == 1:
                symbol = _pick_symbol(symbols)
                return ImpactTarget(
                    kind=TargetKind.SYMBOL,
                    file_path=symbol.file_path,
                    symbol=symbol.name,
                    module=self._module_of(symbol.file_path),
                    display=raw,
                )
            candidates = ", ".join(sorted(p.as_posix() for p in distinct))
            raise ImpactTargetError(
                f"Ambiguous symbol {raw!r}: defined in {candidates}. "
                "Disambiguate with 'path/to/file.py::symbol'."
            )

        # Uniquely named module leaf (e.g. "processor" -> payments/processor.py).
        leaf_matches = [
            path for path in self.index.files if path.stem == raw
        ]
        if len(leaf_matches) == 1:
            path = leaf_matches[0]
            return ImpactTarget(
                kind=TargetKind.MODULE,
                file_path=path,
                module=self._module_of(path),
                display=raw,
            )
        if len(leaf_matches) > 1:
            candidates = ", ".join(sorted(p.as_posix() for p in leaf_matches))
            raise ImpactTargetError(
                f"Ambiguous module name {raw!r}: matches {candidates}."
            )

        raise ImpactTargetError(
            f"No repository file, module, or symbol matches {raw!r}."
        )

    def _resolve_file_symbol(self, raw: str) -> ImpactTarget:
        file_part, _, symbol_part = raw.partition("::")
        if not file_part.strip() or not symbol_part.strip():
            raise ImpactTargetError(
                f"Malformed target {raw!r}; expected 'path/file.py::symbol'."
            )
        anchor = self._resolve_path_or_module(file_part)
        matches = [s for s in self.symbol_index.find(symbol_part) if s.file_path == anchor]
        if not matches:
            raise ImpactTargetError(
                f"Symbol {symbol_part!r} is not defined in {anchor.as_posix()}."
            )
        symbol = _pick_symbol(matches)
        return ImpactTarget(
            kind=TargetKind.SYMBOL_IN_FILE,
            file_path=anchor,
            symbol=symbol.name,
            module=self._module_of(anchor),
            display=(file_part if "/" in file_part or file_part.endswith(".py")
                     else raw),
        )

    def _resolve_path_or_module(self, raw: str) -> Path:
        path = _clean_path(raw)
        if path in self.index.files:
            return path
        module_file = self._modules.get(raw)
        if module_file is not None:
            return module_file
        raise ImpactTargetError(f"No repository file or module matches {raw!r}.")

    # ------------------------------------------------------------------
    # Public analysis entry point
    # ------------------------------------------------------------------

    def analyze(
        self,
        target: str | ImpactTarget,
        *,
        max_depth: int | None = None,
        limit: int | None = None,
    ) -> ImpactResult:
        """Analyze the impact of changing ``target`` (see :meth:`resolve_target`).

        ``max_depth`` / ``limit`` override the constructor's config for this
        call only (shallow; the stateless analyzer is reused). Setting
        ``max_depth=0`` disables reverse traversal.

        Deterministic: the same repository snapshot and target always produce
        the same :class:`ImpactResult`.
        """
        resolved = target if isinstance(target, ImpactTarget) else self.resolve_target(target)
        anchor = resolved.file_path
        effective_depth = (
            self.config.max_depth if max_depth is None else max_depth
        )
        summary: dict = {
            "direct_dependents": 0,
            "indirect_dependents": 0,
            "tests": 0,
            "api_consumers": 0,
            "configuration": 0,
            "reverse_dependencies": 0,
            "max_depth_reached": 0,
            "public_symbols": [],
            "exported": False,
        }

        # Phase 1: reverse traversal (who imports the target, and at what depth).
        reverse_distances = self._reverse_reachable(anchor, effective_depth)
        if reverse_distances:
            summary["max_depth_reached"] = max(reverse_distances.values())

        # Phase 2: symbol-level evidence when a symbol is named.
        symbol_evidence: dict[Path, tuple[str, tuple[str, ...], int]] = {}
        if resolved.symbol is not None:
            symbol_evidence = self._collect_symbol_evidence(resolved)

        # Phase 3: candidate files = reverse dependents + symbol consumers +
        # package exporters (the exporters themselves are affected).
        exporter_items = {}
        if resolved.symbol is not None:
            exporter_items = self._package_exporters(resolved)

        candidates: dict[Path, _PendingItem] = {}

        for path, distance in reverse_distances.items():
            if distance == 1:
                rel = Relationship.DIRECT_DEPENDENCY
                summary["direct_dependents"] += 1
            else:
                rel = Relationship.INDIRECT_DEPENDENCY
                summary["indirect_dependents"] += 1
            pending = _PendingItem(
                relationship=rel,
                reason=(
                    f"imports {resolved.label} directly"
                    if rel is Relationship.DIRECT_DEPENDENCY
                    else
                    f"imports {resolved.label} transitively (depth {distance})"
                ),
                evidence=(EVIDENCE_DEPENDENCY_EDGE,),
                depth=distance,
            )
            self._merge(candidates, path, pending)

        for path, (symbol_name, evidence, depth) in symbol_evidence.items():
            if path in candidates:
                existing = candidates[path]
                existing.evidence = tuple(dict.fromkeys((*existing.evidence, *evidence)))
                existing.symbol = symbol_name
                existing.relationship = _prefer(
                    existing.relationship,
                    Relationship.API_CONSUMER,
                )
                existing.reason = (
                    f"references symbol {symbol_name} defined in "
                    f"{resolved.label}"
                )
            else:
                public = not symbol_name.startswith("_")
                pending = _PendingItem(
                    relationship=Relationship.API_CONSUMER,
                    reason=(
                        f"references symbol {symbol_name} defined in "
                        f"{resolved.label}"
                    ),
                    evidence=evidence,
                    depth=depth,
                    symbol=symbol_name,
                )
                self._merge(candidates, path, pending)
                if public:
                    summary["api_consumers"] += 1

        for path, (symbol_name, reason, evidence, depth) in exporter_items.items():
            pending = _PendingItem(
                relationship=Relationship.API_CONSUMER,
                reason=reason,
                evidence=evidence,
                depth=depth,
                symbol=symbol_name,
            )
            self._merge(candidates, path, pending)
            summary["api_consumers"] += 1

        # Phase 4: classify tests and configuration before finalizing.
        if self.config.include_tests:
            self._classify_tests(candidates, resolved)

        if self.config.include_config:
            self._classify_config(candidates, resolved)
            self._add_non_python_config(candidates, resolved)

        # Phase 5: reverse relationships (what the target itself imports).
        if self.config.include_reverse:
            for dep in self.graph.get_dependencies(anchor):
                if dep == anchor or dep in candidates:
                    continue
                summary["reverse_dependencies"] += 1
                self._merge(
                    candidates,
                    dep,
                    _PendingItem(
                        relationship=Relationship.REVERSE_DEPENDENCY,
                        reason=(
                            f"the changed target {resolved.label} imports "
                            f"this file directly"
                        ),
                        evidence=(EVIDENCE_DEPENDENCY_EDGE,),
                        depth=1,
                    ),
                )

        # Compute per-item risk and finalize ordered items.
        items = self._finalize_items(candidates, resolved.symbol)
        if resolved.symbol is not None and not resolved.symbol.startswith("_"):
            summary["public_symbols"].append(resolved.symbol)
        if resolved.symbol is not None and self._is_exported(resolved):
            summary["exported"] = True
        summary = _count_summary(summary, candidates)
        risk = self._overall_risk(summary)

        if limit is not None:
            items = items[:limit]

        return ImpactResult(
            target=resolved.label,
            kind=resolved.kind.value,
            target_path=resolved.file_path,
            symbol=resolved.symbol,
            module=resolved.module,
            items=tuple(items),
            risk=risk,
            max_depth_reached=summary["max_depth_reached"],
            summary=summary,
        )

    # ------------------------------------------------------------------
    # Graph traversal (bounded)
    # ------------------------------------------------------------------

    def _reverse_reachable(self, start: Path, max_depth: int) -> dict[Path, int]:
        """Bounded BFS over reverse edges; returns {file: distance}."""
        if max_depth <= 0:
            return {}
        distances: dict[Path, int] = {start: 0}
        queue: deque[Path] = deque([start])
        visited: set[Path] = set()
        while queue and len(distances) < self.config.max_nodes:
            node = queue.popleft()
            if node in visited:
                continue
            visited.add(node)
            distance = distances[node]
            if distance >= max_depth:
                continue
            for dependent in self.graph.get_dependents(node):
                if dependent in distances:
                    continue
                if len(distances) >= self.config.max_nodes:
                    break
                distances[dependent] = distance + 1
                if distance + 1 < max_depth:
                    queue.append(dependent)
        distances.pop(start, None)
        return distances

    # ------------------------------------------------------------------
    # Symbol-level evidence
    # ------------------------------------------------------------------

    def _collect_symbol_evidence(
        self, target: ImpactTarget,
    ) -> dict[Path, tuple[str, tuple[str, ...], int]]:
        """Return {consumer_file: (symbol, evidence_tags, depth)}.

        Only files that already import the target's module (a graph edge) or
        that reach it through a package re-export are scanned, so symbol
        matching never walks the whole repository.
        """
        anchor = target.file_path
        symbol = target.symbol or ""
        anchor_module = self._module_of(anchor)
        consumers: dict[Path, tuple[str, tuple[str, ...], int]] = {}

        candidates: set[Path] = set(self.graph.get_dependents(anchor))
        candidates.update(
            dep for dep, _ in self._package_export_consumers(anchor, symbol).items()
        )

        for path in sorted(candidates):
            analysis = self._analysis(path)
            if analysis is None:
                continue
            evidence: list[str] = []
            distance = self.graph_distance(anchor, path)

            # 1. Import of the symbol by name (from-import or import).
            imported = any(
                fi.name == symbol for fi in analysis.from_imports
            )
            if imported:
                evidence.append(EVIDENCE_SYMBOL_IMPORT)

            # 2. Conservative text reference bound to the module (qualified).
            bound = self._bound_module_names(analysis, anchor_module)
            if bound:
                source = self._source(path)
                for name in sorted(bound):
                    if f"{name}.{symbol}" in source:
                        evidence.append(EVIDENCE_SYMBOL_TEXT)
                        break

            # 3. Base-class relationship when the symbol is a class.
            if any(c.base_classes and symbol in c.base_classes for c in analysis.classes):
                evidence.append(EVIDENCE_BASE_CLASS)

            if evidence:
                consumers[path] = (symbol, tuple(dict.fromkeys(evidence)), distance)
        return consumers

    def _package_export_consumers(
        self, anchor: Path, symbol: str,
    ) -> dict[Path, int]:
        """Files that import ``symbol`` through a package ``__init__`` re-export."""
        result: dict[Path, int] = {}
        for init in self._package_inits(anchor):
            init_analysis = self._analysis(init)
            if init_analysis is None:
                continue
            if not any(fi.name == symbol for fi in init_analysis.from_imports):
                continue
            for dependent in self.graph.get_dependents(init):
                dep_analysis = self._analysis(dependent)
                if dep_analysis is None:
                    continue
                if any(fi.name == symbol for fi in dep_analysis.from_imports):
                    result.setdefault(dependent, 0)
        return result

    def _package_exporters(
        self, target: ImpactTarget,
    ) -> dict[Path, tuple[str, str, tuple[str, ...], int]]:
        """Package ``__init__`` files re-exporting the target symbol.

        Returns ``{init_path: (symbol, reason, evidence, depth)}``. When
        ``__init__`` renames a symbol (``... as X``) the re-export is still
        attributable to the public symbol name.
        """
        symbol = target.symbol or ""
        result: dict[Path, tuple[str, str, tuple[str, ...], int]] = {}
        for init in self._package_inits(target.file_path):
            analysis = self._analysis(init)
            if analysis is None:
                continue
            if any(fi.name == symbol for fi in analysis.from_imports):
                result[init] = (
                    symbol,
                    f"package {self._module_of(init)} re-exports {symbol}",
                    (EVIDENCE_PACKAGE_EXPORT,),
                    0,
                )
        return result

    # ------------------------------------------------------------------
    # Test and configuration classification
    # ------------------------------------------------------------------

    def _classify_tests(self, candidates: dict[Path, _PendingItem], target: ImpactTarget) -> None:
        test_files = self._test_matches(target)
        for path, reason in test_files.items():
            pending = _PendingItem(
                relationship=Relationship.TEST,
                reason=reason,
                evidence=(EVIDENCE_TEST_NAME,),
                depth=0,
            )
            self._merge(candidates, path, pending)

    def _test_matches(self, target: ImpactTarget) -> dict[Path, str]:
        """Return {test_file: reason} with deterministic, evidence-based links.

        Only files that look like tests (name/path signals) AND have a concrete
        link to the target (import the target, reference the target symbol, or
        have a filename matching the target module leaf) are returned.
        """
        anchor = target.file_path
        matches: dict[Path, str] = {}

        all_files = list(self.index.files)
        for path in all_files:
            if not self._is_test_file(path):
                continue
            if path in matches:
                continue
            # 1. Imported the target directly.
            if anchor in self.graph.get_dependencies(path):
                matches[path] = (
                    f"test file imports {target.label} directly"
                )
                continue
            # 2. Symbol linkage.
            if target.symbol is not None:
                analysis = self._analysis(path)
                source = self._source(path)
                if analysis is not None and any(
                    fi.name == target.symbol for fi in analysis.from_imports
                ):
                    matches[path] = (
                        f"test file imports symbol {target.symbol}"
                    )
                    continue
                if target.symbol and f".{target.symbol}" in source:
                    matches[path] = (
                        f"test source references symbol {target.symbol}"
                    )
                    continue
            # 3. Filename correspondence with the target module leaf.
            if self._name_matches_module(path, anchor):
                matches[path] = (
                    f"test filename corresponds to module "
                    f"{target.module or anchor.as_posix()}"
                )
        return matches

    def _classify_config(self, candidates: dict[Path, _PendingItem], target: ImpactTarget) -> None:
        for path, pending in list(candidates.items()):
            if pending.relationship is Relationship.TEST:
                continue
            if path.stem in self.config.config_stems or path.name == "setup.py":
                pending.relationship = _prefer(
                    pending.relationship, Relationship.CONFIGURATION
                )
                pending.evidence = tuple(
                    dict.fromkeys((*pending.evidence, EVIDENCE_CONFIG_IMPORT))
                )
                pending.reason = (
                    f"configuration file imports {target.label}"
                )

    def _add_non_python_config(self, candidates: dict[Path, _PendingItem], target: ImpactTarget) -> None:
        module = target.module
        if not module:
            return
        anchor_name = target.file_path.as_posix()
        anchor_no_py = anchor_name[:-3] if anchor_name.endswith(".py") else anchor_name
        for filename in self.config.config_filenames:
            path = self.root / filename
            if not path.is_file():
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except (OSError, ValueError):
                continue
            referenced = module in text or anchor_no_py in text
            if referenced:
                rel_path = Path(filename)
                pending = _PendingItem(
                    relationship=Relationship.CONFIGURATION,
                    reason=(
                        f"declarative config {filename} references module "
                        f"{module}"
                    ),
                    evidence=(EVIDENCE_CONFIG_TEXT,),
                    depth=0,
                )
                self._merge(candidates, rel_path, pending)

    # ------------------------------------------------------------------
    # Finalization and risk
    # ------------------------------------------------------------------

    def _finalize_items(
        self, candidates: dict[Path, _PendingItem], symbol: str | None,
    ) -> list[ImpactItem]:
        items: list[ImpactItem] = []
        for path, pending in candidates.items():
            public = bool(pending.symbol and not pending.symbol.startswith("_"))
            risk = _item_risk(pending.relationship, public)
            items.append(
                ImpactItem(
                    path=path,
                    relationship=pending.relationship,
                    risk=risk,
                    reason=pending.reason,
                    symbol=pending.symbol,
                    evidence=tuple(sorted(pending.evidence)) if pending.evidence else (),
                    depth=pending.depth,
                )
            )
        items.sort(
            key=lambda item: (
                _RELATIONSHIP_BUCKET[item.relationship],
                item.depth,
                item.path.as_posix(),
            )
        )
        return items

    def _overall_risk(self, summary: dict) -> RiskLevel:
        points = 0
        points += min(summary["direct_dependents"], 5) * 2
        points += min(summary["indirect_dependents"], 3) * 1
        points += min(summary["tests"], 2) * 1
        points += 2 if summary["api_consumers"] else 0
        points += 1 if summary["exported"] else 0
        points += 1 if summary["max_depth_reached"] >= 3 else 0
        if points >= 8:
            return RiskLevel.HIGH
        if points >= 4:
            return RiskLevel.MEDIUM
        return RiskLevel.LOW

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _merge(self, candidates: dict[Path, _PendingItem], path: Path, pending: _PendingItem) -> None:
        existing = candidates.get(path)
        if existing is None:
            candidates[path] = pending
            return
        higher = _prefer(existing.relationship, pending.relationship)
        if higher is pending.relationship:
            existing.relationship = pending.relationship
            existing.reason = pending.reason
        existing.evidence = tuple(
            dict.fromkeys((*existing.evidence, *pending.evidence))
        )
        if not existing.symbol:
            existing.symbol = pending.symbol
        if existing.depth == 0:
            existing.depth = pending.depth

    def _analysis(self, path: Path) -> ModuleAnalysis | None:
        try:
            return self.index.analysis_for(path)
        except KeyError:
            return None

    def _source(self, path: Path) -> str:
        try:
            return self.index.source_for(path)
        except KeyError:
            return ""

    def _module_of(self, path: Path) -> str:
        if path.name == "__init__.py":
            return ".".join(path.parts[:-1])
        return ".".join((*path.parts[:-1], path.stem))

    @staticmethod
    def _build_module_map(files) -> dict[str, Path]:
        modules: dict[str, Path] = {}
        for file_path in files:
            if file_path.name == "__init__.py":
                name = ".".join(file_path.parts[:-1])
            else:
                name = ".".join((*file_path.parts[:-1], file_path.stem))
            if not name:
                continue
            modules.setdefault(name, file_path)
        return modules

    def graph_distance(self, anchor: Path, path: Path) -> int:
        """Best-effort distance from ``path`` to ``anchor`` in the reverse graph."""
        distances = self._reverse_reachable(anchor, 1)
        return distances.get(path, 0)

    def _bound_module_names(self, analysis: ModuleAnalysis, anchor_module: str) -> set[str]:
        """Local names bound to ``anchor_module`` in ``analysis``."""
        names: set[str] = set()
        for imp in analysis.imports:
            if imp.module == anchor_module or imp.module.startswith(anchor_module + "."):
                names.add(imp.alias or imp.module.rsplit(".", 1)[-1])
        for fi in analysis.from_imports:
            if fi.module == anchor_module:
                names.add(fi.alias or fi.name)
        return names

    def _package_inits(self, anchor: Path) -> list[Path]:
        """``__init__.py`` files for every ancestor package of ``anchor``."""
        inits: list[Path] = []
        for depth in range(1, len(anchor.parts)):
            candidate = Path(*anchor.parts[:depth]) / "__init__.py"
            if candidate in self.index.files:
                inits.append(candidate)
        return inits

    def _is_exported(self, target: ImpactTarget) -> bool:
        return bool(self._package_exporters(target))

    def _is_test_file(self, path: Path) -> bool:
        if path.name.startswith("test_"):
            return True
        if path.name.endswith("_test.py"):
            return True
        return any(part in _TEST_DIRS for part in path.parts)

    def _name_matches_module(self, test_path: Path, anchor: Path) -> bool:
        leaf = anchor.stem if anchor.name != "__init__.py" else anchor.parts[-2]
        return (
            test_path.name == f"test_{leaf}.py"
            or test_path.name == f"{leaf}_test.py"
            or _any_suffix(f"test_{leaf}", test_path)
        )


def _any_suffix(leaf: str, path: Path) -> bool:
    return path.name.startswith(leaf + "_") or path.name.endswith("_" + leaf)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


@dataclass
class _PendingItem:
    relationship: Relationship
    reason: str
    evidence: tuple[str, ...] = ()
    depth: int = 0
    symbol: str | None = None


def _prefer(a: Relationship, b: Relationship) -> Relationship:
    """Return the higher-priority of two relationships."""
    if _RELATIONSHIP_PRIORITY[b] > _RELATIONSHIP_PRIORITY[a]:
        return b
    return a


def _pick_symbol(symbols) -> object:
    return sorted(symbols, key=lambda s: (s.kind.value, s.line if s.line is not None else 0))[0]


def _clean_path(raw: str) -> Path:
    return Path(raw) if raw.endswith(".py") else Path(raw + ".py")


def _item_risk(relationship: Relationship, public_symbol: bool) -> RiskLevel:
    if relationship is Relationship.DIRECT_DEPENDENCY:
        return RiskLevel.HIGH
    if relationship is Relationship.API_CONSUMER:
        return RiskLevel.HIGH if public_symbol else RiskLevel.MEDIUM
    if relationship is Relationship.TEST:
        return RiskLevel.MEDIUM
    if relationship is Relationship.INDIRECT_DEPENDENCY:
        return RiskLevel.MEDIUM
    if relationship is Relationship.CONFIGURATION:
        return RiskLevel.MEDIUM
    return RiskLevel.LOW


def _count_summary(summary: dict, candidates: dict[Path, _PendingItem]) -> dict:
    """Recount from finalized relationships so summary always matches items."""
    counts = {
        "direct_dependents": 0,
        "indirect_dependents": 0,
        "tests": 0,
        "api_consumers": 0,
        "configuration": 0,
        "reverse_dependencies": 0,
    }
    for pending in candidates.values():
        rel = pending.relationship
        if rel is Relationship.DIRECT_DEPENDENCY:
            counts["direct_dependents"] += 1
        elif rel is Relationship.INDIRECT_DEPENDENCY:
            counts["indirect_dependents"] += 1
        elif rel is Relationship.TEST:
            counts["tests"] += 1
        elif rel is Relationship.API_CONSUMER:
            counts["api_consumers"] += 1
        elif rel is Relationship.CONFIGURATION:
            counts["configuration"] += 1
        elif rel is Relationship.REVERSE_DEPENDENCY:
            counts["reverse_dependencies"] += 1
    summary.update(counts)
    return summary