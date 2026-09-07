"""Deterministic change-plan engine (Milestone 24.1).

Converts a natural-language change request into a bounded, explainable
inspection/change plan.  No LLM; no re-parsing; all traversal bounded;
output deterministic.

The engine reuses the existing shared repository index, dependency graph,
call graph, architecture graph, subsystem discovery, impact analyzer, and
lexical search — it never introduces a second repository scan or parser.
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from repolens.architecture import ArchitectureGraph, ArchitectureGraphBuilder, ArchitectureNodeKind
from repolens.call_graph import CallGraph, CallGraphBuilder
from repolens.graph import DependencyGraph, DependencyGraphBuilder
from repolens.impact import ImpactAnalyzer
from repolens.index import SymbolIndex, SymbolIndexBuilder
from repolens.incremental_index import IncrementalIndexBuilder, RepositoryIndex
from repolens.references import ReferenceIndexBuilder
from repolens.search import CodeSearcher, tokenize
from repolens.subsystems import Subsystem, discover_subsystems, subsystem_of

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_ACTION_VERBS: dict[str, str] = {
    "add": "add",
    "create": "add",
    "implement": "add",
    "remove": "remove",
    "delete": "remove",
    "drop": "remove",
    "modify": "modify",
    "change": "modify",
    "update": "modify",
    "rename": "rename",
    "refactor": "refactor",
    "replace": "replace",
    "fix": "fix",
    "repair": "fix",
    "extend": "extend",
    "migrate": "migrate",
}

# Common English words that are NOT domain terms when standalone.
_STOP_WORDS: frozenset[str] = frozenset({
    "a", "an", "the", "to", "in", "of", "for", "and", "or", "is", "it",
    "my", "our", "its", "with", "from", "by", "on", "at", "as", "be",
    "do", "if", "so", "no", "up", "we", "you", "they", "this", "that",
    "has", "had", "was", "are", "but", "not", "can", "will", "may",
    "new", "old", "all", "some", "any", "each", "than", "then",
    "also", "just", "only", "now", "how", "what", "why", "when", "where",
    "which", "should", "would", "could", "about", "into", "through",
    "during", "before", "after", "above", "below", "between", "support",
})

# Path-like patterns: "foo/bar.py", "foo.bar", "app/services/checkout.py"
_PATH_RE = re.compile(r"[a-zA-Z_][\w./]*(?:\.\w+)")
_MODULE_RE = re.compile(r"[a-zA-Z_]\w*(?:\.[a-zA-Z_]\w*)+")

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class Confidence(str, Enum):
    CONFIRMED = "confirmed"
    LIKELY = "likely"
    POSSIBLE = "possible"


class PlanCategory(str, Enum):
    PRIMARY_TARGET = "primary_target"
    DIRECT_CALLER = "direct_caller"
    INDIRECT_CALLER = "indirect_caller"
    DIRECT_DEPENDENCY = "direct_dependency"
    INDIRECT_DEPENDENCY = "indirect_dependency"
    ARCHITECTURE_ENTRY = "architecture_entry"
    TEST = "test"
    INDIRECT_TEST = "indirect_test"
    SUBSYSTEM_NEIGHBOR = "subsystem_neighbor"
    BROADER_NEIGHBOR = "broader_neighbor"


class TargetKind(str, Enum):
    FILE = "file"
    MODULE = "module"
    PACKAGE = "package"
    SYMBOL = "symbol"


# ---------------------------------------------------------------------------
# Frozen data models
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ChangePlanConfig:
    """Explicit bounds for the change-plan engine."""

    max_target_candidates: int = 30
    max_primary_targets: int = 5
    max_affected_files: int = 50
    max_call_depth: int = 2
    max_dependency_depth: int = 2
    max_tests: int = 15
    max_inspection_items: int = 40
    max_subsystems: int = 10
    max_total_nodes: int = 200
    max_impact_nodes: int = 200
    impact_max_depth: int = 3
    search_limit: int = 20
    arch_max_expanded: int = 50


@dataclass(frozen=True)
class RequestAnalysis:
    """Structured signals extracted from a change request."""

    raw: str
    action: str | None
    tokens: tuple[str, ...]
    domain_terms: tuple[str, ...]
    path_candidates: tuple[str, ...]
    module_candidates: tuple[str, ...]
    symbol_candidates: tuple[str, ...]


@dataclass(frozen=True)
class TargetCandidate:
    """One candidate change target."""

    target: str
    kind: TargetKind
    score: int
    confidence: Confidence
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class PlanItem:
    """One item in the recommended inspection order."""

    path: str
    module: str | None
    symbol: str | None
    category: PlanCategory
    score: int
    confidence: Confidence
    reason: str
    relationship: str
    priority: int


@dataclass(frozen=True)
class ChangePlan:
    """Complete, deterministic change plan."""

    request: str
    targets: tuple[TargetCandidate, ...]
    primary_target: TargetCandidate | None
    affected_files: tuple[PlanItem, ...]
    affected_symbols: tuple[str, ...]
    architecture: tuple[dict[str, Any], ...]
    tests: tuple[PlanItem, ...]
    inspection_order: tuple[PlanItem, ...]
    risk: str
    risk_factors: tuple[str, ...]
    summary: str
    stats: dict[str, Any]


# ---------------------------------------------------------------------------
# Request analysis
# ---------------------------------------------------------------------------


def _detect_action(request: str) -> str | None:
    """Detect the primary action verb in the request."""
    tokens = request.lower().split()
    for tok in tokens:
        tok_clean = re.sub(r"[^a-z]", "", tok)
        if tok_clean in _ACTION_VERBS:
            return _ACTION_VERBS[tok_clean]
    return None


def _extract_path_candidates(request: str) -> tuple[str, ...]:
    """Extract repository-path-like tokens from the request."""
    found = []
    for match in _PATH_RE.finditer(request):
        val = match.group()
        if "/" in val or "." in val:
            found.append(val)
    return tuple(sorted(set(found)))


def _extract_module_candidates(request: str) -> tuple[str, ...]:
    """Extract dotted-module-name tokens from the request."""
    found = []
    for match in _MODULE_RE.finditer(request):
        found.append(match.group())
    return tuple(sorted(set(found)))


def analyze_request(request: str) -> RequestAnalysis:
    """Deterministically extract structured signals from a change request.

    Returns a :class:`RequestAnalysis` with the detected action verb,
    tokenised terms, domain terms (non-stop-words), path candidates, module
    candidates, and symbol candidates.
    """
    action = _detect_action(request)
    raw_tokens = tokenize(request)
    domain_terms = tuple(
        t for t in raw_tokens if t not in _STOP_WORDS and len(t) > 1
    )
    path_candidates = _extract_path_candidates(request)
    module_candidates = _extract_module_candidates(request)
    # Symbol candidates: CamelCase words in the request.
    symbol_candidates = tuple(
        re.findall(r"\b([A-Z][a-zA-Z0-9]+)\b", request)
    )
    return RequestAnalysis(
        raw=request,
        action=action,
        tokens=tuple(raw_tokens),
        domain_terms=domain_terms,
        path_candidates=path_candidates,
        module_candidates=module_candidates,
        symbol_candidates=symbol_candidates,
    )


# ---------------------------------------------------------------------------
# Target discovery helpers
# ---------------------------------------------------------------------------


def _file_score(kind: str, base: int) -> int:
    """Score multiplier for match kind."""
    return {"exact_file": 100, "exact_module": 90, "exact_symbol": 85,
            "exact_package": 80, "search": 60, "architecture": 40,
            "dependency": 30}.get(kind, 10)


def _discover_targets(
    analysis: RequestAnalysis,
    explicit_target: str | None,
    *,
    root: Path,
    index: RepositoryIndex,
    searcher: CodeSearcher,
    symbol_index: SymbolIndex,
    arch_graph: ArchitectureGraph,
    config: ChangePlanConfig,
) -> list[TargetCandidate]:
    """Discover and rank candidate change targets."""
    candidates: dict[str, TargetCandidate] = {}

    def _add(target: str, kind: TargetKind, score: int, confidence: Confidence, *reasons: str) -> None:
        key = target
        existing = candidates.get(key)
        if existing:
            combined_score = max(existing.score, score)
            combined_reasons = tuple(dict.fromkeys(existing.reasons + reasons))
            candidates[key] = TargetCandidate(
                target=key, kind=existing.kind, score=combined_score,
                confidence=existing.confidence, reasons=combined_reasons,
            )
        else:
            candidates[key] = TargetCandidate(target=key, kind=kind, score=score,
                                              confidence=confidence, reasons=reasons)

    # 1. Explicit target (highest priority).
    if explicit_target:
        resolved = _resolve_explicit_target(explicit_target, root, index, symbol_index, arch_graph)
        if resolved:
            kind_str, rel_path, module, pkg = resolved
            if kind_str == "file":
                _add(str(rel_path), TargetKind.FILE, _file_score("exact_file", 100),
                     Confidence.CONFIRMED, "explicit file target")
            elif kind_str == "module":
                _add(module, TargetKind.MODULE, _file_score("exact_module", 90),
                     Confidence.CONFIRMED, "explicit module target")
            elif kind_str == "package":
                _add(pkg, TargetKind.PACKAGE, _file_score("exact_package", 80),
                     Confidence.CONFIRMED, "explicit package target")
            elif kind_str == "symbol":
                _add(module, TargetKind.MODULE, _file_score("exact_symbol", 85),
                     Confidence.CONFIRMED, f"explicit symbol target: {module}")
        else:
            # Could not resolve — still add as a possible target.
            _add(explicit_target, TargetKind.MODULE, 20, Confidence.POSSIBLE,
                 "explicit target (unresolved)")

    # 2. Path candidates from request analysis.
    for path_str in analysis.path_candidates:
        p = root / path_str
        if p.exists() and p.is_file():
            _add(str(p.relative_to(root)), TargetKind.FILE,
                 _file_score("exact_file", 95), Confidence.CONFIRMED,
                 f"file path in request: {path_str}")
        elif p.exists() and p.is_dir():
            _add(str(p.relative_to(root)), TargetKind.PACKAGE,
                 _file_score("exact_package", 80), Confidence.LIKELY,
                 f"package path in request: {path_str}")

    # 3. Module candidates from request analysis.
    for mod in analysis.module_candidates:
        arch_node = arch_graph.get_module_node(mod)
        if arch_node:
            _add(mod, TargetKind.MODULE, _file_score("exact_module", 85),
                 Confidence.CONFIRMED, f"module in request: {mod}")

    # 4. Symbol candidates from request analysis.
    for sym_name in analysis.symbol_candidates:
        syms = symbol_index.find(sym_name)
        if syms:
            for sym in syms[:3]:
                mod = _path_to_module(str(sym.file_path))
                _add(mod, TargetKind.MODULE, _file_score("exact_symbol", 85),
                     Confidence.CONFIRMED, f"symbol '{sym_name}' defined in {sym.file_path}")

    # 5. Domain-term lexical search.
    if analysis.domain_terms:
        query = " ".join(analysis.domain_terms)
        results = searcher.search(query, limit=config.search_limit)
        for sr in results[:10]:
            _add(str(sr.file_path), TargetKind.FILE, _file_score("search", 55),
                 Confidence.LIKELY, f"lexical match for '{query}' (score={sr.score})")

    # 6. Architecture signals from domain terms.
    if analysis.domain_terms:
        from repolens.architecture_retrieval import extract_architecture_signals
        query = " ".join(analysis.domain_terms)
        signals = extract_architecture_signals(query, arch_graph)
        seen_nodes: set[str] = set()
        for sig in signals[:15]:
            if sig.node.id in seen_nodes:
                continue
            seen_nodes.add(sig.node.id)
            if sig.node.kind == ArchitectureNodeKind.MODULE:
                _add(sig.node.id, TargetKind.MODULE, _file_score("architecture", 40),
                     Confidence.LIKELY, f"architecture signal: {sig.reason}")
            elif sig.node.kind == ArchitectureNodeKind.PACKAGE:
                _add(sig.node.id, TargetKind.PACKAGE, _file_score("architecture", 35),
                     Confidence.POSSIBLE, f"architecture signal: {sig.reason}")

    # 7. If nothing found yet, search for the whole request as a fallback.
    if not candidates:
        results = searcher.search(analysis.raw, limit=5)
        for sr in results[:3]:
            _add(str(sr.file_path), TargetKind.FILE, 25, Confidence.POSSIBLE,
                 "fallback lexical search for full request")

    ranked = sorted(candidates.values(), key=lambda c: (-c.score, c.target))
    return ranked[:config.max_target_candidates]


def _resolve_explicit_target(
    target: str, root: Path, index: RepositoryIndex,
    symbol_index: SymbolIndex, arch_graph: ArchitectureGraph,
) -> tuple[str, Path | None, str | None, str | None] | None:
    """Resolve an explicit target string to (kind, path, module, package).

    Returns relative paths (relative to ``root``); all graph/impact layers
    return paths relative to the repository root.
    """
    # Try as file path.
    p = root / target
    if p.exists() and p.is_file():
        return ("file", p.relative_to(root), None, None)
    if p.exists() and p.is_dir():
        return ("package", None, None, str(p.relative_to(root)))

    # Try as dotted module name.
    mod_path = target.replace(".", "/")
    py_file = root / (mod_path + ".py")
    init_file = root / mod_path / "__init__.py"
    if py_file.exists():
        return ("module", py_file.relative_to(root), target, None)
    if init_file.exists():
        return ("module", init_file.relative_to(root), target, None)

    # Try as symbol name.
    syms = symbol_index.find(target)
    if len(syms) == 1:
        sym = syms[0]
        return ("symbol", sym.file_path, _path_to_module(str(sym.file_path)), None)

    # Try as package in architecture graph.
    pkg = arch_graph.get_package_node(target)
    if pkg:
        return ("package", None, None, target)

    # Try as module in architecture graph.
    mod_node = arch_graph.get_module_node(target)
    if mod_node:
        return ("module", None, target, None)

    return None


# ---------------------------------------------------------------------------
# Impact enrichment
# ---------------------------------------------------------------------------


def _enrich_impact(
    targets: list[TargetCandidate],
    *,
    root: Path,
    index: RepositoryIndex,
    dependency_graph: DependencyGraph,
    call_graph: CallGraph,
    symbol_index: SymbolIndex,
    impact_analyzer: ImpactAnalyzer,
    config: ChangePlanConfig,
) -> list[PlanItem]:
    """Collect affected files from the impact analysis for each target."""
    items: list[PlanItem] = []
    seen: set[str] = set()
    for tc in targets[:config.max_primary_targets]:
        try:
            result = impact_analyzer.analyze(
                tc.target, max_depth=config.impact_max_depth,
                limit=config.max_impact_nodes,
            )
        except Exception:
            continue
        for ii in result.items:
            rel_path = str(ii.path)
            if rel_path in seen or rel_path == tc.target:
                continue
            seen.add(rel_path)
            cat = _impact_category(ii.relationship.value, ii.depth)
            items.append(PlanItem(
                path=rel_path,
                module=_path_to_module(rel_path),
                symbol=ii.symbol,
                category=cat,
                score=_impact_score(ii.relationship.value, ii.depth),
                confidence=Confidence.CONFIRMED if ii.confidence == "static" else Confidence.LIKELY,
                reason=ii.reason,
                relationship=ii.relationship.value,
                priority=0,  # assigned later
            ))
    return items[:config.max_affected_files]


def _pkg_of_module(module: str) -> str | None:
    """Return the top-level package of a dotted module id."""
    parts = module.split(".")
    if len(parts) >= 2:
        return ".".join(parts[:2])
    return None


def _impact_category(rel: str, depth: int) -> PlanCategory:
    """Map an impact relationship to a plan category."""
    mapping = {
        "direct_caller": PlanCategory.DIRECT_CALLER,
        "indirect_caller": PlanCategory.INDIRECT_CALLER,
        "direct_dependency": PlanCategory.DIRECT_DEPENDENCY,
        "indirect_dependency": PlanCategory.INDIRECT_DEPENDENCY,
        "direct_callee": PlanCategory.DIRECT_DEPENDENCY,
        "indirect_callee": PlanCategory.INDIRECT_DEPENDENCY,
        "test": PlanCategory.TEST,
        "configuration": PlanCategory.BROADER_NEIGHBOR,
        "api_consumer": PlanCategory.ARCHITECTURE_ENTRY,
        "reverse_dependency": PlanCategory.DIRECT_CALLER if depth == 1 else PlanCategory.INDIRECT_CALLER,
    }
    return mapping.get(rel, PlanCategory.BROADER_NEIGHBOR)


def _impact_score(rel: str, depth: int) -> int:
    """Score for an impact item."""
    base = {
        "direct_caller": 80, "indirect_caller": 50,
        "direct_dependency": 70, "indirect_dependency": 40,
        "direct_callee": 65, "indirect_callee": 35,
        "test": 75, "configuration": 30, "api_consumer": 60,
        "reverse_dependency": 55,
    }.get(rel, 20)
    return max(base - depth * 10, 10)


# ---------------------------------------------------------------------------
# Architecture enrichment
# ---------------------------------------------------------------------------


def _enrich_architecture(
    targets: list[TargetCandidate],
    *,
    arch_graph: ArchitectureGraph,
    subsystems: list[Subsystem],
    config: ChangePlanConfig,
) -> tuple[dict[str, Any], ...]:
    """Collect architecture metadata for each target."""
    results: list[dict[str, Any]] = []
    for tc in targets[:config.max_primary_targets]:
        node = (arch_graph.get_module_node(tc.target)
                or arch_graph.get_file_node(tc.target)
                or arch_graph.get_package_node(tc.target))
        if node is None:
            continue
        file_node = arch_graph.get_file_node(tc.target)
        if file_node is None:
            file_node = node.kind == ArchitectureNodeKind.FILE and node or None
        pkg_nodes = arch_graph.contained_by(file_node) if file_node else []
        pkg = next((n.id for n in pkg_nodes
                    if n.kind == ArchitectureNodeKind.PACKAGE), None)
        sub = subsystem_of(arch_graph, node, subsystems)
        deps = [n.id for n in arch_graph.dependencies_of(node)]
        dep_by = [n.id for n in arch_graph.dependents_of(node)]
        neighbors = [n.id for n in arch_graph.dependencies_of(node)]
        entry = sub.entry_modules[0] if sub and sub.entry_modules else None
        dep_pkgs = sorted({_pkg_of_module(d) for d in deps if _pkg_of_module(d)})
        dep_by_pkgs = sorted({_pkg_of_module(d) for d in dep_by if _pkg_of_module(d)})
        results.append({
            "target": tc.target,
            "kind": node.kind.value,
            "module": node.id,
            "package": pkg,
            "subsystem": sub.id if sub else None,
            "dependencies": deps,
            "dependents": dep_by,
            "neighbors": neighbors,
            "entry_module": entry,
            "dependency_packages": dep_pkgs,
            "dependent_packages": dep_by_pkgs,
        })
    return tuple(results)


# ---------------------------------------------------------------------------
# Test discovery
# ---------------------------------------------------------------------------


def _discover_tests(
    affected_items: list[PlanItem],
    targets: list[TargetCandidate],
    *,
    root: Path,
    arch_graph: ArchitectureGraph,
    dependency_graph: DependencyGraph,
    call_graph: CallGraph,
    subsystems: list[Subsystem],
    config: ChangePlanConfig,
) -> list[PlanItem]:
    """Find tests likely affected by the change."""
    test_items: list[PlanItem] = []
    seen: set[str] = set()
    affected_modules = {item.module for item in affected_items if item.module}
    affected_modules.update(tc.target for tc in targets if tc.kind == TargetKind.MODULE)
    for tc in targets:
        if tc.kind == TargetKind.FILE:
            affected_modules.add(_path_to_module(tc.target))
    test_files = [f for f in arch_graph.files() if "test" in f.id]
    for tf in test_files:
        rel = tf.id
        if rel in seen:
            continue
        reasons: list[str] = []
        tf_mod = rel.replace("/", ".").replace(".py", "")
        if tf_mod.endswith(".__init__"):
            tf_mod = tf_mod[:-9]
        # Check imports of affected modules.
        deps = arch_graph.dependencies_of(tf)
        dep_ids = {d.id for d in deps}
        imported_affected = affected_modules & dep_ids
        if imported_affected:
            reasons.append(f"test: imports affected module ({', '.join(sorted(imported_affected))})")
        # Check if test name matches affected module names.
        tf_name = Path(rel).stem
        for am in affected_modules:
            am_leaf = am.rsplit(".", 1)[-1] if "." in am else am
            if am_leaf and am_leaf in tf_name:
                reasons.append(f"test: matching name for '{am}'")
        # Same subsystem.
        tf_sub = subsystem_of(arch_graph, tf, subsystems)
        target_subs = set()
        for tc in targets:
            n = arch_graph.get_module_node(tc.target) or arch_graph.get_file_node(tc.target)
            if n:
                s = subsystem_of(arch_graph, n, subsystems)
                if s:
                    target_subs.add(s.id)
        if tf_sub and tf_sub.id in target_subs:
            reasons.append(f"test: same subsystem '{tf_sub.id}'")
        if reasons:
            cat = PlanCategory.TEST if imported_affected else PlanCategory.INDIRECT_TEST
            test_items.append(PlanItem(
                path=rel, module=tf_mod, symbol=None, category=cat,
                score=70 if imported_affected else 40,
                confidence=Confidence.CONFIRMED if imported_affected else Confidence.LIKELY,
                reason=reasons[0], relationship="test_of",
                priority=0,
            ))
            seen.add(rel)
    test_items.sort(key=lambda t: (-t.score, t.path))
    return test_items[:config.max_tests]


# ---------------------------------------------------------------------------
# Inspection ordering
# ---------------------------------------------------------------------------


def _build_inspection_order(
    primary: TargetCandidate | None,
    impact_items: list[PlanItem],
    test_items: list[PlanItem],
    arch_info: tuple[dict[str, Any], ...],
    config: ChangePlanConfig,
) -> tuple[PlanItem, ...]:
    """Deterministically prioritise items for inspection."""
    items: list[PlanItem] = []
    rank = 0
    # 1. Primary target.
    if primary:
        rank += 1
        items.append(PlanItem(
            path=primary.target,
            module=_path_to_module(primary.target) if "/" in primary.target else primary.target,
            symbol=None, category=PlanCategory.PRIMARY_TARGET,
            score=100, confidence=primary.confidence,
            reason=primary.reasons[0] if primary.reasons else "primary change target",
            relationship="primary", priority=rank,
        ))
    # 2. Direct callers.
    for item in impact_items:
        if item.category == PlanCategory.DIRECT_CALLER:
            rank += 1
            items.append(item._replace(priority=rank) if hasattr(item, '_replace') else PlanItem(
                path=item.path, module=item.module, symbol=item.symbol,
                category=item.category, score=item.score, confidence=item.confidence,
                reason=item.reason, relationship=item.relationship, priority=rank,
            ))
    # 3. Direct dependencies.
    for item in impact_items:
        if item.category == PlanCategory.DIRECT_DEPENDENCY:
            rank += 1
            items.append(PlanItem(
                path=item.path, module=item.module, symbol=item.symbol,
                category=item.category, score=item.score, confidence=item.confidence,
                reason=item.reason, relationship=item.relationship, priority=rank,
            ))
    # 4. Architecture entry points.
    for ai in arch_info:
        for dep_id in ai.get("dependents", [])[:3]:
            rank += 1
            items.append(PlanItem(
                path=dep_id, module=dep_id, symbol=None,
                category=PlanCategory.ARCHITECTURE_ENTRY, score=55,
                confidence=Confidence.LIKELY,
                reason=f"architecture dependent of {ai['target']}",
                relationship="architecture_dependent", priority=rank,
            ))
    # 5. Tests.
    for item in test_items:
        rank += 1
        items.append(PlanItem(
            path=item.path, module=item.module, symbol=item.symbol,
            category=item.category, score=item.score, confidence=item.confidence,
            reason=item.reason, relationship=item.relationship, priority=rank,
        ))
    # 6. Indirect callers / dependencies.
    for item in impact_items:
        if item.category in (PlanCategory.INDIRECT_CALLER, PlanCategory.INDIRECT_DEPENDENCY,
                             PlanCategory.BROADER_NEIGHBOR, PlanCategory.SUBSYSTEM_NEIGHBOR):
            rank += 1
            items.append(PlanItem(
                path=item.path, module=item.module, symbol=item.symbol,
                category=item.category, score=item.score, confidence=item.confidence,
                reason=item.reason, relationship=item.relationship, priority=rank,
            ))
    return tuple(items[:config.max_inspection_items])


# ---------------------------------------------------------------------------
# Risk assessment
# ---------------------------------------------------------------------------


def _assess_risk(
    impact_items: list[PlanItem],
    test_items: list[PlanItem],
    targets: list[TargetCandidate],
    arch_info: tuple[dict[str, Any], ...],
) -> tuple[str, tuple[str, ...]]:
    """Deterministically compute risk level and contributing factors."""
    factors: list[str] = []
    score = 0
    n_affected = len(impact_items)
    n_callers = sum(1 for i in impact_items if i.category in (PlanCategory.DIRECT_CALLER, PlanCategory.INDIRECT_CALLER))
    n_dependents = sum(1 for i in impact_items if i.relationship in ("reverse_dependency",))
    n_tests = len(test_items)
    n_packages = len(
        {p for ai in arch_info for p in (
            [ai.get("package") if ai.get("package") else None]
            + list(ai.get("dependency_packages", ()))
            + list(ai.get("dependent_packages", ()))
        ) if p}
    )
    n_subsystems = len({ai.get("subsystem") for ai in arch_info if ai.get("subsystem")})
    if n_affected > 20:
        factors.append(f"many affected files ({n_affected})")
        score += 3
    elif n_affected > 10:
        factors.append(f"moderate affected files ({n_affected})")
        score += 2
    elif n_affected > 5:
        factors.append(f"some affected files ({n_affected})")
        score += 1
    if n_callers > 5:
        factors.append(f"many callers ({n_callers})")
        score += 2
    elif n_callers > 2:
        factors.append(f"some callers ({n_callers})")
        score += 1
    if n_dependents > 3:
        factors.append(f"many dependents ({n_dependents})")
        score += 2
    elif n_dependents > 0:
        factors.append(f"some dependents ({n_dependents})")
        score += 1
    if n_packages > 4:
        factors.append(f"cross-package change ({n_packages} packages)")
        score += 2
    elif n_packages > 1:
        factors.append(f"cross-package change ({n_packages} packages)")
        score += 1
    if n_subsystems > 1:
        factors.append(f"cross-subsystem change ({n_subsystems} subsystems)")
        score += 3
    has_public = any(
        ai.get("entry_module") and ai.get("module") == ai.get("entry_module")
        for ai in arch_info
    )
    if has_public:
        factors.append("public/API-facing symbols affected")
        score += 2
    if n_tests == 0:
        factors.append("no tests identified")
        score += 1
    elif n_tests < 2:
        factors.append("limited test coverage")
        score += 1
    unresolved = sum(
        1 for i in impact_items
        if i.reason and "unresolved" in i.reason.lower()
    )
    if unresolved > 0:
        factors.append(f"unresolved references ({unresolved})")
        score += 1
    if score >= 6:
        level = "high"
    elif score >= 3:
        level = "medium"
    else:
        level = "low"
    return level, tuple(factors)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _path_to_module(path: str) -> str:
    """Convert a file path to a dotted module name."""
    mod = path.replace("/", ".").replace(".py", "")
    if mod.endswith(".__init__"):
        mod = mod[:-9]
    return mod


def _build_summary(
    request: str, primary: TargetCandidate | None,
    n_affected: int, n_tests: int, risk: str,
) -> str:
    """Build a one-line summary."""
    target_desc = primary.target if primary else "multiple candidates"
    return (
        f"Change plan for '{request}': "
        f"primary target={target_desc}, "
        f"affected_files={n_affected}, tests={n_tests}, risk={risk}"
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _build_shared_index(root: Path) -> RepositoryIndex:
    """Build the shared incremental index for ``root``.

    Uses a persistent cache below the RepoLens cache base when caching is
    enabled, otherwise an ephemeral in-memory cache — mirroring the MCP
    launcher so a warm engine re-uses previously parsed files.
    """
    if os.environ.get("REPOLENS_CACHE_DISABLED") or os.environ.get("REPOLENS_CACHE_DIR") == "":
        return IncrementalIndexBuilder(root, persist=False).build()
    from repolens.embedding_cache import repository_identity
    from repolens.incremental_index import home_cache_base

    cache_dir = home_cache_base() / "index" / repository_identity(root)
    return IncrementalIndexBuilder(root, cache_dir=cache_dir).build()


class ChangePlanEngine:
    """Deterministic change-plan engine.

    Accepts a natural-language change request (and optional explicit target)
    and produces a bounded, explainable :class:`ChangePlan`.

    Reuses the existing shared repository index — never re-parses.  All
    traversal is bounded by :class:`ChangePlanConfig`.
    """

    def __init__(
        self,
        root: Path | str,
        *,
        index: RepositoryIndex | None = None,
        dependency_graph: DependencyGraph | None = None,
        symbol_index: SymbolIndex | None = None,
        call_graph: CallGraph | None = None,
        arch_graph: ArchitectureGraph | None = None,
        subsystems: list[Subsystem] | None = None,
        impact_analyzer: ImpactAnalyzer | None = None,
        searcher: CodeSearcher | None = None,
        config: ChangePlanConfig | None = None,
    ) -> None:
        self._root = Path(root).resolve()
        self._config = config or ChangePlanConfig()
        start = time.perf_counter()
        # Build or accept the shared index (persistent cache by default so a
        # second engine instance warm-reuses previously parsed files).
        self._index = index or _build_shared_index(self._root)
        self._parsed_files = self._index.stats.files_parsed
        self._dep_graph = dependency_graph or DependencyGraphBuilder(self._root, index=self._index).build()
        self._symbol_index = symbol_index or SymbolIndexBuilder(self._root, index=self._index).build()
        ref_idx = ReferenceIndexBuilder(self._root, index=self._index).build()
        self._call_graph = call_graph or CallGraphBuilder(
            self._root, index=self._index, reference_index=ref_idx,
            symbol_index=self._symbol_index,
        ).build()
        self._arch_graph = arch_graph or ArchitectureGraphBuilder(
            self._root, index=self._index, graph=self._dep_graph,
        ).build()
        self._subsystems = subsystems if subsystems is not None else discover_subsystems(self._arch_graph)
        self._impact_analyzer = impact_analyzer or ImpactAnalyzer(
            self._root, index=self._index, graph=self._dep_graph,
            symbol_index=self._symbol_index, reference_graph=self._call_graph,
        )
        self._searcher = searcher or CodeSearcher(self._root, index=self._index)
        self._build_time = time.perf_counter() - start

    def plan(self, request: str, *, target: str | None = None) -> ChangePlan:
        """Produce a deterministic change plan for ``request``.

        Parameters
        ----------
        request : str
            Natural-language change request.
        target : str, optional
            An explicit target (file, module, package, or symbol name).

        Returns
        -------
        ChangePlan
            A bounded, explainable plan with risk assessment.
        """
        cfg = self._config

        # 1. Request analysis.
        analysis = analyze_request(request)

        # 2. Target discovery.
        targets = _discover_targets(
            analysis, target,
            root=self._root, index=self._index, searcher=self._searcher,
            symbol_index=self._symbol_index, arch_graph=self._arch_graph,
            config=cfg,
        )
        primary = targets[0] if targets else None

        # 3. Impact enrichment.
        impact_items = _enrich_impact(
            targets,
            root=self._root, index=self._index,
            dependency_graph=self._dep_graph, call_graph=self._call_graph,
            symbol_index=self._symbol_index, impact_analyzer=self._impact_analyzer,
            config=cfg,
        )

        # 4. Architecture enrichment.
        arch_info = _enrich_architecture(
            targets, arch_graph=self._arch_graph,
            subsystems=self._subsystems, config=cfg,
        )

        # 5. Test discovery.
        test_items = _discover_tests(
            impact_items, targets,
            root=self._root, arch_graph=self._arch_graph,
            dependency_graph=self._dep_graph, call_graph=self._call_graph,
            subsystems=self._subsystems, config=cfg,
        )

        # 6. Affected symbols.
        affected_symbols = tuple(
            sorted({
                item.symbol for item in impact_items if item.symbol
            } | {tc.target for tc in targets if tc.kind == TargetKind.SYMBOL})
        )

        # 7. Inspection order.
        inspection_order = _build_inspection_order(
            primary, impact_items, test_items, arch_info, cfg,
        )

        # 8. Risk.
        risk, risk_factors = _assess_risk(impact_items, test_items, targets, arch_info)

        # 9. Summary.
        summary = _build_summary(request, primary, len(impact_items), len(test_items), risk)

        return ChangePlan(
            request=request,
            targets=tuple(targets),
            primary_target=primary,
            affected_files=tuple(impact_items),
            affected_symbols=affected_symbols,
            architecture=arch_info,
            tests=tuple(test_items),
            inspection_order=inspection_order,
            risk=risk,
            risk_factors=risk_factors,
            summary=summary,
            stats={
                "target_count": len(targets),
                "affected_file_count": len(impact_items),
                "test_count": len(test_items),
                "inspection_item_count": len(inspection_order),
                "arch_info_count": len(arch_info),
                "parsed_files": self._parsed_files,
                "build_time": round(self._build_time, 4),
            },
        )


def plan_to_context_candidates(
    plan: ChangePlan,
    *,
    limit: int | None = None,
    include_tests: bool = True,
    include_dependencies: bool = True,
    include_callers: bool = True,
    include_callees: bool = True,
    include_architecture: bool = True,
) -> list[dict]:
    """Reusable helper converting a change plan into context-candidate dicts.

    Additive and unused by ``get_context``.  M24.2 integrates change plans
    into context retrieval; the filtering flags let a caller disable whole
    change-plan categories without touching normal retrieval.  Output is
    ordered by inspection priority, bounded by ``limit``, and deterministic.

    Each candidate dict carries ``path``, ``module``, ``symbol`` when known,
    ``score``, ``category``, ``reason``, ``confidence``, ``relationship``, and
    ``priority``.  The ``primary_target`` change-plan item is always retained;
    the inspectable/enricher items are filtered through the ``include_*``
    flags.  The original call surface (``include_tests`` only) is unchanged.
    """
    items = list(plan.inspection_order)

    def _keep(item: PlanItem) -> bool:
        if item.category == PlanCategory.PRIMARY_TARGET:
            return True
        if item.category == PlanCategory.TEST or item.category == PlanCategory.INDIRECT_TEST:
            return include_tests
        if item.category in (PlanCategory.DIRECT_CALLER, PlanCategory.INDIRECT_CALLER):
            return include_callers
        if item.relationship in ("direct_callee", "indirect_callee"):
            return include_callees
        if item.category in (PlanCategory.DIRECT_DEPENDENCY, PlanCategory.INDIRECT_DEPENDENCY):
            return include_dependencies
        if item.category in (
            PlanCategory.ARCHITECTURE_ENTRY,
            PlanCategory.SUBSYSTEM_NEIGHBOR,
            PlanCategory.BROADER_NEIGHBOR,
        ):
            return include_architecture
        return True  # unknown/future categories are retained conservatively

    items = [i for i in items if _keep(i)]
    if limit is not None:
        items = items[:limit]
    return [
        {
            "path": item.path,
            "module": item.module,
            "symbol": item.symbol,
            "score": item.score,
            "priority": item.priority,
            "category": item.category.value,
            "confidence": item.confidence.value,
            "reason": item.reason,
            "relationship": item.relationship,
        }
        for item in items
    ]
