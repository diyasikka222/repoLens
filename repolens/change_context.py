"""Change-aware context support (Milestone 24.2).

This module bridges :class:`~repolens.change_plan.ChangePlanEngine` and the
existing :class:`~repolens.context.ContextEngine` without duplicating either:

- :func:`plan_to_change_candidates` converts a :class:`~repolens.change_plan.ChangePlan`
  into the existing :class:`~repolens.context.ContextCandidate` model
  (no parallel candidate model) with explicit, per-file reasons;
- :class:`ChangeContextOptions` exposes deterministic category filters that
  never touch normal retrieval behaviour;
- :func:`merge_change_candidates` appends change-plan candidates after
  retrieval/dependency/architecture candidates so the engine's dedupe keeps
  the higher-tier role for overlapping files;
- :func:`explain_change_context` answers "why was this file selected?" as a
  deterministic dict (change plan → ranking → budget → firewall);
- :func:`plan_response_payload` renders a plan as a structured, JSON-safe
  response for the MCP ``change_plan`` tool.

All conversion is bounded by the plan/context limits, deterministic, and
reuses only existing structures and existing ranking/budget/firewall stages.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from repolens.change_plan import ChangePlan, PlanCategory, analyze_request
from repolens.context.candidate import (
    INCLUSION_CHANGE_PLAN,
    CandidateRole,
    ContextCandidate,
)
from repolens.context.package import ContextPackage
from repolens.context.tokens import estimate_tokens
from repolens.context.firewall.safe_package import SafeContextPackage

#: Plan categories whose files *use / call / import* the change target.
_DEPENDENT_CATEGORIES = frozenset(
    {
        PlanCategory.DIRECT_CALLER,
        PlanCategory.INDIRECT_CALLER,
        PlanCategory.DIRECT_DEPENDENCY,
        PlanCategory.INDIRECT_DEPENDENCY,
        PlanCategory.TEST,
        PlanCategory.INDIRECT_TEST,
    }
)

#: Plan categories whose files the *change target uses*, plus the primary
#: change target itself.
_DEPENDENCY_CATEGORIES = frozenset(
    {
        PlanCategory.PRIMARY_TARGET,
        PlanCategory.ARCHITECTURE_ENTRY,
        PlanCategory.SUBSYSTEM_NEIGHBOR,
        PlanCategory.BROADER_NEIGHBOR,
    }
)

#: Human-readable explanation of each change-plan category (M24.2).
CATEGORY_EXPLANATIONS: dict[str, str] = {
    "primary_target": "primary change target",
    "direct_caller": "direct caller of the change target",
    "indirect_caller": "indirect caller of the change target",
    "direct_dependency": "direct dependency of the change target",
    "indirect_dependency": "indirect dependency of the change target",
    "architecture_entry": "architecture entry point / API consumer",
    "test": "affected test",
    "indirect_test": "indirectly affected test",
    "subsystem_neighbor": "same-subsystem neighbor",
    "broader_neighbor": "broader neighbourhood / configuration",
}


@dataclass(frozen=True)
class ChangeContextOptions:
    """Deterministic category filters for the change-plan layer.

    These flags only control which change-plan categories are *introduced* by
    the change-plan layer; normal retrieval behaviour is untouched regardless
    of their values.
    """

    include_tests: bool = True
    include_dependencies: bool = True
    include_callers: bool = True
    include_callees: bool = True
    include_architecture: bool = True
    #: Hard cap on change-plan candidates; ``None`` means the plan bound.
    max_files: int | None = None


def _candidate_role(item) -> CandidateRole:
    """Map a plan item to a context-engine candidate role."""
    if item.category in _DEPENDENT_CATEGORIES:
        return CandidateRole.DEPENDENT
    return CandidateRole.DEPENDENCY


def _item_kept(item, options: ChangeContextOptions) -> bool:
    """Apply the deterministic category filters to one plan item."""
    category = item.category
    if category == PlanCategory.PRIMARY_TARGET:
        return True
    if category in (PlanCategory.TEST, PlanCategory.INDIRECT_TEST):
        return options.include_tests
    if category in (PlanCategory.DIRECT_CALLER, PlanCategory.INDIRECT_CALLER):
        return options.include_callers
    if item.relationship in ("direct_callee", "indirect_callee"):
        return options.include_callees
    if category in (PlanCategory.DIRECT_DEPENDENCY, PlanCategory.INDIRECT_DEPENDENCY):
        return options.include_dependencies
    return options.include_architecture  # arch/subsystem/broader neighbour


def plan_to_change_candidates(
    plan: ChangePlan,
    *,
    options: ChangeContextOptions | None = None,
    root: Path | str | None = None,
) -> list[ContextCandidate]:
    """Convert ``plan`` inspection items into :class:`ContextCandidate` objects.

    One candidate per surviving inspection item, ordered by plan priority and
    carrying the change-plan signals that explain why it was introduced.
    ``root`` (the repository root used to read the candidate source text) is
    required; items whose source file cannot be read are skipped so a deleted
    or missing file never becomes a phantom candidate.
    """
    if root is None:
        root = Path(".")
    options = options if options is not None else ChangeContextOptions()
    root_path = Path(root)

    candidates: list[ContextCandidate] = []
    for item in plan.inspection_order:
        if not _item_kept(item, options):
            continue
        path = Path(item.path)
        if not (root_path / path).is_file():
            continue
        try:
            source = (root_path / path).read_text(encoding="utf-8")
        except (OSError, ValueError):
            continue
        candidates.append(
            ContextCandidate(
                path=path,
                source=source,
                role=_candidate_role(item),
                estimated_tokens=estimate_tokens(source),
                selection_reason=f"change-plan: {item.reason}",
                inclusion_reason=INCLUSION_CHANGE_PLAN,
                module=item.module,
                symbol=item.symbol,
                change_score=item.score,
                change_category=item.category.value,
                change_confidence=item.confidence.value,
                change_relationship=item.relationship,
                change_priority=item.priority,
            )
        )
        if options.max_files is not None and len(candidates) >= options.max_files:
            break
    return candidates


def merge_change_candidates(
    existing: list[ContextCandidate],
    change_candidates: list[ContextCandidate],
) -> list[ContextCandidate]:
    """Append change-plan candidates after all existing candidates.

    The engine deduplicates on first occurrence with primary, dependency, and
    architecture candidates added first, so a file introduced by both layers
    keeps its higher-tier retrieval/dependency/architecture role while
    change-plan metadata for surviving change-only files is retained.
    """
    return list(existing) + list(change_candidates)


def explain_change_context(
    plan: ChangePlan | None,
    package: ContextPackage | None,
    path: str | Path,
    *,
    safe_package: SafeContextPackage | None = None,
) -> dict:
    """Explain why ``path`` was (or was not) selected by a change-aware build.

    Deterministic, structured answer to: why was this file selected, what
    target caused it, what relationship does it have, what evidence supports
    inclusion, what priority did it receive, did it survive final ranking,
    did it survive context budgeting, and was it filtered by the firewall.
    No causal relationship is claimed without plan/candidate evidence.
    """
    path_str = str(path)
    rel_path = Path(path_str)

    plan_item = None
    if plan is not None:
        plan_item = next(
            (i for i in plan.inspection_order if i.path == path_str),
            None,
        )

    candidate = None
    selected = False
    excluded = None
    if package is not None:
        by_path = {c.path: c for c in package.selected_files}
        candidate = by_path.get(rel_path)
        selected = candidate is not None
        excluded = next(
            (e for e in package.excluded_candidates if e.path == rel_path),
            None,
        )

    category = plan_item.category.value if plan_item is not None else None
    relationship = (
        candidate.change_relationship
        if candidate is not None and candidate.change_relationship is not None
        else (plan_item.relationship if plan_item is not None else None)
    )
    priority = (
        candidate.change_priority
        if candidate is not None and candidate.change_priority is not None
        else (plan_item.priority if plan_item is not None else None)
    )

    survived_ranking = False
    survived_budget = False
    budget_status = "not_considered"
    if selected:
        survived_ranking = True
        survived_budget = True
        budget_status = "selected"
    elif excluded is not None:
        survived_ranking = True
        survived_budget = False
        budget_status = f"excluded:{excluded.reason}"

    filtered = "unknown"
    if safe_package is not None:
        ok = {c.path for c in safe_package.safe_files}
        blocked = {c.path for c in safe_package.blocked_files}
        if path_str in ok:
            decision = next(
                (c.decision for c in safe_package.safe_files if c.path == path_str),
                "allowed",
            )
            filtered = decision if decision in ("allowed", "redact") else "allowed"
        elif path_str in blocked:
            filtered = "blocked"
        else:
            filtered = "not_present"

    change_plan_source = candidate is not None and any(
        c.path == rel_path for c in package.change_candidates
    )

    evidence = None
    if plan_item is not None:
        evidence = plan_item.reason
    elif candidate is not None:
        evidence = candidate.selection_reason

    target = (
        plan.primary_target.target
        if plan is not None and plan.primary_target is not None
        else None
    )

    return {
        "path": path_str,
        "in_plan": plan_item is not None,
        "category": category,
        "category_explanation": (
            CATEGORY_EXPLANATIONS.get(category) if category is not None else None
        ),
        "relationship": relationship,
        "priority": priority,
        "confidence": (
            plan_item.confidence.value if plan_item is not None else None
        ),
        "target": target,
        "evidence": evidence,
        "selected": selected,
        "survived_ranking": survived_ranking,
        "survived_budget": survived_budget,
        "budget_status": budget_status,
        "role": candidate.role.value if candidate is not None else None,
        "inclusion_reason": (
            candidate.inclusion_reason if candidate is not None else None
        ),
        "change_plan_source": change_plan_source,
        "firewall": filtered,
    }


def plan_response_payload(plan: ChangePlan, request: str) -> dict:
    """Render a :class:`ChangePlan` as a structured, JSON-safe response dict.

    Groups affected files into callers/callees/dependencies/dependents, keeps
    every value JSON-serializable, and never exposes internal Python objects.
    The analysis/signals block is re-derived with the same deterministic
    :func:`~repolens.change_plan.analyze_request` the engine used.
    """
    analysis = analyze_request(request)
    return {
        "request": request,
        "analysis": {
            "action": analysis.action,
            "domain_terms": list(analysis.domain_terms),
            "path_candidates": list(analysis.path_candidates),
            "module_candidates": list(analysis.module_candidates),
            "symbol_candidates": list(analysis.symbol_candidates),
        },
        "primary_target": _target_payload(plan.primary_target),
        "target_candidates": [_target_payload(t) for t in plan.targets],
        "affected_files": [_item_payload(i) for i in plan.affected_files],
        "affected_symbols": list(plan.affected_symbols),
        "callers": [
            _item_payload(i)
            for i in plan.affected_files
            if i.category in (PlanCategory.DIRECT_CALLER, PlanCategory.INDIRECT_CALLER)
        ],
        "callees": [
            _item_payload(i)
            for i in plan.affected_files
            if i.relationship in ("direct_callee", "indirect_callee")
        ],
        "dependencies": [
            _item_payload(i)
            for i in plan.affected_files
            if i.category in (PlanCategory.DIRECT_DEPENDENCY, PlanCategory.INDIRECT_DEPENDENCY)
            and i.relationship not in ("direct_callee", "indirect_callee")
        ],
        "dependents": [
            _item_payload(i)
            for i in plan.affected_files
            if i.relationship in ("reverse_dependency",)
        ],
        "architecture": [dict(a) for a in plan.architecture],
        "tests": [_item_payload(i) for i in plan.tests],
        "inspection_order": [_item_payload(i) for i in plan.inspection_order],
        "risk": plan.risk,
        "risk_factors": list(plan.risk_factors),
        "confidence": (
            plan.primary_target.confidence.value
            if plan.primary_target is not None
            else "unresolved"
        ),
        "summary": plan.summary,
        "statistics": dict(plan.stats),
        "deterministic": True,
    }


def _target_payload(target) -> dict | None:
    if target is None:
        return None
    return {
        "target": target.target,
        "kind": target.kind.value,
        "score": target.score,
        "confidence": target.confidence.value,
        "reasons": list(target.reasons),
    }


def _item_payload(item) -> dict:
    return {
        "path": item.path,
        "module": item.module,
        "symbol": item.symbol,
        "category": item.category.value,
        "score": item.score,
        "confidence": item.confidence.value,
        "reason": item.reason,
        "relationship": item.relationship,
        "priority": item.priority,
    }