"""Structured change-surface scenario diagnostics (Milestone 23, step 2).

Scenario evaluation model for the deterministic change-surface benchmark. A
:class:`ChangeScenario` fixes a set of change targets, a direction, and an
explicit traversal depth; :class:`ExpectedHit` pins exact relationships that
*must* be discovered (path + relationship + direction + depth). An
"exact" scenario additionally requires the surfaced item set to match the
expected set *precisely*, so false/unexpected relationships are reported.

Measurement only: this module never modifies retrieval, ranking, budget,
change-surface semantics, or the MCP contract. It is fully offline and
deterministic — the same repository snapshot and scenario catalog always
produce the same :class:`ScenarioResult` / report.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from repolens.change_surface import (
    ChangeRelationship,
    ChangeSurfaceAnalyzer,
    ChangeSurfaceResult,
    SurfaceDirection,
)
from repolens.impact import ImpactTargetError


class ScenarioOutcome(str, Enum):
    """Final status of a single scenario."""

    PASS = "pass"
    FAIL = "fail"


class ErrorKind(str, Enum):
    """Classification of expected target-resolution failures."""

    UNKNOWN = "unknown"
    AMBIGUOUS = "ambiguous"


#: The canonical item sort policy used to check deterministic ordering.
#: Mirrors ``ChangeSurfaceAnalyzer``'s output ordering (direction, relationship,
#: depth, path, symbol, reason).
def item_sort_key(item) -> tuple:
    return (
        item.direction.value,
        item.relationship.value,
        item.depth,
        item.path.as_posix(),
        item.symbol or "",
        item.reason,
    )


@dataclass(frozen=True)
class ExpectedHit:
    """One relationship that must (or must not) appear on a surface."""

    path: str
    relationship: ChangeRelationship
    direction: SurfaceDirection
    depth: int | None = None


@dataclass(frozen=True)
class ExpectedEvidence:
    """A stable evidence tag an item's ``evidence`` tuple must contain."""

    path: str
    relationship: ChangeRelationship
    direction: SurfaceDirection
    evidence_contains: str
    symbol: str | None = None


@dataclass(frozen=True)
class ChangeScenario:
    """A fixed, deterministic change-surface scenario.

    Attributes:
        id: Stable, machine-readable identifier (e.g. ``file_level``).
        description: One-line human description of what the scenario proves.
        targets: A single target string or a sequence of targets.
        direction: ``"incoming"``, ``"outgoing"``, or ``"both"``.
        max_depth: Traversal depth passed to the analyzer.
        expected: Relationships that must be discovered.
        evidence_checks: Stable evidence-tag assertions on expected items
            (for example ``"base_class"`` on an inheritance item).
        exact: When ``True`` the surfaced item set must equal ``expected``
            exactly (missing *and* unexpected relationships are failures).
        expect_error: Expected target-resolution failure:
            :attr:`ErrorKind.UNKNOWN` or :attr:`ErrorKind.AMBIGUOUS`. ``None``
            means the target must resolve.
    """

    id: str
    description: str
    targets: str | tuple[str, ...]
    direction: str = "both"
    max_depth: int = 2
    expected: tuple[ExpectedHit, ...] = ()
    evidence_checks: tuple[ExpectedEvidence, ...] = ()
    exact: bool = False
    expect_error: ErrorKind | None = None


@dataclass(frozen=True)
class ScenarioResult:
    """One scenario's full, deterministic outcome."""

    id: str
    description: str
    target: str
    direction: str
    max_depth: int
    resolved_target: str | None
    resolved_kinds: tuple[str, ...]
    direct_affected: int
    transitive_affected: int
    total_affected_files: int
    relationship_types: tuple[str, ...]
    affected_paths: tuple[str, ...]
    deterministic_order: bool
    missing_expected: tuple[ExpectedHit, ...]
    missing_evidence: tuple[ExpectedEvidence, ...] = ()
    unexpected: tuple[str, ...] = ()
    error_kind: ErrorKind | None = None
    error_message: str | None = None
    outcome: ScenarioOutcome = ScenarioOutcome.PASS

    @property
    def total_affected_items(self) -> int:
        return self.direct_affected + self.transitive_affected

    @property
    def failed(self) -> bool:
        return self.outcome is ScenarioOutcome.FAIL

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "description": self.description,
            "target": self.target,
            "direction": self.direction,
            "max_depth": self.max_depth,
            "resolved_target": self.resolved_target,
            "resolved_kinds": list(self.resolved_kinds),
            "direct_affected": self.direct_affected,
            "transitive_affected": self.transitive_affected,
            "total_affected_files": self.total_affected_files,
            "total_affected_items": self.total_affected_items,
            "relationship_types": list(self.relationship_types),
            "affected_paths": list(self.affected_paths),
            "deterministic_order": self.deterministic_order,
            "missing_expected": [
                {
                    "path": hit.path,
                    "relationship": hit.relationship.value,
                    "direction": hit.direction.value,
                    "depth": hit.depth,
                }
                for hit in self.missing_expected
            ],
            "missing_evidence": [
                {
                    "path": check.path,
                    "relationship": check.relationship.value,
                    "direction": check.direction.value,
                    "evidence_contains": check.evidence_contains,
                    "symbol": check.symbol,
                }
                for check in self.missing_evidence
            ],
            "unexpected": list(self.unexpected),
            "error_kind": self.error_kind.value if self.error_kind else None,
            "error_message": self.error_message,
            "outcome": self.outcome.value,
        }


@dataclass(frozen=True)
class ChangeSurfaceScenarioReport:
    """Aggregate report over a scenario catalog, with repeatability."""

    scenarios: tuple[ScenarioResult, ...]
    deterministic_repeatable: bool
    repeatability_mismatch: str | None
    summary: dict

    @property
    def passed(self) -> int:
        return sum(1 for scenario in self.scenarios if not scenario.failed)

    @property
    def failed_scenarios(self) -> tuple[ScenarioResult, ...]:
        return tuple(scenario for scenario in self.scenarios if scenario.failed)

    def to_dict(self) -> dict:
        return {
            "deterministic_repeatable": self.deterministic_repeatable,
            "repeatability_mismatch": self.repeatability_mismatch,
            "summary": dict(self.summary),
            "scenarios": [scenario.to_dict() for scenario in self.scenarios],
        }


def run_change_scenario(
    analyzer: ChangeSurfaceAnalyzer,
    scenario: ChangeScenario,
) -> ScenarioResult:
    """Evaluate one :class:`ChangeScenario` against ``analyzer``.

    Deterministic: repeated evaluation of the same scenario over the same
    repository snapshot yields an identical :class:`ScenarioResult`.
    """
    description = scenario.description
    target = _target_label(scenario.targets)
    try:
        result = analyzer.analyze(
            scenario.targets,
            direction=scenario.direction,
            max_depth=scenario.max_depth,
        )
    except ImpactTargetError as exc:
        if scenario.expect_error is None:
            return ScenarioResult(
                id=scenario.id,
                description=description,
                target=target,
                direction=scenario.direction,
                max_depth=scenario.max_depth,
                resolved_target=None,
                resolved_kinds=(),
                direct_affected=0,
                transitive_affected=0,
                total_affected_files=0,
                relationship_types=(),
                affected_paths=(),
                deterministic_order=True,
                missing_expected=scenario.expected,
                unexpected=(),
                error_kind=ErrorKind.UNKNOWN,
                error_message=str(exc),
                outcome=ScenarioOutcome.FAIL,
            )
        kind = _classify_error(str(exc))
        if kind is not scenario.expect_error:
            return ScenarioResult(
                id=scenario.id,
                description=description,
                target=target,
                direction=scenario.direction,
                max_depth=scenario.max_depth,
                resolved_target=None,
                resolved_kinds=(),
                direct_affected=0,
                transitive_affected=0,
                total_affected_files=0,
                relationship_types=(),
                affected_paths=(),
                deterministic_order=True,
                missing_expected=(),
                unexpected=(),
                error_kind=kind,
                error_message=str(exc),
                outcome=ScenarioOutcome.FAIL,
            )
        return ScenarioResult(
            id=scenario.id,
            description=description,
            target=target,
            direction=scenario.direction,
            max_depth=scenario.max_depth,
            resolved_target=None,
            resolved_kinds=(),
            direct_affected=0,
            transitive_affected=0,
            total_affected_files=0,
            relationship_types=(),
            affected_paths=(),
            deterministic_order=True,
            missing_expected=(),
            unexpected=(),
            error_kind=kind,
            error_message=str(exc),
            outcome=ScenarioOutcome.PASS,
        )

    if scenario.expect_error is not None:
        return ScenarioResult(
            id=scenario.id,
            description=description,
            target=target,
            direction=scenario.direction,
            max_depth=scenario.max_depth,
            resolved_target=_resolved_labels(result),
            resolved_kinds=_resolved_kinds(result),
            direct_affected=len(result.directly_affected),
            transitive_affected=len(result.transitively_affected),
            total_affected_files=len(result.files),
            relationship_types=_relationship_types(result),
            affected_paths=_posix_paths(result.files),
            deterministic_order=True,
            missing_expected=(),
            unexpected=(),
            error_kind=None,
            error_message=(
                "target resolved but an error outcome was expected "
                f"(kind={scenario.expect_error.value})"
            ),
            outcome=ScenarioOutcome.FAIL,
        )

    ordered = _is_deterministically_ordered(result)
    present = {_hit_of(item): item.depth for item in result.items}
    by_signature = {_hit_of(item): item for item in result.items}
    missing = tuple(
        hit
        for hit in scenario.expected
        if _hit_of(hit) not in present
        or (hit.depth is not None and present.get(_hit_of(hit)) != hit.depth)
    )

    missing_evidence = tuple(
        check
        for check in scenario.evidence_checks
        if not _evidence_satisfied(by_signature, check)
    )

    unexpected: list[str] = []
    if scenario.exact:
        expected_keys = {_hit_of(hit) for hit in scenario.expected}
        unexpected = sorted(
            (path for path in present if path not in expected_keys),
            key=lambda key: (key[1], key[2], key[0]),
        )

    failed = bool(missing) or bool(missing_evidence) or bool(unexpected) or not ordered
    return ScenarioResult(
        id=scenario.id,
        description=description,
        target=target,
        direction=scenario.direction,
        max_depth=scenario.max_depth,
        resolved_target=_resolved_labels(result),
        resolved_kinds=_resolved_kinds(result),
        direct_affected=len(result.directly_affected),
        transitive_affected=len(result.transitively_affected),
        total_affected_files=len(result.files),
        relationship_types=_relationship_types(result),
        affected_paths=_posix_paths(result.files),
        deterministic_order=ordered,
        missing_expected=missing,
        missing_evidence=missing_evidence,
        unexpected=tuple(unexpected),
        outcome=ScenarioOutcome.FAIL if failed else ScenarioOutcome.PASS,
    )


def summarize_report(
    scenarios: tuple[ScenarioResult, ...],
    *,
    deterministic_repeatable: bool,
    repeatability_mismatch: str | None,
) -> ChangeSurfaceScenarioReport:
    """Build the compact aggregate report for a scenario catalog run."""
    passed = 0
    direct = 0
    transitive = 0
    items = 0
    files = 0
    resolved_targets = 0
    unknown = 0
    ambiguous = 0
    relationship_counts: dict[str, int] = {}
    affected_paths: dict[str, int] = {}
    for scenario in scenarios:
        if not scenario.failed:
            passed += 1
        direct += scenario.direct_affected
        transitive += scenario.transitive_affected
        items += scenario.total_affected_items
        files += scenario.total_affected_files
        if scenario.resolved_target is not None:
            resolved_targets += 1
        if scenario.error_kind is ErrorKind.UNKNOWN:
            unknown += 1
        elif scenario.error_kind is ErrorKind.AMBIGUOUS:
            ambiguous += 1
        for relationship in scenario.relationship_types:
            relationship_counts[relationship] = (
                relationship_counts.get(relationship, 0) + 1
            )
        for path in scenario.affected_paths:
            affected_paths[path] = affected_paths.get(path, 0) + 1
    total = len(scenarios)
    failed = total - passed
    summary = {
        "scenarios_total": total,
        "scenarios_passed": passed,
        "scenarios_failed": failed,
        "total_affected_items": items,
        "total_affected_files": files,
        "resolved_targets": resolved_targets,
        "unresolved_targets": unknown + ambiguous,
        "unknown_targets": unknown,
        "ambiguous_targets": ambiguous,
        "direct_affected": direct,
        "transitive_affected": transitive,
        "relationship_type_counts": dict(
            sorted(relationship_counts.items(), key=lambda kv: kv[0])
        ),
    }
    return ChangeSurfaceScenarioReport(
        scenarios=scenarios,
        deterministic_repeatable=deterministic_repeatable,
        repeatability_mismatch=repeatability_mismatch,
        summary=summary,
    )


def serialize_scenarios(
    scenarios: tuple[ScenarioResult, ...],
) -> tuple[tuple, ...]:
    """Canonical serialization used to prove cross-run repeatability."""
    return tuple(tuple(sorted(scenario.to_dict().items())) for scenario in scenarios)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _evidence_satisfied(
    by_signature: dict[tuple[str, str, str], object],
    check: ExpectedEvidence,
) -> bool:
    item = by_signature.get(
        (check.path, check.relationship.value, check.direction.value)
    )
    if item is None:
        return False
    if check.symbol is not None and item.symbol != check.symbol:
        return False
    return check.evidence_contains in item.evidence


def _hit_of(item) -> tuple[str, str, str]:
    path = item.path
    if hasattr(path, "as_posix"):
        path = path.as_posix()
    return (
        path,
        item.relationship.value,
        item.direction.value,
    )


def _target_label(targets: str | tuple[str, ...]) -> str:
    if isinstance(targets, tuple):
        return ",".join(targets)
    return targets


def _resolved_labels(result: ChangeSurfaceResult) -> str:
    return ",".join(target.display for target in result.targets)


def _resolved_kinds(result: ChangeSurfaceResult) -> tuple[str, ...]:
    return tuple(target.kind for target in result.targets)


def _relationship_types(result: ChangeSurfaceResult) -> tuple[str, ...]:
    return tuple(
        sorted({item.relationship.value for item in result.items})
        if result.items
        else ()
    )


def _posix_paths(paths) -> tuple[str, ...]:
    return tuple(path.as_posix() for path in paths)


def _is_deterministically_ordered(result: ChangeSurfaceResult) -> bool:
    items = result.items
    return all(item_sort_key(items[index]) <= item_sort_key(items[index + 1])
               for index in range(len(items) - 1))


def _classify_error(message: str) -> ErrorKind:
    if "Ambiguous" in message:
        return ErrorKind.AMBIGUOUS
    return ErrorKind.UNKNOWN