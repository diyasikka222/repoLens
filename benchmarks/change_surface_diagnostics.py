"""Change-surface diagnostics benchmark (P26.3, step 2).

Evaluates the deterministic :class:`~repolens.change_surface.ChangeSurfaceAnalyzer`
against a fixed catalog of repository change scenarios built in a throwaway
synthetic repository, so results are reproducible byte-for-byte on any machine.

Every scenario fixes a target (or several), a traversal direction, and a
maximum depth, and carries *explicit expected outcomes*: the relationships
that must be discovered (path + kind + direction + depth), stable evidence
tags those items must carry, and — for controlled scenarios — the exact
surfaced item set (so spurious relationships fail). Unknown and ambiguous
targets are asserted to resolve to the expected deterministic error instead
of crashing.

The benchmark runs the full catalog twice through freshly constructed
analyzers and reports whether the two runs are identical, proving
deterministic repeatability.

Measurement only: it never modifies retrieval, ranking, the context budget,
change-surface semantics, or the MCP contract, and makes no LLM/API calls.

Usage::

    python benchmarks/change_surface_diagnostics.py
    python benchmarks/change_surface_diagnostics.py --json out.json
    python benchmarks/change_surface_diagnostics.py .      # + real-repo smoke

Exit code is 0 when every scenario passes and the runs are repeatable, 1
otherwise.
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

# Keep ``python benchmarks/change_surface_diagnostics.py`` working: that form
# puts ``benchmarks/`` on sys.path but not the repo root.
REPO_ROOT = Path(__file__).resolve().parent.parent
if __package__ in (None, ""):
    sys.path.insert(0, str(REPO_ROOT))

from repolens.change_surface import (
    ChangeRelationship,
    ChangeSurfaceAnalyzer,
    ChangeSurfaceConfig,
    SurfaceDirection,
)
from repolens.change_surface_diagnostics import (
    ChangeScenario,
    ChangeSurfaceScenarioReport,
    ErrorKind,
    ExpectedEvidence,
    ExpectedHit,
    run_change_scenario,
    serialize_scenarios,
    summarize_report,
)

# ---------------------------------------------------------------------------
# Synthetic repository (deterministic scenario fixture)
# ---------------------------------------------------------------------------


def _write_file(root: Path, relative: str, source: str) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")


def build_synthetic_repo(root: Path) -> None:
    """Write the fixed scenario repository. Never executes the written code."""
    _write_file(root, "core/__init__.py", "")
    _write_file(
        root,
        "core/util.py",
        "class Base:\n"
        "    pass\n"
        "\n"
        "def helper():\n"
        "    return 1\n",
    )
    _write_file(
        root,
        "core/engine.py",
        "from core.util import Base, helper\n"
        "class Engine(Base):\n"
        "    def start(self):\n"
        "        return helper()\n"
        "def spin():\n"
        "    return 42\n",
    )
    _write_file(
        root,
        "core/gear.py",
        "from core.engine import Engine\n"
        "def use_engine():\n"
        "    e = Engine()\n"
        "    return e.start()\n",
    )
    _write_file(
        root,
        "core/top.py",
        "from core.gear import use_engine\n"
        "def run_all():\n"
        "    return use_engine()\n",
    )
    _write_file(
        root,
        "core/turbo.py",
        "from core.engine import Engine\n"
        "class Turbo(Engine):\n"
        "    pass\n",
    )
    _write_file(
        root,
        "core/runner.py",
        "from core.util import helper\n"
        "def run():\n"
        "    return helper()\n",
    )
    _write_file(
        root,
        "app/service.py",
        "from core.engine import Engine\n"
        "def make():\n"
        "    return Engine()\n",
    )
    _write_file(
        root,
        "app/launch.py",
        "from core.runner import run\n"
        "def launch():\n"
        "    return run()\n",
    )
    _write_file(
        root,
        "tests/test_engine.py",
        "from core.engine import Engine\n"
        "def test_engine():\n"
        "    assert Engine().start() == 'vroom'\n",
    )
    _write_file(root, "core/ring_a.py", "from core import ring_b\n")
    _write_file(root, "core/ring_b.py", "from core import ring_a\n")
    _write_file(root, "core/useless.py", "VALUE = 7\n")
    _write_file(root, "tools/tool.py", "def tool_a():\n    return 1\n")
    _write_file(root, "scripts/tool.py", "def tool_b():\n    return 2\n")


# ---------------------------------------------------------------------------
# Scenario catalog (explicit expected outcomes)
# ---------------------------------------------------------------------------

IMPORT = ChangeRelationship.IMPORT
CALL = ChangeRelationship.CALL
SYMREF = ChangeRelationship.SYMBOL_REFERENCE
INHERIT = ChangeRelationship.INHERITANCE
OTHER = ChangeRelationship.OTHER
IN = SurfaceDirection.INCOMING
OUT = SurfaceDirection.OUTGOING


def _hits(*specs: tuple) -> tuple[ExpectedHit, ...]:
    return tuple(
        ExpectedHit(path=spec[0], relationship=spec[1], direction=spec[2], depth=spec[3])
        for spec in specs
    )


def _evidence(*specs: tuple) -> tuple[ExpectedEvidence, ...]:
    return tuple(
        ExpectedEvidence(
            path=spec[0],
            relationship=spec[1],
            direction=spec[2],
            evidence_contains=spec[4],
            symbol=spec[3],
        )
        for spec in specs
    )


SCENARIOS: tuple[ChangeScenario, ...] = (
    ChangeScenario(
        id="file_level",
        description="file-level change surfaces the importing/imported surface",
        targets="core/engine.py",
        direction="both",
        max_depth=2,
        expected=_hits(
            ("app/service.py", IMPORT, IN, 1),
            ("core/gear.py", IMPORT, IN, 1),
            ("core/turbo.py", IMPORT, IN, 1),
            ("tests/test_engine.py", IMPORT, IN, 1),
            ("core/top.py", IMPORT, IN, 2),
            ("core/util.py", IMPORT, OUT, 1),
        ),
        exact=True,
    ),
    ChangeScenario(
        id="symbol_level",
        description="symbol-level change surfaces callers, references, bases",
        targets="Engine",
        direction="both",
        max_depth=2,
        expected=_hits(
            ("app/service.py", CALL, IN, 1),
            ("core/gear.py", CALL, IN, 1),
            ("tests/test_engine.py", CALL, IN, 1),
            ("core/top.py", CALL, IN, 2),
            ("app/service.py", IMPORT, IN, 1),
            ("core/gear.py", IMPORT, IN, 1),
            ("core/turbo.py", IMPORT, IN, 1),
            ("tests/test_engine.py", IMPORT, IN, 1),
            ("core/top.py", IMPORT, IN, 2),
            ("core/turbo.py", INHERIT, IN, 1),
            ("tests/test_engine.py", OTHER, IN, 1),
            ("app/service.py", SYMREF, IN, 1),
            ("core/gear.py", SYMREF, IN, 1),
            ("core/util.py", IMPORT, OUT, 1),
            ("core/util.py", INHERIT, OUT, 1),
        ),
        evidence_checks=_evidence(
            ("core/turbo.py", INHERIT, IN, "Engine", "base_class"),
            ("app/service.py", SYMREF, IN, "Engine", "symbol_import"),
        ),
        exact=True,
    ),
    ChangeScenario(
        id="multi_target",
        description="multi-target change merges and de-duplicates surfaces",
        targets=("core/engine.py", "core/runner.py"),
        direction="both",
        max_depth=2,
        expected=_hits(
            ("app/launch.py", IMPORT, IN, 1),
            ("app/service.py", IMPORT, IN, 1),
            ("core/gear.py", IMPORT, IN, 1),
            ("core/turbo.py", IMPORT, IN, 1),
            ("tests/test_engine.py", IMPORT, IN, 1),
            ("core/top.py", IMPORT, IN, 2),
            ("core/util.py", IMPORT, OUT, 1),
        ),
        exact=True,
    ),
    ChangeScenario(
        id="direct_caller_dependent",
        description="direct caller and dependent discovery for a symbol",
        targets="run",
        direction="both",
        max_depth=2,
        expected=_hits(
            ("app/launch.py", CALL, IN, 1),
            ("app/launch.py", IMPORT, IN, 1),
            ("app/launch.py", SYMREF, IN, 1),
            ("core/util.py", CALL, OUT, 1),
            ("core/util.py", IMPORT, OUT, 1),
        ),
        evidence_checks=_evidence(
            ("app/launch.py", SYMREF, IN, "run", "symbol_import"),
        ),
        exact=True,
    ),
    ChangeScenario(
        id="transitive_impact",
        description="transitive impact is depth-tagged (top.py at depth 2)",
        targets="core/engine.py",
        direction="incoming",
        max_depth=2,
        expected=_hits(
            ("app/service.py", IMPORT, IN, 1),
            ("core/gear.py", IMPORT, IN, 1),
            ("core/turbo.py", IMPORT, IN, 1),
            ("tests/test_engine.py", IMPORT, IN, 1),
            ("core/top.py", IMPORT, IN, 2),
        ),
        exact=True,
    ),
    ChangeScenario(
        id="import_dependency",
        description="import dependency discovered for a shared dependency",
        targets="core/util.py",
        direction="incoming",
        max_depth=1,
        expected=_hits(
            ("core/engine.py", IMPORT, IN, 1),
            ("core/runner.py", IMPORT, IN, 1),
        ),
        exact=True,
    ),
    ChangeScenario(
        id="call_dependency",
        description="outgoing call callee discovered; consumer refs stay incoming",
        targets="run",
        direction="outgoing",
        max_depth=2,
        expected=_hits(
            ("core/util.py", CALL, OUT, 1),
            ("core/util.py", IMPORT, OUT, 1),
            ("app/launch.py", SYMREF, IN, 1),
        ),
        exact=True,
    ),
    ChangeScenario(
        id="inheritance_dependency",
        description="inheritance discovered in both directions (subclass + base)",
        targets="Engine",
        direction="both",
        max_depth=2,
        expected=_hits(
            ("core/turbo.py", INHERIT, IN, 1),
            ("core/util.py", INHERIT, OUT, 1),
        ),
        evidence_checks=_evidence(
            ("core/turbo.py", INHERIT, IN, "Engine", "base_class"),
            ("core/util.py", INHERIT, OUT, "Base", "base_class"),
        ),
        exact=False,
    ),
    ChangeScenario(
        id="symbol_reference_dependency",
        description="symbol references separately observable from imports",
        targets="Engine",
        direction="both",
        max_depth=2,
        expected=_hits(
            ("app/service.py", SYMREF, IN, 1),
            ("core/gear.py", SYMREF, IN, 1),
        ),
        evidence_checks=_evidence(
            ("app/service.py", SYMREF, IN, "Engine", "symbol_import"),
        ),
        exact=False,
    ),
    ChangeScenario(
        id="cyclic_dependency",
        description="cyclic imports terminate with bounded, exact surface",
        targets="core/ring_a.py",
        direction="both",
        max_depth=2,
        expected=_hits(
            ("core/ring_b.py", IMPORT, IN, 1),
            ("core/ring_b.py", IMPORT, OUT, 1),
        ),
        exact=True,
    ),
    ChangeScenario(
        id="unknown_target",
        description="unknown target resolves to the documented error, no crash",
        targets="core/nope.py",
        expected=(),
        expect_error=ErrorKind.UNKNOWN,
    ),
    ChangeScenario(
        id="ambiguous_target",
        description="ambiguous module name resolves to the documented error",
        targets="tool",
        expected=(),
        expect_error=ErrorKind.AMBIGUOUS,
    ),
)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_catalog(root: Path) -> tuple:
    """Run every scenario through one fresh analyzer."""
    analyzer = ChangeSurfaceAnalyzer(root)
    return tuple(
        run_change_scenario(analyzer, scenario) for scenario in SCENARIOS
    )


def build_report(root: Path) -> ChangeSurfaceScenarioReport:
    """Run the catalog twice (fresh analyzers) and check repeatability."""
    first = run_catalog(root)
    second = run_catalog(root)
    repeatable = serialize_scenarios(first) == serialize_scenarios(second)
    mismatch = None if repeatable else "scenario outputs differ between runs"
    return summarize_report(
        first,
        deterministic_repeatable=repeatable,
        repeatability_mismatch=mismatch,
    )


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------


def print_report(report: ChangeSurfaceScenarioReport) -> None:
    print(
        "change-surface diagnostics (P26.3): measurement only, no retrieval/"
        "ranking/budget/MCP changes"
    )
    print()
    for index, scenario in enumerate(report.scenarios, start=1):
        label = f"[{index:02d}] {scenario.id}"
        title = f"{scenario.target} {scenario.direction} depth={scenario.max_depth}"
        print(f"{label}  {scenario.description}")
        print(f"  {title}")
        if scenario.resolved_target is None:
            kind = scenario.error_kind.value if scenario.error_kind else "-"
            print(f"  unresolved: kind={kind} error={scenario.error_message}")
        else:
            kinds = ",".join(scenario.resolved_kinds)
            rels = " ".join(sorted(scenario.relationship_types)) or "-"
            print(
                f"  resolved: {scenario.resolved_target} ({kinds})"
                f" | direct={scenario.direct_affected}"
                f" transitive={scenario.transitive_affected}"
                f" files={scenario.total_affected_files}"
                f" items={scenario.total_affected_items}"
                f" rels={rels}"
            )
            print(f"  affected: {' '.join(scenario.affected_paths) or '-'}")
        if scenario.missing_expected:
            print(
                "  MISSING EXPECTED: "
                + "; ".join(
                    f"{hit.path}::{hit.relationship.value}/{hit.direction.value}"
                    for hit in scenario.missing_expected
                )
            )
        if scenario.missing_evidence:
            print(
                "  MISSING EVIDENCE: "
                + "; ".join(
                    f"{check.path}::{check.relationship.value}/{check.direction.value}"
                    f" lacks {check.evidence_contains!r}"
                    for check in scenario.missing_evidence
                )
            )
        if scenario.unexpected:
            unexpected = " ".join(
                f"{key[0]}::{key[1]}/{key[2]}" for key in scenario.unexpected
            )
            print("  UNEXPECTED: " + unexpected)
        verdict = "PASS" if not scenario.failed else "FAIL"
        order = f" deterministic_order={scenario.deterministic_order}" if scenario.resolved_target else ""
        print(f"  -> {verdict}{order}")
        print()

    summary = report.summary
    print("SUMMARY")
    print(
        f"  scenarios: {summary['scenarios_total']} total, "
        f"{summary['scenarios_passed']} passed, {summary['scenarios_failed']} failed"
    )
    print(
        f"  affected: items={summary['total_affected_items']} "
        f"files={summary['total_affected_files']} "
        f"direct={summary['direct_affected']} "
        f"transitive={summary['transitive_affected']}"
    )
    rel_line = " ".join(
        f"{kind}={count}"
        for kind, count in summary["relationship_type_counts"].items()
    )
    print(f"  relationship-type counts: {rel_line or '-'}")
    print(
        f"  targets: resolved={summary['resolved_targets']} "
        f"unresolved={summary['unresolved_targets']} "
        f"(unknown={summary['unknown_targets']} ambiguous={summary['ambiguous_targets']})"
    )
    repeat = (
        "OK before/after (identical outputs across fresh analyzers)"
        if report.deterministic_repeatable
        else f"MISMATCH {report.repeatability_mismatch}"
    )
    print(f"  deterministic repeatability: {repeat}")


# ---------------------------------------------------------------------------
# Real-repository smoke (optional, e.g. ``python ... .``)
# ---------------------------------------------------------------------------


def real_repo_smoke(root: Path) -> None:
    """Run two deterministic samples against a real repository (no exact
    expectations; proves the scenario machinery works at repo scale)."""
    print("REAL-REPOSITORY SMOKE")
    analyzer = ChangeSurfaceAnalyzer(root)
    for target in ("repolens/change_surface.py", "repolens/incremental_index.py"):
        first = analyzer.analyze(target, max_depth=2)
        second = analyzer.analyze(target, max_depth=2)
        assert serialize(first) == serialize(second), (
            f"real-repo analysis not deterministic for {target}"
        )
        assert first.items, f"expected a non-empty surface for {target}"
        kinds = ",".join(sorted({i.relationship.value for i in first.items})) or "-"
        print(
            f"  {target}: resolved={len(first.targets)} items={len(first.items)}"
            f" files={len(first.files)} rels={kinds} deterministic=OK"
        )
    print()


def serialize(result) -> list[tuple]:
    return [
        (item.path.as_posix(), item.relationship.value, item.direction.value,
         item.depth, item.symbol or "")
        for item in result.items
    ]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    json_path: Path | None = None
    real_root: Path | None = None
    args = list(argv)
    if "--json" in args:
        index = args.index("--json")
        json_path = Path(args.pop(index + 1))
        args.pop(index)
    positional = [arg for arg in args if not arg.startswith("-")]
    if positional:
        real_root = Path(positional[0])

    with tempfile.TemporaryDirectory(prefix="repolens-surfdiag-") as tmp:
        root = Path(tmp) / "repo"
        build_synthetic_repo(root)
        started = time.perf_counter()
        report = build_report(root)
        elapsed = time.perf_counter() - started
        print_report(report)
        if real_root is not None and real_root.is_dir():
            print()
            real_repo_smoke(real_root)
        if json_path is not None:
            json_path.write_text(
                json.dumps(report.to_dict(), indent=2),
                encoding="utf-8",
            )
            print(f"json report written to {json_path}")
        print()
        print(f"change-surface diagnostics benchmark finished in {elapsed:.2f}s")

    failed = report.summary["scenarios_failed"] or not report.deterministic_repeatable
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())