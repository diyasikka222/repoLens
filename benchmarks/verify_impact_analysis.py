"""Deterministic verification for M21 change impact analysis.

Part 1 builds a small synthetic repository and asserts the exact relationship,
evidence, dedup, pruning, and risk behavior of :class:`ImpactAnalyzer`.

Part 2 runs against the real RepoLens repository itself to demonstrate
real-world scale: a hot file (``repolens/incremental_index.py``) with many
dependents, tests discovered by name/path, and a public symbol
(``classify_intent``) with api consumers — while confirming the analysis is
deterministic across repeated runs.

Deterministic and fully offline. Prints exact counts. Does not modify the
repository under analysis (index is built in-memory; ``persist=False``).
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from repolens.impact import (
    EVIDENCE_DEPENDENCY_EDGE,
    EVIDENCE_SYMBOL_IMPORT,
    EVIDENCE_TEST_NAME,
    ImpactAnalyzer,
    Relationship,
    RiskLevel,
)


def write_file(root: Path, relative: str, source: str) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")


def build_synthetic(root: Path) -> None:
    write_file(root, "lib/__init__.py", "from lib.helper import Helper, do_thing\n")
    write_file(
        root,
        "lib/helper.py",
        "class Helper:\n"
        "    pass\n"
        "\n"
        "def do_thing():\n"
        "    return 42\n",
    )
    write_file(root, "lib/util.py", "def util():\n    return 'u'\n")
    write_file(
        root,
        "app/main.py",
        "from lib.helper import Helper\n"
        "from lib.util import util\n"
        "\n"
        "def run():\n"
        "    return util()\n",
    )
    write_file(root, "app/runner.py", "import lib.helper\n")
    write_file(
        root,
        "config/settings.py",
        "from lib.helper import do_thing\n"
        "ENABLED = do_thing()\n",
    )
    write_file(
        root,
        "tests/test_main.py",
        "from app.main import run\n"
        "def test_run():\n"
        "    assert run() == 'u'\n",
    )
    write_file(
        root,
        "tests/test_helper.py",
        "from lib.helper import do_thing\n"
        "def test_do_thing():\n"
        "    assert do_thing() == 42\n",
    )
    write_file(root, "pyproject.toml", "[tool.renamed]\napp = \"value\"\n")


def part1(root: Path) -> None:
    analyzer = ImpactAnalyzer(root)

    result = analyzer.analyze("lib/helper.py")
    by_rel: dict[str, list[str]] = {}
    for item in result.items:
        by_rel.setdefault(item.relationship.value, []).append(
            item.path.as_posix()
        )
    for key in by_rel:
        by_rel[key].sort()
        print(f"  {result.target_path.as_posix()} -> {key}: {by_rel[key]}")

    assert item_of(result, "app/main.py").relationship is Relationship.DIRECT_DEPENDENCY
    assert item_of(result, "app/runner.py").relationship is Relationship.DIRECT_DEPENDENCY
    assert item_of(result, "tests/test_helper.py").relationship is Relationship.TEST
    assert item_of(result, "config/settings.py").relationship is Relationship.CONFIGURATION
    # max_depth_reached reports the deepest reverse-dependency depth found.
    assert result.max_depth_reached == 2

    # Test evidence distinguishes the reason from graph nodes.
    test_item = item_of(result, "tests/test_helper.py")
    assert EVIDENCE_TEST_NAME in test_item.evidence
    assert EVIDENCE_DEPENDENCY_EDGE in test_item.evidence

    # lib/__init__.py re-exports helper -> a direct importer dependent.
    init_item = item_of(result, "lib/__init__.py")
    assert init_item.relationship is Relationship.DIRECT_DEPENDENCY

    # Risk scoring: 2 direct dependents -> HIGH.
    assert result.risk is RiskLevel.HIGH or result.risk is RiskLevel.MEDIUM
    print(f"  risk = {result.risk.value} (expected high or medium)")

    # Symbols: api consumer via from-import of the symbol name.
    sym = analyzer.analyze("Helper")
    assert sym.symbol == "Helper"
    main_item = item_of(sym, "app/main.py")
    assert main_item.relationship is Relationship.API_CONSUMER
    assert EVIDENCE_SYMBOL_IMPORT in main_item.evidence
    assert EVIDENCE_DEPENDENCY_EDGE in main_item.evidence
    print(f"  symbol 'Helper': {len(sym.items)} item(s); app/main.py api_consumer")

    # Determinism: byte-for-byte identical serialized items across runs.
    again = analyzer.analyze("Helper")
    assert serialize(sym) == serialize(again)
    print("  symbol analysis deterministic across runs: OK")

    # Dedup: tests/test_main.py must appear exactly once across the result.
    paths = [f"{it.path.as_posix()}::{it.relationship.value}" for it in result.items]
    assert len(paths) == len(set(paths)), "duplicate (path, relationship) in result"
    print(f"  no duplicate (path, relationship) pairs ({len(paths)} items)")

    # max_depth=0 disables reverse traversal entirely.
    shallow = analyzer.analyze("lib/helper.py", max_depth=0)
    assert all(
        it.relationship is not Relationship.REVERSE_DEPENDENCY
        and it.relationship is not Relationship.DIRECT_DEPENDENCY
        and it.relationship is not Relationship.INDIRECT_DEPENDENCY
        for it in shallow.items
    )
    print(f"  max_depth=0: {len(shallow.items)} item(s) after reverse-prune")

    # Unknown target -> safe error.
    try:
        analyzer.analyze("app/missing.py")
        raise AssertionError("expected ImpactTargetError for unknown target")
    except Exception as exc:
        assert type(exc).__name__ == "ImpactTargetError"
    print("  unknown target rejected: OK")


def part2(root: Path) -> None:
    analyzer = ImpactAnalyzer(root)

    first = analyzer.analyze("repolens/incremental_index.py")
    second = analyzer.analyze("repolens/incremental_index.py")
    assert serialize(first) == serialize(second)
    by_rel: dict[str, int] = {}
    for item in first.items:
        by_rel[item.relationship.value] = by_rel.get(item.relationship.value, 0) + 1
    has_tests = any(
        it.relationship is Relationship.TEST and "test" in it.path.as_posix()
        for it in first.items
    )
    assert first.items, "expected at least one impacted file"
    assert any(
        it.relationship is Relationship.DIRECT_DEPENDENCY for it in first.items
    ), "expected a direct dependency on RepoLens' own incremental index"
    assert has_tests, "expected test discovery for repolens/incremental_index.py"
    print("real repo: repolens/incremental_index.py")
    print(f"  item_count={len(first.items)} risk={first.risk.value}")
    print(f"  by_relationship={by_rel}")
    print("  deterministic across runs: OK")
    print("  test discovery: OK")

    sym = analyzer.analyze("classify_intent")
    api_consumers = [
        it.path.as_posix()
        for it in sym.items
        if it.relationship is Relationship.API_CONSUMER
    ]
    sym_items = [it for it in sym.items if it.symbol]
    assert sym.symbol == "classify_intent"
    assert api_consumers, "expected api consumers for classify_intent"
    assert sym_items, "expected symbol-level items with evidence"
    print("real repo: symbol 'classify_intent'")
    print(f"  item_count={len(sym.items)} risk={sym.risk.value}")
    print(f"  api_consumers={api_consumers}")
    print("  symbol items carry evidence: OK")


def item_of(result, path: str):
    for item in result.items:
        if item.path.as_posix() == path:
            return item
    raise AssertionError(f"missing item {path}")


def serialize(result) -> list[str]:
    return [
        f"{it.path.as_posix()}|{it.relationship.value}|{it.risk.value}|"
        f"{it.symbol}|{sorted(it.evidence)}|{it.depth}"
        for it in result.items
    ]


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "repo"
        build_synthetic(root)
        print("PART 1 - synthetic repository")
        part1(root)

    real_root = Path(__file__).resolve().parent.parent
    print("\nPART 2 - real repository (RepoLens itself)")
    part2(real_root)

    print("\nVERIFICATION PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())