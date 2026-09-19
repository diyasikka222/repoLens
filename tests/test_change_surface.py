"""Offline tests for deterministic change-surface reasoning (Milestone 23).

Exercises :class:`ChangeSurfaceAnalyzer`: single-file / single-symbol /
multi-target analysis, direct vs transitive classification, both traversal
directions with relationship metadata (imports, calls, symbol references,
inheritance, other), cycle safety, missing targets, determinism, and stable
ordering.

Fully offline and deterministic: no network, no embeddings, no LLM calls.
"""

from __future__ import annotations

import pytest

from repolens.change_surface import (
    ChangeRelationship,
    ChangeSurfaceAnalyzer,
    ChangeSurfaceConfig,
    SurfaceDirection,
)
from repolens.impact import ImpactTargetError


def write_file(repo, relative: str, source: str) -> None:
    p = repo / relative
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(source, encoding="utf-8")


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    write_file(root, "core/__init__.py", "")
    write_file(
        root,
        "core/util.py",
        "class Base:\n"
        "    pass\n"
        "def helper():\n"
        "    return 1\n",
    )
    write_file(
        root,
        "core/engine.py",
        "from core.util import Base, helper\n"
        "class Engine(Base):\n"
        "    def start(self):\n"
        "        return helper()\n"
        "def spin():\n"
        "    return 42\n",
    )
    write_file(
        root,
        "core/gear.py",
        "from core.engine import Engine\n"
        "def use_engine():\n"
        "    e = Engine()\n"
        "    return e.start()\n",
    )
    write_file(
        root,
        "core/top.py",
        "from core.gear import use_engine\n"
        "def run_all():\n"
        "    return use_engine()\n",
    )
    write_file(
        root,
        "core/turbo.py",
        "from core.engine import Engine\n"
        "class Turbo(Engine):\n"
        "    pass\n",
    )
    write_file(
        root,
        "core/runner.py",
        "from core.util import helper\n"
        "def run():\n"
        "    return helper()\n",
    )
    write_file(
        root,
        "app/service.py",
        "from core.engine import Engine\n"
        "def make():\n"
        "    return Engine()\n",
    )
    write_file(
        root,
        "app/launch.py",
        "from core.runner import run\n"
        "def launch():\n"
        "    return run()\n",
    )
    write_file(
        root,
        "tests/test_engine.py",
        "from core.engine import Engine\n"
        "def test_engine():\n"
        "    assert Engine().start() == 'vroom'\n",
    )
    write_file(root, "core/ring_a.py", "from core import ring_b\n")
    write_file(root, "core/ring_b.py", "from core import ring_a\n")
    write_file(root, "core/useless.py", "VALUE = 7\n")
    return root


@pytest.fixture
def analyzer(repo) -> ChangeSurfaceAnalyzer:
    return ChangeSurfaceAnalyzer(repo)


def surface_item_of(result, path, relationship=None, direction=None):
    for item in result.items:
        if item.path.as_posix() == path:
            if relationship is not None and item.relationship is not relationship:
                continue
            if direction is not None and item.direction is not direction:
                continue
            return item
    return None


def serialize_result(result) -> list[tuple]:
    return [
        (
            item.path.as_posix(),
            item.relationship.value,
            item.direction.value,
            item.depth,
            item.symbol or "",
            tuple(sorted(item.evidence)),
        )
        for item in result.items
    ]


# ---------------------------------------------------------------------------
# Single-file target
# ---------------------------------------------------------------------------


def test_single_file_target(repo, analyzer) -> None:
    result = analyzer.analyze("core/engine.py")
    gear = surface_item_of(
        result, "core/gear.py", ChangeRelationship.IMPORT, SurfaceDirection.INCOMING
    )
    assert gear is not None and gear.depth == 1
    util = surface_item_of(
        result, "core/util.py", ChangeRelationship.IMPORT, SurfaceDirection.OUTGOING
    )
    assert util is not None and util.depth == 1
    assert "core/top.py" in {p.as_posix() for p in result.files}


def test_direct_and_transitive_affects(repo, analyzer) -> None:
    result = analyzer.analyze("core/engine.py")
    top = surface_item_of(
        result, "core/top.py", ChangeRelationship.IMPORT, SurfaceDirection.INCOMING
    )
    assert top is not None and top.depth == 2
    direct = {i.path.as_posix() for i in result.directly_affected}
    transitive = {i.path.as_posix() for i in result.transitively_affected}
    assert "core/gear.py" in direct
    assert "core/top.py" in transitive
    assert direct.isdisjoint(transitive)


def test_max_depth_one_caps_transitive(repo, analyzer) -> None:
    result = analyzer.analyze("core/engine.py", max_depth=1)
    assert surface_item_of(result, "core/gear.py") is not None
    assert surface_item_of(result, "core/top.py") is None


def test_max_depth_zero_disables_traversal(repo, analyzer) -> None:
    result = analyzer.analyze("core/engine.py", max_depth=0)
    assert result.items == ()


def test_direction_outgoing_only(repo, analyzer) -> None:
    result = analyzer.analyze("core/engine.py", direction="outgoing")
    assert result.items
    assert all(item.direction is SurfaceDirection.OUTGOING for item in result.items)
    assert surface_item_of(result, "core/util.py") is not None
    assert surface_item_of(result, "core/gear.py") is None


def test_direction_incoming_only(repo, analyzer) -> None:
    result = analyzer.analyze("core/engine.py", direction="incoming")
    assert result.items
    assert all(item.direction is SurfaceDirection.INCOMING for item in result.items)
    assert surface_item_of(result, "core/gear.py") is not None
    assert surface_item_of(result, "core/util.py") is None


# ---------------------------------------------------------------------------
# Single-symbol target
# ---------------------------------------------------------------------------


def test_symbol_target_reports_symbol_references(repo, analyzer) -> None:
    result = analyzer.analyze("Engine")
    service = surface_item_of(
        result,
        "app/service.py",
        ChangeRelationship.SYMBOL_REFERENCE,
        SurfaceDirection.INCOMING,
    )
    assert service is not None
    assert service.symbol == "Engine"


def test_symbol_target_reports_inheritance(repo, analyzer) -> None:
    result = analyzer.analyze("Engine")
    turbo = surface_item_of(
        result,
        "core/turbo.py",
        ChangeRelationship.INHERITANCE,
        SurfaceDirection.INCOMING,
    )
    assert turbo is not None
    assert turbo.symbol == "Engine"
    assert "base_class" in turbo.evidence


def test_class_target_reports_outgoing_inheritance(repo, analyzer) -> None:
    result = analyzer.analyze("Engine")
    base = surface_item_of(
        result,
        "core/util.py",
        ChangeRelationship.INHERITANCE,
        SurfaceDirection.OUTGOING,
    )
    assert base is not None
    assert base.symbol == "Base"
    assert base.depth == 1


def test_symbol_target_reports_caller_and_callee(repo, analyzer) -> None:
    result = analyzer.analyze("run")
    caller = surface_item_of(
        result,
        "app/launch.py",
        ChangeRelationship.CALL,
        SurfaceDirection.INCOMING,
    )
    assert caller is not None
    assert caller.depth == 1
    assert caller.symbol == "launch"
    callee = surface_item_of(
        result,
        "core/util.py",
        ChangeRelationship.CALL,
        SurfaceDirection.OUTGOING,
    )
    assert callee is not None
    assert callee.depth == 1
    assert callee.symbol == "helper"


def test_indirect_caller_depth(repo, analyzer) -> None:
    result = analyzer.analyze("Engine", max_depth=2)
    top = surface_item_of(
        result, "core/top.py", ChangeRelationship.CALL, SurfaceDirection.INCOMING
    )
    assert top is not None and top.depth == 2


# ---------------------------------------------------------------------------
# Multi-target
# ---------------------------------------------------------------------------


def test_multi_target_merges_and_deduplicates(repo, analyzer) -> None:
    result = analyzer.analyze(["core/engine.py", "core/runner.py"])
    assert len(result.targets) == 2
    assert surface_item_of(result, "core/gear.py") is not None
    assert surface_item_of(result, "app/launch.py") is not None


def test_duplicate_targets_deduplicated(repo, analyzer) -> None:
    result = analyzer.analyze(["core/engine.py", "core/engine.py"])
    assert len(result.targets) == 1
    single = analyzer.analyze("core/engine.py")
    assert serialize_result(result) == serialize_result(single)


# ---------------------------------------------------------------------------
# Cycles, errors, ordering
# ---------------------------------------------------------------------------


def test_cycle_safe(repo, analyzer) -> None:
    result = analyzer.analyze("core/ring_a.py")
    ring_b = surface_item_of(
        result, "core/ring_b.py", ChangeRelationship.IMPORT, SurfaceDirection.INCOMING
    )
    assert ring_b is not None and ring_b.depth == 1
    assert surface_item_of(result, "core/ring_a.py") is None


def test_missing_target_raises(repo, analyzer) -> None:
    with pytest.raises(ImpactTargetError):
        analyzer.analyze("core/nope.py")


def test_unknown_symbol_raises(repo, analyzer) -> None:
    with pytest.raises(ImpactTargetError):
        analyzer.analyze("MissingSymbol")


def test_stable_ordering(repo, analyzer) -> None:
    result = analyzer.analyze("Engine")
    keys = [
        (item.direction.value, item.relationship.value, item.depth)
        for item in result.items
    ]
    assert keys == sorted(keys)


def test_deterministic_across_runs(repo, analyzer) -> None:
    first = analyzer.analyze("Engine")
    second = analyzer.analyze("Engine")
    assert serialize_result(first) == serialize_result(second)
    assert first.to_json(sort_keys=True) == second.to_json(sort_keys=True)


def test_deterministic_across_instances(repo) -> None:
    one = ChangeSurfaceAnalyzer(repo).analyze("core/engine.py")
    two = ChangeSurfaceAnalyzer(repo).analyze("core/engine.py")
    assert serialize_result(one) == serialize_result(two)


# ---------------------------------------------------------------------------
# Configuration and result model
# ---------------------------------------------------------------------------


def test_limit_caps_items(repo, analyzer) -> None:
    result = analyzer.analyze("core/engine.py", limit=2)
    assert len(result.items) == 2


def test_config_disables_calls(repo) -> None:
    analyzer = ChangeSurfaceAnalyzer(
        repo, config=ChangeSurfaceConfig(include_calls=False)
    )
    result = analyzer.analyze("Engine")
    assert all(
        item.relationship is not ChangeRelationship.CALL for item in result.items
    )


def test_result_summary_and_metadata(repo, analyzer) -> None:
    result = analyzer.analyze("core/engine.py")
    assert result.summary["targets"] == 1
    assert result.summary["direct_affected"] > 0
    assert all(item.reason for item in result.items)
    assert all(item.module for item in result.items)
    assert result.max_depth_reached >= 1
    assert result.summary["by_relationship"].get("import", 0) > 0